"""Bounded CIDDS cohort qualification and partition materialisation; no model training."""
from pathlib import Path
import json,hashlib,zipfile,io,gzip,shutil,platform,time,itertools
from datetime import datetime,timezone
from collections import Counter
import numpy as np
import pandas as pd
import projection
STAGE='supplementary_cidds_partitions_20261027'
PARENT='supplementary_external_data_inventory_20261027/run_v1'
SEEDS=[2027,2028,2029,2030,2031]
WINDOW_CAP=1000
CHUNK=250000
CHECKPOINT_ROWS=1000000
FEATURES=['duration_native','protocol_is_tcp','destination_port','unidirectional_packets']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def add(d,v):
 for k,n in v.items():d[str(k)]=d.get(str(k),0)+int(n)
def save_gzip(p,v,local):
 p=Path(p);local=Path(local);local.mkdir(parents=True,exist_ok=True);temp=local/(p.name+'.writing')
 with gzip.open(temp,'wt',encoding='utf-8',compresslevel=1) as f:json.dump(v,f,separators=(',',':'),allow_nan=False)
 p.parent.mkdir(parents=True,exist_ok=True);dst=p.with_suffix(p.suffix+'.tmp');shutil.copyfile(temp,dst);dst.replace(p);temp.unlink()
def read_gzip(p):
 with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
def signature(v):return hashlib.sha256(json.dumps([float(x).hex() for x in v],separators=(',',':')).encode()).hexdigest()
def checked_zip(data):
 z=zipfile.ZipFile(io.BytesIO(data));m=json.loads(z.read('SHA256SUMS.json'))
 for n,h in m.items():
  if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Report checksum mismatch: '+n)
 return z

def update_pool(pool,frame,start,did):
 x,y,block,family,valid,notes=projection.project(frame,'cidds',did);x[x==0]=0.0
 good=valid&(y>=0)&(block!='__INVALID__');ix=np.flatnonzero(good)
 if not len(ix):return notes
 f=pd.DataFrame(x[ix],columns=['a','b','c','d']);f['label']=y[ix];f['block']=block[ix];f['family']=family[ix];f['row']=np.arange(start,start+len(frame),dtype=np.int64)[ix]
 # Float64 equality defines the candidate model representation, not physical event identity.
 f=f.drop_duplicates(['a','b','c','d','label','block'],keep='first')
 f['priority']=pd.util.hash_pandas_object(f[['a','b','c','d']],index=False,hash_key='0123456789abcdef',categorize=False).to_numpy(dtype=np.uint64)
 for (b,c),g in f.groupby(['block','label'],sort=False):
  if len(g)>WINDOW_CAP:
   cutoff=np.partition(g['priority'].to_numpy(dtype=np.uint64),WINDOW_CAP-1)[WINDOW_CAP-1];g=g.loc[g['priority']<=cutoff]
  key=b+' | '+str(c);old={r[0]:r for r in pool.get(key,[])}
  for row in g.itertuples(index=False):
   vals=[row.a,row.b,row.c,row.d];h=signature(vals);r=[h,int(row.priority),int(row.row),*map(float,vals),str(b),str(row.family),int(c)]
   if h not in old or r[2]<old[h][2]:old[h]=r
  pool[key]=sorted(old.values(),key=lambda r:(r[1],r[0]))[:WINDOW_CAP]
 return notes

def build_cohort(open_file,meta,identity,out,local):
 out=Path(out);mark=out.with_suffix('.complete.json');cp=out.with_suffix('.checkpoint.gz')
 if mark.exists():
  m=read(mark)
  if m['identity']!=identity or m['input']!=meta or sha(out)!=m['sha256']:raise ValueError('Completed cohort changed')
  return read_gzip(out)
 s=read_gzip(cp) if cp.exists() else {'identity':identity,'input':meta,'rows':0,'pool':{},'counts':{}}
 if s['identity']!=identity or s['input']!=meta:raise ValueError('Cohort resume identity changed')
 completed=s['rows'];seen=0;last_save=completed
 with open_file() as f:
  for frame in pd.read_csv(f,dtype=str,keep_default_na=False,encoding='utf-8-sig',on_bad_lines='error',chunksize=CHUNK):
   if seen+len(frame)<=completed:seen+=len(frame);continue
   if seen!=s['rows']:raise ValueError('Resume chunk boundary differs')
   notes=update_pool(s['pool'],frame,seen,meta['dataset']);notes.pop('protocol_counts',None);add(s['counts'],notes);s['rows']+=len(frame);seen=s['rows'];print(meta['capture'],f"{seen:,}/{meta['expected_rows']:,} rows",flush=True)
   if seen-last_save>=CHECKPOINT_ROWS:save_gzip(cp,s,local);last_save=seen
 if seen!=meta['expected_rows'] or s['rows']!=meta['expected_rows']:raise ValueError('Raw row count differs')
 save_gzip(out,s,local);write(mark,{'identity':identity,'input':meta,'sha256':sha(out)});return s

def reduce_cohorts(cohorts):
 masks={};rawcounts=[]
 for s in cohorts:
  for group in s['pool'].values():
   for r in group:masks[r[0]]=masks.get(r[0],0)|(1<<r[9])
 conflicts={h for h,v in masks.items() if v==3};reduced=[]
 for s in cohorts:
  first={};sample_count=sum(len(g) for g in s['pool'].values());removed=0
  for group in s['pool'].values():
   for r in group:
    if r[0] in conflicts:removed+=1;continue
    if r[0] not in first or r[2]<first[r[0]][2]:first[r[0]]=r
  rows=sorted(first.values(),key=lambda r:r[0]);reduced.append(rows);rawcounts.append({'capture':s['input']['capture'],'raw_rows':s['rows'],'window_class_sample_occurrences':sample_count,'selected_conflicting_occurrences_removed':removed,'retained_unique_candidate_groups':len(rows),'retained_binary_counts':dict(Counter(str(r[9]) for r in rows))})
 return reduced,conflicts,rawcounts

def role(h,seed):
 v=int(hashlib.sha256((str(seed)+'|'+h).encode()).hexdigest()[:16],16)%10000
 return 'train' if v<3000 else 'calibration' if v<4000 else 'test' if v<5000 else 'environment'
def rank(h,seed,tag):return hashlib.sha256((str(seed)+'|'+tag+'|'+h).encode()).hexdigest()
def balanced(rows,seed,tag,cap):
 out=[]
 for c in [0,1]:out+=sorted([r for r in rows if r[9]==c],key=lambda r:rank(r[0],seed,tag))[:cap]
 return out

def make_partitions(rows,seed,capture):
 roles={k:[] for k in ['train','calibration','test','environment']}
 for r in rows:roles[role(r[0],seed)].append(r)
 selected={k:balanced(roles[k],seed,k,cap) for k,cap in [('train',5000),('calibration',1000),('test',1000)]}
 groups={}
 for r in roles['environment']:groups.setdefault(r[7],[]).append(r)
 env=[];env_names=[];env_class_counts=[]
 for name in sorted(groups):
  ct=Counter(r[9] for r in groups[name])
  if min(ct.get(0,0),ct.get(1,0))<50:continue
  chosen=balanced(groups[name],seed,'environment|'+name,100);env+=chosen;env_names.append(name);env_class_counts.append([sum(r[9]==c for r in chosen) for c in [0,1]])
 selected['environment']=env
 counts={k:[sum(r[9]==c for r in v) for c in [0,1]] for k,v in selected.items()};reasons=[]
 for k,minimum in [('train',500),('calibration',100),('test',100)]:
  if min(counts[k])<minimum:reasons.append(k+'_below_'+str(minimum)+'_per_class')
 if len(env_names)<4:reasons.append('fewer_than_four_environment_windows_with_50_per_class')
 source_ok=not reasons;target_ok=min(counts['test'])>=100
 intersections=[]
 for a,b in itertools.combinations(selected,2):
  n=len({r[0] for r in selected[a]}&{r[0] for r in selected[b]});intersections.append({'left':a,'right':b,'shared_signatures':n})
  if n:raise ValueError('Within-capture role overlap')
 if len({r[0] for r in env})!=len(env):raise ValueError('Environment groups share signatures')
 # Match the environmental group label sizes exactly with disjoint shuffled pseudo-groups.
 pseudo=[];pseudo_gid=[]
 for c in [0,1]:
  ordered=sorted([r for r in env if r[9]==c],key=lambda r:rank(r[0],seed,'pseudo|'+str(c)));pos=0
  for g,ct in enumerate(env_class_counts):
   n=ct[c];pseudo+=ordered[pos:pos+n];pseudo_gid += [g]*n;pos+=n
  if pos!=len(ordered):raise ValueError('Pseudo-group allocation mismatch')
 gid={b:i for i,b in enumerate(env_names)}
 def arrays(prefix,rr):
  return {prefix+'_X':np.asarray([r[3:7] for r in rr],dtype=np.float64).reshape(-1,4),prefix+'_y':np.asarray([r[9] for r in rr],dtype=np.int8),prefix+'_signature':np.asarray([r[0] for r in rr],dtype='S64'),prefix+'_block':np.asarray([r[7] for r in rr],dtype='U19'),prefix+'_family':np.asarray([r[8] for r in rr],dtype=str),prefix+'_raw_row':np.asarray([r[2] for r in rr],dtype=np.int64)}
 payload={'feature_names':np.asarray(FEATURES,dtype=str)}
 for k,rr in selected.items():payload.update(arrays(k,rr))
 payload['environment_group']=np.asarray([gid[r[7]] for r in env],dtype=np.int32);payload.update(arrays('pseudo',pseudo));payload['pseudo_group']=np.asarray(pseudo_gid,dtype=np.int32)
 if set(payload['pseudo_signature'])!=set(payload['environment_signature']):raise ValueError('Pseudo/environment pool mismatch')
 record={'capture':capture,'seed':seed,'source_eligible':source_ok,'target_eligible':target_ok,'failure_reasons':reasons,'role_counts_per_class':counts,'environment_groups':len(env_names),'environment_group_names':env_names,'environment_group_class_counts':env_class_counts,'within_role_intersections':intersections,'global_role_hash_applied':True,'cohort_includes_target_informed_conflict_exclusion':True}
 return payload,record,selected

def save_npz(path,payload):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**payload);tmp.replace(path)
def verify_npz(path):
 with np.load(path,allow_pickle=False) as a:
  assert a['feature_names'].tolist()==FEATURES
  for r in ['train','calibration','test','environment','pseudo']:
   x=a[r+'_X'];y=a[r+'_y'];h=a[r+'_signature'];assert x.shape==(len(y),4) and np.isfinite(x).all() and set(y.tolist())<={0,1}
   assert len(set(h.tolist()))==len(h)
   assert all(signature(v)==s.decode() for v,s in zip(x,h))
  for a1,b1 in itertools.combinations(['train','calibration','test','environment'],2):assert not set(a[a1+'_signature'])&set(a[b1+'_signature'])
  assert set(a['environment_signature'])==set(a['pseudo_signature'])
  for gid in np.unique(a['environment_group']):
   assert np.bincount(a['environment_y'][a['environment_group']==gid],minlength=2).tolist()==np.bincount(a['pseudo_y'][a['pseudo_group']==gid],minlength=2).tolist()
 return True

def run(root,bundle,local_root='/content/paper3_cidds_partitions'):
 root=Path(root);bundle=Path(bundle);local=Path(local_root);local.mkdir(parents=True,exist_ok=True);remote=root/STAGE/'run_v1';parent=root/PARENT;start=time.monotonic();errors=[];cohorts=[];records=[];output_refs=[];cross=[];reduction=[];conflicts=set()
 identity={'engine_sha256':sha(bundle/'engine.py'),'projection_sha256':sha(bundle/'projection.py'),'qualification_report_sha256':sha(bundle/'qualification_report.zip'),'plan_sha256':sha(bundle/'PLAN.md'),'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'seeds':SEEDS,'window_class_cap':WINDOW_CAP,'chunk_rows':CHUNK}
 try:
  remote.mkdir(parents=True,exist_ok=True)
  if (remote/'identity.json').exists() and read(remote/'identity.json')!=identity:raise ValueError('Resume identity changed; use the same notebook/runtime')
  write(remote/'identity.json',identity)
  with checked_zip((bundle/'qualification_report.zip').read_bytes()) as q:
   if json.loads(q.read('summary.json'))['status']!='EXTERNAL_QUALIFICATION_AUDIT_COMPLETE':raise ValueError('Prior qualification incomplete')
   candidates=[x for x in json.loads(q.read('temporal_candidates.json')) if x['preliminary_temporal_source_candidate']]
   if len(candidates)!=4 or any(not x['dataset'].startswith('cidds') for x in candidates):raise ValueError('Unexpected candidate set')
   with checked_zip(q.read('correction_report.zip')) as c:original=c.read('parent_report.zip')
  with checked_zip(original) as p:
   for n in ['identity.json','registry.json','CODE.py','PLAN.md']:
    if sha(parent/n)!=hashlib.sha256(p.read(n)).hexdigest():raise ValueError('Drive parent differs: '+n)
   for did in ['cidds-001','cidds-002']:
    pin=json.loads(p.read('inputs/'+did+'.complete.json'));path=parent/'inputs'/pin['record']['name'];print('Verify raw input',did,flush=True)
    if path.stat().st_size!=pin['hashes']['bytes'] or sha(path)!=pin['hashes']['sha256']:raise ValueError('Raw input differs')
    cache=local/path.name
    if not cache.exists() or sha(cache)!=pin['hashes']['sha256']:shutil.copyfile(path,cache)
    if sha(cache)!=pin['hashes']['sha256']:raise ValueError('Local raw copy differs')
    with zipfile.ZipFile(cache) as z:
     for e in candidates:
      if e['dataset']!=did:continue
      capture=did+'_'+hashlib.sha256(e['member'].encode()).hexdigest()[:16];meta={'dataset':did,'member':e['member'],'capture':capture,'expected_rows':e['rows'],'archive_sha256':pin['hashes']['sha256']};out=remote/'cohorts'/(capture+'.json.gz');cohorts.append(build_cohort(lambda e=e:z.open(e['member']),meta,identity,out,local))
    cache.unlink()
  reduced,conflicts,reduction=reduce_cohorts(cohorts)
  for seed in SEEDS:
   role_sets=[]
   for s,rows in zip(cohorts,reduced):
    capture=s['input']['capture'];payload,rec,selected=make_partitions(rows,seed,capture);out=local/(capture+'_seed_'+str(seed)+'.npz');save_npz(out,payload);verify_npz(out);dst=remote/'partitions'/out.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(out,dst)
    if sha(dst)!=sha(out):raise ValueError('Drive partition copy differs')
    ref={'path':str(dst.relative_to(remote)),'sha256':sha(out),'bytes':out.stat().st_size,'capture':capture,'seed':seed};output_refs.append(ref);rec['partition']=ref;records.append(rec);role_sets.append({k:{r[0] for r in rr} for k,rr in selected.items()});print('Partition verified',capture,seed,'source eligible:',rec['source_eligible'],flush=True)
   for i,j in itertools.combinations(range(len(cohorts)),2):
    for a in ['train','calibration','test','environment']:
     for b in ['train','calibration','test','environment']:
      n=len(role_sets[i][a]&role_sets[j][b]);cross.append({'seed':seed,'left_capture':cohorts[i]['input']['capture'],'right_capture':cohorts[j]['input']['capture'],'left_role':a,'right_role':b,'shared_signatures':n})
      if a!=b and n:raise ValueError('Cross-capture different-role overlap')
 except Exception as ex:
  import traceback;traceback.print_exc();errors.append({'type':type(ex).__name__,'message':str(ex)})
 summary={'status':'CIDDS_PARTITION_QUALIFICATION_COMPLETE' if not errors and len(records)==20 else 'CIDDS_PARTITION_QUALIFICATION_PARTIAL_RESUMABLE','errors':errors,'raw_rows_scanned':sum(s['rows'] for s in cohorts),'captures_processed':len(cohorts),'capture_seed_records':len(records),'source_eligible_records':sum(r['source_eligible'] for r in records),'target_eligible_records':sum(r['target_eligible'] for r in records),'sources_eligible_all_five_seeds':[s['input']['capture'] for s in cohorts if len([r for r in records if r['capture']==s['input']['capture'] and r['source_eligible']])==5],'selected_cohort_conflicting_signatures_removed':len(conflicts),'models_fitted':0,'feature_names':FEATURES,'cohort_scope':'bounded per-window feature-group samples; no full-dataset duplicate absence claim','target_label_informed_cohort_exclusion':True,'independent_site_warning_validation_completed':False,'measured_production_prevalence_established':False,'fine_family_equivalence_certified':False,'session_seconds':time.monotonic()-start}
 reports={'summary.json':summary,'identity.json':identity,'partition_records.json':records,'output_references.json':output_refs,'cohort_reduction.json':reduction,'cross_capture_role_intersections.json':cross}
 for s in cohorts:reports[s['input']['capture']+'_cohort_summary.json']={k:v for k,v in s.items() if k!='pool'}
 for n,v in reports.items():
  write(local/n,v)
  try:write(remote/n,v)
  except OSError:pass
 for n in ['engine.py','projection.py','PLAN.md','qualification_report.zip']:shutil.copyfile(bundle/n,local/n)
 names=sorted(reports)+['engine.py','projection.py','PLAN.md','qualification_report.zip'];write(local/'SHA256SUMS.json',{n:sha(local/n) for n in names});report=local/('cidds_partition_qualification_report_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.zip')
 with zipfile.ZipFile(report,'w',zipfile.ZIP_DEFLATED) as z:
  for n in names+['SHA256SUMS.json']:z.write(local/n,n)
 try:shutil.copyfile(report,remote.parent/report.name)
 except OSError:print('Report retained on Colab; Drive copy unavailable',flush=True)
 print(summary['status'],flush=True);print('Report:',report,flush=True);return report

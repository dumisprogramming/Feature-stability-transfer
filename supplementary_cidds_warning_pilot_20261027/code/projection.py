"""External cohort eligibility audit. No model fits and no certified cross-exporter mapping."""
from pathlib import Path
import json,zipfile,hashlib,io,platform,time,shutil,itertools
from datetime import datetime,timezone
from collections import Counter
import numpy as np
import pandas as pd
STAGE='supplementary_external_qualification_20261027'
PARENT='supplementary_external_data_inventory_20261027/run_v1'
CHUNK=250000
CAP=2000
SCHEMAS={
 'cidds':{'candidate_features':['duration_seconds','protocol_is_tcp','destination_port','unidirectional_packets'],'required':['Duration','Proto','Dst Pt','Packets','Date first seen']},
 'hikari':{'candidate_features':['duration_seconds_candidate','responder_port','forward_packets','backward_packets'],'required':['flow_duration','responp','fwd_pkts_tot','bwd_pkts_tot','Label']}}

def read(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def add(d,v):
 for k,n in v.items():d[str(k)]=d.get(str(k),0)+int(n)
def priorities(start,n,key):
 # Sampling uses file/row identifiers only, not labels, features, or outcomes.
 salt=int(hashlib.sha256(key.encode()).hexdigest()[:16],16)
 with np.errstate(over='ignore'):
  v=np.arange(start,start+n,dtype=np.uint64)+np.uint64(salt);v=(v^(v>>30))*np.uint64(0xbf58476d1ce4e5b9);v=(v^(v>>27))*np.uint64(0x94d049bb133111eb);v=v^(v>>31)
 return v

def project(frame,kind,dataset):
 cols=list(frame.columns);notes={};n=len(frame)
 if kind=='cidds':
  found=[c for c in cols if c.strip().lower() in ['class','label']]
  if len(found)!=1:raise ValueError('Ambiguous/missing CIDDS label')
  lab=frame[found[0]].str.strip().str.lower();y=np.where(lab.eq('normal'),0,np.where(lab.isin(['attacker','victim']),1,-1))
  proto=frame['Proto'].str.strip().str.upper();notes['protocol_counts']=proto.value_counts().to_dict();supported=proto.isin(['TCP','UDP']).to_numpy()
  nums=[pd.to_numeric(frame[c],errors='coerce').to_numpy(dtype=float) for c in ['Duration','Dst Pt','Packets']]
  dur,port,pkt=nums;x=np.column_stack([dur,proto.eq('TCP').astype(float),port,pkt]);ints=[(port,65535),(pkt,2**53-1)]
  t=pd.to_datetime(frame['Date first seen'],format='mixed',errors='coerce');block=t.dt.floor('30min').dt.strftime('%Y-%m-%d %H:%M:%S').fillna('__INVALID__').to_numpy();family=frame['attackType'].str.strip().to_numpy()
 else:
  lab=frame['Label'].str.strip();y=np.where(lab.eq('0'),0,np.where(lab.eq('1'),1,-1));supported=np.ones(n,dtype=bool)
  s=frame['flow_duration'].str.strip()
  if dataset=='hikari_2022':
   syntax=s.str.fullmatch(r'[+-]?\d+ days? \d{1,2}:\d{2}:\d{2}(?:\.\d+)?');dur=pd.to_timedelta(s.where(syntax),errors='coerce').dt.total_seconds().to_numpy(dtype=float);notes['duration_syntax_rejected']=int((~syntax).sum())
  else:dur=pd.to_numeric(s,errors='coerce').to_numpy(dtype=float)
  port,fwd,bwd=[pd.to_numeric(frame[c],errors='coerce').to_numpy(dtype=float) for c in ['responp','fwd_pkts_tot','bwd_pkts_tot']];x=np.column_stack([dur,port,fwd,bwd]);ints=[(port,65535),(fwd,2**53-1),(bwd,2**53-1)];block=np.full(n,'__NO_TIMESTAMP__',dtype=object)
  family=frame['traffic_category' if dataset=='hikari_2021' else 'attack_category'].str.strip().to_numpy()
 valid=np.isfinite(x).all(axis=1)&(dur>=0)&supported
 notes['nonfinite_feature_rows']=int((~np.isfinite(x).all(axis=1)).sum());notes['negative_duration_rows']=int((dur<0).sum());notes['unsupported_protocol_rows']=int((~supported).sum())
 for j,(v,upper) in enumerate(ints):
  good=np.isfinite(v)&(v>=0)&(v<=upper)&(v==np.floor(v));notes['invalid_integer_field_'+str(j)]=int((~good).sum());valid&=good
 notes['known_label_rows']=int((y>=0).sum());notes['unknown_label_rows']=int((y<0).sum());notes['candidate_valid_known_rows']=int((valid&(y>=0)).sum())
 return x,y,block,family,valid,notes

def select(old,frame,priority,cap):
 if len(frame)>cap:
  # lexsort handles priority ties using the unique row ordinal.
  ix=np.lexsort((frame[:,0].astype(np.int64),priority))[:cap];frame=frame[ix];priority=priority[ix]
 items=old+[[int(p),int(r[0]),*[float(v) for v in r[1:5]],str(r[5]),str(r[6])] for p,r in zip(priority,frame)]
 items.sort(key=lambda r:(r[0],r[1]));return items[:cap]

def audit_csv(open_file,meta,expected_rows,out,identity):
 out=Path(out);cp=out.with_suffix('.checkpoint.json');mark=out.with_suffix('.complete.json')
 if mark.exists():
  m=read(mark)
  if m['identity']!=identity or m['input']!=meta or m['sha256']!=sha(out):raise ValueError('Changed completion record')
  return read(out)
 s=read(cp) if cp.exists() else {'identity':identity,'input':meta,'rows':0,'columns':None,'counts':{},'protocol_counts':{},'candidate_time_class_counts':{},'candidate_family_class_counts':{},'sample_0':[],'sample_1':[],'chunks':0}
 if s['identity']!=identity or s['input']!=meta:raise ValueError('Changed resume identity')
 seen=0;completed=s['rows'];kind=meta['kind'];did=meta['dataset'];key=did+'/'+meta['member']
 with open_file() as f:
  for frame in pd.read_csv(f,dtype=str,keep_default_na=False,on_bad_lines='error',encoding='utf-8-sig',chunksize=CHUNK):
   if seen+len(frame)<=completed:seen+=len(frame);continue
   if seen!=s['rows']:raise ValueError('Resume chunk boundary mismatch')
   cols=list(frame.columns)
   if s['columns'] is not None and s['columns']!=cols:raise ValueError('Changed CSV schema')
   s['columns']=cols;x,y,block,family,valid,notes=project(frame,kind,did);add(s['protocol_counts'],notes.pop('protocol_counts',{}));add(s['counts'],notes);priority=priorities(seen,len(frame),key);row=np.arange(seen,seen+len(frame),dtype=np.int64)
   known=valid&(y>=0)
   if known.any():
    add(s['candidate_time_class_counts'],pd.Series(block[known]+' | '+y[known].astype(str)).value_counts().to_dict());add(s['candidate_family_class_counts'],pd.Series(family[known]+' | '+y[known].astype(str)).value_counts().to_dict())
   for c in [0,1]:
    ix=np.flatnonzero(valid&(y==c))
    if len(ix):
     # Keep only bounded candidate rows before building the mixed object array.
     if len(ix)>CAP:ix=ix[np.lexsort((row[ix],priority[ix]))[:CAP]]
     a=np.empty((len(ix),7),dtype=object);a[:,0]=row[ix];a[:,1:5]=x[ix];a[:,5]=block[ix];a[:,6]=family[ix];s['sample_'+str(c)]=select(s['sample_'+str(c)],a,priority[ix],CAP)
   s['rows']+=len(frame);seen=s['rows'];s['chunks']+=1;write(cp,s);print(did,meta['member'],f'{seen:,}/{expected_rows:,} rows',flush=True)
 if seen!=expected_rows or s['rows']!=expected_rows:raise ValueError('Inventory row count mismatch')
 assert sum(s['candidate_time_class_counts'].values())==s['counts']['candidate_valid_known_rows']==sum(s['candidate_family_class_counts'].values())
 s['sampling_scope']='bottom-2000 row-priority candidates per binary class per file; not a whole-dataset duplicate audit';s['model_fit_performed']=False
 write(out,s);write(mark,{'identity':identity,'input':meta,'sha256':sha(out)});return s

def fingerprints(s):
 # Label-independent model-projection equivalence, not proof that two flows are the same event.
 result={}
 for c in [0,1]:
  for r in s['sample_'+str(c)]:
   key=hashlib.sha256(json.dumps([float(v).hex() for v in r[2:6]],separators=(',',':')).encode()).hexdigest();result.setdefault(key,[]).append(c)
 return result

def eligibility(s):
 blocks={}
 for k,n in s['candidate_time_class_counts'].items():
  b,c=k.rsplit(' | ',1);blocks.setdefault(b,{'0':0,'1':0})[c]=n
 eligible=[b for b,n in blocks.items() if not b.startswith('__') and min(n.values())>=100]
 return {'dataset':s['input']['dataset'],'member':s['input']['member'],'rows':s['rows'],'valid_known_candidate_rows':s['counts']['candidate_valid_known_rows'],'clock_blocks_with_100_each_class':len(eligible),'candidate_clock_blocks':sorted(eligible),'preliminary_temporal_source_candidate':len(eligible)>=8,'eligibility_is_before_role_assignment_and_duplicate_exclusion':True,'time_partition_certified':False,'timezone_certified':False}

def run(root,bundle,local_root='/content/paper3_external_qualification'):
 root=Path(root);bundle=Path(bundle);local=Path(local_root);local.mkdir(parents=True,exist_ok=True);parent=root/PARENT;remote=root/STAGE/'run_v1';start=time.monotonic();errors=[];results=[];pins=[]
 identity={'engine_sha256':sha(bundle/'engine.py'),'correction_report_sha256':sha(bundle/'correction_report.zip'),'plan_sha256':sha(bundle/'PLAN.md'),'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'chunksize':CHUNK,'cap_per_class_file':CAP}
 try:
  remote.mkdir(parents=True,exist_ok=True)
  if (remote/'identity.json').exists() and read(remote/'identity.json')!=identity:raise ValueError('Resume identity changed: use the same notebook/runtime')
  write(remote/'identity.json',identity)
  with zipfile.ZipFile(bundle/'correction_report.zip') as corr:
   for n,h in json.loads(corr.read('SHA256SUMS.json')).items():
    if hashlib.sha256(corr.read(n)).hexdigest()!=h:raise ValueError('Correction report changed')
   if json.loads(corr.read('summary.json'))['status']!='CORRECTION_AUDIT_COMPLETE':raise ValueError('Correction incomplete')
   original=corr.read('parent_report.zip')
  with zipfile.ZipFile(io.BytesIO(original)) as p:
   for n,h in json.loads(p.read('SHA256SUMS.json')).items():
    if hashlib.sha256(p.read(n)).hexdigest()!=h:raise ValueError('Parent report changed')
   index=json.loads(p.read('inventory_index.json'))
   for n in ['identity.json','registry.json','CODE.py','PLAN.md']:
    if sha(parent/n)!=hashlib.sha256(p.read(n)).hexdigest():raise ValueError('Drive parent differs: '+n)
   for did in ['cidds-001','cidds-002','hikari_2021','hikari_2022']:
    pin=json.loads(p.read('inputs/'+did+'.complete.json'));path=parent/'inputs'/pin['record']['name'];print('Verify existing input',did,flush=True)
    if path.stat().st_size!=pin['hashes']['bytes'] or sha(path)!=pin['hashes']['sha256']:raise ValueError('Input differs: '+did)
    pins.append(pin);cache=local/path.name
    if not cache.exists() or sha(cache)!=pin['hashes']['sha256']:shutil.copyfile(path,cache)
    if sha(cache)!=pin['hashes']['sha256']:raise ValueError('Local copy differs')
    entries=[x for x in index if x['id']==did and x['kind'] in ['cidds_flow','hikari_flow']]
    def process(entry,opener):
     meta={'dataset':did,'member':entry['member'],'kind':'cidds' if did.startswith('cidds') else 'hikari','input_sha256':pin['hashes']['sha256']};key=did+'_'+hashlib.sha256(entry['member'].encode()).hexdigest()[:16];out=remote/(key+'.json');s=audit_csv(opener,meta,entry['rows'],out,identity);shutil.copyfile(out,local/out.name);shutil.copyfile(out.with_suffix('.complete.json'),local/out.with_suffix('.complete.json').name);results.append(s)
    if cache.suffix=='.zip':
     with zipfile.ZipFile(cache) as z:
      for e in entries:process(e,lambda e=e:z.open(e['member']))
    else:
     if len(entries)!=1:raise ValueError('Unexpected plain CSV index')
     process(entries[0],lambda:cache.open('rb'))
    cache.unlink()
  if len(results)!=12:raise ValueError('Expected 12 flow CSV inventories')
 except Exception as ex:
  import traceback;traceback.print_exc();errors.append({'type':type(ex).__name__,'message':str(ex)})
 # Compact summaries contain no row-level features, addresses, UIDs, or fingerprint lists.
 elig=[eligibility(s) for s in results];overlap=[];within=[];fps=[fingerprints(s) for s in results]
 for s,f in zip(results,fps):
  selected=sum(len(s['sample_'+str(c)]) for c in [0,1]);within.append({'dataset':s['input']['dataset'],'member':s['input']['member'],'selected_rows':selected,'unique_projection_signatures':len(f),'projection_duplicate_excess':selected-len(f),'binary_conflicting_signatures':sum(len(set(v))>1 for v in f.values())})
 for a,b in itertools.combinations(range(len(results)),2):
  if results[a]['input']['kind']!=results[b]['input']['kind']:continue
  common=set(fps[a])&set(fps[b]);overlap.append({'left':results[a]['input'],'right':results[b]['input'],'shared_selected_projection_signatures':len(common),'binary_conflicting_shared_signatures':sum(len(set(fps[a][h]+fps[b][h]))>1 for h in common)})
 summary={'status':'EXTERNAL_QUALIFICATION_AUDIT_COMPLETE' if not errors and len(results)==12 else 'EXTERNAL_QUALIFICATION_AUDIT_PARTIAL_RESUMABLE','errors':errors,'files_audited':len(results),'rows_scanned':sum(s['rows'] for s in results),'models_fitted':0,'overlap_scope':'bounded candidate model-projection samples; never full-dataset absence-of-overlap evidence','cross_provider_feature_equivalence_certified':False,'temporal_partitions_certified':False,'independent_site_warning_validation_completed':False,'measured_production_prevalence_established':False,'fine_family_equivalence_certified':False,'session_seconds':time.monotonic()-start}
 reports={'summary.json':summary,'identity.json':identity,'verified_input_pins.json':pins,'feature_schemas.json':SCHEMAS,'temporal_candidates.json':elig,'within_sample_overlap.json':within,'between_sample_overlap.json':overlap}
 for s in results:
  d={k:v for k,v in s.items() if k not in ['sample_0','sample_1']};reports[s['input']['dataset']+'_'+hashlib.sha256(s['input']['member'].encode()).hexdigest()[:16]+'_summary.json']=d
 for n,v in reports.items():
  write(local/n,v)
  try:write(remote/n,v)
  except OSError:pass
 for n in ['engine.py','PLAN.md','correction_report.zip']:shutil.copyfile(bundle/n,local/n)
 names=sorted(reports)+['engine.py','PLAN.md','correction_report.zip'];write(local/'SHA256SUMS.json',{n:sha(local/n) for n in names})
 report=local/('external_qualification_report_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.zip')
 with zipfile.ZipFile(report,'w',zipfile.ZIP_DEFLATED) as z:
  for n in names+['SHA256SUMS.json']:z.write(local/n,n)
 try:shutil.copyfile(report,remote.parent/report.name)
 except OSError:print('Report retained on Colab; Drive copy unavailable',flush=True)
 print(summary['status'],flush=True);print('Report:',report,flush=True);return report

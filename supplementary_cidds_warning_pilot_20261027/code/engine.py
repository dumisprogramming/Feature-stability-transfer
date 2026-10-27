"""Fixed supplementary CIDDS LR transfer pilot; no warning-utility inference from two sources."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zipfile,io,shutil,platform,time,warnings,itertools
import numpy as np
import scipy,sklearn,threadpoolctl
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import partition
STAGE='supplementary_cidds_warning_pilot_20261027'
INPUT_STAGE='supplementary_cidds_partitions_20261027/run_v1'
SEEDS=[2027,2028,2029,2030,2031]
REPEATS=5

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def save_npz(p,**a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp.npz');np.savez_compressed(t,**a);t.replace(p)
def load_npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def checked_zip(b):
 z=zipfile.ZipFile(io.BytesIO(b))
 for n,h in json.loads(z.read('SHA256SUMS.json')).items():
  if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Report hash mismatch: '+n)
 return z

def confusion(y,p,t):
 y=np.asarray(y);pred=np.asarray(p)>=t
 if not set(y.tolist())<={0,1} or not np.isfinite(p).all():raise ValueError('Invalid labels/probabilities')
 return np.asarray([np.sum((y==0)&~pred),np.sum((y==0)&pred),np.sum((y==1)&~pred),np.sum((y==1)&pred)],dtype=np.int64)
def ba_counts(c):
 tn,fp,fn,tp=map(int,c)
 if tn+fp==0 or fn+tp==0:raise ValueError('Both classes required')
 return .5*(tn/(tn+fp)+tp/(tp+fn))
def metrics(y,p,t):
 c=confusion(y,p,t);tn,fp,fn,tp=map(int,c)
 return {'rows':len(y),'tn':tn,'fp':fp,'fn':fn,'tp':tp,'balanced_accuracy':ba_counts(c),'false_positive_rate':fp/(tn+fp),'attack_recall':tp/(tp+fn),'threshold':float(t)}
def threshold(y,p):
 y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
 n0=int((y==0).sum());n1=int((y==1).sum())
 if not n0 or not n1 or not np.isfinite(p).all():raise ValueError('Invalid calibration panel')
 order=np.argsort(-p,kind='stable');s=p[order];yy=y[order];ends=np.flatnonzero(np.r_[s[:-1]!=s[1:],True]);tp=np.cumsum(yy)[ends];fp=ends+1-tp
 scores=np.r_[.5,.5*(tp/n1+(n0-fp)/n0)];candidates=np.r_[np.nextafter(float(s[0]),np.inf),s[ends]];best=float(scores.max());chosen=float(candidates[np.abs(scores-best)<=1e-12].max())
 return chosen,best

def fit(x,y,cx,cy,seed):
 scaler=StandardScaler().fit(x);model=LogisticRegression(C=1.0,solver='lbfgs',tol=1e-8,max_iter=5000,class_weight=None,fit_intercept=True,random_state=seed)
 with warnings.catch_warnings():
  warnings.simplefilter('error',ConvergenceWarning)
  with threadpool_limits(limits=1):model.fit(scaler.transform(x),y)
 if model.classes_.tolist()!=[0,1]:raise ValueError('Unexpected model classes')
 numeric={'mean':scaler.mean_,'scale':scaler.scale_,'coef':model.coef_[0],'intercept':np.asarray(float(model.intercept_[0]))}
 p=predict(numeric,cx);original=model.predict_proba(scaler.transform(cx))[:,1]
 if not np.allclose(p,original,rtol=0,atol=1e-12):raise ValueError('Numeric model differs from sklearn probabilities')
 t,b=threshold(cy,p);numeric['threshold']=np.asarray(t)
 info={'seed':seed,'train_rows':len(y),'train_class_counts':np.bincount(y,minlength=2).tolist(),'iterations':int(model.n_iter_[0]),'calibration_best_balanced_accuracy':b,'threshold':t,'numeric_vs_sklearn_max_difference':float(np.max(np.abs(p-original))),'parameters':{'C':1.0,'solver':'lbfgs','tol':1e-8,'max_iter':5000,'class_weight':None,'fit_intercept':True,'preprocessing':'StandardScaler fit on source training rows only'}}
 return numeric,p,info

def predict(model,x):
 with threadpool_limits(limits=1):return expit(((x-model['mean'])/model['scale'])@model['coef']+float(model['intercept']))
def seed_for(tag):return int(hashlib.sha256(tag.encode()).hexdigest()[:16],16)
def importance(model,x,y,tag):
 t=float(model['threshold']);base=confusion(y,predict(model,x),t);counts=np.empty((REPEATS,4,4),dtype=np.int64)
 for repeat in range(REPEATS):
  for f in range(4):
   order=np.random.default_rng(seed_for(tag+'|'+str(repeat)+'|'+str(f))).permutation(len(y));xx=x.copy();xx[:,f]=xx[order,f];counts[repeat,f]=confusion(y,predict(model,xx),t)
 n0=int(base[0]+base[1]);n1=int(base[2]+base[3]);numerator=np.sum((int(base[3])-counts[:,:,3])*n0+(int(base[0])-counts[:,:,0])*n1,axis=0,dtype=np.int64);denominator=2*n0*n1*REPEATS
 return {'baseline_counts':base,'permuted_counts':counts,'importance_numerator':numerator,'importance_denominator':np.asarray(denominator,dtype=np.int64),'importance_mean':numerator/denominator}

def stability(groups):
 pairs=[];values=[]
 for i,j in itertools.combinations(range(len(groups)),2):
  a=rankdata(groups[i]['importance_numerator'],method='average');b=rankdata(groups[j]['importance_numerator'],method='average')
  rho=None if np.ptp(a)==0 or np.ptp(b)==0 else float(np.clip(np.corrcoef(a,b)[0,1],-1,1));pairs.append({'left_group':i,'right_group':j,'rho':rho})
  if rho is not None:values.append(rho)
 ok=bool(pairs) and len(values)==len(pairs)
 return {'groups':len(groups),'pairs':len(pairs),'undefined_pairs':len(pairs)-len(values),'mean_spearman':float(np.mean(values)) if ok else None,'instability':float(1-np.mean(values)) if ok else None,'pairwise':pairs}

def checkpoint(folder,identity,meta,build):
 folder=Path(folder);mark=folder/'complete.json'
 if mark.exists():
  m=read(mark)
  if m['identity']!=identity or m['meta']!=meta:raise ValueError('Checkpoint identity changed: '+str(folder))
  for n,h in m['files'].items():
   if sha(folder/n)!=h:raise ValueError('Checkpoint artifact changed: '+str(folder/n))
  return m
 folder.mkdir(parents=True,exist_ok=True);names=build(folder);m={'identity':identity,'meta':meta,'files':{n:sha(folder/n) for n in names}};write(mark,m);return m

def validate_inputs(inputs,records):
 # Recompute role assignments and actual overlaps from the saved arrays, rather than trusting report counters.
 result=[]
 for seed in SEEDS:
  sources=[r for r in records if r['seed']==seed];sets=[];labels={}
  for rec in sources:
   a=inputs[(rec['capture'],seed)];d={};ct={}
   for r in ['train','calibration','test','environment']:
    h=[v.decode() for v in a[r+'_signature']];d[r]=set(h);ct[r]=np.bincount(a[r+'_y'],minlength=2).tolist()
    if ct[r]!=rec['role_counts_per_class'][r]:raise ValueError('Partition class counts changed')
    for sig,y in zip(h,a[r+'_y']):
     if partition.role(sig,seed)!=r:raise ValueError('Partition global role assignment changed')
     if sig in labels and labels[sig]!=int(y):raise ValueError('Selected input label conflict remains')
     labels[sig]=int(y)
   source_ok=all(min(ct[r])>=n for r,n in [('train',500),('calibration',100),('test',100)]) and len(np.unique(a['environment_group']))>=4
   if source_ok!=rec['source_eligible'] or (min(ct['test'])>=100)!=rec['target_eligible']:raise ValueError('Eligibility changed')
   sets.append((rec['capture'],d))
  checks=0;shared=0
  for (ca,a),(cb,b) in itertools.combinations(sets,2):
   for r in a:
    for s in b:
     n=len(a[r]&b[s]);checks+=1
     if r!=s and n:raise ValueError('Different-role cross-capture overlap')
     if r==s:shared+=n
  result.append({'seed':seed,'cross_capture_role_checks':checks,'different_role_nonzero':0,'same_role_shared_signature_pair_sum':shared})
 return result

def run(root,bundle,local_root='/content/paper3_cidds_warning_pilot'):
 root=Path(root);bundle=Path(bundle);local=Path(local_root);local.mkdir(parents=True,exist_ok=True);remote=root/STAGE/'run_v1';input_root=root/INPUT_STAGE;start=time.monotonic();errors=[];checkpoints=[];evaluations=[];warning_records=[];preflight=[];input_refs=[];inputs={};records=[];fits_ready=0
 identity={'engine_sha256':sha(bundle/'engine.py'),'partition_code_sha256':sha(bundle/'partition.py'),'projection_sha256':sha(bundle/'projection.py'),'plan_sha256':sha(bundle/'PLAN.md'),'verifier_sha256':sha(bundle/'verify_report.py'),'partition_report_sha256':sha(bundle/'partition_report.zip'),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pandas':partition.pd.__version__,'threadpoolctl':threadpoolctl.__version__,'repeats':REPEATS,'seeds':SEEDS}
 try:
  remote.mkdir(parents=True,exist_ok=True)
  if (remote/'identity.json').exists() and read(remote/'identity.json')!=identity:raise ValueError('Resume identity changed: use the same notebook and runtime versions')
  write(remote/'identity.json',identity)
  with checked_zip((bundle/'partition_report.zip').read_bytes()) as p:
   summary=json.loads(p.read('summary.json'));records=json.loads(p.read('partition_records.json'));input_refs=json.loads(p.read('output_references.json'))
   if summary['status']!='CIDDS_PARTITION_QUALIFICATION_COMPLETE' or len(input_refs)!=20:raise ValueError('Partition qualification incomplete')
   if hashlib.sha256(p.read('engine.py')).hexdigest()!=identity['partition_code_sha256'] or hashlib.sha256(p.read('projection.py')).hexdigest()!=identity['projection_sha256']:raise ValueError('Partition code provenance differs')
   source_ids=summary['sources_eligible_all_five_seeds']
  independently_eligible=sorted({r['capture'] for r in records if sum(t['source_eligible'] for t in records if t['capture']==r['capture'])==5})
  if len(source_ids)!=2 or sorted(source_ids)!=independently_eligible:raise ValueError('Expected exactly two sources passing all five seeds')
  for ref in input_refs:
   path=input_root/ref['path'];cache=local/'inputs'/Path(ref['path']).name;cache.parent.mkdir(parents=True,exist_ok=True)
   if path.stat().st_size!=ref['bytes'] or sha(path)!=ref['sha256']:raise ValueError('Partition input changed: '+ref['path'])
   if not cache.exists() or sha(cache)!=ref['sha256']:shutil.copyfile(path,cache)
   if sha(cache)!=ref['sha256']:raise ValueError('Local partition copy differs')
   partition.verify_npz(cache);inputs[(ref['capture'],ref['seed'])]=load_npz(cache)
  preflight=validate_inputs(inputs,records);print('Verified 20 saved partitions and recomputed all cross-capture role checks.',flush=True)
  for source in source_ids:
   for seed in SEEDS:
    a=inputs[(source,seed)];ref=next(r for r in input_refs if r['capture']==source and r['seed']==seed);job=source+'_seed_'+str(seed);out=remote/'outputs'/job
    def build_fit(folder):
     model,cp,info=fit(a['train_X'],a['train_y'],a['calibration_X'],a['calibration_y'],seed);save_npz(folder/'model.npz',**model);save_npz(folder/'calibration.npz',y=a['calibration_y'],probability=cp);write(folder/'fit.json',info);return ['model.npz','calibration.npz','fit.json']
    fm=checkpoint(out/'fit',identity,{'source':source,'seed':seed,'source_partition_sha256':ref['sha256']},build_fit);checkpoints.append(out/'fit');fits_ready+=1;model=load_npz(out/'fit/model.npz');model_hash=fm['files']['model.npz'];print('Fit ready',job,flush=True)
    group_results={}
    for kind in ['environment','pseudo']:
     groups=[]
     for gid in sorted(np.unique(a[kind+'_group']).tolist()):
      idx=a[kind+'_group']==gid;folder=out/kind/('group_'+str(gid));meta={'source':source,'seed':seed,'kind':kind,'group':gid,'source_partition_sha256':ref['sha256'],'model_sha256':model_hash}
      def build_group(folder,idx=idx,kind=kind,gid=gid):
       g=importance(model,a[kind+'_X'][idx],a[kind+'_y'][idx],job+'|'+kind+'|'+str(gid));save_npz(folder/'importance.npz',**g);return ['importance.npz']
      checkpoint(folder,identity,meta,build_group);checkpoints.append(folder);groups.append(load_npz(folder/'importance.npz'))
     group_results[kind]=stability(groups);print('Stability ready',job,kind,'groups',len(groups),flush=True)
    environmental=group_results['environment']['instability'];sampling=group_results['pseudo']['instability'];gap=None if environmental is None or sampling is None else environmental-sampling
    warning_records.append({'source':source,'seed':seed,'environmental_instability':environmental,'sampling_instability':sampling,'sampling_environment_gap':gap,'environment':group_results['environment'],'pseudo':group_results['pseudo']})
    targets=[r for r in records if r['seed']==seed and r['target_eligible'] and r['capture']!=source]
    if len(targets)!=3:raise ValueError('Expected exactly three external target captures')
    for target in [source]+[r['capture'] for r in targets]:
     panel=inputs[(target,seed)];target_ref=next(r for r in input_refs if r['capture']==target and r['seed']==seed);folder=out/'evaluations'/target;meta={'source':source,'target':target,'seed':seed,'model_sha256':model_hash,'target_partition_sha256':target_ref['sha256']}
     def build_eval(folder,panel=panel):
      pp=predict(model,panel['test_X']);mm=metrics(panel['test_y'],pp,float(model['threshold']));save_npz(folder/'predictions.npz',y=panel['test_y'],probability=pp);write(folder/'metrics.json',mm);return ['predictions.npz','metrics.json']
     checkpoint(folder,identity,meta,build_eval);checkpoints.append(folder);mm=read(folder/'metrics.json');evaluations.append({**meta,**mm,'is_source_test':target==source})
    print('Completed source job',job,'| total external evaluations',sum(not r['is_source_test'] for r in evaluations),flush=True)
  if len(warning_records)!=10 or len(evaluations)!=40:raise ValueError('Expected output jobs missing')
 except Exception as ex:
  import traceback;traceback.print_exc();errors.append({'type':type(ex).__name__,'message':str(ex)})
 source_stats=[];directional=[]
 for source in sorted({r['source'] for r in evaluations}):
  own=[r['balanced_accuracy'] for r in evaluations if r['source']==source and r['is_source_test']];ext=[r for r in evaluations if r['source']==source and not r['is_source_test']];warn=[r for r in warning_records if r['source']==source]
  losses=[next(s['balanced_accuracy'] for s in evaluations if s['source']==source and s['seed']==r['seed'] and s['is_source_test'])-r['balanced_accuracy'] for r in ext]
  def mean_warning(key):
   v=[r[key] for r in warn];return float(np.mean(v)) if v and all(x is not None for x in v) else None
  source_stats.append({'source':source,'source_tests':len(own),'external_evaluations':len(ext),'mean_source_ba':float(np.mean(own)) if own else None,'mean_target_ba':float(np.mean([r['balanced_accuracy'] for r in ext])) if ext else None,'mean_loss':float(np.mean(losses)) if losses else None,**{'mean_'+key:mean_warning(key) for key in ['environmental_instability','sampling_instability','sampling_environment_gap']}})
  for target in sorted({r['target'] for r in ext}):
   rr=[r for r in ext if r['target']==target];ll=[next(s['balanced_accuracy'] for s in evaluations if s['source']==source and s['seed']==r['seed'] and s['is_source_test'])-r['balanced_accuracy'] for r in rr];directional.append({'source':source,'target':target,'seeds':len(rr),'mean_target_ba':float(np.mean([r['balanced_accuracy'] for r in rr])),'mean_loss':float(np.mean(ll))})
 source_eval=[r for r in evaluations if r['is_source_test']];target_eval=[r for r in evaluations if not r['is_source_test']];losses=[next(s['balanced_accuracy'] for s in source_eval if s['source']==r['source'] and s['seed']==r['seed'])-r['balanced_accuracy'] for r in target_eval]
 summary={'status':'CIDDS_WARNING_PILOT_COMPLETE' if not errors and len(evaluations)==40 and len(warning_records)==10 else 'CIDDS_WARNING_PILOT_PARTIAL_RESUMABLE','errors':errors,'source_fits_completed':fits_ready,'source_tests':len(source_eval),'external_evaluations':len(target_eval),'warning_records':len(warning_records),'fully_defined_warning_records':sum(all(r[k] is not None for k in ['environmental_instability','sampling_instability','sampling_environment_gap']) for r in warning_records),'mean_source_ba':float(np.mean([r['balanced_accuracy'] for r in source_eval])) if source_eval else None,'mean_target_ba':float(np.mean([r['balanced_accuracy'] for r in target_eval])) if target_eval else None,'mean_loss':float(np.mean(losses)) if losses else None,'degraded_external_jobs':sum(v>0 for v in losses),'improved_external_jobs':sum(v<0 for v in losses),'tied_external_jobs':sum(v==0 for v in losses),'predictive_warning_utility_validation_performed':False,'reason_no_warning_utility_inference':'Only two eligible related source periods; seeds do not increase independent source count. No correlation p-values, regression utility claims, or general warning-utility inference.','post_hoc_target_label_informed_cohort':True,'independent_site_warning_validation_completed':False,'measured_production_prevalence_established':False,'fine_family_equivalence_certified':False,'session_seconds':time.monotonic()-start}
 reports={'summary.json':summary,'identity.json':identity,'evaluations.json':evaluations,'warnings.json':warning_records,'source_capture_summary.json':source_stats,'directional_summary.json':directional,'partition_preflight.json':preflight,'input_references.json':input_refs}
 for n,v in reports.items():
  write(local/n,v)
  try:write(remote/n,v)
  except OSError:pass
 for n in ['engine.py','partition.py','projection.py','verify_report.py','PLAN.md','partition_report.zip']:shutil.copyfile(bundle/n,local/n)
 # Include only explicit completed artifact references, never raw features/partition inputs.
 included={}
 for folder in checkpoints:
  try:
   m=read(folder/'complete.json')
   for n,h in m['files'].items():
    if sha(folder/n)!=h:raise ValueError('Output changed during report construction')
    included[str((folder/n).relative_to(remote))]=folder/n
   included[str((folder/'complete.json').relative_to(remote))]=folder/'complete.json'
  except OSError as ex:summary['status']='CIDDS_WARNING_PILOT_PARTIAL_RESUMABLE';summary['errors'].append({'type':type(ex).__name__,'message':'Drive checkpoint unavailable during report collection'});write(local/'summary.json',summary);break
 names=sorted(reports)+['engine.py','partition.py','projection.py','verify_report.py','PLAN.md','partition_report.zip'];manifest={n:sha(local/n) for n in names};manifest.update({n:sha(p) for n,p in included.items()});write(local/'SHA256SUMS.json',manifest)
 report=local/('cidds_warning_pilot_report_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.zip')
 with zipfile.ZipFile(report,'w',zipfile.ZIP_DEFLATED) as z:
  for n in names+['SHA256SUMS.json']:z.write(local/n,n)
  for n,p in included.items():z.write(p,n)
 try:shutil.copyfile(report,remote.parent/report.name)
 except OSError:print('Report retained on Colab; Drive copy unavailable',flush=True)
 print(summary['status'],flush=True);print('Report:',report,flush=True);return report

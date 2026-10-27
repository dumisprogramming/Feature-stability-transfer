"""Bounded post hoc CTU warning pilot; no raw download and no target adaptation."""
from pathlib import Path
from itertools import combinations
from datetime import datetime,timezone
import json,hashlib,shutil,csv,platform,sys,time,zipfile,warnings,os
import numpy as np
import scipy,sklearn,joblib,threadpoolctl
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import roc_curve
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
STAGE='supplementary_ctu13_warning_pilot_20261027'
FEATURES=['protocol_is_tcp','destination_port','duration_seconds','total_packets','total_bytes','source_bytes']
DEFAULT={'repeats':5,'C':1.,'tol':1e-8,'max_iter':5000,'ridge_alpha':1.,'synthetic_test':False}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def savez(p,**a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.npz.tmp')
 with t.open('wb') as f:np.savez_compressed(f,**a)
 t.replace(p)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def table(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:p.write_text('');return
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def safe_rho(a,b):
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
 if len(a)<2 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):return None
 a=rankdata(a);b=rankdata(b);a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
 return float(np.clip(a@b/den,-1,1)) if den else None
def metrics(y,p,threshold):
 y=np.asarray(y);pred=np.asarray(p)>=threshold
 tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum());tp=int(((y==1)&pred).sum())
 if not (tn+fp and tp+fn):raise ValueError('Both known classes required')
 precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn);f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
 return {'rows':len(y),'tn':tn,'fp':fp,'fn':fn,'tp':tp,'ba':.5*(tn/(tn+fp)+recall),'precision':precision,'recall':recall,'f1':f1}
def choose_threshold(y,p):
 fpr,tpr,thresholds=roc_curve(y,p,drop_intermediate=False);thresholds[0]=np.nextafter(float(np.max(p)),np.inf);scores=.5*(1+tpr-fpr);best=float(np.max(scores));chosen=float(np.max(thresholds[np.abs(scores-best)<=1e-12]));m=metrics(y,p,chosen)
 if abs(m['ba']-best)>1e-12:raise ValueError('Threshold enumeration/metric disagreement')
 return chosen,thresholds,scores,m

def role(signatures,seed):
 vals=np.array([int.from_bytes(hashlib.sha256(f'ctu-qualification-v1|global-role|{seed}|'.encode()+np.asarray(s,dtype='S32').tobytes()).digest()[:8],'big')%100 for s in signatures]);return np.searchsorted([30,40,50,100],vals,side='right')
def random_seed(source,seed,group,feature,repeat):return int.from_bytes(hashlib.sha256(f'ctu-warning-v1|{source}|{seed}|{group}|{feature}|{repeat}'.encode()).digest()[:8],'big')

class Workspace:
 def __init__(self,root,context,source,plan,local_root,config):
  self.context=context;self.config=config;self.parent=Path(root)/context['parent_stage']/'run_v1';self.remote=Path(root)/STAGE/'run_v1';self.local=Path(local_root)/STAGE/'run_v1'
  if not self.parent.is_dir():raise FileNotFoundError('Qualification folder missing: '+str(self.parent))
  self.remote.mkdir(parents=True,exist_ok=True);self.local.mkdir(parents=True,exist_ok=True)
  self.identity={'code_sha256':hashlib.sha256(source.encode()).hexdigest(),'plan_sha256':hashlib.sha256(plan.encode()).hexdigest(),'context_sha256':digest(context),'config':config,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__,'threadpoolctl':threadpoolctl.__version__,'machine':platform.machine()}
  if (self.remote/'identity.json').exists() and read(self.remote/'identity.json')!=self.identity:raise ValueError('Resume identity changed. Use the same notebook and Python/package versions; do not edit the lock.')
  write(self.local/'identity.json',self.identity);write(self.local/'context.json',context);(self.local/'CODE.py').write_text(source);(self.local/'PLAN.md').write_text(plan)
  for n in ['identity.json','context.json','CODE.py','PLAN.md']:self.upload(n)
 def upload(self,n):
  src=self.local/n;dst=self.remote/n;dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix(dst.suffix+'.tmp');shutil.copyfile(src,tmp)
  if sha(tmp)!=sha(src):raise IOError('Drive copy mismatch: '+n)
  tmp.replace(dst)
 def finish(self,marker,outputs,details):
  for n in outputs:self.upload(n)
  v={'identity':self.identity,'outputs':{n:sha(self.local/n) for n in outputs},'details':details,'details_sha256':digest(details)};write(self.local/marker,v);self.upload(marker)
 def cached(self,marker):
  src=self.remote/marker
  if not src.exists():return None
  m=read(src)
  if m['identity']!=self.identity or digest(m['details'])!=m['details_sha256']:raise ValueError('Checkpoint differs: '+marker)
  for n,h in m['outputs'].items():
   p=self.remote/n
   if sha(p)!=h:raise ValueError('Checkpoint output changed: '+n)
   q=self.local/n;q.parent.mkdir(parents=True,exist_ok=True)
   if not q.exists() or sha(q)!=h:shutil.copyfile(p,q)
  q=self.local/marker;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,q);return m['details']
 def report(self,status,errors):
  fits=list(self.local.glob('source_*/seed_*/fit.complete.json'));unitfiles=list(self.local.glob('source_*/seed_*/unit.json'));evfiles=list(self.local.glob('source_*/seed_*/test_*/metrics.json'))
  v={'status':status,'errors':errors,'completed_fit_records_restored_this_session':len(fits),'completed_warning_units_restored_this_session':len(unitfiles),'completed_test_records_restored_this_session':len(evfiles),'fresh_fits_this_session':getattr(self,'fresh_fits',0),'models_planned':len(self.context['sources'])*len(self.context['seeds']),'external_evaluations_planned':len(self.context['directions'])*len(self.context['seeds']),'post_hoc_label_informed_cohort':True,'general_independent_site_validation_completed':False,'natural_production_prevalence_established':False,'full_warning_utility_established':False,'session_seconds':time.monotonic()-self.started}
  write(self.local/'summary.json',v)
  try:self.upload('summary.json')
  except OSError:print('Drive unavailable; report retained locally.',flush=True)
  files=[p for p in self.local.rglob('*') if p.is_file() and 'cache' not in p.parts and p.name!='SHA256SUMS.json' and p.suffix in ['.json','.md','.py','.csv','.npz']];write(self.local/'SHA256SUMS.json',{str(p.relative_to(self.local)):sha(p) for p in sorted(files)});stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f');zp=self.local.parent/f'ctu13_warning_pilot_report_{stamp}.zip'
  with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
   for p in files+[self.local/'SHA256SUMS.json']:z.write(p,str(p.relative_to(self.local)))
  try:
   dst=self.remote.parent/zp.name;tmp=dst.with_suffix('.zip.tmp');shutil.copyfile(zp,tmp)
   if sha(tmp)!=sha(zp):raise IOError('Report copy mismatch')
   tmp.replace(dst);print('Drive report:',dst,flush=True)
  except OSError:print('Use the local report download.',flush=True)
  print(status,flush=True);print('Local report:',zp,flush=True);return zp

def validate_inputs(w):
 c=w.context
 if read(w.parent/'identity.json')!=c['parent_identity']:raise ValueError('Qualification identity changed')
 for i,(n,h) in enumerate(c['parent_pins'].items()):
  if sha(w.parent/n)!=h:raise ValueError('Qualification record changed: '+n)
  if (i+1)%50==0:print('Verified qualification records',i+1,'/',len(c['parent_pins']),flush=True)
 panels={};union={seed:{r:[] for r in ['train','calibration','test','environment']} for seed in c['seeds']};localchecks=0
 for i,rec in enumerate(c['panel_records']):
  sid,seed,qual=rec['scenario'],rec['seed'],rec['qualification'];panelset={}
  for rn,meta in qual['panels'].items():
   rel=meta['path'];p=w.parent/rel;q=w.local/'cache'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
   if sha(q)!=meta['sha256']:raise ValueError('Selected input panel changed: '+rel)
   a=load(q);x,y,s=a['x'],a['y'],a['signature'];n=len(y)
   if x.dtype!=np.dtype('float64') or x.shape!=(n,6) or not np.all(np.isfinite(x)) or np.any(x<0):raise ValueError('Invalid selected model matrix')
   if s.dtype!=np.dtype('S32') or len(s)!=n or len(np.unique(s))!=n or n!=meta['rows']:raise ValueError('Selected signature/count mismatch')
   if not set(y.tolist())<={0,1} or int((y==0).sum())!=meta['benign'] or int((y==1).sum())!=meta['attack']:raise ValueError('Known-label/count mismatch')
   if not np.all(np.isin(x[:,0],[0,1])) or np.any(x[:,1]>65535) or np.any(x[:,[1,3,4,5]]!=np.floor(x[:,[1,3,4,5]])) or np.any(x[:,5]>x[:,4]):raise ValueError('Numeric representation domain mismatch')
   if any(len(a[k])!=n for k in a):raise ValueError('Panel array lengths disagree')
   if n and not np.all(role(s,seed)==['train','calibration','test','environment'].index(rn)):raise ValueError('Global role assignments disagree')
   if rn=='environment':
    groups=qual['environment_groups'];quota=qual['environment_quota_per_class']
    for gn in ['actual_group','pseudo_group']:
     if set(a[gn].tolist())!=set(range(groups)):raise ValueError('Group index mismatch')
     for g in range(groups):
      if any(np.count_nonzero((a[gn]==g)&(y==z))!=quota for z in [0,1]):raise ValueError('Group quota mismatch')
    keptblocks=[v['block'] for v in qual['environment_windows'] if v['retained']]
    if len(keptblocks)!=groups:raise ValueError('Retained window count differs')
    for g,block in enumerate(keptblocks):
     if set(a['block'][a['actual_group']==g].tolist())!={block}:raise ValueError('Actual time-window membership differs')
    expectedpseudo=np.full(n,-1,dtype='int32')
    for clsid in [0,1]:
     ids=np.flatnonzero(y==clsid);order=sorted(ids.tolist(),key=lambda i:hashlib.sha256(f'ctu-qualification-v1|pseudo-{sid}-class-{clsid}|{seed}|'.encode()+np.asarray(s[i],dtype='S32').tobytes()).digest());expectedpseudo[np.asarray(order,dtype='int64')]=np.repeat(np.arange(groups),quota)
    if not np.array_equal(expectedpseudo,a['pseudo_group']):raise ValueError('Matched pseudo assignment differs')
   union[seed][rn].append(s);panelset[rn]=a;panels[(sid,seed,rn)]=a
  for u,v in combinations(panelset,2):
   if np.intersect1d(panelset[u]['signature'],panelset[v]['signature']).size:raise ValueError('Within-capture role overlap')
   localchecks+=1
  if (i+1)%10==0:print('Verified selected panel sets',i+1,'/',len(c['panel_records']),flush=True)
 checks=[]
 for seed,roles in union.items():
  u={k:np.unique(np.concatenate(v)) for k,v in roles.items()}
  for a,b in combinations(u,2):
   count=len(np.intersect1d(u[a],u[b],assume_unique=True));checks.append({'seed':seed,'roles':[a,b],'intersection':count})
   if count:raise ValueError('Cross-capture cross-role overlap')
 v={'pinned_records':len(c['parent_pins']),'panel_arrays_verified':len(panels),'within_capture_role_pairs_checked':localchecks,'cross_capture_role_pairs':checks,'raw_recovery_independently_replayed':False,'unknown_used':0,'support_thresholds_relaxed':False};write(w.local/'input_validation.json',v);w.finish('inputs.complete.json',['input_validation.json'],v);return panels

def unit_tag(s,seed):return f'source_{s:02d}/seed_{seed}'
def numeric_predict(model,x):
 from scipy.special import expit
 z=(x-model['scaler_mean'])/model['scaler_scale'];return expit(z@model['coef']+model['intercept'][0])
def fit_source(w,s,seed,panels):
 tag=unit_tag(s,seed);cached=w.cached(tag+'/fit.complete.json')
 if cached is not None:return load(w.local/(tag+'/fit.npz')),cached
 train=panels[(s,seed,'train')];cal=panels[(s,seed,'calibration')];x=train['x'];y=train['y'];scaler=StandardScaler().fit(x);lr=LogisticRegression(C=w.config['C'],solver='lbfgs',tol=w.config['tol'],max_iter=w.config['max_iter'],fit_intercept=True)
 with threadpool_limits(limits=1),warnings.catch_warnings():
  warnings.simplefilter('error',ConvergenceWarning);lr.fit(scaler.transform(x),y)
 if not np.array_equal(lr.classes_,[0,1]) or not np.all(np.isfinite(lr.coef_)):raise ValueError('Invalid fitted classifier')
 m={'scaler_mean':scaler.mean_,'scaler_scale':scaler.scale_,'coef':lr.coef_[0],'intercept':lr.intercept_};p=lr.predict_proba(scaler.transform(cal['x']))[:,1];ind=numeric_predict(m,cal['x']);difference=float(np.max(np.abs(p-ind)))
 if difference>1e-12:raise ValueError('Independent numerical classifier disagrees')
 p=ind;threshold,ts,bs,cm=choose_threshold(cal['y'],p);m['threshold']=np.array([threshold]);savez(w.local/(tag+'/fit.npz'),**m);savez(w.local/(tag+'/calibration.npz'),y=cal['y'],probability=p,threshold_candidates=ts,ba_candidates=bs)
 d={'source':s,'seed':seed,'threshold':threshold,'n_iter':int(lr.n_iter_[0]),'training_rows':len(y),'calibration':cm,'calibration_max_forward_difference':difference,'training_constant_columns':[FEATURES[i] for i in range(6) if np.ptp(x[:,i])==0],'source_only_fit':True};write(w.local/(tag+'/fit.json'),d);w.finish(tag+'/fit.complete.json',[tag+'/fit.npz',tag+'/calibration.npz',tag+'/fit.json'],d);w.fresh_fits+=1;return m,d

def importance(w,s,seed,kind,g,arr,m):
 tag=unit_tag(s,seed)+f'/importance/{kind}_{g:03d}';old=w.cached(tag+'.complete.json')
 if old is not None:return load(w.local/(tag+'.npz'))['drops'],old
 ids=np.flatnonzero(arr[kind+'_group']==g);x=arr['x'][ids];y=arr['y'][ids];threshold=float(m['threshold'][0]);base=metrics(y,numeric_predict(m,x),threshold);drops=np.empty((6,w.config['repeats']))
 for j in range(6):
  for k in range(w.config['repeats']):
   perm=np.random.default_rng(random_seed(s,seed,g,j,k)).permutation(len(y));xp=x.copy();xp[:,j]=x[perm,j];drops[j,k]=base['ba']-metrics(y,numeric_predict(m,xp),threshold)['ba']
 d={'source':s,'seed':seed,'kind':kind,'group':g,'baseline':base,'importance_mean':drops.mean(axis=1).tolist(),'repeats':w.config['repeats']};savez(w.local/(tag+'.npz'),drops=drops);write(w.local/(tag+'.json'),d);w.finish(tag+'.complete.json',[tag+'.npz',tag+'.json'],d);print(f'  Source {s} seed {seed}: {kind} group {g+1}',flush=True);return drops,d

def score_panel(w,s,seed,t,a,m):
 tag=unit_tag(s,seed)+f'/test_{t:02d}';old=w.cached(tag+'/evaluation.complete.json')
 if old is not None:return old
 p=numeric_predict(m,a['x']);threshold=float(m['threshold'][0]);v={'source':s,'seed':seed,'target':t,'self_test':s==t,'threshold':threshold,**metrics(a['y'],p,threshold)};savez(w.local/(tag+'/probabilities.npz'),y=a['y'],probability=p);write(w.local/(tag+'/metrics.json'),v);w.finish(tag+'/evaluation.complete.json',[tag+'/probabilities.npz',tag+'/metrics.json'],v);return v

def aggregate_warning(vectors):
 pairs=[{'a':a,'b':b,'rho':safe_rho(vectors[a],vectors[b])} for a,b in combinations(range(len(vectors)),2)];valid=[p['rho'] for p in pairs if p['rho'] is not None];instability=1-float(np.mean(valid)) if pairs and len(valid)==len(pairs) else None
 return {'instability':instability,'total_pairs':len(pairs),'finite_pairs':len(valid),'pairs':pairs}
def run_unit(w,s,seed,panels):
 tag=unit_tag(s,seed);m,fit=fit_source(w,s,seed,panels);a=panels[(s,seed,'environment')];n=len(np.unique(a['actual_group']));vectors={}
 for kind in ['actual','pseudo']:
  vectors[kind]=[importance(w,s,seed,kind,g,a,m)[0].mean(axis=1) for g in range(n)]
 env=aggregate_warning(vectors['actual']);sampling=aggregate_warning(vectors['pseudo']);gap=env['instability']-sampling['instability'] if env['instability'] is not None and sampling['instability'] is not None else None
 evaluations=[score_panel(w,s,seed,t,panels[(t,seed,'test')],m) for t in sorted(set(w.context['targets']+[s]))];source=next(v for v in evaluations if v['target']==s);losses=[{'source':s,'seed':seed,'target':v['target'],'source_ba':source['ba'],'target_ba':v['ba'],'loss':source['ba']-v['ba']} for v in evaluations if not v['self_test']]
 d={'source':s,'seed':seed,'environment_groups':n,'environment_instability':env['instability'],'sampling_instability':sampling['instability'],'gap':gap,'warning_available':gap is not None,'environment_pairs':env,'sampling_pairs':sampling,'source_ba':source['ba'],'source_error':1-source['ba'],'mean_loss':float(np.mean([v['loss'] for v in losses])),'target_count':len(losses),'target_losses':losses}
 write(w.local/(tag+'/unit.json'),d);w.finish(tag+'/unit.complete.json',[tag+'/unit.json'],d);return d

SPECS={'intercept_only':[],'source_error':['source_error'],'environment_only':['environment_instability'],'gap_only':['gap'],'environment_and_gap':['environment_instability','gap']}
def fit_predict_spec(train,test,features,alpha):
 yy=np.array([r['outcome'] for r in train]);xx=np.array([[r[n] for n in features] for r in train],dtype=float);xt=np.array([[r[n] for n in features] for r in test],dtype=float)
 if not features:return np.full(len(test),yy.mean()),{'training_mean':float(yy.mean()),'features':[]}
 sc=StandardScaler().fit(xx);reg=Ridge(alpha=alpha).fit(sc.transform(xx),yy);pred=reg.predict(sc.transform(xt));params={'features':features,'scaler_mean':sc.mean_.tolist(),'scaler_scale':sc.scale_.tolist(),'coef':reg.coef_.tolist(),'intercept':float(reg.intercept_),'alpha':alpha}
 if not np.allclose((xt-sc.mean_)/sc.scale_@reg.coef_+reg.intercept_,pred,atol=1e-12,rtol=0):raise ValueError('Ridge numerical replay differs')
 return pred,params

def predictive_check(units,context,alpha):
 complete=[u for u in units if u['warning_available']];excluded=[{'source':u['source'],'seed':u['seed'],'reason':'Undefined environmental or sampling rank stability'} for u in units if not u['warning_available']];groups=[];assigned=set()
 for g in context['same_documented_binary_groups']:
  srcs=sorted(set(g['scenarios'])&set(context['sources']))
  if srcs:groups.append(srcs);assigned.update(srcs)
 groups.extend([[s] for s in context['sources'] if s not in assigned]);schemes={'leave_source_capture_out':[[s] for s in context['sources']],'leave_documented_specimen_group_out':groups};predictions=[];fits=[];unavailable=[]
 for scheme,folds in schemes.items():
  for held in folds:
   train=[];test=[]
   for u in complete:
    isheld=u['source'] in held;ls=[v for v in u['target_losses'] if isheld or v['target'] not in held]
    if not ls:continue
    rec={k:u[k] for k in ['source','seed','source_error','environment_instability','gap']};rec.update(outcome=float(np.mean([v['loss'] for v in ls])),targets=[v['target'] for v in ls]);(test if isheld else train).append(rec)
   if len(set(t['source'] for t in train))<2 or not test:
    unavailable.append({'scheme':scheme,'held_sources':held,'reason':'Fewer than two training captures or no complete-case held-out unit'});continue
   if any(t['source'] in held or set(t['targets'])&set(held) for t in train):raise ValueError('Strict capture-role exclusion failed')
   for name,cols in SPECS.items():
    pp,params=fit_predict_spec(train,test,cols,alpha);fits.append({'scheme':scheme,'held_sources':held,'specification':name,'train_units':train,'test_unit_ids':[[t['source'],t['seed']] for t in test],'fit':params})
    for t,p in zip(test,pp):predictions.append({'scheme':scheme,'held_sources':','.join(map(str,held)),'specification':name,'source':t['source'],'seed':t['seed'],'observed_loss':t['outcome'],'predicted_loss':float(p),'absolute_error':abs(t['outcome']-float(p)),'squared_error':(t['outcome']-float(p))**2})
 per_source=[];overall=[]
 for scheme in schemes:
  for spec in SPECS:
   a=[p for p in predictions if p['scheme']==scheme and p['specification']==spec];source_rows=[]
   for s in sorted(set(p['source'] for p in a)):
    z=[p for p in a if p['source']==s];r={'scheme':scheme,'specification':spec,'source':s,'units':len(z),'mae':float(np.mean([p['absolute_error'] for p in z])),'rmse':float(np.sqrt(np.mean([p['squared_error'] for p in z])))};source_rows.append(r);per_source.append(r)
   if source_rows:overall.append({'scheme':scheme,'specification':spec,'source_captures':len(source_rows),'units':len(a),'macro_mae':float(np.mean([r['mae'] for r in source_rows])),'macro_rmse':float(np.mean([r['rmse'] for r in source_rows]))})
 for r in overall:
  base=next(x['macro_mae'] for x in overall if x['scheme']==r['scheme'] and x['specification']=='intercept_only');r['mae_improvement_vs_intercept']=base-r['macro_mae']
 return {'predictions':predictions,'fits':fits,'per_source':per_source,'summary':overall,'excluded_units':excluded,'unavailable_folds':unavailable,'schemes':schemes,'independent_institution_inference':False,'natural_prevalence':False,'general_warning_utility_established':False}

def analysis(w,units):
 outcomes=[v for u in units for v in u['target_losses']];captures=[];associations=[]
 for s in w.context['sources']:
  a=[u for u in units if u['source']==s];r={'source':s,'seed_units':len(a),'mean_loss':float(np.mean([u['mean_loss'] for u in a])),'mean_source_ba':float(np.mean([u['source_ba'] for u in a]))}
  for key in ['sampling_instability','environment_instability','gap']:r[key]=float(np.mean([u[key] for u in a])) if all(u[key] is not None for u in a) else None
  captures.append(r)
 for seed in [None]+w.context['seeds']:
  a=captures if seed is None else [u for u in units if u['seed']==seed]
  for k in ['sampling_instability','environment_instability','gap']:
   valid=[u for u in a if u[k] is not None];rho=safe_rho([u[k] for u in valid],[u['mean_loss'] for u in valid]) if len(valid)>=3 else None;associations.append({'level':'capture_seed_average' if seed is None else 'within_seed','seed':seed,'warning':k,'captures':len(valid),'rho':rho,'inference':'descriptive_no_p_value'})
 groups=w.context['same_documented_binary_groups'];same=lambda s,t:any(s in g['scenarios'] and t in g['scenarios'] for g in groups)
 for o in outcomes:o['provider_same_documented_binary']=same(o['source'],o['target'])
 prediction=predictive_check(units,w.context,w.config['ridge_alpha']);flat=[{k:u[k] for k in ['source','seed','environment_groups','environment_instability','sampling_instability','gap','warning_available','source_ba','source_error','mean_loss','target_count']} for u in units]
 outputs=[]
 for n,v in [('source_units.csv',flat),('directed_losses.csv',outcomes),('capture_summary.csv',captures),('associations.csv',associations),('prediction_units.csv',prediction['predictions']),('prediction_per_source.csv',prediction['per_source']),('prediction_summary.csv',prediction['summary'])]:table(w.local/n,v);outputs.append(n)
 write(w.local/'predictive_check.json',prediction);outputs.append('predictive_check.json');v={'source_units':len(units),'source_captures':len(captures),'external_evaluations':len(outcomes),'warnings_available':sum(u['warning_available'] for u in units),'warning_units_missing':sum(not u['warning_available'] for u in units),'same_documented_binary_cases':sum(o['provider_same_documented_binary'] for o in outcomes),'predictive_fits':len(prediction['fits']),'unavailable_prediction_folds':len(prediction['unavailable_folds']),'models_fitted_total':len(units),'broader_research_requests_fully_closed':False};write(w.local/'analysis.json',v);outputs.append('analysis.json');w.finish('analysis.complete.json',outputs,v);return v

def run(root,context,source,plan,local_root='/content/paper3_ctu_warning_pilot',config=None):
 config=dict(DEFAULT if config is None else config);w=Workspace(root,context,source,plan,local_root,config);w.started=time.monotonic();w.fresh_fits=0;errors=[]
 try:
  panels=validate_inputs(w);units=[];total=len(context['sources'])*len(context['seeds']);print('Planned source fits:',total,'External evaluations:',len(context['directions'])*len(context['seeds']),flush=True)
  with threadpool_limits(limits=1):
   for s in context['sources']:
    for seed in context['seeds']:
     units.append(run_unit(w,s,seed,panels));print('Completed source units',len(units),'/',total,flush=True)
  v=analysis(w,units);print(json.dumps(v,indent=2),flush=True)
 except Exception as ex:
  import traceback;traceback.print_exc();errors.append({'type':type(ex).__name__,'message':str(ex)})
 return w.report('CTU13_WARNING_PILOT_PARTIAL_RESUMABLE' if errors else 'CTU13_WARNING_PILOT_COMPLETE',errors)

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--local-root',default='/content/paper3_ctu_warning_pilot');args=parser.parse_args();here=Path(__file__).parent
 try:
  report=run(args.root,read(here/'context.json'),(here/'engine.py').read_text(),(here/'PLAN.md').read_text(),args.local_root);print('REPORT_ZIP='+str(report),flush=True)
 except Exception as ex:
  print('INITIALIZATION_BLOCKED: '+type(ex).__name__+': '+str(ex),flush=True);sys.exit(2)

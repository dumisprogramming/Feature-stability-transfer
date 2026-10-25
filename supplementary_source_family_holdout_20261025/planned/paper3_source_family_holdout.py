"""Post hoc source-family exclusion with fixed external panels. Originals read only."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zipfile,shutil,platform,warnings,gc,time
import numpy as np,pandas as pd,joblib,sklearn,xgboost
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,confusion_matrix,balanced_accuracy_score

PLAN='''Post hoc source-family exclusion experiment. Earlier study outcomes are known.
All four original datasets, seeds 2027-2031, 49 original predictors and three original classifier configurations are retained. Exact family strings are dataset-specific; no cross-dataset equivalence is assumed. Before fitting, select every family having at least 50 attack rows in original training AND source-test partitions in all five seeds, and at least 50 retained benign and 50 retained attack calibration/training rows after removal in all seeds. Include a newly fitted full-training reference for each source/seed/model. Retain every family eligibility decision, including unsupported families.
For each eligible family, remove its attack rows from BOTH source training and source calibration, fit a new median imputer on retained training only, fit a new scaler on retained training only for logistic regression, clone the original estimator configuration and seed, and fit a new classifier. Use remaining calibration only for the original balanced-accuracy threshold rule. Do not use source-test or target outcomes to fit, tune or calibrate. Historical source models and files remain unchanged.
Primary diagnostic: recall of the excluded family on the original disjoint source-test partition, compared with the full-training refit on exactly those rows. Retain support counts, source binary confusion counts, all-family source-test recalls and retained known-family recalls. This is source-family novelty within the same capture, not independent-domain family novelty.
Secondary diagnostic: binary performance on all three audited unstratified external target panels (100000 rows each), using unchanged panels and the newly fitted source model/threshold. Retain confusion counts and descriptive paired differences versus the full-training refit. Exact family strings are not harmonised across external datasets, and their binary target outcomes do not establish recall of a semantically matched unseen target family. No warning scores are recomputed; historical warnings must NOT be assigned to newly fitted models. Thus this experiment is a detector training ablation, not validation of family-specific stability-warning utility.
No target adaptation, hyperparameter tuning, class rebalancing, clipping, new significance test or equivalence claim. Sparse families remain in eligibility records rather than silently dropped. Save a model checkpoint before target scoring, then save each evaluation independently; resume only identical software/code/plan/context/input hashes. Validate all source and target inputs before fitting. Hash saved model and prediction files. Store outputs in a new stage; never modify original experiment folders. Report actual fits and evaluations, failures and exclusions. A complete run requires all planned reference/family fits and external evaluations.
'''
STAGE='supplementary_source_family_holdout_20261025'
WORK_BASE=Path('/content')
MODELS=['LogisticRegression','RandomForest','XGBoost'];SEEDS=list(range(2027,2032))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def js(p):return json.loads(Path(p).read_text())
def clean(f,features):
 a=f[features].to_numpy(np.float64,copy=True);a[~np.isfinite(a)]=np.nan
 with np.errstate(over='ignore'):a=a.astype(np.float32)
 a[~np.isfinite(a)]=np.nan;return a
def keys(f,features):
 a=clean(f,features).astype('<f4');a[a==0]=0.;a[np.isnan(a)]=np.nan
 return np.ascontiguousarray(a).view(np.dtype((np.void,4*len(features)))).ravel()
def choose_threshold(y,p):
 fp,tp,th=roc_curve(y,p,drop_intermediate=False);ok=np.isfinite(th)&(th>=0)&(th<=1);th=th[ok];ba=(tp[ok]+1-fp[ok])/2
 ix=np.flatnonzero(np.isclose(ba,ba.max(),atol=1e-12,rtol=0));return float(th[ix[np.argmin(abs(th[ix]-.5))]])
def stats(y,pred):
 tn,fp,fn,tp=map(int,np.bincount(2*np.asarray(y,int)+np.asarray(pred,int),minlength=4))
 if min(tn+fp,fn+tp)<=0:raise ValueError('Evaluation needs both classes')
 ba=(tn/(tn+fp)+tp/(tp+fn))/2
 if abs(ba-balanced_accuracy_score(y,pred))>1e-12:raise ValueError('Independent BA check failed')
 return dict(tn=tn,fp=fp,fn=fn,tp=tp,balanced_accuracy=ba,attack_recall=tp/(tp+fn),false_positive_rate=fp/(tn+fp))
def attach_metadata(f,meta):
 if not f.row_id.is_unique:raise ValueError('Source row IDs not unique')
 m=meta.set_index('row_id',verify_integrity=True).loc[f.row_id]
 if not np.array_equal(m.Label.to_numpy(),f.Label.to_numpy()):raise ValueError('Metadata labels disagree')
 if m.Attack.isna().any() or m.Attack.astype(str).str.strip().eq('').any():raise ValueError('Missing family metadata')
 out=f.copy();out['family']=m.Attack.astype(str).to_numpy();return out
def inventory(frames):
 families=sorted(set().union(*[set(v['train'].loc[v['train'].Label.eq(1),'family'])|set(v['source_test'].loc[v['source_test'].Label.eq(1),'family']) for v in frames.values()]))
 rows=[];eligible=[]
 for family in families:
  valid=True
  for seed,v in frames.items():
   tr,ca,te=[v[s] for s in ['train','calibration','source_test']]
   nh=lambda x:int((x.Label.eq(1)&x.family.eq(family)).sum())
   trk=tr[~(tr.Label.eq(1)&tr.family.eq(family))];cak=ca[~(ca.Label.eq(1)&ca.family.eq(family))]
   support=[int(trk.Label.eq(c).sum()) for c in [0,1]]+[int(cak.Label.eq(c).sum()) for c in [0,1]]
   ok=nh(tr)>=50 and nh(te)>=50 and min(support)>=50;valid&=ok
   rows.append(dict(seed=seed,family=family,train_family_rows=nh(tr),calibration_family_rows=nh(ca),test_family_rows=nh(te),retained_train_benign=support[0],retained_train_attack=support[1],retained_cal_benign=support[2],retained_cal_attack=support[3],seed_supported=ok))
  if valid:eligible.append(family)
 for r in rows:r['eligible_all_five_seeds']=r['family'] in eligible
 return eligible,rows

class Workspace:
 def __init__(self,root,context,source):
  import pyarrow,scipy
  self.root=Path(root);self.context=context;self.local=WORK_BASE/STAGE/'run_v1';self.remote=self.root/STAGE/'run_v1';self.cache=WORK_BASE/'paper3_family_input_cache'
  self.local.mkdir(parents=True,exist_ok=True);self.cache.mkdir(parents=True,exist_ok=True)
  self.identity=dict(code_sha256=hashlib.sha256(source.encode()).hexdigest(),plan_sha256=hashlib.sha256(PLAN.encode()).hexdigest(),context_sha256=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest(),software=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,xgboost=xgboost.__version__,joblib=joblib.__version__,pyarrow=pyarrow.__version__,scipy=scipy.__version__))
 def upload(self,rel):
  a=self.local/rel;b=self.remote/rel;b.parent.mkdir(parents=True,exist_ok=True);q=b.with_suffix(b.suffix+'.tmp');shutil.copyfile(a,q)
  if sha(a)!=sha(q):raise IOError('Drive copy differs')
  q.replace(b)
 def restore(self,rel):
  a=self.local/rel;b=self.remote/rel
  if not a.exists() and b.is_file():a.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(b,a)
  return a
 def setup(self,source):
  if sklearn.__version__!='1.6.1' or xgboost.__version__!='3.4.1':raise ValueError('Original estimator versions required; rerun setup in a fresh runtime')
  p=self.restore('identity.json')
  if p.exists() and js(p)!=self.identity:raise ValueError('Existing stage differs in code/software/context. Do not overwrite it.')
  write(p,self.identity);write(self.local/'CONTEXT.json',self.context)
  (self.local/'EXPERIMENT_CODE.py').write_text(source);(self.local/'PLAN.md').write_text(PLAN)
  for rel in ['identity.json','CONTEXT.json','EXPERIMENT_CODE.py','PLAN.md']:self.upload(rel)
 def input(self,rel):
  h=self.context['pins'][rel];p=self.cache/rel
  if p.is_file() and sha(p)==h:return p
  p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');shutil.copyfile(self.root/rel,q)
  if sha(q)!=h:raise ValueError('Pinned input differs: '+rel)
  q.replace(p);return p
 def cached(self,marker):
  p=self.restore(marker)
  if not p.exists():return None
  r=js(p)
  if r['identity']!=self.identity:raise ValueError('Checkpoint identity differs')
  for rel,h in r['outputs'].items():
   q=self.restore(rel)
   if not q.is_file() or sha(q)!=h:raise ValueError('Checkpoint output differs: '+rel)
  return r['details']
 def finish(self,marker,files,record):
  write(self.local/marker,dict(identity=self.identity,outputs={r:sha(self.local/r) for r in files},details=record))
  for rel in files:self.upload(rel)
  self.upload(marker)
 def package(self,report):
  write(self.local/'verification_summary.json',report)
  files=[p for p in self.local.rglob('*') if p.is_file() and p.suffix not in ['.parquet','.joblib','.zip','.tmp'] and p.name!='report_hashes.json']
  write(self.local/'report_hashes.json',{str(p.relative_to(self.local)):sha(p) for p in files});files.append(self.local/'report_hashes.json')
  p=self.local/(STAGE+'_report_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.zip')
  with zipfile.ZipFile(p,'x',zipfile.ZIP_DEFLATED) as z:
   for f in files:z.write(f,str(f.relative_to(self.local)))
  try:self.upload('verification_summary.json');self.upload('report_hashes.json');shutil.copyfile(p,self.remote/p.name)
  except Exception as e:print('Drive report copy failed; download local report:',str(e))
  print(json.dumps(report,indent=2));print('Results:',self.remote);return p

def run(root,context,source):
 w=Workspace(root,context,source);report=dict(status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),models_completed=0,evaluations_completed=0,new_fits_this_session=0)
 records=[];evaluations=[];families=[];allframes={}
 try:
  w.setup(source);features=js(w.input('Paper3_Materialized/feature_schema_v1.json'))['predictors_in_order']
  if len(features)!=49:raise ValueError('Expected49features')
  # Preflight every pinned source/model/target input before any new fit.
  for i,rel in enumerate(context['pins']):
   w.input(rel)
   if i%25==0:print('Verified inputs',i+1,'/',len(context['pins']),flush=True)
  for ds in context['datasets']:
   # Locate the exact historical metadata checkpoint, never pick the latest arbitrarily.
   rc=context['recovery'][ds];parent=w.root/'supplementary_raw_metadata_recovery_20261024'
   matches=[p for p in parent.glob('*/'+ds+'/complete.json') if sha(p)==rc['marker_sha256']]
   if len(matches)!=1:raise ValueError('Expected one exact metadata checkpoint for '+ds)
   marker=js(matches[0]);rel=rc['lookup_relative'];mp=matches[0].parent/rel
   if sha(mp)!=rc['lookup_sha256']:raise ValueError('Metadata hash differs')
   meta=pd.read_parquet(mp);frames={}
   for seed in SEEDS:
    frame={s:attach_metadata(pd.read_parquet(w.input(f'Paper3_Sanitized/{ds}/seed_{seed}/{s}.parquet')),meta) for s in ['train','calibration','source_test']}
    for a,b in [('train','calibration'),('train','source_test'),('calibration','source_test')]:
     if np.intersect1d(keys(frame[a],features),keys(frame[b],features)).size:raise ValueError('Source partition signature overlap')
    frames[seed]=frame
   eligible,rows=inventory(frames)
   if eligible!=context['eligible_families'][ds]:raise ValueError('Family eligibility differs from recovered inventory')
   families +=[dict(source_dataset=ds,**r) for r in rows];allframes[ds]=frames
  pd.DataFrame(families).to_csv(w.local/'family_eligibility.csv',index=False);w.upload('family_eligibility.csv')
  planned=sum((1+len(context['eligible_families'][ds]))*len(SEEDS)*len(MODELS) for ds in context['datasets'])
  expected_evaluations=planned*len(context['datasets'])
  write(w.local/'execution_plan.json',dict(planned_fits=planned,planned_evaluations=expected_evaluations,eligible_families=context['eligible_families']));w.upload('execution_plan.json')
  print('Planned fits:',planned,'Evaluations:',expected_evaluations,flush=True)
  for ds in context['datasets']:
   for seed in SEEDS:
    v=allframes[ds][seed];tr,ca,te=[v[s] for s in ['train','calibration','source_test']]
    for name in MODELS:
     original=joblib.load(w.input(f'Paper3_Source_Models/{ds}/seed_{seed}/{name}_model.joblib'))
     if original['dataset']!=ds or int(original['seed'])!=seed or original['model_name']!=name or list(original['features'])!=features:raise ValueError('Historical model identity differs')
     for family in [None]+context['eligible_families'][ds]:
      token='full_reference' if family is None else 'exclude_'+hashlib.sha256(family.encode()).hexdigest()[:16]
      folder=f'jobs/{ds}/seed_{seed}/{name}/{token}';marker=folder+'/fit.json';record=w.cached(marker)
      if record is None:
       start=time.monotonic();train=tr if family is None else tr[~(tr.Label.eq(1)&tr.family.eq(family))];cal=ca if family is None else ca[~(ca.Label.eq(1)&ca.family.eq(family))]
       if family is not None and ((train.Label.eq(1)&train.family.eq(family)).any() or (cal.Label.eq(1)&cal.family.eq(family)).any()):raise ValueError('Excluded family present in fitting/calibration')
       im=SimpleImputer(strategy='median');xt=im.fit_transform(clean(train,features)).astype(np.float32);xc=im.transform(clean(cal,features)).astype(np.float32);scaler=None
       if name=='LogisticRegression':scaler=StandardScaler();xt=scaler.fit_transform(xt).astype(np.float32);xc=scaler.transform(xc).astype(np.float32)
       if xt.shape[1]!=49 or not np.isfinite(xt).all() or not np.isfinite(xc).all():raise ValueError('Invalid transformed features')
       est=clone(original['model'])
       with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always');est.fit(xt,train.Label.to_numpy(np.int8))
       if any(issubclass(x.category,sklearn.exceptions.ConvergenceWarning) for x in warns):raise ValueError('Convergence warning: do not omit failed fit')
       cp=est.predict_proba(xc)[:,1]
       if not np.isfinite(cp).all() or ((cp<0)|(cp>1)).any():raise ValueError('Invalid calibration probabilities')
       th=choose_threshold(cal.Label.to_numpy(np.int8),cp)
       bundle=dict(model=est,imputer=im,scaler=scaler,threshold=th,features=features)
       p=w.local/folder;p.mkdir(parents=True,exist_ok=True);joblib.dump(bundle,p/'model.joblib',compress=3)
       record=dict(source_dataset=ds,seed=seed,model=name,condition=token,excluded_family=family,training_rows=len(train),calibration_rows=len(cal),threshold=th,training_row_ids_sha256=hashlib.sha256(train.row_id.to_numpy('<i8').tobytes()).hexdigest(),calibration_row_ids_sha256=hashlib.sha256(cal.row_id.to_numpy('<i8').tobytes()).hexdigest(),excluded_family_training_rows=0 if family is not None else None,excluded_family_calibration_rows=0 if family is not None else None,elapsed_seconds=time.monotonic()-start,warnings=[str(x.message) for x in warns])
       w.finish(marker,[folder+'/model.joblib'],record);report['new_fits_this_session']+=1;del xt,xc,cp,est,bundle;gc.collect()
      records.append(record);report['models_completed']=len(records)
      bundle=None
      for target in [ds]+[x for x in context['datasets'] if x!=ds]:
       emarker=folder+'/eval_'+target+'.json';ev=w.cached(emarker)
       if ev is None:
        if bundle is None:bundle=joblib.load(w.restore(folder+'/model.joblib'))
        if target==ds:f=te;truth=te.Label.to_numpy(np.int8)
        else:
         prefix=f'supplementary_natural_prevalence_20261024/run_v1/jobs/{ds}__to__{target}/seed_{seed}'
         f=pd.read_parquet(w.input(prefix+'/features.parquet'));lab=pd.read_parquet(w.input(prefix+'/labels.parquet'))
         if not f.row_id.equals(lab.row_id) or len(f)!=100000:raise ValueError('Target panel alignment/size differs')
         truth=lab.Label.to_numpy(np.int8)
        xx=bundle['imputer'].transform(clean(f,features)).astype(np.float32)
        if bundle['scaler'] is not None:xx=bundle['scaler'].transform(xx).astype(np.float32)
        prob=bundle['model'].predict_proba(xx)[:,1]
        if not np.isfinite(prob).all() or ((prob<0)|(prob>1)).any():raise ValueError('Invalid evaluation probabilities')
        pred=(prob>=bundle['threshold']).astype(np.int8)
        ev=dict(source_dataset=ds,target_dataset=target,seed=seed,model=name,condition=token,excluded_family=family,rows=len(f),**stats(truth,pred))
        if target==ds:
         familyrows=[]
         for ff in sorted(te.loc[te.Label.eq(1),'family'].unique()):
          mask=(te.Label.eq(1)&te.family.eq(ff)).to_numpy();n=int(mask.sum());det=int(pred[mask].sum());familyrows.append(dict(family=ff,support=n,detected=det,missed=n-det,recall=det/n,excluded_from_fit=ff==family))
         ev['source_test_family_recalls']=familyrows
        rel=folder+'/predictions_'+target+'.parquet';pd.DataFrame(dict(row_id=f.row_id.to_numpy(),y_true=truth,probability_attack=prob,y_pred=pred)).to_parquet(w.local/rel,index=False)
        w.finish(emarker,[rel],ev);del xx,prob,pred,f;gc.collect()
       evaluations.append(ev);report['evaluations_completed']=len(evaluations)
      write(w.local/'progress.json',report);w.upload('progress.json')
      print('Completed models',len(records),'/',planned,'| evaluations',len(evaluations),'|',ds,seed,name,token,flush=True)
      del bundle;gc.collect()
  flat=[{k:v for k,v in r.items() if k!='source_test_family_recalls'} for r in evaluations];scores=pd.DataFrame(flat)
  refs=scores[scores.condition.eq('full_reference')].set_index(['source_dataset','target_dataset','seed','model'])
  scores['ba_change_vs_full_refit']=[float(r.balanced_accuracy-refs.loc[(r.source_dataset,r.target_dataset,r.seed,r.model),'balanced_accuracy']) for _,r in scores.iterrows()];scores.to_csv(w.local/'scores.csv',index=False);w.upload('scores.csv')
  famrows=[]
  for r in evaluations:
   for f in r.get('source_test_family_recalls',[]):famrows.append({k:r[k] for k in ['source_dataset','seed','model','condition','excluded_family']}|f)
  fam=pd.DataFrame(famrows);ref=fam[fam.condition.eq('full_reference')].set_index(['source_dataset','seed','model','family']).recall
  fam['recall_change_vs_full_refit']=[float(r.recall-ref.loc[(r.source_dataset,r.seed,r.model,r.family)]) for _,r in fam.iterrows()];fam.to_csv(w.local/'source_family_scores.csv',index=False);w.upload('source_family_scores.csv')
  pd.DataFrame(records).to_csv(w.local/'fit_records.csv',index=False);w.upload('fit_records.csv')
  if len(records)!=planned or len(evaluations)!=expected_evaluations:raise ValueError('Incomplete experiment grid')
  report.update(status='PASS',planned_models=planned,planned_evaluations=expected_evaluations,eligible_dataset_family_combinations=sum(map(len,context['eligible_families'].values())),completed_utc=datetime.now(timezone.utc).isoformat(),no_new_warning_validation=True,no_independent_new_dataset=True)
 except Exception as e:report.update(status='INCOMPLETE',error_type=type(e).__name__,error=str(e));print('STOPPED:',type(e).__name__,str(e),flush=True)
 return w.package(report)

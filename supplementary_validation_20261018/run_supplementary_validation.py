"""Supplementary saved-result validation. Does not modify historical inputs.
Usage: python run_supplementary_validation.py --repo /path/to/repository --output /new/output
Run first with --check-only to validate all fold partitions without fitting.
"""
import argparse, hashlib, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

KEY=['source_dataset','target_dataset','seed','model']
UNIT=['source_dataset','seed','model']
WARN=['environmental_instability','sampling_environment_gap','sampling_instability']
SPECS={
 'mean_only':[], 'model_only':[],
 'model_plus_sampling_instability':['sampling_instability'],
 'model_plus_environmental_instability':['environmental_instability'],
 'model_plus_sampling_environment_gap':['sampling_environment_gap'],
 'model_plus_primary_combined':['environmental_instability','sampling_environment_gap']}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def rho(a,b):
 if len(np.unique(a))<2 or len(np.unique(b))<2:return np.nan
 return float(spearmanr(a,b).statistic)
def aggregate(frame):
 return frame.groupby(UNIT,as_index=False).agg(loss=('balanced_accuracy_loss','mean'),targets=('target_dataset','nunique'),**{w:(w,'first') for w in WARN})
def fold(frame,h):
 train_rows=frame[(frame.source_dataset!=h)&(frame.target_dataset!=h)]
 test_rows=frame[frame.source_dataset==h]
 nseed=frame.seed.nunique()
 assert h not in set(train_rows.source_dataset)|set(train_rows.target_dataset)
 assert len(train_rows)==6*nseed*3 and len(test_rows)==3*nseed*3
 tr,te=aggregate(train_rows),aggregate(test_rows)
 assert len(tr)==3*nseed*3 and len(te)==nseed*3
 assert tr.targets.eq(2).all() and te.targets.eq(3).all()
 return tr,te

def design(train,test,cols,model):
 a=[];b=[];scales={}
 if model:
  for m in ['RandomForest','XGBoost']:
   a.append(train.model.eq(m).to_numpy(float));b.append(test.model.eq(m).to_numpy(float))
 for w in cols:
  mean=float(train[w].mean());sd=float(train[w].std(ddof=0))
  if sd==0:sd=1.0
  scales[w]={'training_mean':mean,'training_population_sd_used':sd}
  a.append((train[w].to_numpy()-mean)/sd);b.append((test[w].to_numpy()-mean)/sd)
 return np.column_stack(a),np.column_stack(b),scales

def validate(frame,label):
 predictions=[];diagnostics=[]
 for h in sorted(frame.source_dataset.unique()):
  tr,te=fold(frame,h)
  for spec,cols in SPECS.items():
   if spec=='mean_only':
    yhat=np.full(len(te),tr.loss.mean());rank=1;condition=1.0;parameters=1;scales={}
   else:
    a,b,scales=design(tr,te,cols,True)
    augmented=np.column_stack([np.ones(len(a)),a])
    rank=int(np.linalg.matrix_rank(augmented));parameters=augmented.shape[1]
    condition=float(np.linalg.cond(augmented))
    est=LinearRegression().fit(a,tr.loss.to_numpy());yhat=est.predict(b)
   assert np.isfinite(yhat).all()
   diagnostics.append({'run':label,'held_out_dataset':h,'specification':spec,'train_units':len(tr),'test_units':len(te),'design_rank':rank,'parameters_with_intercept':parameters,'rank_deficient':rank<parameters,'condition_number':condition if np.isfinite(condition) else None,'scaling':scales})
   for (_,r),pred in zip(te.iterrows(),yhat):
    predictions.append({'run':label,'held_out_dataset':h,'source_dataset':r.source_dataset,'seed':int(r.seed),'model':r.model,'specification':spec,'actual_loss':float(r.loss),'predicted_loss':float(pred)})
 out=pd.DataFrame(predictions)
 assert len(out)==frame.seed.nunique()*3*4*6
 return out,diagnostics

def scores(pred):
 rows=[]
 for level,groups in [('overall',['run','specification']),('fold',['run','held_out_dataset','specification']),('model',['run','model','specification'])]:
  for key,g in pred.groupby(groups,sort=True):
   row=dict(zip(groups,key));err=g.actual_loss-g.predicted_loss
   row.update(level=level,units=len(g),mae=float(np.abs(err).mean()),rmse=float(np.sqrt((err**2).mean())),spearman_rho=rho(g.actual_loss,g.predicted_loss))
   rows.append(row)
 s=pd.DataFrame(rows)
 scope=['run','level','held_out_dataset','model']
 for c in scope:s[c]=s[c].fillna('ALL')
 for base,col in [('model_only','mae_improvement_vs_model_only'),('model_plus_sampling_instability','mae_improvement_vs_sampling_baseline')]:
  mapping=s[s.specification==base].set_index(scope).mae
  s[col]=[float(mapping.loc[tuple(r[c] for c in scope)])-r.mae for _,r in s.iterrows()]
 return s

def associations(frame,label):
 u=aggregate(frame)
 means=u.groupby(['source_dataset','model'],as_index=False)[['loss']+WARN].mean()
 return [{'deletion':label,'predictor':w,'source_model_units':len(means),'spearman_rho':rho(means[w],means.loss)} for w in WARN]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
 src=args.repo/'Paper3_Final_Evaluation/transfer_analysis_all_180.csv';f=pd.read_csv(src)
 assert len(f)==180 and not f.duplicated(KEY).any()
 datasets=sorted(f.source_dataset.unique());seeds=sorted(f.seed.unique());models=sorted(f.model.unique())
 assert len(datasets)==4 and seeds==list(range(2027,2032)) and models==['LogisticRegression','RandomForest','XGBoost']
 import itertools
 assert set(map(tuple,f[KEY].to_numpy()))==set((s,t,k,m) for s,t in itertools.permutations(datasets,2) for k in seeds for m in models)
 assert np.isfinite(f[['balanced_accuracy_loss','source_test_balanced_accuracy','target_balanced_accuracy']+WARN].to_numpy()).all()
 assert np.allclose(f.balanced_accuracy_loss,f.source_test_balanced_accuracy-f.target_balanced_accuracy,rtol=0,atol=1e-12)
 assert (f.groupby(UNIT)[WARN].nunique()==1).all().all()
 checks=[]
 for omitted in [None]+seeds:
  sub=f if omitted is None else f[f.seed!=omitted]
  for h in datasets:
   tr,te=fold(sub,h);checks.append({'omitted_seed':None if omitted is None else int(omitted),'held_out_dataset':h,'train_units':len(tr),'test_units':len(te)})
 # Meaningful leakage test: changing held-out-source outcomes cannot change training targets or scaling.
 h=datasets[0];tr,te=fold(f,h);changed=f.copy();changed.loc[changed.source_dataset==h,'balanced_accuracy_loss']+=100
 tr2,te2=fold(changed,h);pd.testing.assert_frame_equal(tr,tr2)
 _,_,scale1=design(tr,te,WARN[:2],True);_,_,scale2=design(tr2,te2,WARN[:2],True);assert scale1==scale2
 # Changing test warning values must not change training-fold scaling.
 te3=te.copy();te3[WARN]=te3[WARN]+100
 assert design(tr,te3,WARN[:2],True)[2]==scale1
 print('PASS: complete input grid, numeric consistency, 24 fold partitions, held-out outcome and scaling isolation')
 if args.check_only:return
 args.output.mkdir(parents=True,exist_ok=True)
 if (args.output/'predictions.csv').exists():raise RuntimeError('Refusing to overwrite completed results; choose a new output directory')
 plan=Path(__file__).with_name('Paper3_Supplementary_Validation_Plan.md')
 inputs=[src,args.repo/'Paper3_Protocol/amendment_A3_confirmatory_analysis_plan.json',plan,Path(__file__)]
 manifest={'started_utc':datetime.now(timezone.utc).isoformat(),'analysis_status':'supplementary_after_original_results_known','input_hashes':{p.name:sha(p) for p in inputs},'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__},'structural_checks':checks,'deviations':[]}
 dump(args.output/'execution_manifest.json',manifest) # Written before any fit.
 pp=[];dd=[];aa=[]
 p,d=validate(f,'all_seeds');pp.append(p);dd+=d
 for h in datasets:aa+=associations(f[(f.source_dataset!=h)&(f.target_dataset!=h)],'dataset:'+h)
 for seed in seeds:
  sub=f[f.seed!=seed];label='omit_seed:'+str(seed)
  p,d=validate(sub,label);pp.append(p);dd+=d;aa+=associations(sub,label)
 pred=pd.concat(pp,ignore_index=True);sc=scores(pred)
 pred.to_csv(args.output/'predictions.csv',index=False);sc.to_csv(args.output/'scores.csv',index=False);pd.DataFrame(aa).to_csv(args.output/'association_sensitivity.csv',index=False);dump(args.output/'design_diagnostics.json',dd)
 assert len(pred)==1800 and len(aa)==27 and len(dd)==144
 assert all(x['design_rank']<=x['parameters_with_intercept'] for x in dd)
 # Equal fold weighting and pooled errors must agree for these equal-sized folds.
 for (run,spec),g in sc[sc.level=='fold'].groupby(['run','specification']):
  overall=sc[(sc.run==run)&(sc.specification==spec)&(sc.level=='overall')].iloc[0]
  assert np.isclose(g.mae.mean(),overall.mae,rtol=0,atol=1e-12)
  assert np.isclose(np.sqrt((g.rmse**2).mean()),overall.rmse,rtol=0,atol=1e-12)
 manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(),prediction_rows=len(pred),rank_deficient_designs=sum(x['rank_deficient'] for x in dd),output_hashes={p.name:sha(p) for p in args.output.iterdir() if p.is_file() and p.name!='execution_manifest.json'},validation_checks='PASS')
 dump(args.output/'execution_manifest.json',manifest)
 print('PASS: execution and score-consistency checks; outputs:',args.output)
 print(sc[(sc.run=='all_seeds')&(sc.level=='overall')][['specification','mae','rmse','mae_improvement_vs_model_only']].to_string(index=False))

if __name__=='__main__':main()

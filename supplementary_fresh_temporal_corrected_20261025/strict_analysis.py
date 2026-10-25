"""Fixed, post hoc, dataset-role-excluded prediction comparisons."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from sklearn.linear_model import LinearRegression
KEY=['source_dataset','seed','model']
WARN=['sampling_instability','environmental_instability','sampling_environment_gap']
BASE_SPECS={'mean_only':[],'model_only':[],'model_plus_sampling':WARN[:1],
 'model_plus_environment':WARN[1:2],'model_plus_gap':WARN[2:],
 'model_plus_both_primary':WARN[1:]}
def analyse(f,out,extra_specs=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 specs={**BASE_SPECS,**(extra_specs or {})};cols=list(dict.fromkeys(c for v in specs.values() for c in v))
 if len(f)!=180 or f.duplicated(KEY+['target_dataset']).any():raise ValueError('Expected complete unique 180-job input')
 if not np.isfinite(f[cols+['balanced_accuracy_loss']].to_numpy()).all():raise ValueError('Nonfinite analysis input')
 if not f.groupby(KEY)[cols].nunique().eq(1).all().all():raise ValueError('Source indicators change with target')
 def aggregate(a):return a.groupby(KEY,as_index=False).agg(loss=('balanced_accuracy_loss','mean'),**{c:(c,'first') for c in cols})
 def design(t,e,cs,name):
  a=[np.ones(len(t))];b=[np.ones(len(e))]
  if name!='mean_only':
   for m in ['RandomForest','XGBoost']:a.append(t.model.eq(m).to_numpy(float));b.append(e.model.eq(m).to_numpy(float))
  for c in cs:
   mu=t[c].mean();sd=t[c].std(ddof=0);sd=sd if sd>0 else 1.
   a.append((t[c].to_numpy()-mu)/sd);b.append((e[c].to_numpy()-mu)/sd)
  return np.column_stack(a),np.column_stack(b)
 predictions=[];diagnostics=[]
 for omit in [None]+list(range(2027,2032)):
  sub=f if omit is None else f[f.seed!=omit];run='all_seeds' if omit is None else 'omit_seed:'+str(omit)
  for held in sorted(sub.source_dataset.unique()):
   train=sub[(sub.source_dataset!=held)&(sub.target_dataset!=held)];test=sub[sub.source_dataset==held]
   if held in set(train.source_dataset)|set(train.target_dataset):raise ValueError('Held-out dataset role leakage')
   if not train.groupby(KEY).target_dataset.nunique().eq(2).all() or not test.groupby(KEY).target_dataset.nunique().eq(3).all():raise ValueError('Unexpected target counts')
   t,e=aggregate(train),aggregate(test)
   changed=sub.copy();changed.loc[(changed.source_dataset==held)|(changed.target_dataset==held),'balanced_accuracy_loss']+=1000
   pd.testing.assert_frame_equal(t,aggregate(changed[(changed.source_dataset!=held)&(changed.target_dataset!=held)]))
   for name,cs in specs.items():
    a,b=design(t,e,cs,name);mut=e.copy()
    for c in cs:mut[c]+=1000
    if not np.array_equal(a,design(t,mut,cs,name)[0]):raise ValueError('Test values changed training scaling')
    coef=np.linalg.lstsq(a,t.loss,rcond=None)[0];prediction=b@coef
    independent=LinearRegression(fit_intercept=False).fit(a,t.loss).predict(b)
    diff=float(np.max(np.abs(prediction-independent)));rank=int(np.linalg.matrix_rank(a))
    if diff>1e-10 or rank!=a.shape[1]:raise ValueError('Independent fit/rank check failed')
    diagnostics.append({'run':run,'held_out':held,'specification':name,'training_units':len(t),'test_units':len(e),'rank':rank,'columns':a.shape[1],'maximum_independent_difference':diff})
    for (_,r),v in zip(e.iterrows(),prediction):predictions.append({'run':run,'held_out':held,'source_dataset':held,'seed':int(r.seed),'model':r.model,'specification':name,'actual_loss':float(r.loss),'predicted_loss':float(v)})
 pred=pd.DataFrame(predictions);rows=[]
 for level,keys in [('overall',['run','specification']),('fold',['run','held_out','specification'])]:
  for key,g in pred.groupby(keys):
   er=g.actual_loss-g.predicted_loss;rows.append({**dict(zip(keys,key)),'level':level,'units':len(g),'mae':float(abs(er).mean()),'rmse':float(np.sqrt((er**2).mean()))})
 sc=pd.DataFrame(rows).fillna('ALL');scope=['run','level','held_out'];ref=sc[sc.specification=='model_only'].set_index(scope).mae
 sc['mae_improvement_vs_model_only']=[float(ref.loc[tuple(r[k] for k in scope)])-r.mae for _,r in sc.iterrows()]
 unit=aggregate(f).groupby(['source_dataset','model'],as_index=False)[['loss']+cols].mean()
 associations=[{'indicator':c,'units':len(unit),'spearman_rho':float(unit[c].rank().corr(unit.loss.rank())),'inference':'descriptive_only'} for c in cols]
 f.to_csv(out/'transfer_input_180.csv',index=False);pred.to_csv(out/'predictions.csv',index=False);sc.to_csv(out/'scores.csv',index=False)
 for name,value in [('design_diagnostics.json',diagnostics),('descriptive_associations.json',associations)]:
  (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
 return {'status':'PASS','fits':len(diagnostics),'predictions':len(pred),'maximum_independent_difference':max(x['maximum_independent_difference'] for x in diagnostics),'no_new_detector_training':True},sc

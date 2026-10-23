import argparse,hashlib,json,platform,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd,scipy,sklearn
from sklearn.linear_model import LinearRegression
C=['temporal_ba_sd','sampling_ba_sd','source_environment_error'];W=['environmental_instability','sampling_environment_gap']
S={'model_only':[],'model_plus_primary_warnings':W,'model_plus_temporal_ba_sd':C[:1],'model_plus_sampling_ba_sd':C[1:2],'model_plus_source_environment_error':C[2:],'model_plus_all_performance_controls':C,'performance_controls_plus_primary_warnings':C+W}
U=['source_dataset','seed','model']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def aggregate(f):return f.groupby(U,as_index=False).agg(loss=('balanced_accuracy_loss','mean'),**{c:(c,'first') for c in C+W})
def design(t,e,cols):
 a=[np.ones(len(t))];b=[np.ones(len(e))]
 for m in ['RandomForest','XGBoost']:a.append(t.model.eq(m).to_numpy(float));b.append(e.model.eq(m).to_numpy(float))
 for c in cols:
  mu=t[c].mean();sd=t[c].std(ddof=0);sd=sd if sd>0 else 1.
  a.append((t[c].to_numpy()-mu)/sd);b.append((e[c].to_numpy()-mu)/sd)
 return np.column_stack(a),np.column_stack(b)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output
 if out.exists():raise RuntimeError('Choose a fresh output directory')
 out.mkdir(parents=True)
 repo=args.repo;src=repo/'Paper3_Final_Evaluation/transfer_analysis_all_180.csv';warn=repo/'Paper3_Source_Stability/source_only_warning_scores.csv';files=sorted((repo/'Paper3_Source_Stability').rglob('permutation_importance.parquet'))
 assert len(files)==60
 manifest={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'post_hoc_after_original_outcomes_known','hashes':{str(p.relative_to(repo)):sha(p) for p in [src,warn]+files},'plan_sha256':sha(Path(__file__).with_name('PLAN.md')),'script_sha256':sha(__file__),'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}}
 write(out/'manifest.json',manifest)
 ww=pd.read_csv(warn);records=[];groups=[]
 for p in files:
  f=pd.read_parquet(p);s=json.load(open(p.parent/'stability_complete.json'));assert sha(p)==s['permutation_importance_sha256']
  key={k:f[k].iloc[0] for k in ['dataset','seed','model']};assert len(f[['dataset','seed','model']].drop_duplicates())==1
  key['seed']=int(key['seed']);r={'source_dataset':key['dataset'],'seed':key['seed'],'model':key['model']}
  for g,sub in f.groupby('group_type'):
   assert sub.groupby('group_id').feature.nunique().eq(49).all()
   assert sub.groupby('group_id').baseline_balanced_accuracy.nunique().eq(1).all()
   z=sub.drop_duplicates('group_id');assert z.rows.eq(4000).all() and z.benign_rows.eq(2000).all() and z.attack_rows.eq(2000).all()
   wrow=ww[(ww.dataset==key['dataset'])&(ww.seed==key['seed'])&(ww.model==key['model'])].iloc[0]
   assert np.isclose(z.baseline_balanced_accuracy.mean(),wrow['mean_'+g+'_baseline_ba'],atol=1e-12,rtol=0)
   r['temporal_ba_sd' if g=='environmental' else 'sampling_ba_sd']=float(z.baseline_balanced_accuracy.std(ddof=0))
   if g=='environmental':r['source_environment_error']=float(1-z.baseline_balanced_accuracy.mean())
   for _,row in z.iterrows():groups.append({**r,'group_type':g,'group_id':row.group_id,'balanced_accuracy':float(row.baseline_balanced_accuracy)})
  records.append(r)
 controls=pd.DataFrame(records);assert not controls.duplicated(U).any();assert len(controls)==60
 f=pd.read_csv(src).merge(controls,on=U,validate='many_to_one');assert len(f)==180
 assert np.isfinite(f[C+W+['balanced_accuracy_loss']]).all().all()
 controls.to_csv(out/'source_performance_controls.csv',index=False);pd.DataFrame(groups).drop(columns=C,errors='ignore').to_csv(out/'source_group_scores.csv',index=False)
 predictions=[];diagnostics=[];maxdiff=0.
 for omitted in [None]+sorted(f.seed.unique()):
  sub=f if omitted is None else f[f.seed!=omitted];run='all_seeds' if omitted is None else 'omit_seed:'+str(omitted)
  for h in sorted(sub.source_dataset.unique()):
   train=sub[(sub.source_dataset!=h)&(sub.target_dataset!=h)];test=sub[sub.source_dataset==h];t,e=aggregate(train),aggregate(test)
   assert h not in set(train.source_dataset)|set(train.target_dataset)
   assert len(t)==3*sub.seed.nunique()*3 and len(e)==sub.seed.nunique()*3
   # Held-out source outcome mutation cannot alter training loss or design.
   mutated=sub.copy();mutated.loc[mutated.source_dataset==h,'balanced_accuracy_loss']+=100
   pd.testing.assert_frame_equal(t,aggregate(mutated[(mutated.source_dataset!=h)&(mutated.target_dataset!=h)]))
   for spec,cols in S.items():
    a,b=design(t,e,cols);changed=e.copy();changed[cols]=changed[cols]+100 if cols else changed[cols]
    assert np.array_equal(a,design(t,changed,cols)[0])
    est=LinearRegression(fit_intercept=False).fit(a,t.loss);pred=est.predict(b)
    indep=b@np.linalg.lstsq(a,t.loss,rcond=None)[0];dif=float(np.max(abs(pred-indep)));maxdiff=max(maxdiff,dif);assert dif<1e-10
    rank=int(np.linalg.matrix_rank(a));assert rank==a.shape[1]
    diagnostics.append({'run':run,'held_out':h,'specification':spec,'training_units':len(t),'test_units':len(e),'rank':rank,'columns':a.shape[1],'condition_number':float(np.linalg.cond(a)),'numpy_difference':dif})
    for (_,row),v in zip(e.iterrows(),pred):predictions.append({'run':run,'held_out':h,'seed':int(row.seed),'model':row.model,'specification':spec,'actual_loss':float(row.loss),'predicted_loss':float(v)})
 pred=pd.DataFrame(predictions);score=[]
 for level,keys in [('overall',['run','specification']),('fold',['run','held_out','specification'])]:
  for key,g in pred.groupby(keys):
   error=g.actual_loss-g.predicted_loss;score.append({**dict(zip(keys,key)),'level':level,'units':len(g),'mae':float(abs(error).mean()),'rmse':float(np.sqrt((error**2).mean()))})
 sc=pd.DataFrame(score).fillna('ALL');scope=['run','level','held_out']
 for base,col in [('model_only','mae_improvement_vs_model_only'),('model_plus_all_performance_controls','mae_improvement_vs_performance_controls')]:
  bas=sc[sc.specification==base].set_index(scope).mae
  sc[col]=[float(bas.loc[tuple(r[k] for k in scope)])-r.mae for _,r in sc.iterrows()]
 pred.to_csv(out/'predictions.csv',index=False);sc.to_csv(out/'scores.csv',index=False);write(out/'design_diagnostics.json',diagnostics)
 # Compare established specifications with the prior saved validation as an independent external reference if available.
 manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(),source_files_verified=len(files),source_model_units=len(controls),fits=len(diagnostics),predictions=len(pred),max_independent_prediction_difference=maxdiff,checks='PASS',output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})
 write(out/'manifest.json',manifest)
 print(sc[(sc.run=='all_seeds')&(sc.level=='overall')].to_string(index=False))
 print('PASS',len(pred),'predictions;',len(diagnostics),'full-rank fits; maximum independent difference',maxdiff)
if __name__=='__main__':main()

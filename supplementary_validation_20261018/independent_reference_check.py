import json
from pathlib import Path
import numpy as np,pandas as pd
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
r=args.output;f=pd.read_csv(args.repo/'Paper3_Final_Evaluation/transfer_analysis_all_180.csv');p=pd.read_csv(r/'predictions.csv')
cols={'mean_only':[],'model_only':[],'model_plus_sampling_instability':['sampling_instability'],'model_plus_environmental_instability':['environmental_instability'],'model_plus_sampling_environment_gap':['sampling_environment_gap'],'model_plus_primary_combined':['environmental_instability','sampling_environment_gap']}
errors=[];groups=0
for (run,h,spec),g in p.groupby(['run','held_out_dataset','specification']):
 frame=f if run=='all_seeds' else f[f.seed!=int(run.split(':')[1])]
 t=frame[(frame.source_dataset!=h)&(frame.target_dataset!=h)]
 train=t.groupby(['source_dataset','seed','model']).agg(y=('balanced_accuracy_loss','mean'),**{c:(c,'mean') for c in cols[spec]}).reset_index()
 test=frame[frame.source_dataset==h].groupby(['seed','model']).agg(y=('balanced_accuracy_loss','mean'),**{c:(c,'mean') for c in cols[spec]}).reset_index()
 def matrix(df):
  vectors=[np.ones(len(df))]
  if spec!='mean_only':vectors += [df.model.eq(m).to_numpy(float) for m in ['RandomForest','XGBoost']]
  vectors += [df[c].to_numpy(float) for c in cols[spec]]
  return np.column_stack(vectors)
 coef=np.linalg.lstsq(matrix(train),train.y.to_numpy(),rcond=None)[0];test['reference']=matrix(test)@coef
 joined=g.merge(test,on=['seed','model'],validate='one_to_one')
 assert len(joined)==len(g)
 assert np.allclose(joined.actual_loss,joined.y,atol=1e-12,rtol=0)
 errors.extend(np.abs(joined.predicted_loss-joined.reference));groups+=1
assert max(errors)<1e-10
out={'reference':'Independent NumPy least-squares fit on unscaled covariates and independently grouped training data','fit_groups_checked':groups,'predictions_checked':len(errors),'maximum_absolute_prediction_difference':float(max(errors)),'status':'PASS'}
(r/'independent_check.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

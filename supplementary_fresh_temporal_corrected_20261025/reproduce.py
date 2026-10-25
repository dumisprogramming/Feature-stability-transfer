from pathlib import Path
import zipfile,json,hashlib,io,itertools,sys
import numpy as np,pandas as pd
from strict_analysis import analyse
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent
R=BASE/'recomputed'/datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S_%f');R.mkdir(parents=True)
z=zipfile.ZipFile(BASE/'inputs/final_report.zip')
def js(n):return json.loads(z.read(n))
def csv(n):return pd.read_csv(io.BytesIO(z.read(n)),float_precision='round_trip')
hs=js('report_hashes.json');assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in hs.items());print('HASHES',len(hs),'MATCH')
identity=js('run_identity.json');assert hashlib.sha256(z.read('EXPERIMENT_CODE.py')).hexdigest()==identity['code_sha256'];assert hashlib.sha256(z.read('PLAN.md')).hexdigest()==identity['plan_sha256']
previous=zipfile.ZipFile(BASE/'inputs/previous_corrected_report.zip');original=zipfile.ZipFile(BASE/'inputs/original_superseded_temporal_report.zip')
meta=js('CONTINUATION_INPUTS.json');repair=js('REPAIR_INPUTS.json');assert meta['identity']==json.loads(previous.read('run_identity.json'));assert repair['old_identity']==json.loads(original.read('run_identity.json'))
assert all(hashlib.sha256(previous.read(n)).hexdigest()==h for n,h in meta['hashes'].items());assert all(hashlib.sha256(original.read(n)).hexdigest()==h for n,h in repair['old_hashes'].items())
legacy=imported=new=0
for n in z.namelist():
 if n.startswith(('importance/','samples/','replays/')) and n.endswith('.json'):
  r=js(n);assert r['identity']==identity;d=r['details'];assert hashlib.sha256(json.dumps(d,sort_keys=True,allow_nan=False).encode()).hexdigest()==r['details_sha256']
  for f,h in r['outputs'].items():
   if f in z.namelist():assert hashlib.sha256(z.read(f)).hexdigest()==h
  if n.startswith('importance/'):
   if d.get('imported_corrected_checkpoint'):
    provenance=d['imported_corrected_checkpoint'];assert provenance['sha256']==meta['hashes'][n];pdeta=json.loads(previous.read(n))['details'];assert all(d[k]==v for k,v in pdeta.items());imported+=1
   if d.get('reused_from_legacy'):
    rel=d['reused_from_legacy'];assert d['legacy_record_sha256']==repair['old_hashes'][rel];old=json.loads(original.read(rel))['details'];assert all(d[k]==v for k,v in old.items());legacy+=1
   else:new+=1
print('PROVENANCE legacy',legacy,'imported',imported,'not_legacy',new)
units=csv('fresh_warning_scores_240.csv');assert len(units)==240 and not units.duplicated(['source_dataset','seed','model','hours']).any()
for r in units.itertuples():
 st={};undef={}
 for kind in ['environmental','sampling']:
  ranked=[]
  for code in range(r.groups):
   d=js(f'importance/{r.source_dataset}/seed_{r.seed}/{r.model}/{r.hours}h/{kind}_{code}.json')['details'];assert (d['rows'],d['benign'],d['attack'])==(4000,2000,2000);assert 0<=d['baseline_balanced_accuracy']<=1
   a=np.array(d['importance_repeats']);assert a.shape==(49,3) and np.isfinite(a).all();ranked.append(pd.Series(-a.mean(axis=1)).rank(method='average').to_numpy())
  values=[];u=0
  for a,b in itertools.combinations(ranked,2):
   if np.std(a)==0 or np.std(b)==0:values.append(0.);u+=1
   else:values.append(float(np.corrcoef(a,b)[0,1]))
  st[kind]=np.mean(values);undef[kind]=u;assert len(values)==r.pairs_per_type
 assert abs(1-st['environmental']-r.environmental_instability)<1e-12 and abs(1-st['sampling']-r.sampling_instability)<1e-12 and abs(st['sampling']-st['environmental']-r.sampling_environment_gap)<1e-12
 assert undef['sampling']==r.undefined_sampling_pairs and undef['environmental']==r.undefined_environmental_pairs
print('REBUILT all240 warnings')
samples=csv('sampling_summary_80.csv');assert len(samples)==80 and samples.source_signature_overlap.eq(0).all() and samples.source_row_id_overlap.eq(0).all() and samples.rows.eq(samples.groups*4000).all()
for ds in samples.dataset.unique():
 w=csv(f'samples/{ds}/window_counts.csv');assert np.array_equal(w.eligible_at_2000_each_class, (w.remaining_benign>=2000)&(w.remaining_attack>=2000))
 counts=w.groupby(['seed','hours']).eligible_at_2000_each_class.sum();a=samples[samples.dataset==ds].set_index(['seed','hours']).sort_index();assert np.array_equal(counts.sort_index().values,a.groups.values)
replays=[js(n)['details'] for n in z.namelist() if n.startswith('replays/')];assert len(replays)==60 and all(d['source_decisions_identical'] and d['maximum_probability_difference']<=1e-7 for d in replays);print('REPLAYS',len(replays),'maxdiff',max(d['maximum_probability_difference'] for d in replays))
allresults=[]
for label in ['balanced','unstratified']:
 for h in [3,6,12,24]:
  prefix=f'analysis_{label}/{h}h';f=csv(prefix+'/transfer_input_180.csv');a=f.merge(units[units.hours==h],on=['source_dataset','seed','model'],suffixes=('_input','_verified'),validate='many_to_one')
  for warn in ['sampling_instability','environmental_instability','sampling_environment_gap']:assert np.max(abs(a[warn+'_input']-a[warn+'_verified']))<1e-12
  expected=pd.DataFrame(js('OUTCOME_RECORDS.json')[label]).set_index(['source_dataset','target_dataset','seed','model']).sort_index();actual=f.set_index(['source_dataset','target_dataset','seed','model']).sort_index();assert np.max(abs(actual.balanced_accuracy_loss-expected.balanced_accuracy_loss))<1e-12
  checked,scores=analyse(f,R/'recomputed'/label/str(h));old=csv(prefix+'/predictions.csv');newp=pd.read_csv(R/'recomputed'/label/str(h)/'predictions.csv',float_precision='round_trip');assert np.max(abs(old.predicted_loss-newp.predicted_loss))<1e-10
  oldsc=csv(prefix+'/scores.csv');assert np.max(abs(oldsc.mae-scores.mae))<1e-12 and np.max(abs(oldsc.rmse-scores.rmse))<1e-12
  focus=scores[(scores.run=='all_seeds')&(scores.level=='overall')].copy();focus['outcomes']=label;focus['hours']=h;allresults.append(focus)
  primary=scores[scores.specification=='model_plus_both_primary'];print('RESULT',label,h,'fold improvements',primary[(primary.run=='all_seeds')&(primary.level=='fold')].mae_improvement_vs_model_only.round(6).tolist(),'seed deletion improvement range',primary[(primary.run!='all_seeds')&(primary.level=='overall')].mae_improvement_vs_model_only.agg(['min','max']).tolist())
results=pd.concat(allresults);print(results.pivot(index=['outcomes','hours'],columns='specification',values='mae').to_string());results.to_csv(R/'summary_scores.csv',index=False)
print('PASS all1152 independent fits,14400 predictions; report omits raw sampled Parquets, so actual per-row exclusions/importance predictions not independently replayed here.')

print("Outputs:",R)
print("Unstratified-outcome analyses are PROVISIONAL pending a separate precision-matched target exclusion audit.")

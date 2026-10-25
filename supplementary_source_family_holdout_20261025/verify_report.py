"""Independent saved-report arithmetic and provenance checks; no raw-data replay."""
from pathlib import Path
from datetime import datetime,timezone
import json,zipfile,hashlib,math,csv,io,itertools,sys,collections

B=Path(__file__).resolve().parent
default_report=B/'inputs/family_report.zip'
if not default_report.exists():default_report=B.parent/'upload/supplementary_source_family_holdout_20261025_report_20261025_170254_728555.zip'
REPORT=Path(sys.argv[1]) if len(sys.argv)>1 else default_report
O=Path(sys.argv[2]) if len(sys.argv)>2 else B/'outputs'/datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S_%f');O.mkdir(parents=True,exist_ok=True)
def near(a,b):return math.isclose(float(a),float(b),abs_tol=1e-12,rel_tol=1e-12)
def digest(b):return hashlib.sha256(b).hexdigest()
def condition(family):return 'full_reference' if family is None else 'exclude_'+digest(family.encode())[:16]
def rkey(r):return (r['source_dataset'],int(r['seed']),r['model'],r['condition'])
def ekey(r):return rkey(r)+(r['target_dataset'],)
def fkey(r):return rkey(r)+(r['family'],)
def mean(v):return sum(v)/len(v)
with zipfile.ZipFile(REPORT) as z:
 assert z.testzip() is None
 names=z.namelist();assert len(names)==len(set(names))==2262
 manifest=json.loads(z.read('report_hashes.json'))
 assert set(manifest)==set(names)-{'report_hashes.json'} and len(manifest)==2261
 for n,h in manifest.items():assert digest(z.read(n))==h,n
 js=lambda n:json.loads(z.read(n))
 csvread=lambda n:list(csv.DictReader(io.StringIO(z.read(n).decode())))
 summary=js('verification_summary.json');identity=js('identity.json');context=js('CONTEXT.json');execution=js('execution_plan.json')
 assert summary['status']=='PASS' and summary['models_completed']==450 and summary['evaluations_completed']==1800
 assert digest(z.read('EXPERIMENT_CODE.py'))==identity['code_sha256']
 assert digest(z.read('PLAN.md'))==identity['plan_sha256']
 assert digest(json.dumps(context,sort_keys=True).encode())==identity['context_sha256']
 # If the original prepared package is present, compare it byte for byte as well.
 planned=B/'planned'
 planned_checked=False
 if (planned/'paper3_source_family_holdout.py').exists():
  assert (planned/'paper3_source_family_holdout.py').read_bytes()==z.read('EXPERIMENT_CODE.py')
  assert json.loads((planned/'family_holdout_context.json').read_text())==context
  planned_checked=True
 ds=context['datasets'];models=['LogisticRegression','RandomForest','XGBoost'];seeds=list(range(2027,2032))
 assert len(context['pins'])==241
 expected={(d,s,m,condition(f)) for d in ds for s in seeds for m in models for f in [None]+context['eligible_families'][d]}
 assert len(expected)==450 and execution['planned_fits']==450 and execution['planned_evaluations']==1800
 assert sum(map(len,context['eligible_families'].values()))==26
 fits={};ev={}
 for n in names:
  if n.startswith('jobs/') and n.endswith('.json'):
   a=js(n);assert a['identity']==identity,n
   for p,h in a['outputs'].items():assert len(h)==64 and p.startswith('jobs/')
   r=a['details']
   if n.endswith('/fit.json'):
    k=rkey(r);assert k not in fits;fits[k]=r
    assert n.endswith('/'+r['condition']+'/fit.json')
    assert len(a['outputs'])==1 and next(iter(a['outputs'])).endswith('/model.joblib')
    assert not r['warnings'] and 0<=r['threshold']<=1
    f=r['excluded_family'];assert condition(f)==r['condition']
    if f is not None:assert r['excluded_family_training_rows']==r['excluded_family_calibration_rows']==0
   else:
    k=ekey(r);assert k not in ev;ev[k]=r
    assert n.endswith('/'+r['condition']+'/eval_'+r['target_dataset']+'.json')
    assert len(a['outputs'])==1 and next(iter(a['outputs'])).endswith('/predictions_'+r['target_dataset']+'.parquet')
 assert set(fits)==expected
 assert set(ev)=={k+(t,) for k in expected for t in ds}
 fitscsv=csvread('fit_records.csv');scores=csvread('scores.csv');familycsv=csvread('source_family_scores.csv');eligibility=csvread('family_eligibility.csv')
 assert len(fitscsv)==450 and len(scores)==1800
 assert len({rkey(r) for r in fitscsv})==450 and len({ekey(r) for r in scores})==1800
 for r in fitscsv:
  j=fits[rkey(r)]
  for n in ['training_rows','calibration_rows','threshold','elapsed_seconds']:assert near(r[n],j[n])
  for n in ['training_row_ids_sha256','calibration_row_ids_sha256']:assert r[n]==j[n]
 # Rebuild every support-based eligibility decision, including unsupported families.
 eligible=collections.defaultdict(list)
 byfam=collections.defaultdict(list)
 for r in eligibility:
  d=r['source_dataset'];f=r['family'];s=int(r['seed']);byfam[d,f].append(r)
  good=min(int(r[n]) for n in ['train_family_rows','test_family_rows','retained_train_benign','retained_train_attack','retained_cal_benign','retained_cal_attack'])>=50
  assert (r['seed_supported']=='True')==good
  for m in models:
   base=fits[(d,s,m,'full_reference')]
   assert base['training_rows']==sum(int(r[n]) for n in ['train_family_rows','retained_train_benign','retained_train_attack'])
   assert base['calibration_rows']==sum(int(r[n]) for n in ['calibration_family_rows','retained_cal_benign','retained_cal_attack'])
   if f in context['eligible_families'][d]:
    fit=fits[d,s,m,condition(f)]
    assert fit['training_rows']==sum(int(r[n]) for n in ['retained_train_benign','retained_train_attack'])
    assert fit['calibration_rows']==sum(int(r[n]) for n in ['retained_cal_benign','retained_cal_attack'])
 for (d,f),rr in byfam.items():
  assert len(rr)==5 and {int(r['seed']) for r in rr}==set(seeds)
  good=all(r['seed_supported']=='True' for r in rr)
  assert all((r['eligible_all_five_seeds']=='True')==good for r in rr)
  if good:eligible[d].append(f)
 assert {d:sorted(eligible[d]) for d in ds}==context['eligible_families']
 familyrecords={}
 for k,r in ev.items():
  tn,fp,fn,tp=[r[n] for n in ['tn','fp','fn','tp']]
  assert all(isinstance(x,int) and x>=0 for x in [tn,fp,fn,tp])
  assert sum([tn,fp,fn,tp])==r['rows'] and min(tn+fp,fn+tp)>0
  assert near(r['balanced_accuracy'],((tn/(tn+fp))+(tp/(tp+fn)))/2)
  assert near(r['attack_recall'],tp/(tp+fn)) and near(r['false_positive_rate'],fp/(tn+fp))
  base=ev[(r['source_dataset'],r['seed'],r['model'],'full_reference',r['target_dataset'])]
  assert tn+fp==base['tn']+base['fp'] and fn+tp==base['fn']+base['tp']
  if r['source_dataset']!=r['target_dataset']:assert r['rows']==100000
  else:
   family=r['source_test_family_recalls']
   assert sum(v['detected'] for v in family)==tp and sum(v['missed'] for v in family)==fn
   assert sum(v['support'] for v in family)==tp+fn
   for v in family:
    assert v['support']==v['detected']+v['missed']>0
    assert near(v['recall'],v['detected']/v['support'])
    assert v['excluded_from_fit']==(v['family']==r['excluded_family'])
    fk=rkey(r)+(v['family'],);assert fk not in familyrecords;familyrecords[fk]=v
    orig=ev[(r['source_dataset'],r['seed'],r['model'],'full_reference',r['source_dataset'])]
    ref=next(vv for vv in orig['source_test_family_recalls'] if vv['family']==v['family']);assert v['support']==ref['support']
 for r in scores:
  j=ev[ekey(r)]
  for n in ['rows','tn','fp','fn','tp','balanced_accuracy','attack_recall','false_positive_rate']:assert near(r[n],j[n])
  base=ev[(r['source_dataset'],int(r['seed']),r['model'],'full_reference',r['target_dataset'])]
  assert near(r['ba_change_vs_full_refit'],j['balanced_accuracy']-base['balanced_accuracy'])
 assert len(familycsv)==len(familyrecords) and len({fkey(r) for r in familycsv})==len(familyrecords)
 primary=[]
 for r in familycsv:
  j=familyrecords[fkey(r)]
  for n in ['support','detected','missed','recall']:assert near(r[n],j[n])
  assert (r['excluded_from_fit']=='True')==j['excluded_from_fit']
  ref=familyrecords[r['source_dataset'],int(r['seed']),r['model'],'full_reference',r['family']]
  delta=j['recall']-ref['recall'];assert near(delta,r['recall_change_vs_full_refit'])
  if j['excluded_from_fit']:
   primary.append(dict(source_dataset=r['source_dataset'],seed=int(r['seed']),model=r['model'],family=r['family'],support=j['support'],reference_recall=ref['recall'],excluded_recall=j['recall'],recall_change=delta))
 assert len(primary)==390
 external=[]
 for k,r in ev.items():
  if r['excluded_family'] is not None and r['source_dataset']!=r['target_dataset']:
   ref=ev[(r['source_dataset'],r['seed'],r['model'],'full_reference',r['target_dataset'])]
   external.append(dict(source_dataset=r['source_dataset'],seed=r['seed'],model=r['model'],family=r['excluded_family'],target_dataset=r['target_dataset'],reference_ba=ref['balanced_accuracy'],excluded_ba=r['balanced_accuracy'],ba_change=r['balanced_accuracy']-ref['balanced_accuracy']))
 assert len(external)==1170
 def savecsv(name,rows):
  with (O/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 savecsv('held_family_pairs_390.csv',primary);savecsv('external_pairs_1170.csv',external)
 datasetmodel=[]
 for d,m in itertools.product(ds,models):
  pp=[r for r in primary if r['source_dataset']==d and r['model']==m];ee=[r for r in external if r['source_dataset']==d and r['model']==m]
  datasetmodel.append(dict(source_dataset=d,model=m,families=len(context['eligible_families'][d]),held_family_pairs=len(pp),reference_recall=mean([r['reference_recall'] for r in pp]),excluded_recall=mean([r['excluded_recall'] for r in pp]),recall_change=mean([r['recall_change'] for r in pp]),external_pairs=len(ee),reference_ba=mean([r['reference_ba'] for r in ee]),excluded_ba=mean([r['excluded_ba'] for r in ee]),ba_change=mean([r['ba_change'] for r in ee])))
 savecsv('dataset_model_summary.csv',datasetmodel)
 familymeans=[]
 for d in ds:
  for f in context['eligible_families'][d]:
   pp=[r for r in primary if r['source_dataset']==d and r['family']==f];ee=[r for r in external if r['source_dataset']==d and r['family']==f]
   familymeans.append(dict(source_dataset=d,family=f,reference_recall=mean([r['reference_recall'] for r in pp]),excluded_recall=mean([r['excluded_recall'] for r in pp]),recall_change=mean([r['recall_change'] for r in pp]),external_ba_change=mean([r['ba_change'] for r in ee])))
 savecsv('dataset_family_summary_26.csv',familymeans)
 reference_comparison=None
 recovery_checked=False
 recovery=B/'inputs/recovery_report.zip'
 if not recovery.exists():recovery=B.parent/'upload/recovery_report_20261024_002449_541452.zip'
 if recovery.exists():
  with zipfile.ZipFile(recovery) as rz:
   for d in ds:
    marker_bytes=rz.read(d+'/complete.json');marker=json.loads(marker_bytes)
    assert digest(marker_bytes)==context['recovery'][d]['marker_sha256']
    assert marker['output_hashes'][context['recovery'][d]['lookup_relative']]==context['recovery'][d]['lookup_sha256']
    rr=list(csv.DictReader(io.StringIO(rz.read(d+'/saved_role_family_counts.csv').decode())))
    for r in eligibility:
     if r['source_dataset']!=d:continue
     for split,col in [('train','train_family_rows'),('calibration','calibration_family_rows'),('source_test','test_family_rows')]:
      n=sum(int(a['rows']) for a in rr if a['role']==f"seed_{r['seed']}/{split}" and a['label']=='1' and a['family']==r['family'])
      assert n==int(r[col])
    for p,h in context['pins'].items():
     if p.startswith('Paper3_Sanitized/'+d+'/'):assert marker['input_hashes'][p]==h
   recovery_checked=True
 old=B/'inputs/natural_prevalence_report.zip'
 if not old.exists():old=B.parent/'upload/natural_prevalence_report_20261024_015652_450517.zip'
 if old.exists():
  with zipfile.ZipFile(old) as zz:
   rr=list(csv.DictReader(io.StringIO(zz.read('natural_prevalence_metrics_180.csv').decode())))
   diffs=[];countdifferent=0
   for r in rr:
    j=ev[(r['source_dataset'],int(r['seed']),r['model'],'full_reference',r['target_dataset'])]
    diffs.append(abs(j['balanced_accuracy']-float(r['balanced_accuracy'])))
    countdifferent+=int(any(j[n]!=int(r[n]) for n in ['tn','fp','fn','tp']))
   reference_comparison={'jobs':len(rr),'maximum_ba_difference':max(diffs),'jobs_with_different_confusion_counts':countdifferent}
 out=dict(status='VERIFIED_SAVED_REPORT',report_sha256=digest(REPORT.read_bytes()),report_hashes_matched=2261,fit_markers=450,evaluation_markers=1800,family_score_records=len(familyrecords),eligibility_records=len(eligibility),eligible_dataset_family_combinations=26,excluded_fit_pairs=390,external_pairs=1170,prepared_code_and_context_match=planned_checked,recovery_support_counts_and_source_pins_checked=recovery_checked,elapsed_seconds=(datetime.fromisoformat(summary['completed_utc'])-datetime.fromisoformat(summary['started_utc'])).total_seconds(),mean_held_family_reference_recall=mean([r['reference_recall'] for r in primary]),mean_held_family_excluded_recall=mean([r['excluded_recall'] for r in primary]),mean_held_family_recall_change=mean([r['recall_change'] for r in primary]),held_family_lower=sum(r['recall_change'] < -1e-12 for r in primary),held_family_higher=sum(r['recall_change'] > 1e-12 for r in primary),held_family_unchanged=sum(abs(r['recall_change'])<=1e-12 for r in primary),mean_external_reference_ba=mean([r['reference_ba'] for r in external]),mean_external_excluded_ba=mean([r['excluded_ba'] for r in external]),mean_external_ba_change=mean([r['ba_change'] for r in external]),external_ba_lower=sum(r['ba_change'] < -1e-12 for r in external),external_ba_higher=sum(r['ba_change'] > 1e-12 for r in external),external_ba_unchanged=sum(abs(r['ba_change'])<=1e-12 for r in external),original_reference_comparison=reference_comparison,raw_training_or_predictions_independently_replayed=False)
 (O/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));print('DATASET MODEL SUMMARY');print(json.dumps(datasetmodel,indent=2));print('LARGEST FAMILY DROPS');print(json.dumps(sorted(familymeans,key=lambda r:r['recall_change'])[:5],indent=2))

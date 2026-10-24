"""Post hoc family-resolved detection recall from unchanged saved predictions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, zipfile, platform
import numpy as np
import pandas as pd

DATASETS=['NF_BoT_IoT_v3','NF_CSE_CIC_IDS2018_v3','NF_ToN_IoT_v3','NF_UNSW_NB15_v3']
MODELS=['LogisticRegression','RandomForest','XGBoost']
EXPECTED_METRICS_SHA='0f997deebc76fe4ad3ba0fc693622a13abc43573e93e41da05e707438e1595b7'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def read_json(p):return json.loads(Path(p).read_text())
def write_json(p,v):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def require_hash(p,expected):
    actual=sha(p)
    if not expected or actual!=expected:raise ValueError('File hash differs from verified reference: '+str(p))
    return actual

def family_counts(families,predictions,universe):
    """Retain absent families as n=0, recall missing; verify pandas against NumPy."""
    g=pd.DataFrame({'family':families,'detected':predictions}).groupby('family',sort=True).detected.agg(['count','sum'])
    rows=[]
    for family in universe:
        mask=np.asarray(families)==family
        n=int(mask.sum());tp=int(np.asarray(predictions)[mask].sum())
        if family in g.index:
            if int(g.loc[family,'count'])!=n or int(g.loc[family,'sum'])!=tp:raise ValueError('Independent family count disagreement')
        elif n or tp:raise ValueError('Family grouping lost observations')
        rows.append({'family':family,'n_attack':n,'detected':tp,'missed':n-tp,'recall':tp/n if n else np.nan,
            'support_status':'NO_SUPPORT' if n==0 else ('SMALL_GROUP_LT_50' if n<50 else 'N_AT_LEAST_50')})
    if sum(x['n_attack'] for x in rows)!=len(families):raise ValueError('Family universe does not cover all attack rows')
    return rows

def evaluate(root,recovery_root,output):
    root=Path(root).resolve();recovery_root=Path(recovery_root).resolve();output=Path(output).resolve()
    if output.parent!=root/'supplementary_attack_family_evaluation_20261024':raise ValueError('Use the separate family evaluation output folder')
    if not root.is_dir() or not recovery_root.is_dir():raise FileNotFoundError('Paper 3 or metadata recovery folder missing')
    output.mkdir(parents=True,exist_ok=False)
    plan='''Post hoc family-resolved transfer evaluation plan, recorded before computing these family results.
Use all 180 original target jobs (four datasets, three source classifiers, five seeds, twelve transfer directions); no retraining, calibration, threshold changes or job selection.
Use the verified raw-metadata recovery lookup and original candidate label vault. Keep exact attack family labels separate by target dataset; do not harmonise names or label attacks seen/unseen across collections.
For each job and each raw target attack family report n, detected, missed and recall TP_family/n_family. Retain n=0 with undefined recall, not zero. Retain small families; flag n<50 only as a descriptive display aid, not an inferential threshold.
Reproduce original target confusion counts, binary attack/benign recalls and balanced accuracy before accepting a job. Independently check group counts with NumPy.
Summaries use equal-weight mean/min/max recall over jobs with support and state numbers of jobs, smallest/largest family sample counts, zero-support and small-support jobs. Sum-of-job observations are repeated evaluation counts, not unique flows; no pooled independent-flow confidence interval is calculated.
Report every family for every transfer/model/seed, including negative and unsupported results. No p values, superiority tests, family-specific warning calibration or new confirmatory claim.
This evaluates a fixed binary detector's recall within target attack families. It is not family classification, leave-one-family-out training, open-set detection, natural-prevalence validation or proof that source stability predicts family-specific failure.
'''
    (output/'PLAN.md').write_text(plan);(output/'EVALUATION_CODE.py').write_text(EVALUATION_SOURCE)
    write_json(output/'PINNED_RECOVERY_INPUTS.json',PINNED_RECOVERY)
    write_json(output/'progress.json',{'status':'RUNNING','completed_jobs':0})
    try:
        metrics_path=root/'Paper3_Final_Evaluation/target_metrics_all_180.csv'
        metrics_sha=require_hash(metrics_path,EXPECTED_METRICS_SHA)
        original=pd.read_csv(metrics_path)
        keys=['source_dataset','target_dataset','seed','model']
        expected={(s,t,seed,m) for s in DATASETS for t in DATASETS if s!=t for seed in range(2027,2032) for m in MODELS}
        if len(original)!=180 or original.duplicated(keys).any() or set(map(tuple,original[keys].to_numpy()))!=expected:
            raise ValueError('Original target metrics job keys differ from the expected 180 jobs')
        orig=original.set_index(keys,verify_integrity=True)
        families_by_target={};lookups={};vaults={};inputs=[];coverage=[]
        for dataset in DATASETS:
            pin=PINNED_RECOVERY[dataset];folder=recovery_root/dataset
            require_hash(folder/'complete.json',pin['complete_sha256'])
            checkpoint=read_json(folder/'complete.json')
            lookup_path=folder/pin['lookup_relative_path']
            require_hash(lookup_path,pin['lookup_sha256'])
            rawpath=folder/pin['raw_family_relative_path'];require_hash(rawpath,pin['raw_family_sha256'])
            raw=pd.read_csv(rawpath,keep_default_na=False)
            if not set(raw.label.astype(str)).issubset({'0','1'}) or raw.family.astype(str).str.strip().eq('').any():
                raise ValueError('Missing families or invalid raw family labels')
            universe=sorted(raw.loc[raw.label.astype(str)=='1','family'].astype(str).unique())
            if not universe:raise ValueError('No raw attack family inventory')
            families_by_target[dataset]=universe
            recovered=pd.read_parquet(lookup_path,columns=['row_id','Label','Attack'])
            if not recovered.row_id.is_unique or not recovered.Label.isin([0,1]).all() or recovered.Attack.isna().any():
                raise ValueError('Recovered lookup IDs/labels/families are invalid')
            recovered=recovered.set_index('row_id',verify_integrity=True)
            vp=root/'Paper3_Target_Label_Vault'/dataset/'target_candidate_labels.parquet'
            vkey=str(vp.relative_to(root));require_hash(vp,checkpoint['input_hashes'][vkey])
            vault=pd.read_parquet(vp,columns=['row_id','Label']).set_index('row_id',verify_integrity=True)
            if len(vault)!=100000 or vault.Label.value_counts().to_dict()!={0:50000,1:50000}:raise ValueError('Original candidate quotas differ')
            aligned=recovered.reindex(vault.index)
            if aligned.Label.isna().any() or not np.array_equal(aligned.Label.to_numpy(),vault.Label.to_numpy()):raise ValueError('Candidate labels differ from recovered raw labels')
            if not set(aligned.loc[aligned.Label==1,'Attack'].astype(str)).issubset(universe):raise ValueError('Candidate family outside raw inventory')
            lookups[dataset]=recovered;vaults[dataset]=vault
            for family in universe:
                coverage.append({'target_dataset':dataset,'family':family,
                    'raw_attack_rows':int(raw.loc[(raw.label.astype(str)=='1')&(raw.family==family),'rows'].sum()),
                    'candidate_attack_rows':int(((aligned.Label==1)&(aligned.Attack==family)).sum())})
            inputs.append({'dataset':dataset,'lookup_sha256':pin['lookup_sha256'],'candidate_labels_sha256':sha(vp)})
        rows=[];checks=[];prediction_inputs=[]
        for source,target,seed,model in sorted(expected):
            key=(source,target,seed,model);old=orig.loc[key]
            base=root/'Paper3_Target_Predictions'/(source+'__to__'+target)/f'seed_{seed}'
            p=base/(model+'_target_predictions.parquet');digest=require_hash(p,str(old.prediction_file_sha256))
            mpath=base/(model+'_prediction_manifest.json');manifest=read_json(mpath)
            if manifest['prediction_file_sha256']!=digest:raise ValueError('Prediction manifest and original result hash disagree')
            f=pd.read_parquet(p,columns=['row_id','attack_probability','predicted_label'])
            if len(f)!=80000 or not f.row_id.is_unique or not f.predicted_label.isin([0,1]).all():raise ValueError('Invalid prediction panel')
            pro=f.attack_probability.to_numpy(float);pred=f.predicted_label.to_numpy(int);threshold=float(manifest['model_threshold'])
            if not (np.isfinite(pro).all() and ((pro>=0)&(pro<=1)).all() and np.isfinite(threshold) and 0<=threshold<=1):raise ValueError('Invalid probabilities or threshold')
            if not np.array_equal((pro>=threshold).astype(int),pred):raise ValueError('Stored predictions disagree with fixed threshold')
            labels=vaults[target].Label.reindex(f.row_id)
            recovered=lookups[target].reindex(f.row_id)
            if labels.isna().any() or recovered.Label.isna().any() or not np.array_equal(labels.to_numpy(),recovered.Label.to_numpy()):raise ValueError('Prediction/label/family join failed')
            y=labels.to_numpy(int);attack=y==1;benign=y==0
            if int(attack.sum())!=40000 or int(benign.sum())!=40000:raise ValueError('Target job quotas differ')
            confusion=np.bincount(2*y+pred,minlength=4);tn,fp,fn,tp=map(int,confusion)
            for name,value in zip(['tn','fp','fn','tp'],[tn,fp,fn,tp]):
                if int(old[name])!=value:raise ValueError('Original confusion count mismatch: '+name)
            br=tn/(tn+fp);ar=tp/(tp+fn);ba=(br+ar)/2
            for name,value in [('target_benign_recall',br),('target_attack_recall',ar),('target_balanced_accuracy',ba)]:
                if not np.isclose(value,float(old[name]),rtol=0,atol=1e-12):raise ValueError('Original metric mismatch: '+name)
            counts=family_counts(recovered.Attack.to_numpy(dtype=str)[attack],pred[attack],families_by_target[target])
            if sum(x['detected'] for x in counts)!=tp or sum(x['missed'] for x in counts)!=fn:raise ValueError('Family decomposition differs from binary outcomes')
            for rec in counts:rows.append(dict(zip(keys,key))|rec)
            checks.append(dict(zip(keys,key))|{'tn':tn,'fp':fp,'fn':fn,'tp':tp,'benign_recall':br,'attack_recall':ar,'balanced_accuracy':ba,
                'original_counts_and_recalls_match':True,'family_decomposition_matches':True,'independent_family_counts_match':True})
            prediction_inputs.append(dict(zip(keys,key))|{'prediction_sha256':digest,'manifest_sha256':sha(mpath),'threshold':threshold})
            write_json(output/'progress.json',{'status':'RUNNING','completed_jobs':len(checks),'expected_jobs':180})
            if len(checks)%15==0:print('Verified and evaluated:',len(checks),'/ 180 jobs',flush=True)
        frame=pd.DataFrame(rows)
        frame.to_csv(output/'family_recall_per_job.csv',index=False)
        pd.DataFrame(checks).to_csv(output/'binary_reproduction_checks.csv',index=False)
        pd.DataFrame(coverage).to_csv(output/'family_coverage.csv',index=False)
        def summarise(group_keys):
            result=[]
            for key,g in frame.groupby(group_keys,sort=True):
                if not isinstance(key,tuple):key=(key,)
                valid=g[g.n_attack>0]
                result.append(dict(zip(group_keys,key))|{'jobs_total':len(g),'jobs_with_support':len(valid),
                    'zero_support_jobs':int((g.n_attack==0).sum()),'small_support_jobs_lt_50':int(((g.n_attack>0)&(g.n_attack<50)).sum()),
                    'minimum_rows_per_job':int(g.n_attack.min()),'maximum_rows_per_job':int(g.n_attack.max()),
                    'mean_recall_over_supported_jobs':float(valid.recall.mean()) if len(valid) else np.nan,
                    'minimum_recall_over_supported_jobs':float(valid.recall.min()) if len(valid) else np.nan,
                    'maximum_recall_over_supported_jobs':float(valid.recall.max()) if len(valid) else np.nan})
            return pd.DataFrame(result)
        summarise(['source_dataset','target_dataset','model','family']).to_csv(output/'family_recall_by_direction_model.csv',index=False)
        summarise(['target_dataset','model','family']).to_csv(output/'family_recall_by_target_model.csv',index=False)
        write_json(output/'input_manifest.json',{'original_target_metrics_sha256':metrics_sha,'recovery_inputs':inputs,'prediction_inputs':prediction_inputs})
        summary={'status':'PASS','scope':'post_hoc_target_family_resolved_binary_detection','jobs_completed':len(checks),
            'family_job_records_including_no_support':len(frame),'small_support_family_job_records':int(((frame.n_attack>0)&(frame.n_attack<50)).sum()),
            'zero_support_family_job_records':int((frame.n_attack==0).sum()),'original_binary_jobs_reproduced':len(checks),
            'no_retraining':True,'no_threshold_changes':True,'no_new_p_values_or_confidence_intervals':True,
            'capture_prevalence_not_evaluated':True,'family_specific_warning_prediction_not_evaluated':True,
            'source_family_holdout_not_evaluated':True,'pandas_version':pd.__version__,'numpy_version':np.__version__,'python':platform.python_version(),
            'code_sha256':hashlib.sha256(EVALUATION_SOURCE.encode()).hexdigest(),'plan_sha256':sha(output/'PLAN.md'),
            'interpretation':'Descriptive target-family detection recall. Shared datasets, overlapping seed panels and rare families limit interpretation; exact family strings are not harmonised across datasets.'}
        write_json(output/'verification_summary.json',summary)
        (output/'README.md').write_text('''# Post hoc target family recall
Use family_recall_per_job.csv for every job/family, including n=0 and n<50. Recall is missing when no attack row from that family occurs; it is not zero. All small families are retained. The 50-row flag is a descriptive aid only.
The direction/model table averages supported seed jobs equally (up to five). The target/model table averages supported source/seed jobs equally (up to fifteen). These are descriptive job averages, not pooled independent-flow estimates or confidence intervals. Do not interpret repeated rows across jobs/seeds as unique flows.
Family names are exact target dataset labels. This is binary attack detection within each family, not multiclass classification, a family-held-out experiment or evidence that a source-only warning predicts family-specific failure. Balanced original panels remain balanced. No new operational prevalence evaluation is performed.
Original models, predictions and thresholds are unchanged. Reproduction checks compare counts and recalls to all 180 original target jobs. Recovered label alignment does not replay all 49 raw feature values.
''')
        write_json(output/'progress.json',{'status':'PASS','completed_jobs':len(checks)})
    except Exception as e:
        write_json(output/'verification_summary.json',{'status':'FAILED','error':str(e),'completed_jobs':len(locals().get('checks',[]))})
        write_json(output/'progress.json',{'status':'FAILED','error':str(e)})
        print('FAILED:',str(e),flush=True)
    hashes={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output/'output_hashes.json',hashes)
    package=output.with_suffix('.zip')
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.iterdir()):
            if p.is_file():z.write(p,output.name+'/'+p.name)
    print('Result status:',read_json(output/'verification_summary.json')['status']);print('Results ZIP:',package)
    return package

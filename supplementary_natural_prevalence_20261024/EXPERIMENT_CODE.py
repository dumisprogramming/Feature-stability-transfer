"""Post hoc unstratified target sampling, fixed-source prediction and evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zipfile,shutil,platform,gc,warnings
import numpy as np
import pandas as pd
import joblib,sklearn
from sklearn.metrics import average_precision_score,matthews_corrcoef

DATASETS=['NF_UNSW_NB15_v3','NF_BoT_IoT_v3','NF_CSE_CIC_IDS2018_v3','NF_ToN_IoT_v3']
MODELS=['LogisticRegression','RandomForest','XGBoost']
SEEDS=list(range(2027,2032))
META={'row_id','Label','block_id_ms','pseudo_environment_id','actual_environment_id'}
N_TARGET=100000
POOL_SIZE=200000
CHUNK_SIZE=100000
PLAN='''Post hoc natural-capture-proportion target evaluation plan, recorded before new sampling and predictions.
Evaluate all twelve directed transfers, five seeds and three original classifiers (180 jobs). No retraining, target tuning, threshold adjustment or new warning regression.
For each target/seed, take the 200,000 smallest deterministic, label-independent SplitMix64 priorities among valid binary-labelled raw rows. Preserve predictors and labels without class quotas. For each source/seed, exclude rows whose cleaned 49-predictor float64 byte signature matches any original source training, calibration, source-test or environment row. Retain the 100,000 smallest priorities after exclusion. If fewer than 100,000 remain, stop that target/direction rather than silently reduce panel size or change the plan.
Because the oversample contains the lowest priorities in the entire raw target, enough retained rows imply the selected panel is exactly the lowest-priority 100,000 eligible raw rows. No extra quota is imposed on either class. Priorities differ by target and seed, not by classifier; classifiers share a panel. Different sources can exclude different rows. Repeated flows within a target are retained unless their signature occurs in the source; these are flow-level samples, not independent session samples.
The eligible population is the raw benchmark conditional on valid binary labels and source-overlap exclusions. Its class proportion may differ from the complete raw capture. Report complete raw capture proportion, oversample proportion and selected proportion. These are benchmark captures, not a claim about production prevalence. Do not use reweighted balanced summaries as substitutes for new predictions.
Apply unchanged source preprocessing and model-specific calibration thresholds. Replay every original source-test prediction first: binary decisions must match exactly and maximum probability difference must be <=1e-7. Keep the original feature order, float32 prediction transform and float64 overlap signatures. Do not bypass compatibility or integrity failures.
Write separate target feature/label files. Generate predictions using the feature file, write and checksum predictions, then join labels for scoring. Original outcomes are already known: this is a post hoc run, not a historical pre-label confirmatory freeze.
Report per-job n, class counts, observed capture/sample prevalence, precision, attack/benign recall, false-positive rate, accuracy, balanced accuracy, F1, MCC and average precision (stepwise PR-AUC). Undefined ratios remain missing. Summarise means/ranges descriptively; no independent-flow confidence intervals or superiority tests. Pooled job observations are not unique flows. The original balanced experiment remains unchanged.
Checkpoint complete source replays, target pools and individual jobs. Reuse only with matching code/configuration/software/input and output checksums. Interrupted raw scans restart their target, while completed stages are reused. Keep local outputs so a Drive failure can still produce an error-report ZIP.
'''

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def parquet(p,f):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');f.to_parquet(t,index=False);t.replace(p)
def canonical_keys(frame,features):
    a=frame[features].to_numpy(np.float64,copy=True);a[~np.isfinite(a)]=np.nan;a[a==0]=0.;a[np.isnan(a)]=np.nan
    a=np.ascontiguousarray(a.astype('<f8',copy=False))
    return a.view(np.dtype((np.void,8*len(features)))).ravel().copy()
def priorities(target,seed,ids):
    salt=int.from_bytes(hashlib.sha256(f'paper3-natural-v1|{target}|{seed}'.encode()).digest()[:8],'little')
    x=np.asarray(ids,dtype=np.uint64)^np.uint64(salt)
    with np.errstate(over='ignore'):
        x=x+np.uint64(0x9E3779B97F4A7C15);x=(x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB);x=x^(x>>np.uint64(31))
    return x
def retain(existing,incoming,k):
    f=incoming if existing is None else pd.concat([existing,incoming],ignore_index=True)
    if len(f)>k:f=f.iloc[np.argpartition(f.selection_key.to_numpy(np.uint64),k-1)[:k]]
    return f.reset_index(drop=True)
def select_panel(pool,source_keys,features,n):
    keys=canonical_keys(pool,features);eligible=~np.isin(keys,source_keys)
    f=pool.loc[eligible].sort_values(['selection_key','row_id'],kind='stable')
    if len(f)<n:raise ValueError(f'Oversample insufficient after source exclusion: {len(f)} < {n}. No panel size or quota was changed.')
    selected=f.iloc[:n].sort_values('row_id').reset_index(drop=True)
    if np.isin(canonical_keys(selected,features),source_keys).any():raise ValueError('Final exact source overlap')
    return selected,int((~eligible).sum())
def scores(y,pred,prob):
    y=np.asarray(y,int);pred=np.asarray(pred,int);prob=np.asarray(prob,float)
    tn,fp,fn,tp=map(int,np.bincount(2*y+pred,minlength=4))
    div=lambda a,b:a/b if b else None
    ar=div(tp,tp+fn);br=div(tn,tn+fp);precision=div(tp,tp+fp)
    return {'rows':len(y),'benign_rows':tn+fp,'attack_rows':tp+fn,'attack_fraction':float(y.mean()),
        'tn':tn,'fp':fp,'fn':fn,'tp':tp,'precision':precision,'attack_recall':ar,'benign_recall':br,
        'false_positive_rate':div(fp,tn+fp),'accuracy':div(tp+tn,len(y)),
        'balanced_accuracy':(ar+br)/2 if ar is not None and br is not None else None,
        'f1':div(2*tp,2*tp+fp+fn),'mcc':float(matthews_corrcoef(y,pred)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)>0 else None,
        'pr_auc_average_precision':float(average_precision_score(y,prob)) if len(np.unique(y))==2 else None,
        'class_support':'BOTH_CLASSES' if len(np.unique(y))==2 else 'ONE_CLASS_ONLY'}
def transformed(prep,f,features,scaled):
    x=prep['imputer'].transform(f[features].to_numpy(np.float32)).astype(np.float32)
    if scaled:x=prep['logistic_regression_scaler'].transform(x).astype(np.float32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite transformed values')
    return x

def run_natural(drive_root,local_root,local_inputs):
    import xgboost,scipy,pyarrow
    drive_root=Path(drive_root);local_root=Path(local_root);local_inputs=Path(local_inputs)
    local_root.mkdir(parents=True,exist_ok=True);local_inputs.mkdir(parents=True,exist_ok=True)
    remote=drive_root/'supplementary_natural_prevalence_20261024/run_v1'
    report={'status':'RUNNING','started_utc':datetime.now(timezone.utc).isoformat(),'completed_jobs':0,'errors':[],'model_load_warnings':[]}
    software={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'xgboost':xgboost.__version__,'joblib':joblib.__version__,'scipy':scipy.__version__,'pyarrow':pyarrow.__version__}
    config={'software':software,'target_rows':N_TARGET,'pool_rows':POOL_SIZE,'chunk_size':CHUNK_SIZE,'seeds':SEEDS,'datasets':DATASETS,
        'code_sha256':hashlib.sha256(EXPERIMENT_SOURCE.encode()).hexdigest(),'plan_sha256':hashlib.sha256(PLAN.encode()).hexdigest(),
        'raw_info_sha256':hashlib.sha256(json.dumps(RAW_INFO,sort_keys=True).encode()).hexdigest(),
        'pinned_inputs_sha256':hashlib.sha256(json.dumps(PINNED_INPUTS,sort_keys=True).encode()).hexdigest()}
    def upload(rel):
        src=local_root/rel;dst=remote/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        temp=dst.with_suffix(dst.suffix+'.tmp');shutil.copyfile(src,temp)
        if sha(src)!=sha(temp):raise IOError('Checkpoint copy checksum failed: '+str(rel))
        temp.replace(dst)
    def restore(rel):
        p=local_root/rel
        if not p.exists() and (remote/rel).is_file():
            p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(remote/rel,p)
        return p
    def finish(marker,identity,rels,details):
        record={'identity':identity,'output_hashes':{str(r):sha(local_root/r) for r in rels},'details':details}
        write(local_root/marker,record)
        for r in rels:upload(r)
        upload(marker) # Completion marker copied last.
        return record
    def cached(marker,identity):
        p=restore(marker)
        if not p.exists():return None
        c=js(p)
        if c['identity']!=identity:raise ValueError('Checkpoint identity changed: '+str(marker))
        for r,h in c['output_hashes'].items():
            q=restore(r)
            if not q.is_file() or sha(q)!=h:raise ValueError('Checkpoint output changed: '+r)
        return c
    def stage(rel):
        rel=str(rel);expected=PINNED_INPUTS.get(rel)
        if not expected:raise ValueError('Input has no verified pinned hash: '+rel)
        dst=local_inputs/rel
        if dst.exists() and sha(dst)==expected:return dst
        dst.parent.mkdir(parents=True,exist_ok=True);temp=dst.with_suffix(dst.suffix+'.tmp')
        shutil.copyfile(drive_root/rel,temp)
        if sha(temp)!=expected:raise ValueError('Historical input hash differs: '+rel)
        temp.replace(dst);return dst
    def load_source(source,seed):
        base=f'Paper3_Source_Models/{source}/seed_{seed}'
        meta=js(stage(base+'/source_seed_complete.json'))
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');prep=joblib.load(stage(base+'/source_preprocessor.joblib'))
        report['model_load_warnings']+=list(dict.fromkeys(str(w.message) for w in ws))
        if list(prep['features'])!=features:raise ValueError('Preprocessor feature order mismatch')
        return base,meta,prep
    def load_model(base,meta,source,seed,model):
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');bundle=joblib.load(stage(base+'/'+model+'_model.joblib'))
        report['model_load_warnings']+=list(dict.fromkeys(str(w.message) for w in ws))
        if bundle['dataset']!=source or int(bundle['seed'])!=seed or bundle['model_name']!=model or list(bundle['features'])!=features:raise ValueError('Model identity/schema mismatch')
        scaled=bool(bundle['uses_standard_scaler'])
        if scaled!=(model=='LogisticRegression'):raise ValueError('Unexpected scaler choice')
        threshold=float(bundle['threshold']);info=meta['models'][model]
        if not np.isfinite(threshold) or not 0<=threshold<=1 or abs(threshold-float(info['source_test_metrics']['threshold']))>1e-15:raise ValueError('Threshold changed')
        if not np.array_equal(bundle['model'].classes_,[0,1]):raise ValueError('Unexpected estimator classes')
        return bundle['model'],threshold,scaled
    (local_root/'PLAN.md').write_text(PLAN)
    (local_root/'EXPERIMENT_CODE.py').write_text(EXPERIMENT_SOURCE)
    write(local_root/'PINNED_INPUTS.json',PINNED_INPUTS)
    try:
        if sklearn.__version__!='1.6.1' or xgboost.__version__!='3.4.1':raise ValueError('Run the dependency cell first; original sklearn/XGBoost versions are required')
        lock=restore('run_identity.json')
        if lock.exists() and js(lock)!=config:raise ValueError('Code/config/software differs from existing run; restore original runtime or use an explicitly new run')
        write(lock,config);upload('run_identity.json')
        (local_root/'PLAN.md').write_text(PLAN);(local_root/'EXPERIMENT_CODE.py').write_text(EXPERIMENT_SOURCE)
        write(local_root/'PINNED_INPUTS.json',PINNED_INPUTS)
        for r in ['PLAN.md','EXPERIMENT_CODE.py','PINNED_INPUTS.json']:upload(r)
        features=js(stage('Paper3_Materialized/feature_schema_v1.json'))['predictors_in_order']
        if len(features)!=49:raise ValueError('Expected 49 predictors')
        # Replay all source-test predictions before scanning target archives.
        compatibility=[]
        for source in DATASETS:
            for seed in SEEDS:
                base,meta,prep=load_source(source,seed)
                test=pd.read_parquet(stage(f'Paper3_Sanitized/{source}/seed_{seed}/source_test.parquet'))
                if [c for c in test.columns if c not in META]!=features:raise ValueError('Source feature order differs')
                for model in MODELS:
                    marker=f'compatibility/{source}/seed_{seed}/{model}.json'
                    c=cached(marker,config)
                    if c is None:
                        est,threshold,scaled=load_model(base,meta,source,seed,model)
                        saved=pd.read_parquet(stage(base+'/'+model+'_source_test_predictions.parquet')).set_index('row_id',verify_integrity=True).loc[test.row_id]
                        if len(saved)!=len(test) or not np.array_equal(saved.y_true.to_numpy(),test.Label.to_numpy()):raise ValueError('Source replay labels differ')
                        prob=est.predict_proba(transformed(prep,test,features,scaled))[:,1]
                        delta=float(np.max(np.abs(prob-saved.probability_attack.to_numpy())))
                        if delta>1e-7 or not np.array_equal((prob>=threshold).astype(int),saved.y_pred.to_numpy()):raise ValueError('Source replay failed; no target scan started')
                        c=finish(marker,config,[],{'source':source,'seed':seed,'model':model,'maximum_probability_difference':delta,'binary_predictions_match':True})
                        del est
                    compatibility.append(c['details']);print('Source replay verified:',source,seed,model,flush=True)
                del prep,test;gc.collect()
        write(local_root/'source_replay_checks.json',compatibility);upload('source_replay_checks.json')
        metrics=[];panel_summary=[]
        for target in DATASETS:
            raw=RAW_INFO[target];ziprel=raw['raw_zip_name'];pool_marker=f'pools/{target}/complete.json'
            c=cached(pool_marker,config)
            if c is None:
                archive=stage(ziprel);pools={s:None for s in SEEDS};cutoffs={s:None for s in SEEDS};offset=0;class_counts={0:0,1:0};invalid=0
                print('Scanning unstratified raw target:',target,flush=True)
                with zipfile.ZipFile(archive) as z:
                    with z.open(raw['raw_csv_member']) as stream:
                        for chunk_no,chunk in enumerate(pd.read_csv(stream,usecols=features+['Label'],chunksize=CHUNK_SIZE,low_memory=False),1):
                            y=pd.to_numeric(chunk.Label,errors='coerce').to_numpy();valid=np.isin(y,[0,1]);ids=np.arange(offset,offset+len(chunk),dtype=np.int64)
                            for k in [0,1]:class_counts[k]+=int((y==k).sum())
                            invalid+=int((~valid).sum())
                            for seed in SEEDS:
                                key=priorities(target,seed,ids);mask=valid.copy()
                                if cutoffs[seed] is not None:mask&=key<cutoffs[seed]
                                if mask.any():
                                    inc=chunk.loc[mask,features].copy();inc.insert(0,'row_id',ids[mask]);inc.insert(1,'selection_key',key[mask]);inc['Label']=y[mask].astype('int8')
                                    pools[seed]=retain(pools[seed],inc,POOL_SIZE)
                                    if len(pools[seed])==POOL_SIZE:cutoffs[seed]=pools[seed].selection_key.max()
                            offset+=len(chunk)
                            write(local_root/'progress.json',{'status':'SCANNING_TARGET','target':target,'rows_scanned':offset,'completed_jobs':len(metrics)})
                            if chunk_no%10==0:print(target,format(offset,','),'rows scanned',flush=True)
                if offset!=raw['raw_rows'] or class_counts!={int(k):v for k,v in raw['raw_class_counts'].items()} or invalid:raise ValueError('Raw counts differ from verified metadata scan')
                rels=[]
                for seed,pool in pools.items():
                    if pool is None or len(pool)!=POOL_SIZE or not pool.row_id.is_unique or not pool.selection_key.is_unique:raise ValueError('Incomplete/invalid oversample')
                    # Clean exactly as used for original source-target signature comparisons.
                    pool[features]=pool[features].astype(np.float64).replace([np.inf,-np.inf],np.nan)
                    rel=f'pools/{target}/seed_{seed}.parquet';parquet(local_root/rel,pool.sort_values('selection_key'));rels.append(rel)
                finish(pool_marker,config,rels,{'raw_rows':offset,'class_counts':{str(k):v for k,v in class_counts.items()}})
                del pools,pool;gc.collect()
            for source in DATASETS:
                if source==target:continue
                for seed in SEEDS:
                    prefix=f'jobs/{source}__to__{target}/seed_{seed}'
                    allcached=[cached(prefix+'/'+m+'.json',config) for m in MODELS]
                    if all(x is not None for x in allcached):
                        for x in allcached:metrics.append(x['details']['metrics'])
                        report['completed_jobs']=len(metrics)
                        panel_summary.append(allcached[0]['details']['panel']);print('Reused all three jobs:',source,target,seed,flush=True);continue
                    # Exact byte-signature exclusion uses all four original source partitions.
                    parts=[]
                    for split in ['train','calibration','source_test','environment']:
                        p=stage(f'Paper3_Sanitized/{source}/seed_{seed}/{split}.parquet');sf=pd.read_parquet(p)
                        if [c for c in sf.columns if c not in META]!=features:raise ValueError('Source schema differs')
                        parts.append(canonical_keys(sf,features));del sf
                    source_keys=np.unique(np.concatenate(parts));del parts
                    pool=pd.read_parquet(local_root/f'pools/{target}/seed_{seed}.parquet')
                    selected,excluded=select_panel(pool,source_keys,features,N_TARGET)
                    panel={'source_dataset':source,'target_dataset':target,'seed':seed,'pool_rows':len(pool),'pool_rows_excluded_by_source_signature':excluded,
                        'selected_rows':len(selected),'pool_attack_fraction':float(pool.Label.mean()),'selected_attack_fraction':float(selected.Label.mean()),
                        'raw_capture_attack_fraction':raw['raw_class_counts']['1']/raw['raw_rows'],'source_overlap_rows':0,'class_quotas_used':False}
                    fp=prefix+'/features.parquet';lp=prefix+'/labels.parquet'
                    # On partial resume these deterministic files must reproduce the earlier checkpoint hashes.
                    parquet(local_root/fp,selected[['row_id']+features]);parquet(local_root/lp,selected[['row_id','Label']])
                    panel['feature_sha256']=sha(local_root/fp);panel['label_sha256']=sha(local_root/lp)
                    del source_keys,pool,selected;gc.collect()
                    base,meta,prep=load_source(source,seed)
                    for model,old in zip(MODELS,allcached):
                        marker=prefix+'/'+model+'.json'
                        if old is not None:
                            if old['details']['panel']!=panel:raise ValueError('Rebuilt panel differs from partial checkpoint')
                            metrics.append(old['details']['metrics']);report['completed_jobs']=len(metrics);continue
                        est,threshold,scaled=load_model(base,meta,source,seed,model)
                        feature_frame=pd.read_parquet(local_root/fp)
                        prob=est.predict_proba(transformed(prep,feature_frame,features,scaled))[:,1]
                        if len(prob)!=N_TARGET or not np.isfinite(prob).all() or not ((prob>=0)&(prob<=1)).all():raise ValueError('Invalid target probabilities')
                        pred=(prob>=threshold).astype('int8');prediction=pd.DataFrame({'row_id':feature_frame.row_id,'attack_probability':prob,'predicted_label':pred})
                        pr=prefix+'/'+model+'_predictions.parquet';parquet(local_root/pr,prediction)
                        prediction_sha=sha(local_root/pr) # Predictions written before this new scoring label join.
                        labels=pd.read_parquet(local_root/lp)
                        if not prediction.row_id.equals(labels.row_id):raise ValueError('Target label/prediction alignment failed')
                        score=scores(labels.Label,pred,prob)
                        # Independent boolean count check.
                        yy=labels.Label.to_numpy(int)
                        for k,value in {'tn':int(((yy==0)&(pred==0)).sum()),'fp':int(((yy==0)&(pred==1)).sum()),'fn':int(((yy==1)&(pred==0)).sum()),'tp':int(((yy==1)&(pred==1)).sum())}.items():
                            if score[k]!=value:raise ValueError('Independent confusion calculation differs')
                        score.update({'source_dataset':source,'target_dataset':target,'seed':seed,'model':model,'threshold':threshold,
                            'raw_capture_attack_fraction':panel['raw_capture_attack_fraction'],'prediction_sha256':prediction_sha})
                        finish(marker,config,[fp,lp,pr],{'metrics':score,'panel':panel,'predictions_written_before_new_scoring_join':True})
                        metrics.append(score);report['completed_jobs']=len(metrics)
                        write(local_root/'progress.json',report);print('Completed:',len(metrics),'/ 180 |',source,'to',target,seed,model,flush=True)
                        del est,feature_frame,prediction,labels,prob,pred;gc.collect()
                    panel_summary.append(panel);del prep;gc.collect()
        if len(metrics)!=180 or len(panel_summary)!=60:raise ValueError('Incomplete expected experiment')
        pd.DataFrame(metrics).to_csv(local_root/'natural_prevalence_metrics_180.csv',index=False)
        pd.DataFrame(panel_summary).to_csv(local_root/'panel_sampling_summary_60.csv',index=False)
        numeric=['precision','attack_recall','false_positive_rate','balanced_accuracy','pr_auc_average_precision']
        pd.DataFrame(metrics).groupby(['source_dataset','target_dataset','model'])[numeric].agg(['mean','min','max','count']).to_csv(local_root/'direction_model_descriptive_summary.csv')
        report.update(status='PASS',completed_jobs=180,panels=60,source_model_replays=60,detector_retraining=False,threshold_changes=False,
            natural_capture_sampling=True,production_prevalence_claim=False,completed_utc=datetime.now(timezone.utc).isoformat())
    except Exception as e:
        report.update(status='INCOMPLETE',error_type=type(e).__name__,error=str(e));print('STOPPED:',type(e).__name__,str(e),flush=True)
    write(local_root/'verification_summary.json',report);write(local_root/'progress.json',report)
    # Report generation uses local disk, even if the Drive mount failed.
    files=['PLAN.md','EXPERIMENT_CODE.py','PINNED_INPUTS.json','run_identity.json','source_replay_checks.json','natural_prevalence_metrics_180.csv',
        'panel_sampling_summary_60.csv','direction_model_descriptive_summary.csv','verification_summary.json','progress.json']
    write(local_root/'report_hashes.json',{r:sha(local_root/r) for r in files if (local_root/r).is_file()});files.append('report_hashes.json')
    package=local_root/('natural_prevalence_report_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'.zip')
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
        for r in files:
            if (local_root/r).is_file():z.write(local_root/r,r)
        for p in sorted((local_root/'jobs').rglob('*.json')) if (local_root/'jobs').exists() else []:
            z.write(p,str(p.relative_to(local_root)))
    try:
        for r in files:
            if (local_root/r).is_file():upload(r)
        shutil.copyfile(package,remote/package.name)
        print('Drive results folder:',remote)
    except Exception as e:print('Drive save unavailable. Keep this runtime and download the LOCAL report ZIP:',type(e).__name__,str(e))
    print('Status:',report['status']);print('Download ZIP:',package)
    return package

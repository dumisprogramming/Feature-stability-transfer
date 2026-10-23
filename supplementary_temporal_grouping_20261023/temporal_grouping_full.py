"""Post-hoc temporal regrouping on original sampled source rows. No retraining."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,itertools,json,platform,time,warnings
import numpy as np,pandas as pd,scipy,sklearn,joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LinearRegression

DATASETS=['NF_BoT_IoT_v3','NF_CSE_CIC_IDS2018_v3','NF_ToN_IoT_v3','NF_UNSW_NB15_v3']
MODELS=['LogisticRegression','RandomForest','XGBoost']
KEY=['source_dataset','seed','model']
WARN=['sampling_instability','environmental_instability','sampling_environment_gap']
SPECS={'mean_only':[],'model_only':[],'model_plus_sampling':WARN[:1],
       'model_plus_environment':WARN[1:2],'model_plus_gap':WARN[2:],
       'model_plus_both_primary':WARN[1:]}

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def state(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:4],'little')
def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def seal(path,obj):
    obj=dict(obj);obj.pop('payload_sha256',None);obj['payload_sha256']=digest(obj);write(path,obj)
def readsealed(path):
    x=json.loads(Path(path).read_text());h=x.pop('payload_sha256');assert digest(x)==h,'Checkpoint checksum mismatch';return x
def score_at(threshold):
    def score(model,x,y):return balanced_accuracy_score(y,(model.predict_proba(x)[:,1]>=threshold).astype(np.int8))
    return score
def paired_stability(vectors):
    ranks=[pd.Series(-np.asarray(v)).rank(method='average').to_numpy() for v in vectors]
    values=[];undefined=0
    for a,b in itertools.combinations(ranks,2):
        if np.std(a)==0 or np.std(b)==0:values.append(0.);undefined+=1
        else:values.append(float(np.corrcoef(a,b)[0,1]))
    assert len(values)>0
    return float(np.mean(values)),undefined,len(values)
def matched_groups(frame,hours,dataset,seed):
    actual=frame.block_id_ms.to_numpy(np.int64)//(hours*3600000)*(hours*3600000)
    starts=np.sort(np.unique(actual));codes=np.searchsorted(starts,actual)
    # Same membership -> same pseudo-assignment and permutation states at both widths.
    membership=digest({'row_ids':frame.row_id.astype('int64').tolist(),'actual_codes':codes.tolist()})
    pseudo=np.full(len(frame),-1,dtype=np.int64);y=frame.Label.to_numpy(np.int8)
    rng=np.random.default_rng(state('TEMPORAL_REGROUPING_FULL_V1',dataset,seed,membership,'pseudo'))
    for label in [0,1]:
        positions=rng.permutation(np.flatnonzero(y==label));offset=0
        for code in range(len(starts)):
            count=int(np.sum((codes==code)&(y==label)));pseudo[positions[offset:offset+count]]=code;offset+=count
        assert offset==len(positions)
    assert np.all(pseudo>=0)
    for code in range(len(starts)):
        a=np.bincount(y[codes==code],minlength=2);b=np.bincount(y[pseudo==code],minlength=2)
        assert np.array_equal(a,b) and np.all(a>=2000)
    return codes,pseudo,membership,starts

def predictive_analysis(root,out,units):
    # All new source scores are saved and hashed before reading the saved transfer table.
    units.to_csv(out/'source_warning_scores_all_widths.csv',index=False)
    frozen=sha(out/'source_warning_scores_all_widths.csv')
    write(out/'source_scores_record.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'sha256':frozen,'post_hoc_after_original_outcomes_known':True,'not_a_historical_pre_label_seal':True})
    transfer_path=root/'Paper3_Final_Evaluation/transfer_analysis_all_180.csv'
    transfers=pd.read_csv(transfer_path);assert len(transfers)==180
    predictions=[];diagnostics=[];associations=[]
    for hours in [6,12,24]:
        controls=units[units.hours==hours].rename(columns={'dataset':'source_dataset'})
        assert len(controls)==60 and not controls.duplicated(KEY).any()
        f=transfers.drop(columns=WARN,errors='ignore').merge(controls[KEY+WARN],on=KEY,validate='many_to_one')
        assert len(f)==180 and np.isfinite(f[WARN+['balanced_accuracy_loss']]).all().all()
        def aggregate(a):
            return a.groupby(KEY,as_index=False).agg(loss=('balanced_accuracy_loss','mean'),**{c:(c,'first') for c in WARN})
        combined=aggregate(f).groupby(['source_dataset','model'],as_index=False)[['loss']+WARN].mean()
        assert len(combined)==12
        for c in WARN:
            rho=combined[c].rank().corr(combined.loss.rank())
            associations.append({'hours':hours,'warning':c,'units':12,'spearman_rho':None if pd.isna(rho) else float(rho),'inference':'descriptive_only_no_p_value'})
        for omit in [None]+list(range(2027,2032)):
            sub=f if omit is None else f[f.seed!=omit];run='all_seeds' if omit is None else 'omit_seed:'+str(omit)
            for held in sorted(sub.source_dataset.unique()):
                train=sub[(sub.source_dataset!=held)&(sub.target_dataset!=held)];test=sub[sub.source_dataset==held]
                assert held not in set(train.source_dataset)|set(train.target_dataset)
                assert train.groupby(KEY).target_dataset.nunique().eq(2).all() and test.groupby(KEY).target_dataset.nunique().eq(3).all()
                t,e=aggregate(train),aggregate(test)
                assert len(t)==(45 if omit is None else 36) and len(e)==(15 if omit is None else 12)
                for spec,cols in SPECS.items():
                    a=[np.ones(len(t))];b=[np.ones(len(e))]
                    if spec!='mean_only':
                        for model in ['RandomForest','XGBoost']:
                            a.append(t.model.eq(model).to_numpy(float));b.append(e.model.eq(model).to_numpy(float))
                    for c in cols:
                        mu=float(t[c].mean());sd=float(t[c].std(ddof=0));sd=sd if sd>0 else 1.
                        a.append((t[c].to_numpy()-mu)/sd);b.append((e[c].to_numpy()-mu)/sd)
                    a,b=np.column_stack(a),np.column_stack(b)
                    pred=LinearRegression(fit_intercept=False).fit(a,t.loss).predict(b)
                    direct=b@np.linalg.lstsq(a,t.loss,rcond=None)[0];diff=float(np.max(abs(pred-direct)));assert diff<1e-9
                    rank=int(np.linalg.matrix_rank(a));condition=float(np.linalg.cond(a))
                    diagnostics.append({'hours':hours,'run':run,'held_out':held,'specification':spec,'rank':rank,'columns':a.shape[1],
                        'full_rank':rank==a.shape[1],'condition_number':condition if np.isfinite(condition) else None,'independent_difference':diff})
                    for (_,r),v in zip(e.iterrows(),pred):
                        predictions.append({'hours':hours,'run':run,'held_out':held,'seed':int(r.seed),'model':r.model,
                            'specification':spec,'actual_loss':float(r.loss),'predicted_loss':float(v)})
    pred=pd.DataFrame(predictions);rows=[]
    for level,keys in [('overall',['hours','run','specification']),('fold',['hours','run','held_out','specification'])]:
        for key,g in pred.groupby(keys):
            error=g.actual_loss-g.predicted_loss
            rows.append({**dict(zip(keys,key)),'level':level,'units':len(g),'mae':float(abs(error).mean()),'rmse':float(np.sqrt((error**2).mean()))})
    scores=pd.DataFrame(rows).fillna('ALL');scope=['hours','run','level','held_out']
    reference=scores[scores.specification=='model_only'].set_index(scope).mae
    scores['mae_improvement_vs_model_only']=[float(reference.loc[tuple(r[k] for k in scope)])-r.mae for _,r in scores.iterrows()]
    old=scores[(scores.hours==6)&(scores.run=='all_seeds')&(scores.level=='overall')].set_index('specification')
    assert abs(old.loc['model_only','mae']-0.05009957252970506)<1e-10
    assert abs(old.loc['model_plus_both_primary','mae']-0.06689810412254844)<1e-10
    assert len(pred)==5400 and len(diagnostics)==432
    pred.to_csv(out/'strict_validation_predictions.csv',index=False);scores.to_csv(out/'strict_validation_scores.csv',index=False)
    write(out/'strict_validation_diagnostics.json',diagnostics);write(out/'descriptive_associations.json',associations)
    return {'predictions':len(pred),'regression_fits':len(diagnostics),'rank_deficient_fits':sum(not x['full_rank'] for x in diagnostics),
            'source_scores_sha256':frozen,'saved_transfer_table_sha256':sha(transfer_path),'six_hour_reference_reproduced':True}

def run_full(root,out,plan_text,script_text):
    import xgboost
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
    for name,content in [('PLAN.md',plan_text),('temporal_grouping_full.py',script_text)]:
        path=out/name
        if path.exists() and path.read_text()!=content:raise RuntimeError('Existing run uses different plan/code. Use its original notebook or a new folder.')
        if not path.exists():path.write_text(content)
    software={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
              'sklearn':sklearn.__version__,'joblib':joblib.__version__,'xgboost':xgboost.__version__}
    session={'status':'RUNNING','started_utc':datetime.now(timezone.utc).isoformat(),'software':software,
             'plan_sha256':sha(out/'PLAN.md'),'script_sha256':sha(out/'temporal_grouping_full.py'),
             'scope':'post_hoc_regrouping_of_original_sampled_rows','detector_retraining':False,'input_hashes':{},'warnings':[]}
    identity={'software':software,'plan_sha256':session['plan_sha256'],'script_sha256':session['script_sha256']}
    lock=out/'run_identity.json'
    if lock.exists():assert readsealed(lock)==identity,'Software/code/plan differs from existing run. Restore the recorded runtime or use a new run folder.'
    else:seal(lock,identity)
    def record(path,expected=None):
        value=sha(path)
        if expected is not None:assert value==expected,'Historical input hash mismatch: '+str(path)
        session['input_hashes'][str(path.relative_to(root))]=value;return value
    def load(path,expected):
        record(path,expected)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');result=joblib.load(path)
        session['warnings'].extend({'file':path.name,'warning':str(x.message)} for x in caught)
        return result
    rows=[];group_rows=[];completed_refs=0;new_jobs=0;reused_jobs=0;total_refs=0
    # Source-only preflight records the whole intended workload before any new PFI.
    for ds,seed in itertools.product(DATASETS,range(2027,2032)):
        f=root/'Paper3_Sanitized'/ds/f'seed_{seed}'/'environment.parquet'
        q=pd.read_parquet(f,columns=['block_id_ms'])
        for hours in [12,24]:total_refs+=len(np.unique(q.block_id_ms.to_numpy(np.int64)//(hours*3600000)))*2*3
    session['total_group_references']=total_refs;write(out/'progress.json',session)
    for dataset,seed in itertools.product(DATASETS,range(2027,2032)):
        base=root/'Paper3_Source_Models'/dataset/f'seed_{seed}';meta_path=base/'source_seed_complete.json'
        record(meta_path);meta=json.loads(meta_path.read_text())
        schema=root/'Paper3_Materialized/feature_schema_v1.json';record(schema,meta['feature_schema_sha256'])
        features=json.loads(schema.read_text())['predictors_in_order'];assert len(features)==49
        env_path=root/'Paper3_Sanitized'/dataset/f'seed_{seed}'/'environment.parquet'
        sanitization=env_path.with_name('sanitization_manifest.json');record(sanitization,meta['input_sanitization_manifest_sha256'])
        audit=json.loads(sanitization.read_text());env_hash=record(env_path,audit['output_files']['environment']['sha256'])
        test_path=env_path.with_name('source_test.parquet');record(test_path,audit['output_files']['source_test']['sha256'])
        env=pd.read_parquet(env_path);test=pd.read_parquet(test_path)
        assert env.row_id.is_unique and test.row_id.is_unique and not set(env.row_id)&set(test.row_id)
        assert np.all(env.block_id_ms.to_numpy(np.int64)%(6*3600000)==0)
        assert env.Label.isin([0,1]).all() and test.Label.isin([0,1]).all()
        prep=load(base/'source_preprocessor.joblib',meta['preprocessor']['sha256']);assert list(prep['features'])==features
        def transform(frame):
            raw=frame[features].to_numpy(np.float32);x=prep['imputer'].transform(raw).astype(np.float32)
            xs=prep['logistic_regression_scaler'].transform(x).astype(np.float32)
            assert np.isfinite(x).all() and np.isfinite(xs).all();return x,xs
        ex,tx=transform(env),transform(test);y=env.Label.to_numpy(np.int8)
        groups={hours:matched_groups(env,hours,dataset,seed) for hours in [12,24]}
        for model_name in MODELS:
            info=meta['models'][model_name];artifact=load(base/(model_name+'_model.joblib'),info['files']['model']['sha256'])
            model=artifact['model'];threshold=float(artifact['threshold']);scaled=int(bool(artifact['uses_standard_scaler']))
            assert abs(threshold-info['source_test_metrics']['threshold'])<1e-15
            x=ex[scaled];xt=tx[scaled]
            oldpath=root/'Paper3_Source_Stability'/dataset/f'seed_{seed}'/model_name/'stability_complete.json'
            record(oldpath);original=json.loads(oldpath.read_text())
            pfipath=oldpath.with_name('permutation_importance.parquet');record(pfipath,original['permutation_importance_sha256'])
            storedpath=base/(model_name+'_source_test_predictions.parquet');record(storedpath,info['files']['source_test_predictions']['sha256'])
            check_identity={**identity,'model_sha256':sha(base/(model_name+'_model.joblib')),'preprocessor_sha256':sha(base/'source_preprocessor.joblib'),
                'environment_sha256':env_hash,'source_test_sha256':sha(test_path),'saved_predictions_sha256':sha(storedpath),'saved_importance_sha256':sha(pfipath)}
            check_path=out/'compatibility'/dataset/f'seed_{seed}'/(model_name+'.json')
            if check_path.exists():
                check=readsealed(check_path);assert check['identity']==check_identity and check['status']=='PASS'
            else:
                print('Checking original model/PFI:',dataset,seed,model_name,flush=True)
                saved=pd.read_parquet(storedpath).set_index('row_id');assert saved.index.is_unique and len(saved)==len(test)
                saved=saved.loc[test.row_id];assert np.array_equal(saved.y_true.to_numpy(),test.Label.to_numpy())
                probability=model.predict_proba(xt)[:,1];pdiff=float(np.max(abs(probability-saved.probability_attack.to_numpy())))
                assert pdiff<=1e-7 and np.array_equal((probability>=threshold).astype(np.int8),saved.y_pred.to_numpy()),'Source-test replay mismatch'
                group=str(sorted(env.actual_environment_id.unique())[0]);positions=np.flatnonzero(env.actual_environment_id.astype(str).to_numpy()==group)
                old=pd.read_parquet(pfipath);ref=old[(old.group_type=='environmental')&(old.group_id.astype(str)==group)].set_index('feature').loc[features]
                assert len(ref)==49 and len(positions)==4000
                pfi=permutation_importance(model,x[positions],y[positions],scoring=score_at(threshold),n_repeats=3,
                    random_state=state(dataset,seed,model_name,'environmental',group,'PFI'),n_jobs=1)
                difference=float(np.max(abs(pfi.importances-ref[['repeat_1','repeat_2','repeat_3']].to_numpy())))
                bd=abs(score_at(threshold)(model,x[positions],y[positions])-float(ref.baseline_balanced_accuracy.iloc[0]))
                assert difference<=1e-8 and bd<=1e-12,'Original six-hour PFI replay mismatch'
                seal(check_path,{'identity':check_identity,'status':'PASS','probability_max_difference':pdiff,'importance_max_difference':difference,'baseline_difference':bd})
            w=original['warning_scores'];rows.append({'dataset':dataset,'seed':seed,'model':model_name,'hours':6,
                'sampling_instability':1-float(w['sampling_stability']),'environmental_instability':float(w['environmental_instability']),
                'sampling_environment_gap':float(w['sampling_environment_gap']),'environmental_groups':int(w['environmental_groups']),
                'sampling_groups':int(w['sampling_groups']),'score_origin':'historical_six_hour'})
            for hours,(actual,pseudo,membership,starts) in groups.items():
                means={};undefined={};paircounts={}
                for kind,codes in [('environmental',actual),('sampling',pseudo)]:
                    vectors=[]
                    for code,start in enumerate(starts):
                        positions=np.flatnonzero(codes==code);row_hash=hashlib.sha256(env.row_id.to_numpy(np.int64)[positions].tobytes()).hexdigest()
                        ji={**check_identity,'group_type':kind,'selected_rows_sha256':row_hash,'rows':len(positions),'permutation_repeats':3}
                        # Width omitted from cache/RNG: identical row sets are computed once.
                        job_path=out/'groups'/dataset/f'seed_{seed}'/model_name/(kind+'_'+row_hash+'.json')
                        if job_path.exists():
                            result=readsealed(job_path);assert result['identity']==ji and result['status']=='PASS';reused_jobs+=1
                        else:
                            print(f'{completed_refs}/{total_refs} START {dataset} {seed} {model_name} {hours}h {kind} group {code+1}/{len(starts)} rows={len(positions)}',flush=True)
                            begin=time.perf_counter()
                            pfi=permutation_importance(model,x[positions],y[positions],scoring=score_at(threshold),n_repeats=3,
                                random_state=state('TEMPORAL_REGROUPING_FULL_V1',dataset,seed,model_name,kind,row_hash,'PFI'),n_jobs=1)
                            result={'identity':ji,'status':'PASS','importance_repeats':pfi.importances.tolist(),
                                'baseline_balanced_accuracy':float(score_at(threshold)(model,x[positions],y[positions])),
                                'elapsed_seconds':time.perf_counter()-begin,'created_utc':datetime.now(timezone.utc).isoformat()}
                            seal(job_path,result);new_jobs+=1
                        importance=np.asarray(result['importance_repeats']);assert importance.shape==(49,3) and np.isfinite(importance).all()
                        vectors.append(importance.mean(axis=1))
                        group_rows.append({'dataset':dataset,'seed':seed,'model':model_name,'hours':hours,'group_type':kind,
                            'group_start_ms':int(start),'rows':len(positions),'benign':int(np.sum(y[positions]==0)),
                            'attack':int(np.sum(y[positions]==1)),'baseline_balanced_accuracy':result['baseline_balanced_accuracy'],
                            'checkpoint':str(job_path.relative_to(out))})
                        completed_refs+=1
                        session.update(completed_group_references=completed_refs,new_group_jobs=new_jobs,reused_group_jobs=reused_jobs,current_dataset=dataset,current_seed=seed,current_model=model_name)
                        write(out/'progress.json',session)
                        print(f'{completed_refs}/{total_refs} DONE; new={new_jobs}, cached={reused_jobs}',flush=True)
                    means[kind],undefined[kind],paircounts[kind]=paired_stability(vectors)
                rows.append({'dataset':dataset,'seed':seed,'model':model_name,'hours':hours,
                    'sampling_instability':1-means['sampling'],'environmental_instability':1-means['environmental'],
                    'sampling_environment_gap':means['sampling']-means['environmental'],
                    'environmental_groups':len(starts),'sampling_groups':len(starts),
                    'undefined_environmental_pairs':undefined['environmental'],'undefined_sampling_pairs':undefined['sampling'],
                    'environmental_pairs':paircounts['environmental'],'sampling_pairs':paircounts['sampling'],
                    'score_origin':'post_hoc_regrouped_original_rows'})
    assert completed_refs==total_refs
    units=pd.DataFrame(rows);assert len(units)==180
    # The audited CSE partition is identical at 12 and 24 hours: scores must match exactly.
    a=units[(units.dataset=='NF_CSE_CIC_IDS2018_v3')&(units.hours==12)].sort_values(['seed','model'])
    b=units[(units.dataset=='NF_CSE_CIC_IDS2018_v3')&(units.hours==24)].sort_values(['seed','model'])
    assert np.array_equal(a[WARN].to_numpy(),b[WARN].to_numpy()),'Identical CSE partitions should yield identical warnings'
    pd.DataFrame(group_rows).to_csv(out/'group_inventory.csv',index=False)
    analysis=predictive_analysis(root,out,units)
    session.update(status='FULL_PASS' if analysis['rank_deficient_fits']==0 else 'COMPLETED_WITH_RANK_DEFICIENCY',
        completed_utc=datetime.now(timezone.utc).isoformat(),analysis=analysis,identical_cse_12h_24h_verified=True,
        original_frozen_files_modified=False,full_raw_data_alternative_width_experiment=False)
    write(out/'progress.json',session)
    write(out/'output_checksums.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='output_checksums.json' and p.suffix!='.tmp'})
    print(session['status'],completed_refs,'group references;',analysis,flush=True)
    return session

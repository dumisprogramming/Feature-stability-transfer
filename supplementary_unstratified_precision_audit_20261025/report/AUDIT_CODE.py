"""Read-only precision audit of saved unstratified target panels."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,zipfile,shutil,gc,platform
import numpy as np,pandas as pd
DATASETS=['NF_UNSW_NB15_v3','NF_BoT_IoT_v3','NF_CSE_CIC_IDS2018_v3','NF_ToN_IoT_v3']
SEEDS=list(range(2027,2032));MODELS=['LogisticRegression','RandomForest','XGBoost']
SPLITS=['train','calibration','source_test','environment']
STAGE='supplementary_natural_prevalence_20261024/run_v1'
N_TARGET=100000;POOL_SIZE=200000

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text())
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def canonical(frame,features,bits):
 with np.errstate(over='ignore',invalid='ignore'):
  a=frame[features].to_numpy(np.float64,copy=True);a[~np.isfinite(a)]=np.nan;a=a.astype('<f4' if bits==32 else '<f8')
 a[~np.isfinite(a)]=np.nan;a[a==0]=0.;a[np.isnan(a)]=np.nan
 return np.ascontiguousarray(a).view(np.dtype((np.void,(bits//8)*len(features)))).ravel().copy()
def priorities(target,seed,ids):
 salt=int.from_bytes(hashlib.sha256(f'paper3-natural-v1|{target}|{seed}'.encode()).digest()[:8],'little')
 x=np.asarray(ids,dtype=np.uint64)^np.uint64(salt)
 with np.errstate(over='ignore'):
  x=x+np.uint64(0x9E3779B97F4A7C15);x=(x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9);x=(x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB);x=x^(x>>np.uint64(31))
 return x

def inspect_panel(pool,panel,source_keys32,source_keys64,part_keys32,features,source,target,seed,n=N_TARGET):
 if source==target:raise ValueError('Self transfer is not a directed external panel')
 if len(panel)!=n or not panel.row_id.is_unique or not panel.Label.isin([0,1]).all():raise ValueError('Invalid saved panel')
 if not pool.row_id.is_unique or not pool.selection_key.is_unique or not pool.Label.isin([0,1]).all():raise ValueError('Invalid saved target pool')
 ids=panel.row_id.to_numpy(np.int64);indexed=pool.set_index('row_id',verify_integrity=True)
 if not np.isin(ids,indexed.index).all():raise ValueError('Panel has rows outside the saved pool')
 matching=indexed.loc[ids].reset_index()
 if not np.array_equal(panel.Label.to_numpy(),matching.Label.to_numpy()):raise ValueError('Pool/panel labels differ')
 if not np.array_equal(panel[features].to_numpy(np.float64),matching[features].to_numpy(np.float64),equal_nan=True):raise ValueError('Pool/panel feature values differ')
 legacy_mask=~np.isin(canonical(pool,features,64),source_keys64)
 legacy=pool[legacy_mask].sort_values(['selection_key','row_id'],kind='stable').iloc[:n].sort_values('row_id')
 if len(legacy)!=n or not np.array_equal(legacy.row_id.to_numpy(np.int64),ids):raise ValueError('Original float64 selection does not reproduce saved panel')
 signature=canonical(panel,features,32);overlap=np.isin(signature,source_keys32);by_split=[]
 for split,keys in part_keys32.items():
  hit=np.isin(signature,keys)
  by_split.append({'source_dataset':source,'target_dataset':target,'seed':seed,'source_partition':split,'matching_target_row_occurrences':int(hit.sum()),'benign_matching_rows':int((hit&panel.Label.eq(0).to_numpy()).sum()),'attack_matching_rows':int((hit&panel.Label.eq(1).to_numpy()).sum()),'example_target_row_ids':','.join(map(str,ids[hit][:20]))})
 eligible=~np.isin(canonical(pool,features,32),source_keys32);candidate=pool[eligible].sort_values(['selection_key','row_id'],kind='stable')
 enough=len(candidate)>=n;changed=None;corrected_hash=None;corrected_fraction=None
 if enough:
  corrected=candidate.iloc[:n].sort_values('row_id');newids=corrected.row_id.to_numpy('<i8');changed=int((~np.isin(newids,ids)).sum());corrected_hash=hashlib.sha256(newids.tobytes()).hexdigest();corrected_fraction=float(corrected.Label.mean())
  if np.isin(canonical(corrected,features,32),source_keys32).any():raise ValueError('Corrected candidate exclusion check failed')
 return {'source_dataset':source,'target_dataset':target,'seed':seed,'panel_rows':n,'source_float32_signature_overlap_rows':int(overlap.sum()),'legacy_float64_signature_overlap_rows':int(np.isin(canonical(panel,features,64),source_keys64).sum()),'source_overlap_fraction':float(overlap.mean()),'original_selection_reproduced':True,'pool_rows':len(pool),'corrected_eligible_pool_rows':len(candidate),'corrected_pool_sufficient':enough,'corrected_panel_replacement_rows':changed,'original_panel_row_ids_sha256':hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),'corrected_panel_row_ids_sha256':corrected_hash,'original_attack_fraction':float(panel.Label.mean()),'corrected_attack_fraction':corrected_fraction,'requires_panel_repair':bool(overlap.any() or not enough or changed!=0),'cross_dataset_row_ids_compared':False},by_split

def run_audit(root,expected,source,local_base=Path('/content')):
 root=Path(root);stamp=datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S_%f');out=Path(local_base)/'paper3_unstratified_precision_audit_20261025'/stamp;out.mkdir(parents=True)
 summary={'status':'RUNNING','started_utc':datetime.now(timezone.utc).isoformat(),'checked_panels':0,'new_model_fits':0,'new_model_predictions':0,'raw_archive_scans':0,'original_files_modified':False,'cross_dataset_row_ids_are_not_compared':True,'raw_global_bottom_k_not_independently_replayed':True,'pool_hashes_newly_recorded_not_historically_pinned':True};rows=[];split_rows=[];inputs={};pool_records=[];source_dtypes=[]
 (out/'AUDIT_CODE.py').write_text(source);write(out/'EXPECTED_CONTEXT.json',expected)
 def checked(rel,known=None):
  p=root/rel
  if not p.is_file():raise FileNotFoundError('Required saved input missing: '+str(p))
  value=sha(p)
  if known is not None and value!=known:raise ValueError('Input checksum differs: '+rel)
  inputs[rel]=value;return p
 try:
  identity=js(checked(STAGE+'/run_identity.json',expected['identity_sha256']))
  if identity!=expected['run_identity']:raise ValueError('Natural-prevalence run identity differs')
  if identity['target_rows']!=N_TARGET or identity['pool_rows']!=POOL_SIZE:raise ValueError('Panel/pool sizes differ from original configuration')
  features=js(checked('Paper3_Materialized/feature_schema_v1.json',expected['source_pins']['Paper3_Materialized/feature_schema_v1.json']))['predictors_in_order']
  if len(features)!=49:raise ValueError('Expected49predictors')
  for seed in SEEDS:
   sources={}
   for source_ds in DATASETS:
    keys32={};keys64=[]
    for split in SPLITS:
     rel=f'Paper3_Sanitized/{source_ds}/seed_{seed}/{split}.parquet';f=pd.read_parquet(checked(rel,expected['source_pins'][rel]));keys32[split]=np.unique(canonical(f,features,32));keys64.append(canonical(f,features,64));source_dtypes.append({'dataset':source_ds,'seed':seed,'partition':split,'feature_dtypes':str(dict(f[features].dtypes.astype(str).value_counts()))})
    sources[source_ds]=(np.unique(np.concatenate(list(keys32.values()))),np.unique(np.concatenate(keys64)),keys32)
   for target in DATASETS:
    marker_rel=f'{STAGE}/pools/{target}/complete.json';marker_path=checked(marker_rel);marker=js(marker_path)
    if marker['identity']!=identity:raise ValueError('Saved pool checkpoint identity differs')
    raw=expected['raw_info'][target]
    if marker['details']['raw_rows']!=raw['raw_rows'] or marker['details']['class_counts']!=raw['raw_class_counts']:raise ValueError('Saved pool raw inventory differs')
    pool_rel=f'pools/{target}/seed_{seed}.parquet';pool=pd.read_parquet(checked(STAGE+'/'+pool_rel,marker['output_hashes'][pool_rel]))
    if len(pool)!=POOL_SIZE:raise ValueError('Expected fixed200000-row oversample')
    if not ((pool.row_id>=0)&(pool.row_id<raw['raw_rows'])).all():raise ValueError('Target pool row IDs outside its raw capture')
    if not np.array_equal(pool.selection_key.to_numpy(np.uint64),priorities(target,seed,pool.row_id.to_numpy(np.int64))):raise ValueError('Pool priorities do not reproduce')
    pool_records.append({'target_dataset':target,'seed':seed,'pool_marker_sha256':inputs[marker_rel],'pool_sha256':inputs[STAGE+'/'+pool_rel],'rows':len(pool),'metadata_identity_matches':True,'historical_pool_hash_in_received_report':False})
    for source_ds in DATASETS:
     if source_ds==target:continue
     key=f'{source_ds}__to__{target}/seed_{seed}';context=expected['panels'][key];prefix='jobs/'+key
     for name in MODELS:
      rel=f'{prefix}/{name}.json';job=js(checked(STAGE+'/'+rel,expected['job_hashes'][rel]))
      if job['identity']!=identity or job['details']['panel']!=context['panel']:raise ValueError('Model job panel identity differs')
      for n in ['features.parquet','labels.parquet']:
       if job['output_hashes'][prefix+'/'+n]!=context['file_hashes'][prefix+'/'+n]:raise ValueError('Classifiers disagree on panel hashes')
     f=pd.read_parquet(checked(STAGE+'/'+prefix+'/features.parquet',context['file_hashes'][prefix+'/features.parquet']));labels=pd.read_parquet(checked(STAGE+'/'+prefix+'/labels.parquet',context['file_hashes'][prefix+'/labels.parquet']))
     if not f.row_id.equals(labels.row_id):raise ValueError('Feature/label row alignment differs')
     panel=f.copy();panel['Label']=labels.Label.to_numpy()
     if abs(float(panel.Label.mean())-float(context['panel']['selected_attack_fraction']))>1e-12:raise ValueError('Panel class fraction differs')
     union32,union64,parts=sources[source_ds];record,detail=inspect_panel(pool,panel,union32,union64,parts,features,source_ds,target,seed,n=N_TARGET);rows.append(record);split_rows+=detail;summary['checked_panels']=len(rows)
     pd.DataFrame(rows).to_csv(out/'panel_audit_60.csv',index=False);pd.DataFrame(split_rows).to_csv(out/'overlap_by_source_partition.csv',index=False);write(out/'progress.json',summary)
     print(f'Checked {len(rows)}/60 | {source_ds} -> {target} seed{seed} | overlap={record["source_float32_signature_overlap_rows"]} | corrected replacements={record["corrected_panel_replacement_rows"]}',flush=True)
     del f,labels,panel;gc.collect()
    del pool;gc.collect()
   del sources;gc.collect()
  grid=len(DATASETS)*(len(DATASETS)-1)*len(SEEDS)
  if len(rows)!=grid:raise ValueError('Incomplete expected panel grid')
  f=pd.DataFrame(rows)
  summary.update(status='AUDIT_COMPLETED',affected_panels=int(f.requires_panel_repair.sum()),panels_with_source_overlap=int(f.source_float32_signature_overlap_rows.gt(0).sum()),source_overlapping_target_row_occurrences=int(f.source_float32_signature_overlap_rows.sum()),target_row_occurrences=int(f.panel_rows.sum()),panels_requiring_expanded_pool=int((~f.corrected_pool_sufficient).sum()),all_original_float64_selections_reproduced=bool(f.original_selection_reproduced.all()),completed_utc=datetime.now(timezone.utc).isoformat())
 except Exception as e:summary.update(status='INCOMPLETE',error_type=type(e).__name__,error=str(e));print('STOPPED',type(e).__name__,str(e),flush=True)
 write(out/'input_hashes.json',inputs);write(out/'pool_checkpoint_provenance.json',pool_records);write(out/'source_feature_dtypes.json',source_dtypes);write(out/'verification_summary.json',summary)
 (out/'README.md').write_text('Read-only audit of saved unstratified target panels. Cross-dataset row-ID integers are different namespaces and are NOT compared; canonical model-precision49-feature signatures are compared against all four original source partitions. Original feature/label bytes and job markers are pinned to the received report. Saved200000-row pool hashes are checked against their matching-identity Drive completion markers, which were not included in the earlier downloaded report: their hashes are newly recorded, not independent historical certification. Pool priorities and old float64 panel selection are reconstructed; corrected float32 eligibility and lowest-priority100000-row selection are compared without writing repaired panels or running detectors. Adequate candidate pool support plus agreement with original raw-bottom-k provenance supports the corrected-selection comparison, but no full raw archive scan independently replays global bottom-k membership. Zero observed overlap alone is not a complete raw-data replication. Repeated rows across directions/seeds are occurrences, not independent flows. Return this report before deciding whether repair is needed. No original file is modified.\n')
 files=[p for p in out.rglob('*') if p.is_file()];write(out/'report_hashes.json',{str(p.relative_to(out)):sha(p) for p in files});files.append(out/'report_hashes.json')
 archive=out.parent/('paper3_unstratified_precision_audit_20261025_'+stamp+'.zip')
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,str(p.relative_to(out)))
 remote=root/'supplementary_unstratified_precision_audit_20261025'/stamp
 try:
  remote.mkdir(parents=True,exist_ok=False)
  for p in files:shutil.copyfile(p,remote/p.name)
  shutil.copyfile(archive,remote/archive.name);print('Audit saved:',remote)
 except Exception as e:print('Drive copy unavailable; download local report before closing runtime:',type(e).__name__,str(e))
 print(json.dumps(summary,indent=2));print('Report ZIP:',archive);return archive

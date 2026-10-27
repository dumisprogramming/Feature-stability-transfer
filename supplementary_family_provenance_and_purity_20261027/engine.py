"""Post hoc representative-label family exclusion using pinned six-field panels."""
from pathlib import Path
from datetime import datetime,timezone
from itertools import combinations
import json,hashlib,zipfile,shutil,time,platform,warnings,traceback
import numpy as np,pandas as pd,scipy,sklearn,joblib
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
PARENT_STAGE='supplementary_common_representation_partition_20261026'
STAGE='supplementary_pure_family_sensitivity_20261026'
SEED=2027;REPEATS=3
FEATURES=['destination_port','duration_us','fwd_packets','bwd_packets','fwd_packet_bytes','bwd_packet_bytes']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def read(p):return json.loads(Path(p).read_text())
def state(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')
def keys(x):
 x=np.asarray(x)
 if x.dtype!=np.float64 or x.ndim!=2 or x.shape[1]!=6 or not np.isfinite(x).all() or (x<0).any() or (x!=np.floor(x)).any() or (x>2**53-1).any() or (x[:,0]>65535).any():raise ValueError('Panel does not satisfy the audited six-integer/float64 representation')
 a=np.ascontiguousarray(x.astype('<i8'));return a.view(np.dtype((np.void,48))).ravel()
def classcheck(y,expected):
 y=np.asarray(y)
 if not np.isin(y,[0,1]).all() or y.ndim!=1:raise ValueError('Invalid binary label array')
 if int((y==0).sum())!=expected['benign'] or int((y==1).sum())!=expected['attack']:raise ValueError('Panel class counts differ from manifest')
def metrics(y,p):
 y=np.asarray(y);p=np.asarray(p)
 if not np.isin(y,[0,1]).all() or not np.isin(p,[0,1]).all() or y.shape!=p.shape or not (y==0).any() or not (y==1).any():raise ValueError('Invalid metric inputs')
 tn=int(((y==0)&(p==0)).sum());fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum());tp=int(((y==1)&(p==1)).sum());r0=tn/(tn+fp);r1=tp/(tp+fn);f0=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.;f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.;den=((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
 return dict(rows=len(y),tn=tn,fp=fp,fn=fn,tp=tp,balanced_accuracy=(r0+r1)/2,benign_recall=r0,attack_recall=r1,attack_precision=tp/(tp+fp) if tp+fp else 0.,macro_f1=(f0+f1)/2,mcc=(tp*tn-fp*fn)/den if den else 0.)
def ba(y,p):return metrics(y,p)['balanced_accuracy']
def threshold(y,p):
 fp,tp,t=roc_curve(y,p,drop_intermediate=False);ok=np.isfinite(t)&(t>=0)&(t<=1);t=t[ok];v=(1-fp[ok]+tp[ok])/2;i=np.flatnonzero(np.isclose(v,v.max(),atol=1e-12,rtol=0));return float(t[i[np.argmin(abs(t[i]-.5))]])
def stability(vectors):
 ranks=[rankdata(v,method='average') for v in vectors];pairs=[];constant=0
 for a,b in combinations(ranks,2):
  if np.std(a)==0 or np.std(b)==0:pairs.append(0.);constant+=1
  else:pairs.append(float(np.corrcoef(a,b)[0,1]))
 if not pairs:raise ValueError('At least two importance groups required')
 return {'stability':float(np.mean(pairs)),'pair_count':len(pairs),'constant_pairs':constant,'pairwise_correlations':pairs}
class Workspace:
 def __init__(self,root,context,source,plan,local_base,config):
  self.root=Path(root);self.local=Path(local_base)/STAGE/'run_v1';self.remote=self.root/'Paper3_DataSet'/STAGE/'run_v1';self.parent=self.root/'Paper3_DataSet'/PARENT_STAGE;self.local.mkdir(parents=True,exist_ok=True);self.remote.mkdir(parents=True,exist_ok=True)
  self.identity={'code_sha256':hashlib.sha256(source.encode()).hexdigest(),'plan_sha256':hashlib.sha256(plan.encode()).hexdigest(),'context_sha256':digest(context),'config':config,'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__}}
  if (self.remote/'identity.json').exists() and read(self.remote/'identity.json')!=self.identity:raise ValueError('Resume identity differs; retain old stage rather than mixing versions')
  write(self.local/'identity.json',self.identity);write(self.local/'context.json',context);(self.local/'PLAN.md').write_text(plan);(self.local/'CODE.py').write_text(source)
  for n in ['identity.json','context.json','PLAN.md','CODE.py']:self.upload(n)
 def upload(self,rel):
  src=self.local/rel;dst=self.remote/rel;dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix(dst.suffix+'.tmp');shutil.copyfile(src,tmp)
  if sha(tmp)!=sha(src):raise ValueError('Drive output copy mismatch: '+rel)
  tmp.replace(dst)
 def finish(self,marker,outputs,details):
  rec={'identity':self.identity,'outputs':{n:sha(self.local/n) for n in outputs},'details':details,'details_sha256':digest(details)};write(self.local/marker,rec)
  for n in outputs:self.upload(n)
  self.upload(marker)
 def cached(self,marker):
  src=self.remote/marker
  if not src.exists():return None
  rec=read(src)
  if rec['identity']!=self.identity or digest(rec['details'])!=rec['details_sha256']:raise ValueError('Checkpoint identity/details differ')
  for n,h in rec['outputs'].items():
   p=self.remote/n
   if sha(p)!=h:raise ValueError('Checkpoint output differs: '+n)
   dest=self.local/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
  shutil.copyfile(src,self.local/marker);return rec['details']
 def input(self,name,expected):
  p=self.parent/'panels'/(name+'.npz');dest=self.local/'input_cache'/(name+'.npz');dest.parent.mkdir(exist_ok=True)
  if not dest.exists() or sha(dest)!=expected:
   shutil.copyfile(p,dest)
   if sha(dest)!=expected:raise ValueError('Pinned panel differs: '+name)
  return dest
 def report(self,summary):
  write(self.local/'summary.json',summary)
  try:self.upload('summary.json')
  except OSError as e:print('Drive summary unavailable; local report retained:',e)
  files=[p for p in self.local.rglob('*') if p.is_file() and 'input_cache' not in p.parts and p.suffix in ['.json','.py','.md','.txt','.csv'] and p.name!='SHA256SUMS.json'];write(self.local/'SHA256SUMS.json',{str(p.relative_to(self.local)):sha(p) for p in files});stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f');zp=self.local.parent/f'pure_family_sensitivity_report_{stamp}.zip'
  with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
   for p in files+[self.local/'SHA256SUMS.json']:z.write(p,str(p.relative_to(self.local)))
  try:
   dest=self.remote.parent/zp.name;tmp=dest.with_suffix('.zip.tmp');shutil.copyfile(zp,tmp)
   if sha(tmp)!=sha(zp):raise ValueError('Report ZIP copy differs')
   tmp.replace(dest);print('Drive report:',dest,flush=True)
  except OSError as e:print('Drive report copy unavailable; use local download:',e)
  return zp

def load_panel(path,expected,with_labels=True):
 with np.load(path,allow_pickle=False) as z:
  required={'x','y','file_index','row_index','block','family'}
  if not required.issubset(z.files):raise ValueError('Missing panel arrays')
  data={k:z[k] for k in ['x','file_index','row_index','block']}
  if with_labels:data.update(y=z['y'],family=z['family'])
 n=expected['rows'];sig=keys(data['x'])
 if len(sig)!=n or len(np.unique(sig))!=n:raise ValueError('Panel size or unique-signature gate failed')
 if any(len(v)!=n for v in data.values()):raise ValueError('Panel array length mismatch')
 fi=data['file_index'];ri=data['row_index']
 if fi.dtype.kind not in 'iu' or ri.dtype.kind not in 'iu' or (fi<0).any() or (ri<0).any() or len(set(zip(fi.tolist(),ri.tolist())))!=n:raise ValueError('Invalid representative row locations')
 if with_labels:
  classcheck(data['y'],expected)
  if not np.all((data['family']=='Benign')==(data['y']==0)):raise ValueError('Source representative family/binary label mismatch')
 return data,sig

def transform(bundle,x):
 v=bundle['scaler'].transform(bundle['imputer'].transform(x))
 if v.dtype!=np.float64 or v.shape!=x.shape or not np.isfinite(v).all():raise ValueError('Invalid float64 preprocessing output')
 return v

def probabilities(bundle,x):
 p=bundle['model'].predict_proba(transform(bundle,x))[:,1]
 if not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid model probabilities')
 return p

CONDITIONS=['full_reference','family_excluded','size_matched_control']
FAMILY_ORDER=['GoldenEye','Hulk','SSH Patator']

def normal_label(s):return ''.join(c for c in str(s).casefold() if c.isalnum())
def family_mask(values,aliases):return np.isin(np.array([normal_label(x) for x in values]),[normal_label(x) for x in aliases])
def index_hash(idx):return hashlib.sha256(np.asarray(idx,dtype='<i8').tobytes()).hexdigest()
def select_roles(panels,candidate,condition):
 result={};details={}
 for role in ['train','calibration']:
  d=panels[role];held=family_mask(d['family'],candidate['aliases']['source']);keep=np.flatnonzero(~held)
  if condition=='full_reference':idx=np.arange(len(d['y']))
  elif condition=='family_excluded':idx=keep
  elif condition=='size_matched_control':
   chosen=[]
   for label in [0,1]:
    pool=np.flatnonzero(d['y']==label);n=int((d['y'][keep]==label).sum())
    if n>len(pool):raise ValueError('Invalid matched size')
    chosen.extend(np.random.default_rng(state('family-size-control',SEED,candidate['candidate_tool_label_family'],role,label)).choice(pool,size=n,replace=False).tolist())
   idx=np.sort(chosen)
  else:raise ValueError('Unknown fitting condition')
  yy=d['y'][idx]
  if any(int((yy==label).sum())<50 for label in [0,1]):raise ValueError('Fitting/calibration class support below 50')
  held_count=int(held[idx].sum())
  if condition=='family_excluded' and held_count:raise ValueError('Excluded representative family remains')
  result[role]=idx;details[role]={'rows':len(idx),'benign':int((yy==0).sum()),'attack':int((yy==1).sum()),'candidate_family_representatives':held_count,'row_positions_sha256':index_hash(idx)}
 return result,details

def environmental_design(env,candidate,quota=500):
 held=family_mask(env['family'],candidate['aliases']['source']);blocks=np.sort(np.unique(env['block']));selected=[];actual=[];retained=[]
 for block in blocks:
  take=(env['block']==block)&~held;parts=[np.flatnonzero(take&(env['y']==label)) for label in [0,1]]
  if min(map(len,parts))<quota:continue
  k=len(retained);retained.append(str(block))
  for label,positions in enumerate(parts):
   # Production panels have exactly the audited quota in each eligible class.
   chosen=np.sort(np.random.default_rng(state('family-env-selection',SEED,candidate['candidate_tool_label_family'],str(block),label)).choice(positions,size=quota,replace=False))
   selected.extend(chosen.tolist());actual.extend([k]*quota)
 selected=np.asarray(selected,dtype=int);actual=np.asarray(actual,dtype=int);ng=len(retained)
 if ng<2:raise ValueError('Fewer than two retained source environments')
 pseudo=np.full(len(selected),-1,dtype=int);yy=env['y'][selected]
 for label in [0,1]:
  positions=np.random.default_rng(state('family-pseudo',SEED,candidate['candidate_tool_label_family'],label)).permutation(np.flatnonzero(yy==label))
  for k in range(ng):pseudo[positions[k*quota:(k+1)*quota]]=k
 for kind,assignment in [('environment',actual),('sampling',pseudo)]:
  for k in range(ng):
   if any(int(((assignment==k)&(yy==label)).sum())!=quota for label in [0,1]):raise ValueError('Unbalanced importance group')
 if held[selected].any():raise ValueError('Held family remains in environmental representatives')
 return {'selected':selected,'environment':actual,'sampling':pseudo,'blocks':retained,'groups':ng,'quota':quota,'selected_sha256':index_hash(selected),'environment_sha256':index_hash(actual),'sampling_sha256':index_hash(pseudo)}

def fit_bundle(panels,indices):
 tr,ca=panels['train'],panels['calibration'];ti,ci=indices['train'],indices['calibration'];im=SimpleImputer(strategy='median');xt=im.fit_transform(tr['x'][ti]);xc=im.transform(ca['x'][ci]);sc=StandardScaler();xt=sc.fit_transform(xt);xc=sc.transform(xc)
 if xt.dtype!=np.float64 or xc.dtype!=np.float64:raise ValueError('Preprocessing precision changed')
 model=LogisticRegression(C=1.,solver='lbfgs',max_iter=3000,random_state=SEED)
 with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always');model.fit(xt,tr['y'][ti])
 if any(issubclass(x.category,ConvergenceWarning) for x in caught):raise ValueError('Convergence warning; stop without retuning')
 cp=model.predict_proba(xc)[:,1]
 if not np.isfinite(cp).all():raise ValueError('Invalid calibration probabilities')
 th=threshold(ca['y'][ci],cp);bundle={'model':model,'imputer':im,'scaler':sc,'features':FEATURES,'threshold':th,'dtype':'float64'}
 return bundle,{'threshold':th,'calibration_metrics':metrics(ca['y'][ci],cp>=th),'estimator_parameters':model.get_params(deep=False),'n_iter':model.n_iter_.tolist(),'warnings':[str(x.message) for x in caught]}

def importance_record(bundle,x,y,idx,family,kind,k,block,inputs):
 xx=x[idx].copy();yy=y[idx];base=ba(yy,bundle['model'].predict_proba(xx)[:,1]>=bundle['threshold']);arr=np.empty((6,REPEATS))
 for j in range(6):
  original=xx[:,j].copy()
  for repeat in range(REPEATS):
   order=np.random.default_rng(state('family-pfi',SEED,family,kind,k,j,repeat)).permutation(len(xx));xx[:,j]=original[order];arr[j,repeat]=base-ba(yy,bundle['model'].predict_proba(xx)[:,1]>=bundle['threshold'])
  xx[:,j]=original
 return {'partition':kind,'group':k,'block':block,'rows':len(idx),'benign':int((yy==0).sum()),'attack':int((yy==1).sum()),'baseline_ba':base,'importance_repeats':arr.tolist(),'importance_mean':arr.mean(axis=1).tolist(),'inputs':inputs}

def family_recall(data,pred,aliases):
 take=family_mask(data['family'],aliases)&(data['y']==1);n=int(take.sum());det=int(pred[take].sum())
 if n<50:raise ValueError('Selected candidate family has fewer than 50 representatives')
 return {'rows':n,'detected':det,'missed':n-det,'recall':det/n,'miss_rate':1-det/n,'representative_labels_only':True}

def check_reference(evaluation,expected):
 for role in ['source_test','target']:
  for k in ['tn','fp','fn','tp']:
   if evaluation[role]['metrics'][k]!=expected[role][k]:raise ValueError('Full reference does not reproduce supplied pilot '+role+'/'+k)

def purify_panels(w,context,panels):
 path=w.root/context['provenance_report_relative_path'];local=w.local/'input_cache'/'provenance_report.zip'
 if not local.exists() or sha(local)!=context['provenance_report_sha256']:
  shutil.copyfile(path,local)
  if sha(local)!=context['provenance_report_sha256']:raise ValueError('Provenance report differs from reviewed audit')
 with zipfile.ZipFile(local) as z:
  manifest=json.loads(z.read('SHA256SUMS.json'))
  for name in ['selected_pattern_index.json','pattern_provenance.json']:
   if hashlib.sha256(z.read(name)).hexdigest()!=manifest[name]:raise ValueError('Provenance member hash differs')
  index=json.loads(z.read('selected_pattern_index.json'));provenance=json.loads(z.read('pattern_provenance.json'))
 expected_total=sum(context['manifest'][role]['rows'] for role in panels)
 if len(index)!=len(provenance) or len(index)!=expected_total:raise ValueError('Provenance coverage differs')
 records={};seen=set()
 for m,p in zip(index,provenance):
  pid=m['pattern_id']
  if pid in seen or p['pattern_id']!=pid or m['role']!=p['role'] or m['representative_family']!=p['representative_family']:raise ValueError('Provenance index alignment differs')
  seen.add(pid);role=m['role'];row=m['panel_row'];data=panels[role]
  if not 0<=row<len(data['y']) or (role,row) in records:raise ValueError('Invalid duplicate provenance panel row')
  key=bytes(np.ascontiguousarray(data['x'][row].astype('<i8')).view('V48')[0])
  if hashlib.sha256(key).hexdigest()!=m['signature_sha256'] or str(data['family'][row])!=m['representative_family'] or int(data['y'][row])!=m['binary_label'] or int(data['file_index'][row])!=m['file_index'] or int(data['row_index'][row])!=m['row_index']:raise ValueError('Provenance does not match pinned selected row')
  own=p['raw_own_family_counts']
  if not own or any(not isinstance(n,int) or n<1 for n in own.values()) or own.get(m['representative_family'],0)<1 or p['mixed_exact_families']!=(len(own)>1):raise ValueError('Invalid own-capture family counts')
  binary={0 if label==('BENIGN' if role=='target' else 'Benign') else 1 for label in own}
  if p['binary_conflict']!=(len(binary)>1):raise ValueError('Invalid binary conflict flag')
  records[(role,row)]=p
 removed={};purified={};counts={}
 for role,data in panels.items():
  mask=np.array([len(records[(role,i)]['raw_own_family_counts'])==1 and not records[(role,i)]['binary_conflict'] for i in range(len(data['y']))])
  removed[role]=np.flatnonzero(~mask).tolist()
  if removed[role]!=context['purity_removed_panel_rows'][role]:raise ValueError('Purity mask differs from fixed plan')
  purified[role]={k:v[mask] for k,v in data.items()};y=purified[role]['y'];counts[role]={'rows':len(y),'benign':int((y==0).sum()),'attack':int((y==1).sum())}
  if counts[role]!=context['purified_panel_counts'][role]:raise ValueError('Purified panel class counts differ')
  for i in np.flatnonzero(mask):
   if set(records[(role,int(i))]['raw_own_family_counts'])!={str(data['family'][i])}:raise ValueError('Retained pattern lacks exact own-label purity')
 details={'provenance_report_sha256':context['provenance_report_sha256'],'removed_panel_rows':removed,'purified_panel_counts':counts,'input_panel_hashes':{role:context['manifest'][role]['sha256'] for role in panels},'fitting_and_environment_indices_refer_to_purified_panel_order':True,'target_restriction_uses_previously_known_raw_labels':True,'all_retained_patterns_have_one_exact_own_capture_family_label':True,'semantic_equivalence_established':False}
 write(w.local/'purity_selection.json',details);w.upload('purity_selection.json');return purified,details

def run_family(root,context,source,plan,local_base=Path('/content')):
 config={'families':FAMILY_ORDER,'conditions':CONDITIONS,'seed':SEED,'features':6,'pfi_repeats':REPEATS,'environment_class_quota':500,'classifier':'LogisticRegression','post_hoc':True,'representative_label_ablation':True,'own_capture_single_family_patterns_only':True,'target_label_informed_post_hoc_restriction':True}
 w=Workspace(root,context,source,plan,local_base,config);(w.local/'error.txt').unlink(missing_ok=True)
 started=time.monotonic();summary={'status':'RUNNING','new_fits_this_session':0,'new_importance_groups_this_session':0,'new_evaluations_this_session':0,'prior_outcomes_known':True,'semantic_equivalence_established':False,'production_validation_completed':False,'no_warning_predictive_utility_inference':True,'purity_filtered_sensitivity':True}
 m=context['manifest'];records=[];source_seals={};fits={};designs={};candidates=context['selected_candidates']
 try:
  if [c['candidate_tool_label_family'] for c in candidates]!=FAMILY_ORDER:raise ValueError('Family order differs')
  if sha(w.parent/'panel_manifest.json')!=context['manifest_sha256'] or sha(w.parent/'identity.json')!=context['parent_identity_sha256'] or read(w.parent/'panel_manifest.json')!=m or m['features']!=FEATURES:raise ValueError('Parent identity or representation differs')
  paths={n:w.input(n,m[n]['sha256']) for n in ['train','calibration','source_test','environment','target']};panels={};sigs={};locations=set()
  for name in ['train','calibration','source_test','environment']:
   panels[name],sigs[name]=load_panel(paths[name],m[name]);d=panels[name];ids=set(zip(d['file_index'].tolist(),d['row_index'].tolist()))
   if locations&ids:raise ValueError('Source row locations overlap')
   locations.update(ids)
  _,sigs['target']=load_panel(paths['target'],m['target'],with_labels=False)
  for a,b in combinations(sigs,2):
   if np.intersect1d(sigs[a],sigs[b]).size:raise ValueError('Selected source/target signatures overlap')
  # Purity is a planned, target-label-informed cohort restriction after earlier outcomes were known.
  with np.load(paths['target'],allow_pickle=False) as z:panels['target']={k:z[k] for k in ['x','y','family','file_index','row_index','block']}
  classcheck(panels['target']['y'],m['target'])
  panels,purity=purify_panels(w,context,panels);target_for_eval=panels.pop('target');purity_hash=sha(w.local/'purity_selection.json');new_full_threshold=None
  env=panels['environment']
  for candidate in candidates:
   family=candidate['candidate_tool_label_family'];slug=family.replace(' ','_');design=environmental_design(env,candidate)
   for role in panels:
    if int(family_mask(panels[role]['family'],candidate['aliases']['source']).sum())!=candidate['family_counts'][role]:raise ValueError('Audited source family counts differ')
   if design['groups']!=candidate['expected_pure_environment_groups']:raise ValueError('Purified eligible group count differs for '+family)
   expected_blocks=[g['block'] for g in candidate['retained_environment_by_block'] if g['maximum_balanced_rows_per_class']>=500]
   if design['blocks']!=expected_blocks:raise ValueError('Eligible source windows differ from support audit')
   designs[family]=design
   rel=f'{slug}/environment_selection.npz';(w.local/slug).mkdir(exist_ok=True);np.savez_compressed(w.local/rel,selected=design['selected'],actual=design['environment'],pseudo=design['sampling']);w.upload(rel)
   for condition in CONDITIONS:
    tag=f'{slug}/{condition}';(w.local/tag).mkdir(parents=True,exist_ok=True);indices,selection=select_roles(panels,candidate,condition)
    expected={'family':family,'condition':condition,'selection':selection,'parent_manifest_sha256':context['manifest_sha256'],'purity_selection_sha256':purity_hash}
    marker=tag+'/fit.complete.json';fit=w.cached(marker)
    if fit is None:
     bundle,details=fit_bundle(panels,indices)
     joblib.dump(bundle,w.local/(tag+'/model.joblib'),compress=3)
     np.savez_compressed(w.local/(tag+'/fitting_indices.npz'),**indices)
     fit={'inputs':expected,'model_sha256':sha(w.local/(tag+'/model.joblib')),**details};write(w.local/(tag+'/fit_details.json'),fit);w.finish(marker,[tag+'/model.joblib',tag+'/fit_details.json',tag+'/fitting_indices.npz'],fit);summary['new_fits_this_session']+=1
    if fit['inputs']!=expected:raise ValueError('Cached fit selection differs')
    bundle=joblib.load(w.local/(tag+'/model.joblib'))
    if fit['model_sha256']!=sha(w.local/(tag+'/model.joblib')):raise ValueError('Model hash differs')
    if condition=='full_reference':
     if new_full_threshold is None:new_full_threshold=fit['threshold']
     elif abs(fit['threshold']-new_full_threshold)>1e-12:raise ValueError('New pure full-reference thresholds disagree')
    fits[tag]=fit;x=transform(bundle,env['x'][design['selected']]);y=env['y'][design['selected']];by_type={};group_hashes={}
    with threadpool_limits(limits=1):
     for kind in ['environment','sampling']:
      vectors=[]
      for k in range(design['groups']):
       idx=np.flatnonzero(design[kind]==k);rel=tag+f'/groups/{kind}_{k:03d}.json';marker=rel.replace('.json','.complete.json');inputs={'model_sha256':fit['model_sha256'],'source_environment_panel_sha256':m['environment']['sha256'],'selected_rows_sha256':design['selected_sha256'],'assignment_sha256':design[kind+'_sha256'],'within_selected_positions_sha256':index_hash(idx)};rec=w.cached(marker)
       if rec is None:
        rec=importance_record(bundle,x,y,idx,family,kind,k,design['blocks'][k] if kind=='environment' else None,inputs);write(w.local/rel,rec);w.finish(marker,[rel],rec);summary['new_importance_groups_this_session']+=1
       a=np.asarray(rec['importance_repeats'])
       if rec['inputs']!=inputs or a.shape!=(6,3) or not np.isfinite(a).all() or not np.allclose(a.mean(axis=1),rec['importance_mean'],atol=1e-14,rtol=0) or rec['benign']!=500 or rec['attack']!=500:raise ValueError('Importance record or quota differs')
       vectors.append(a.mean(axis=1));group_hashes[rel]=sha(w.local/rel)
      by_type[kind]=stability(vectors)
    se=by_type['environment']['stability'];ss=by_type['sampling']['stability'];warning={'environmental_instability':1-se,'sampling_instability':1-ss,'sampling_environment_gap':ss-se,'stability_details':by_type,'environment_groups':design['groups'],'environment_rows':len(design['selected']),'class_quota_per_group':500,'family_representatives_removed_from_all_conditions_environment':True}
    seal={'family':family,'condition':condition,'fit_marker_sha256':sha(w.local/(tag+'/fit.complete.json')),'model_sha256':fit['model_sha256'],'group_hashes':group_hashes,'warnings':warning,'environment_blocks':design['blocks'],'selection_sha256':design['selected_sha256'],'source_family_labels_are_representatives':True}
    cached=w.cached(tag+'/warnings.complete.json')
    if cached is None:write(w.local/(tag+'/warnings.json'),seal);w.finish(tag+'/warnings.complete.json',[tag+'/warnings.json',slug+'/environment_selection.npz'],seal)
    elif cached!=seal:raise ValueError('Recomputed source warning seal differs')
    source_seals[tag]=sha(w.local/(tag+'/warnings.complete.json'));print('Source fit and warnings completed',len(source_seals),'/ 9',family,condition,flush=True)
  all_source={'source_warning_seals':source_seals,'all_nine_fits_and_warnings_completed_before_new_target_scoring':True,'prior_outcomes_known':True}
  cached=w.cached('all_source.complete.json')
  if cached is None:write(w.local/'all_source.json',all_source);w.finish('all_source.complete.json',['all_source.json'],all_source)
  elif cached!=all_source:raise ValueError('Global source seal differs')
  print('All source warnings sealed. Beginning fixed source-test and target scoring.',flush=True)
  target=target_for_eval;classcheck(target['y'],context['purified_panel_counts']['target']);new_full_reference=None
  if not np.array_equal(target['family']=='BENIGN',target['y']==0):raise ValueError('Target labels differ')
  for candidate in candidates:
   family=candidate['candidate_tool_label_family'];slug=family.replace(' ','_')
   if int(family_mask(target['family'],candidate['aliases']['target']).sum())!=candidate['family_counts']['target']:raise ValueError('Audited target family support differs')
   for condition in CONDITIONS:
    tag=f'{slug}/{condition}';fit=fits[tag];bundle=joblib.load(w.local/(tag+'/model.joblib'));evaluation={}
    for role,data in [('source_test',panels['source_test']),('target',target)]:
     inputs={'source_warning_seal_sha256':source_seals[tag],'all_source_seal_sha256':sha(w.local/'all_source.complete.json'),'panel_sha256':m[role]['sha256'],'model_sha256':fit['model_sha256'],'purity_selection_sha256':purity_hash};marker=tag+'/'+role+'.complete.json';ev=w.cached(marker)
     if ev is None:
      with threadpool_limits(limits=1):prob=probabilities(bundle,data['x'])
      pred=(prob>=bundle['threshold']).astype(np.int8);aliases=candidate['aliases']['target' if role=='target' else 'source'];ev={'role':role,'inputs':inputs,'metrics':metrics(data['y'],pred),'candidate_family':family_recall(data,pred,aliases)}
      np.savez_compressed(w.local/(tag+'/'+role+'_predictions.npz'),probability=prob,predicted=pred,y=data['y']);write(w.local/(tag+'/'+role+'_evaluation.json'),ev);w.finish(marker,[tag+'/'+role+'_evaluation.json',tag+'/'+role+'_predictions.npz'],ev);summary['new_evaluations_this_session']+=1
     if ev['inputs']!=inputs:raise ValueError('Evaluation input identity differs')
     evaluation[role]=ev
    if condition=='full_reference':
     if new_full_reference is None:new_full_reference={role:evaluation[role]['metrics'] for role in ['source_test','target']}
     else:check_reference(evaluation,new_full_reference)
    warnings_=read(w.local/(tag+'/warnings.json'))['warnings'];records.append({'family':family,'condition':condition,'selection':fit['inputs']['selection'],'threshold':fit['threshold'],'source_test':evaluation['source_test'],'target':evaluation['target'],'balanced_accuracy_loss':evaluation['source_test']['metrics']['balanced_accuracy']-evaluation['target']['metrics']['balanced_accuracy'],'warnings':warnings_})
    print('Evaluated family/condition',len(records),'/ 9',family,condition,flush=True)
  contrasts=[]
  for family in FAMILY_ORDER:
   rr={r['condition']:r for r in records if r['family']==family};ex=rr['family_excluded']
   for comparator in ['full_reference','size_matched_control']:
    ref=rr[comparator];contrasts.append({'family':family,'contrast':'family_excluded_minus_'+comparator,'source_family_recall_change':ex['source_test']['candidate_family']['recall']-ref['source_test']['candidate_family']['recall'],'target_family_recall_change':ex['target']['candidate_family']['recall']-ref['target']['candidate_family']['recall'],'target_balanced_accuracy_change':ex['target']['metrics']['balanced_accuracy']-ref['target']['metrics']['balanced_accuracy'],'environmental_instability_change':ex['warnings']['environmental_instability']-ref['warnings']['environmental_instability'],'sampling_instability_change':ex['warnings']['sampling_instability']-ref['warnings']['sampling_instability'],'gap_change':ex['warnings']['sampling_environment_gap']-ref['warnings']['sampling_environment_gap']})
  final={'records':records,'paired_descriptive_contrasts':contrasts,'one_source_one_target_one_seed_one_classifier':True,'candidate_tool_label_correspondence_only':True,'own_capture_single_family_signature_filter_applied':True,'raw_purity_audited_only_in_pinned_files':True,'target_label_informed_post_hoc_restriction':True,'no_predictive_warning_utility_test':True,'no_p_values_or_meta_regression':True,'no_production_prevalence_validation':True};write(w.local/'family_results.json',final);w.finish('experiment.complete.json',['family_results.json'],final)
  summary.update(status='PURE_FAMILY_SENSITIVITY_COMPLETE',fits_completed=9,importance_groups_completed=sum(6*d['groups'] for d in designs.values()),permutation_scores_completed=sum(108*d['groups'] for d in designs.values()),source_target_evaluations_completed=18,family_condition_records=9,paired_contrasts=6,groups_per_family_condition={family:d['groups'] for family,d in designs.items()})
 except Exception as e:
  summary.update(status='PURE_FAMILY_SENSITIVITY_BLOCKED_RESUMABLE',error=str(e));(w.local/'error.txt').write_text(traceback.format_exc())
  try:w.upload('error.txt')
  except OSError:pass
 summary['session_elapsed_seconds']=time.monotonic()-started;print(json.dumps(summary,indent=2),flush=True);return w.report(summary)

import tempfile,copy,json,zipfile,sys,importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score,confusion_matrix,matthews_corrcoef
from threadpoolctl import threadpool_limits
import engine as e

def fixture(root):
 parent=root/'Paper3_DataSet'/e.PARENT_STAGE;(parent/'panels').mkdir(parents=True)
 source_labels=['DoS attacks-GoldenEye','DoS attacks-Hulk','SSH-Bruteforce'];target_labels=['DoS GoldenEye','DoS Hulk','SSH-Patator'];rng=np.random.default_rng(193);panels={};manifest={'features':e.FEATURES,'source_environment_groups':9}
 for i,role in enumerate(['train','calibration','source_test','environment','target']):
  n=1000 if role=='train' else 9000 if role=='environment' else 600
  y=np.tile([0,1],n//2);x=rng.integers(0,100,size=(n,6)).astype(np.float64);x[:,0]+=y*20;x[:,1]=np.arange(n)+100000*i
  family=np.full(n,'Other',dtype='<U40');family[y==0]='BENIGN' if role=='target' else 'Benign'
  labels=target_labels if role=='target' else source_labels
  if role=='environment':
   block=np.repeat([f'b{j}' for j in range(9)],1000)
   for j,label in enumerate(labels):family[(block==f'b{j}')&(y==1)]=label
  else:
   block=np.full(n,'unused');attack=np.flatnonzero(y==1);quota=100 if role=='train' else 60
   for j,label in enumerate(labels):family[attack[j*quota:(j+1)*quota]]=label
  d={'x':x,'y':y,'family':family,'block':block,'file_index':np.full(n,i),'row_index':np.arange(n)};panels[role]=d
  path=parent/'panels'/f'{role}.npz';np.savez_compressed(path,**d);manifest[role]={'rows':n,'benign':n//2,'attack':n//2,'sha256':e.sha(path)}
 e.write(parent/'panel_manifest.json',manifest);e.write(parent/'identity.json',{'synthetic':True})
 cs=[]
 for j,name in enumerate(e.FAMILY_ORDER):
  aliases={'source':[source_labels[j]],'target':[target_labels[j]]};counts={k:int(e.family_mask(d['family'],aliases['target' if k=='target' else 'source']).sum()) for k,d in panels.items()};design=e.environmental_design(panels['environment'],{'candidate_tool_label_family':name,'aliases':aliases})
  blocks=[{'block':f'b{k}','maximum_balanced_rows_per_class':0 if k==j else 500} for k in range(9)]
  cs.append({'candidate_tool_label_family':name,'aliases':aliases,'family_counts':counts,'retained_environment_by_block':blocks})
 # Independent full reference and brute-force calibration threshold.
 im=SimpleImputer(strategy='median');sc=StandardScaler();xx=sc.fit_transform(im.fit_transform(panels['train']['x']));model=LogisticRegression(C=1.,solver='lbfgs',max_iter=3000,random_state=2027)
 with threadpool_limits(limits=1):model.fit(xx,panels['train']['y'])
 cp=model.predict_proba(sc.transform(im.transform(panels['calibration']['x'])))[:,1];ts=np.unique(cp);scores=np.array([balanced_accuracy_score(panels['calibration']['y'],cp>=v) for v in ts]);eligible=ts[np.isclose(scores,scores.max(),atol=1e-12,rtol=0)];th=float(sorted(eligible,key=lambda v:(abs(v-.5),-v))[0]);ref={'threshold':th};preds={}
 for role in ['source_test','target']:
  pred=(model.predict_proba(sc.transform(im.transform(panels[role]['x'])))[:,1]>=th).astype(int);preds[role]=pred;tn,fp,fn,tp=confusion_matrix(panels[role]['y'],pred,labels=[0,1]).ravel();ref[role]={'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
 return {'manifest':manifest,'manifest_sha256':e.sha(parent/'panel_manifest.json'),'parent_identity_sha256':e.sha(parent/'identity.json'),'selected_candidates':cs,'full_reference':ref},panels

def unpack(path):
 with zipfile.ZipFile(path) as z:
  s=json.loads(z.read('summary.json'));final=json.loads(z.read('family_results.json')) if 'family_results.json' in z.namelist() else None
  manifest=json.loads(z.read('SHA256SUMS.json'));import hashlib
  assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
 return s,final

def test():
 with tempfile.TemporaryDirectory() as tmp:
  base=Path(tmp);root=base/'drive';context,panels=fixture(root);code=Path(e.__file__).read_text();plan='Synthetic integration test only.'
  zp=e.run_family(root,context,code,plan,base/'local');s,f=unpack(zp);assert s['status']=='COMMON_FAMILY_EXCLUSION_COMPLETE',s;assert s['new_fits_this_session']==9 and s['new_importance_groups_this_session']==144 and s['new_evaluations_this_session']==18
  remote=root/'Paper3_DataSet'/e.STAGE/'run_v1'
  for rec in f['records']:
   fam=rec['family'];cond=rec['condition'];tag=fam.replace(' ','_')+'/'+cond;candidate=next(c for c in context['selected_candidates'] if c['candidate_tool_label_family']==fam);indices,selection=e.select_roles(panels,candidate,cond)
   assert rec['selection']==selection
   if cond=='family_excluded':assert not e.family_mask(panels['train']['family'][indices['train']],candidate['aliases']['source']).any()
   for role in ['source_test','target']:
    with np.load(remote/(tag+'/'+role+'_predictions.npz')) as z:y=z['y'];pred=z['predicted']
    assert abs(rec[role]['metrics']['balanced_accuracy']-balanced_accuracy_score(y,pred))<1e-14
    assert abs(rec[role]['metrics']['mcc']-matthews_corrcoef(y,pred))<1e-14
    take=e.family_mask(panels[role]['family'],candidate['aliases']['target' if role=='target' else 'source']);assert abs(rec[role]['candidate_family']['recall']-pred[take].mean())<1e-14
   for kind in ['environment','sampling']:
    vec=[]
    for i in range(8):
     d=e.read(remote/(tag+f'/groups/{kind}_{i:03d}.json'));a=np.array(d['importance_repeats']);assert np.allclose(a.mean(axis=1),d['importance_mean'],rtol=0,atol=1e-14);vec.append(a.mean(axis=1))
    from itertools import combinations
    vals=[0. if np.ptp(a)==0 or np.ptp(b)==0 else float(spearmanr(a,b).statistic) for a,b in combinations(vec,2)]
    assert np.allclose(vals,rec['warnings']['stability_details'][kind]['pairwise_correlations'],rtol=0,atol=1e-14)
  # Matched controls have exactly the excluded condition's class sizes.
  for candidate in context['selected_candidates']:
   _,excluded=e.select_roles(panels,candidate,'family_excluded');_,matched=e.select_roles(panels,candidate,'size_matched_control')
   for role in ['train','calibration']:
    assert (excluded[role]['benign'],excluded[role]['attack'])==(matched[role]['benign'],matched[role]['attack'])
  zp=e.run_family(root,context,code,plan,base/'fresh_local');s,_=unpack(zp);assert s['status']=='COMMON_FAMILY_EXCLUSION_COMPLETE';assert s['new_fits_this_session']==s['new_importance_groups_this_session']==s['new_evaluations_this_session']==0
  changed=remote/'GoldenEye/full_reference/groups/environment_000.json';d=e.read(changed);d['baseline_ba']+=.01;e.write(changed,d)
  zp=e.run_family(root,context,code,plan,base/'tamper_local');s,_=unpack(zp);assert s['status']=='COMMON_FAMILY_EXCLUSION_BLOCKED_RESUMABLE'
 print('PASS synthetic full 9-fit/144-group/18-evaluation run, independent threshold/metrics/ranks, exact resume without refitting, matched class counts, and rejection of changed group output. Real experiment not run here.')
if __name__=='__main__':test()

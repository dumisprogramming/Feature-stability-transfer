"""Compact-result arithmetic verification; no feature/raw prediction replay."""
import pathlib,json,hashlib,csv,sys,itertools,math
import numpy as np

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def rank(a):
 a=np.asarray(a,dtype=float);o=np.argsort(a,kind='stable');out=np.empty(len(a));i=0
 while i<len(a):
  j=i+1
  while j<len(a) and a[o[j]]==a[o[i]]:j+=1
  out[o[i:j]]=(i+j-1)/2+1;i=j
 return out

def rho(a,b):
 a=rank(a);b=rank(b);a-=np.mean(a);b-=np.mean(b);den=np.linalg.norm(a)*np.linalg.norm(b)
 return float(np.clip(np.dot(a,b)/den,-1,1)) if den else None

def close(a,b):
 if a is None or b is None:assert a is b,(a,b)
 else:assert abs(a-b)<1e-10,(a,b)

def verify(root):
 root=pathlib.Path(root);read=lambda n:json.loads((root/n).read_text());sha=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
 h=read('SHA256SUMS.json')
 for n,v in h.items():assert sha(n)==v,n
 ident=read('identity.json');assert ident['code_sha256']==sha('CODE.py') and ident['plan_sha256']==sha('PLAN.md') and ident['context_sha256']==digest(read('context.json'))
 checkpoints=0;refs=0
 for f in root.rglob('*.complete.json'):
  m=json.loads(f.read_text());assert m['identity']==ident and digest(m['details'])==m['details_sha256'];checkpoints+=1
  for n,v in m['outputs'].items():assert sha(n)==v;refs+=1
 summary=read('summary.json');units=[];metricsets=0;pairsets=0;group_arrays=0;calibrations=0
 def arrays(n):
  with np.load(root/n,allow_pickle=False) as z:return {k:z[k] for k in z.files}
 def met(y,p,t):
  pred=p>=t;c=[int(((y==yy)&(pred==pp)).sum()) for yy,pp in [(0,0),(0,1),(1,0),(1,1)]];tn,fp,fn,tp=c
  return dict(rows=len(y),tn=tn,fp=fp,fn=fn,tp=tp,ba=.5*(tn/(tn+fp)+tp/(tp+fn)),recall=tp/(tp+fn),precision=tp/(tp+fp) if tp+fp else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 for f in sorted(root.glob('source_*/seed_*/fit.json')):
  rel=str(f.parent.relative_to(root));d=read(str(f.relative_to(root)));a=arrays(rel+'/calibration.npz');m=arrays(rel+'/fit.npz');close(d['threshold'],float(m['threshold'][0]));y,p=a['y'],a['probability'];t=d['threshold']
  assert set(y.tolist())=={0,1} and np.all(np.isfinite(p)) and np.all((p>=0)&(p<=1))
  for k,v in met(y,p,t).items():close(v,d['calibration'][k])
  # Independent optimal-BA search by sorting probabilities within each class.
  candidates=np.r_[np.unique(p),np.nextafter(p.max(),np.inf)];p0=np.sort(p[y==0]);p1=np.sort(p[y==1]);scores=.5*(np.searchsorted(p0,candidates,side='left')/len(p0)+1-np.searchsorted(p1,candidates,side='left')/len(p1));best=scores.max();threshold=candidates[np.abs(scores-best)<=1e-12].max();close(threshold,t);calibrations+=1
 for f in sorted(root.glob('source_*/seed_*/test_*/metrics.json')):
  rel=str(f.parent.relative_to(root));d=read(str(f.relative_to(root)));a=arrays(rel+'/probabilities.npz');assert set(a['y'].tolist())=={0,1};assert np.all(np.isfinite(a['probability'])) and np.all((a['probability']>=0)&(a['probability']<=1))
  for k,v in met(a['y'],a['probability'],d['threshold']).items():close(v,d[k])
  metricsets+=1
 for f in sorted(root.glob('source_*/seed_*/unit.json')):
  rel=str(f.parent.relative_to(root));u=read(str(f.relative_to(root)));units.append(u)
  for kind,key in [('actual','environment_pairs'),('pseudo','sampling_pairs')]:
   vectors=[]
   for g in range(u['environment_groups']):
    name=rel+f'/importance/{kind}_{g:03d}';a=arrays(name+'.npz');d=read(name+'.json');assert a['drops'].shape==(6,ident['config']['repeats']);assert np.all(np.isfinite(a['drops']));np.testing.assert_allclose(a['drops'].mean(axis=1),d['importance_mean'],atol=1e-12);vectors.append(a['drops'].mean(axis=1));group_arrays+=1
   pairs=[(i,j,rho(vectors[i],vectors[j])) for i,j in itertools.combinations(range(len(vectors)),2)];saved=u[key];assert saved['total_pairs']==len(pairs);assert saved['finite_pairs']==sum(r is not None for i,j,r in pairs)
   for (i,j,r),p in zip(pairs,saved['pairs']):assert (p['a'],p['b'])==(i,j);close(r,p['rho']);pairsets+=1
   inst=1-np.mean([r for i,j,r in pairs]) if pairs and all(r is not None for i,j,r in pairs) else None;close(inst,saved['instability']);close(inst,u['environment_instability' if kind=='actual' else 'sampling_instability'])
  e,s=u['environment_instability'],u['sampling_instability'];gap=e-s if e is not None and s is not None else None;close(gap,u['gap']);assert u['warning_available']==(gap is not None)
  src=read(rel+f'/test_{u["source"]:02d}/metrics.json');close(src['ba'],u['source_ba']);close(1-src['ba'],u['source_error'])
  for v in u['target_losses']:
   target=read(rel+f'/test_{v["target"]:02d}/metrics.json');close(target['ba'],v['target_ba']);close(src['ba']-target['ba'],v['loss'])
  close(np.mean([v['loss'] for v in u['target_losses']]),u['mean_loss'])
 fits=0;predictions=0
 if summary['status']=='CTU13_WARNING_PILOT_COMPLETE':
  c=read('context.json');assert len(units)==len(c['sources'])*len(c['seeds']);assert metricsets==len(units)*len(c['targets']);pred=read('predictive_check.json');lookup={(u['source'],u['seed']):u for u in units};common={k for k,u in lookup.items() if u['warning_available']}
  for f in pred['fits']:
   held=f['held_sources'];expectedtrain=[];expectedtest=[]
   for k in common:
    u=lookup[k];ls=[v for v in u['target_losses'] if u['source'] in held or v['target'] not in held]
    if ls:(expectedtest if u['source'] in held else expectedtrain).append(k)
   train=f['train_units'];assert {(t['source'],t['seed']) for t in train}==set(expectedtrain);assert {tuple(k) for k in f['test_unit_ids']}==set(expectedtest)
   for t in train:
    u=lookup[(t['source'],t['seed'])];ls=[v for v in u['target_losses'] if v['target'] not in held];assert not set(t['targets'])&set(held);assert sorted(t['targets'])==sorted(v['target'] for v in ls);close(t['outcome'],np.mean([v['loss'] for v in ls]));assert t['source'] not in held
   yy=np.array([t['outcome'] for t in train]);cols=f['fit']['features']
   pp=[p for p in pred['predictions'] if p['scheme']==f['scheme'] and p['specification']==f['specification'] and p['held_sources']==','.join(map(str,held))];assert len(pp)==len(expectedtest)
   if cols:
    xx=np.array([[t[n] for n in cols] for t in train]);mu=xx.mean(axis=0);scale=xx.std(axis=0);scale[scale==0]=1;z=(xx-mu)/scale;coef=np.linalg.solve(z.T@z+f['fit']['alpha']*np.eye(len(cols)),z.T@(yy-yy.mean()));inter=float(yy.mean())
    np.testing.assert_allclose(coef,f['fit']['coef'],atol=1e-10);np.testing.assert_allclose(mu,f['fit']['scaler_mean'],atol=1e-12);np.testing.assert_allclose(scale,f['fit']['scaler_scale'],atol=1e-12)
   for p in pp:
    u=lookup[(p['source'],p['seed'])];close(p['observed_loss'],u['mean_loss']);prediction=float(((np.array([u[n] for n in cols])-mu)/scale)@coef+inter) if cols else float(yy.mean());close(prediction,p['predicted_loss']);close(abs(prediction-u['mean_loss']),p['absolute_error']);close((prediction-u['mean_loss'])**2,p['squared_error']);predictions+=1
   fits+=1
  for r in pred['per_source']:
   a=[p for p in pred['predictions'] if p['scheme']==r['scheme'] and p['specification']==r['specification'] and p['source']==r['source']];assert len(a)==r['units'];close(np.mean([p['absolute_error'] for p in a]),r['mae']);close(np.sqrt(np.mean([p['squared_error'] for p in a])),r['rmse'])
  for r in pred['summary']:
   a=[p for p in pred['per_source'] if p['scheme']==r['scheme'] and p['specification']==r['specification']];close(np.mean([p['mae'] for p in a]),r['macro_mae']);close(np.mean([p['rmse'] for p in a]),r['macro_rmse']);base=next(v['macro_mae'] for v in pred['summary'] if v['scheme']==r['scheme'] and v['specification']=='intercept_only');close(base-r['macro_mae'],r['mae_improvement_vs_intercept'])
 return {'status':'PASS','report_status':summary['status'],'file_hashes_checked':len(h),'checkpoint_details_checked':checkpoints,'checkpoint_output_refs_checked':refs,'source_units_checked':len(units),'calibration_thresholds_checked':calibrations,'evaluation_metric_sets_checked':metricsets,'importance_group_arrays_checked':group_arrays,'rank_pairs_checked':pairsets,'strict_regression_fits_recomputed':fits,'strict_predictions_recomputed':predictions,'raw_or_input_feature_predictions_independently_replayed':False}
if __name__=='__main__':
 result=verify(sys.argv[1]);print(json.dumps(result,indent=2))
 if len(sys.argv)>2:pathlib.Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')

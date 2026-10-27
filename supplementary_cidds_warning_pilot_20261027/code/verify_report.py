"""Independent saved-result arithmetic verifier. Does not replay omitted raw partitions or refit models."""
import json,io,zipfile,hashlib,math,sys,itertools
import numpy as np

def close(a,b):
 if a is None or b is None:assert a is b
 else:assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-11),(a,b)
def counts(y,p,t):
 pred=p>=t;return [int(np.sum((y==0)&~pred)),int(np.sum((y==0)&pred)),int(np.sum((y==1)&~pred)),int(np.sum((y==1)&pred))]
def ba(c):
 tn,fp,fn,tp=map(int,c);assert min(tn,fp,fn,tp)>=0 and tn+fp>0 and fn+tp>0
 return .5*(tn/(tn+fp)+tp/(tp+fn))
def arrays(z,n):
 with np.load(io.BytesIO(z.read(n)),allow_pickle=False) as a:return {k:a[k] for k in a.files}
def ranks(v):
 order=sorted(range(len(v)),key=lambda i:int(v[i]));r=[0.0]*len(v);start=0
 while start<len(v):
  end=start+1
  while end<len(v) and int(v[order[end]])==int(v[order[start]]):end+=1
  for k in order[start:end]:r[k]=(start+1+end)/2
  start=end
 return r
def rho(a,b):
 a=ranks(a);b=ranks(b);ma=sum(a)/len(a);mb=sum(b)/len(b);va=sum((v-ma)**2 for v in a);vb=sum((v-mb)**2 for v in b)
 return None if not va or not vb else sum((x-ma)*(y-mb) for x,y in zip(a,b))/math.sqrt(va*vb)
def verify(path):
 with zipfile.ZipFile(path) as z:
  manifest=json.loads(z.read('SHA256SUMS.json'));assert len(manifest)==len(set(manifest))
  for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
  ident=json.loads(z.read('identity.json'))
  for n,k in [('engine.py','engine_sha256'),('partition.py','partition_code_sha256'),('projection.py','projection_sha256'),('PLAN.md','plan_sha256'),('verify_report.py','verifier_sha256'),('partition_report.zip','partition_report_sha256')]:assert hashlib.sha256(z.read(n)).hexdigest()==ident[k]
  nested=z.read('partition_report.zip');nested_counts=[]
  for next_name in ['qualification_report.zip','correction_report.zip','parent_report.zip',None]:
   with zipfile.ZipFile(io.BytesIO(nested)) as q:
    mm=json.loads(q.read('SHA256SUMS.json'))
    for n,h in mm.items():assert hashlib.sha256(q.read(n)).hexdigest()==h
    if next_name=='qualification_report.zip':
     partition_records=json.loads(q.read('partition_records.json'));partition_refs=json.loads(q.read('output_references.json'))
    nested_counts.append(len(mm));nested=q.read(next_name) if next_name else None
  evaluations=json.loads(z.read('evaluations.json'));warnings=json.loads(z.read('warnings.json'));summary=json.loads(z.read('summary.json'));group_map={};model_map={};eval_map={};checkpoint_count=0;calibration_checks=0;group_checks=0
  for n in manifest:
   if not n.endswith('/complete.json'):continue
   cp=json.loads(z.read(n));checkpoint_count+=1;assert cp['identity']==ident;prefix=n[:-len('complete.json')];meta=cp['meta']
   for f,h in cp['files'].items():assert hashlib.sha256(z.read(prefix+f)).hexdigest()==h
   if 'model.npz' in cp['files']:
    model=arrays(z,prefix+'model.npz');cal=arrays(z,prefix+'calibration.npz');info=json.loads(z.read(prefix+'fit.json'));t=float(model['threshold']);pp=cal['probability'];yy=cal['y'];assert np.isfinite(pp).all() and ((pp>=0)&(pp<=1)).all() and np.isfinite(t);assert model['coef'].shape==(4,) and model['mean'].shape==(4,) and model['scale'].shape==(4,) and (model['scale']>0).all()
    candidates=np.r_[np.unique(pp),np.nextafter(float(pp.max()),np.inf)];scores=[ba(counts(yy,pp,v)) for v in candidates];best=max(scores);opt=max(float(v) for v,s in zip(candidates,scores) if abs(s-best)<=1e-12);close(opt,t);close(info['threshold'],t);close(info['calibration_best_balanced_accuracy'],best);model_map[(meta['source'],meta['seed'])]=(t,cp['files']['model.npz']);calibration_checks+=1
    expected=next(r for r in partition_records if r['capture']==meta['source'] and r['seed']==meta['seed']);assert info['train_class_counts']==expected['role_counts_per_class']['train'];assert info['train_rows']==sum(info['train_class_counts']);assert np.bincount(yy,minlength=2).tolist()==expected['role_counts_per_class']['calibration'];assert meta['source_partition_sha256']==expected['partition']['sha256']
   elif 'importance.npz' in cp['files']:
    a=arrays(z,prefix+'importance.npz');base=a['baseline_counts'];pc=a['permuted_counts'];assert base.shape==(4,) and pc.shape==(5,4,4);n0=int(base[0]+base[1]);n1=int(base[2]+base[3]);assert n0>=50 and n1>=50
    num=[]
    for f in range(4):
     total=0
     for repeat in range(5):
      cc=pc[repeat,f];ba(cc);assert int(cc[0]+cc[1])==n0 and int(cc[2]+cc[3])==n1;total+=(int(base[3])-int(cc[3]))*n0+(int(base[0])-int(cc[0]))*n1
     num.append(total)
    assert a['importance_numerator'].tolist()==num;assert int(a['importance_denominator'])==2*n0*n1*5
    assert np.allclose(a['importance_mean'],np.asarray(num)/(2*n0*n1*5),rtol=0,atol=1e-14);group_map[(meta['source'],meta['seed'],meta['kind'],meta['group'])]=num;group_checks+=1
   elif 'predictions.npz' in cp['files']:
    a=arrays(z,prefix+'predictions.npz');mm=json.loads(z.read(prefix+'metrics.json'));pp=a['probability'];yy=a['y'];assert np.isfinite(pp).all() and ((pp>=0)&(pp<=1)).all() and set(yy.tolist())<={0,1};cc=counts(yy,pp,mm['threshold']);assert [mm[k] for k in ['tn','fp','fn','tp']]==cc;close(ba(cc),mm['balanced_accuracy']);close(mm['false_positive_rate'],cc[1]/(cc[0]+cc[1]));close(mm['attack_recall'],cc[3]/(cc[2]+cc[3]));assert mm['rows']==len(yy);eval_map[(meta['source'],meta['seed'],meta['target'])]=(meta,mm)
  for r in evaluations:
   expected=next(t for t in partition_records if t['capture']==r['target'] and t['seed']==r['seed']);assert [r['tn']+r['fp'],r['fn']+r['tp']]==expected['role_counts_per_class']['test'];assert r['target_partition_sha256']==expected['partition']['sha256']
   meta,mm=eval_map[(r['source'],r['seed'],r['target'])]
   for k,v in mm.items():assert r[k]==v
   assert r['is_source_test']==(r['source']==r['target']);close(r['threshold'],model_map[(r['source'],r['seed'])][0]);assert r['model_sha256']==model_map[(r['source'],r['seed'])][1]
  pair_checks=0
  for w in warnings:
   for kind,field in [('environment','environment'),('pseudo','pseudo')]:
    keys=sorted(k for k in group_map if k[:3]==(w['source'],w['seed'],kind));v=[group_map[k] for k in keys];expected=[(i,j,rho(v[i],v[j])) for i,j in itertools.combinations(range(len(v)),2)];s=w[field];assert s['groups']==len(v) and s['pairs']==len(expected) and s['undefined_pairs']==sum(r is None for i,j,r in expected);assert len(s['pairwise'])==len(expected)
    for (i,j,r),p in zip(expected,s['pairwise']):assert p['left_group']==i and p['right_group']==j;close(p['rho'],r);pair_checks+=1
    mean=sum(r for i,j,r in expected)/len(expected) if expected and all(r is not None for i,j,r in expected) else None;close(s['mean_spearman'],mean);close(s['instability'],None if mean is None else 1-mean)
   env=w['environment']['instability'];sam=w['pseudo']['instability'];close(w['environmental_instability'],env);close(w['sampling_instability'],sam);close(w['sampling_environment_gap'],None if env is None or sam is None else env-sam)
  own=[r for r in evaluations if r['is_source_test']];external=[r for r in evaluations if not r['is_source_test']];losses=[next(s['balanced_accuracy'] for s in own if s['source']==r['source'] and s['seed']==r['seed'])-r['balanced_accuracy'] for r in external]
  if summary['status']=='CIDDS_WARNING_PILOT_COMPLETE':
   assert len(model_map)==10 and len(own)==10 and len(external)==30 and len(warnings)==10;assert len({(r['source'],r['target']) for r in external})==6
  assert summary['source_fits_completed']==len(model_map);assert summary['source_tests']==len(own);assert summary['external_evaluations']==len(external);assert summary['warning_records']==len(warnings)
  if own:close(summary['mean_source_ba'],np.mean([r['balanced_accuracy'] for r in own]))
  if external:close(summary['mean_target_ba'],np.mean([r['balanced_accuracy'] for r in external]));close(summary['mean_loss'],np.mean(losses))
  assert summary['degraded_external_jobs']==sum(v>0 for v in losses);assert summary['improved_external_jobs']==sum(v<0 for v in losses);assert summary['tied_external_jobs']==sum(v==0 for v in losses)
  for d in json.loads(z.read('directional_summary.json')):
   rr=[r for r in external if r['source']==d['source'] and r['target']==d['target']];assert len(rr)==d['seeds'];close(d['mean_target_ba'],np.mean([r['balanced_accuracy'] for r in rr]));close(d['mean_loss'],np.mean([next(s['balanced_accuracy'] for s in own if s['source']==r['source'] and s['seed']==r['seed'])-r['balanced_accuracy'] for r in rr]))
  for d in json.loads(z.read('source_capture_summary.json')):
   ss=[r for r in own if r['source']==d['source']];tt=[r for r in external if r['source']==d['source']];ww=[w for w in warnings if w['source']==d['source']];assert d['source_tests']==len(ss) and d['external_evaluations']==len(tt)
   if ss:close(d['mean_source_ba'],np.mean([r['balanced_accuracy'] for r in ss]))
   if tt:close(d['mean_target_ba'],np.mean([r['balanced_accuracy'] for r in tt]));close(d['mean_loss'],np.mean([next(s['balanced_accuracy'] for s in ss if s['seed']==r['seed'])-r['balanced_accuracy'] for r in tt]))
   for k in ['environmental_instability','sampling_instability','sampling_environment_gap']:
    vv=[w[k] for w in ww];close(d['mean_'+k],np.mean(vv) if vv and all(v is not None for v in vv) else None)
  for k in ['predictive_warning_utility_validation_performed','independent_site_warning_validation_completed','measured_production_prevalence_established','fine_family_equivalence_certified']:assert summary[k] is False
  return {'status':'PASS','pilot_status':summary['status'],'file_hashes_verified':len(manifest),'nested_report_hash_counts':nested_counts,'checkpoints_verified':checkpoint_count,'calibration_optima_recomputed':calibration_checks,'metric_sets_recomputed':len(eval_map),'importance_groups_recomputed':group_checks,'rank_pairs_recomputed':pair_checks,'limits':'Saved-result arithmetic and hash continuity only; raw partitions, model fitting and feature permutations are not independently replayed.'}
if __name__=='__main__':
 try:print(json.dumps(verify(sys.argv[1]),indent=2))
 except Exception as ex:print('VERIFICATION FAILED:',type(ex).__name__,str(ex));raise

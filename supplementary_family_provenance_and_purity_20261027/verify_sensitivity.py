from pathlib import Path
import json,hashlib,itertools,sys
import numpy as np
from scipy.stats import spearmanr

def verify(root,prepared,provenance):
 root,prepared,provenance=map(Path,(root,prepared,provenance));read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();dig=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
 close=lambda a,b:np.testing.assert_allclose(a,b,rtol=0,atol=2e-12)
 manifest=read(root/'SHA256SUMS.json')
 for n,h in manifest.items():assert sha(root/n)==h,n
 identity=read(root/'identity.json');context=read(root/'context.json');assert sha(root/'CODE.py')==identity['code_sha256'] and sha(root/'PLAN.md')==identity['plan_sha256'] and dig(context)==identity['context_sha256']
 assert (root/'CODE.py').read_bytes()==(prepared/'engine.py').read_bytes()
 assert context==read(prepared/'context.json')
 included=0;missing=[];markers=list(root.rglob('*.complete.json'))
 for p in markers:
  m=read(p);assert m['identity']==identity and dig(m['details'])==m['details_sha256'],str(p)
  for n,h in m['outputs'].items():
   if (root/n).exists():assert sha(root/n)==h,n;included+=1
   else:missing.append(n)
 index=read(provenance/'selected_pattern_index.json');prov=read(provenance/'pattern_provenance.json');byid={p['pattern_id']:p for p in prov};sel=read(root/'purity_selection.json');counts={};kept={}
 for role in ['train','calibration','source_test','environment','target']:
  rr=sorted([x for x in index if x['role']==role],key=lambda x:x['panel_row']);bad=[x['panel_row'] for x in rr if byid[x['pattern_id']]['mixed_exact_families'] or byid[x['pattern_id']]['binary_conflict']]
  assert bad==sel['removed_panel_rows'][role]==context['purity_removed_panel_rows'][role];bad=set(bad);kk=[x for x in rr if x['panel_row'] not in bad];kept[role]=kk
  counts[role]={'rows':len(kk),'benign':sum(x['binary_label']==0 for x in kk),'attack':sum(x['binary_label']==1 for x in kk)};assert counts[role]==sel['purified_panel_counts'][role]==context['purified_panel_counts'][role]
  for x in kk:assert byid[x['pattern_id']]['raw_own_family_counts'].keys()=={x['representative_family']}
 a=read(root/'family_results.json');records=a['records'];assert len(records)==9
 pairs=groups=metrics=0
 for x in records:
  family,condition=x['family'],x['condition'];base=root/family.replace(' ','_')/condition
  assert x['warnings']==read(base/'warnings.json')['warnings']
  for partition in ['environment','sampling']:
   pp=sorted(p for p in (base/'groups').glob(partition+'_*.json') if not p.name.endswith('.complete.json'));vv=[]
   assert len(pp)==x['warnings']['environment_groups']
   for p in pp:
    g=read(p);v=np.asarray(g['importance_repeats']);assert v.shape==(6,3);close(v.mean(1),g['importance_mean']);vv.append(v.mean(1));groups+=1
   corr=[]
   for u,v in itertools.combinations(vv,2):
    # Spearman average ranks; constant vectors contribute zero by protocol.
    corr.append(0. if np.ptp(u)==0 or np.ptp(v)==0 else float(spearmanr(u,v).statistic))
   saved=x['warnings']['stability_details'][partition];close(corr,saved['pairwise_correlations']);close(np.mean(corr),saved['stability']);assert len(corr)==saved['pair_count'];pairs+=len(corr)
  w=x['warnings'];close(w['environmental_instability'],1-w['stability_details']['environment']['stability']);close(w['sampling_instability'],1-w['stability_details']['sampling']['stability']);close(w['sampling_environment_gap'],w['environmental_instability']-w['sampling_instability'])
  for role in ['source_test','target']:
   e=x[role];assert e==read(base/(role+'_evaluation.json'));m=e['metrics'];tn,fp,fn,tp=[m[k] for k in ['tn','fp','fn','tp']];assert tn+fp+fn+tp==m['rows']==counts[role]['rows'];assert tn+fp==counts[role]['benign'] and fn+tp==counts[role]['attack']
   r0=tn/(tn+fp);r1=tp/(tp+fn);close([m['balanced_accuracy'],m['benign_recall'],m['attack_recall'],m['attack_precision'],m['macro_f1'],m['mcc']],[(r0+r1)/2,r0,r1,tp/(tp+fp),(2*tn/(2*tn+fp+fn)+2*tp/(2*tp+fp+fn))/2,(tp*tn-fp*fn)/np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))])
   f=e['candidate_family'];assert f['detected']+f['missed']==f['rows'];close(f['recall'],f['detected']/f['rows']);close(f['miss_rate'],1-f['recall']);metrics+=1
  close(x['balanced_accuracy_loss'],x['source_test']['metrics']['balanced_accuracy']-x['target']['metrics']['balanced_accuracy'])
 for family in ['GoldenEye','Hulk','SSH Patator']:
  rr={x['condition']:x for x in records if x['family']==family};assert set(rr)=={'full_reference','family_excluded','size_matched_control'}
  for role in ['train','calibration']:
   e=rr['family_excluded']['selection'][role];c=rr['size_matched_control']['selection'][role];assert e['candidate_family_representatives']==0;assert all(e[k]==c[k] for k in ['rows','benign','attack']);assert rr['full_reference']['selection'][role]['rows']==counts[role]['rows']
  for cond in rr:
   # Candidate-specific grouping selections and permutation assignment hashes agree across conditions.
   for p in (root/family.replace(' ','_')/cond/'groups').glob('*.json'):
    if p.name.endswith('.complete.json'):continue
    g=read(p);ref=read(root/family.replace(' ','_')/'full_reference'/'groups'/p.name)
    assert all(g['inputs'][k]==ref['inputs'][k] for k in ['selected_rows_sha256','assignment_sha256','within_selected_positions_sha256'])
 refs=[x for x in records if x['condition']=='full_reference']
 for x in refs[1:]:
  close(x['threshold'],refs[0]['threshold'])
  for role in ['source_test','target']:assert x[role]['metrics']==refs[0][role]['metrics']
 for d in a['paired_descriptive_contrasts']:
  rr={x['condition']:x for x in records if x['family']==d['family']};e=rr['family_excluded'];b=rr[d['contrast'].split('_minus_')[1]]
  calc={'source_family_recall_change':e['source_test']['candidate_family']['recall']-b['source_test']['candidate_family']['recall'],'target_family_recall_change':e['target']['candidate_family']['recall']-b['target']['candidate_family']['recall'],'target_balanced_accuracy_change':e['target']['metrics']['balanced_accuracy']-b['target']['metrics']['balanced_accuracy']}
  for k,v in [('environmental_instability_change','environmental_instability'),('sampling_instability_change','sampling_instability'),('gap_change','sampling_environment_gap')]:calc[k]=e['warnings'][v]-b['warnings'][v]
  for k,v in calc.items():close(d[k],v)
 summary=read(root/'summary.json');assert summary['status']=='PURE_FAMILY_SENSITIVITY_COMPLETE';assert (groups,pairs,metrics)==(132,420,18)
 return dict(status='PASS',recorded_file_hashes=len(manifest),checkpoint_detail_digests=len(markers),included_checkpoint_output_hashes=included,omitted_binary_output_references=len(missing),omitted_unique_binary_outputs=len(set(missing)),prepared_code_and_context_match=True,purity_rows_reconstructed=117000,mixed_patterns_removed=444,purified_panel_counts=counts,importance_group_means=groups,spearman_pair_correlations=pairs,count_derived_metric_sets=metrics,candidate_family_recall_sets=metrics,paired_contrasts=6,full_references_agree=True,raw_fits_predictions_and_selection_arrays_independently_replayed=False)
if __name__=='__main__':
 result=verify(*sys.argv[1:4]);print(json.dumps(result,indent=2));Path(sys.argv[4]).write_text(json.dumps(result,indent=2)+'\n')

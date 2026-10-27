from pathlib import Path
from collections import defaultdict,Counter
import json,hashlib,sys

def verify(p):
 p=Path(p);read=lambda n:json.loads((p/n).read_text());sha=lambda n:hashlib.sha256((p/n).read_bytes()).hexdigest();digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
 manifest=read('SHA256SUMS.json');assert all(sha(n)==v for n,v in manifest.items())
 identity=read('identity.json');context=read('context.json');assert identity['context_sha256']==digest(context) and identity['code_sha256']==sha('CODE.py') and identity['plan_sha256']==sha('PLAN.md')
 markers=list(p.glob('*.complete.json'));output_checks=0
 for path in markers:
  d=json.loads(path.read_text());assert d['identity']==identity and digest(d['details'])==d['details_sha256']
  for n,v in d['outputs'].items():assert sha(n)==v;output_checks+=1
 index=read('selected_pattern_index.json');patterns=read('pattern_provenance.json');saved=read('family_provenance_summary.json');assert len(index)==len(patterns)==117000
 assert len({m['signature_sha256'] for m in index})==117000
 for i,m in enumerate(index):assert m['pattern_id']==i
 sides={s:defaultdict(Counter) for s in ['source','target']};first={s:{} for s in sides};hits={s:set() for s in sides};inputrows={s:0 for s in sides}
 for side,n in [('source',10),('target',8)]:
  for fi in range(n):
   d=read(f'{side}_{fi:02d}.json');assert d['file_index']==fi;cnt=d['counts'];assert cnt['retained_rows']==cnt['retained_benign']+cnt['retained_attack'];assert cnt['input_rows']==cnt['retained_rows']+cnt['excluded_rows'];inputrows[side]+=cnt['input_rows']
   if side=='source':
    c=context['raw_sources'][fi];assert d['input_sha256']==c['sha256'] and d['input_size_bytes']==c['size_bytes'];assert cnt['input_rows']==c['expected_rows'] and cnt['retained_benign']==c['expected_valid_benign'] and cnt['retained_attack']==c['expected_valid_attack']
   else:assert d['member']==context['target_members'][fi] and d['archive_sha256']==context['target_archive_sha256'] and cnt==context['target_counts'][fi]
   expected={m['pattern_id'] for m in index if (m['role']=='target')==(side=='target') and m['file_index']==fi};assert set(d['representative_row_hits'])==expected;assert not hits[side]&expected;hits[side]|=expected
   for rec in d['occurrences']:
    pid=rec['pattern_id'];label=rec['family'];assert 0<=pid<len(index) and rec['occurrences']>0;sides[side][pid][label]+=rec['occurrences'];where=(fi,rec['first_row_index']);first[side][pid]=min(first[side].get(pid,where),where)
 for m,r in zip(index,patterns):
  pid=m['pattern_id'];side='target' if m['role']=='target' else 'source';other='source' if side=='target' else 'target';own=sides[side][pid]
  assert r['pattern_id']==pid and r['role']==m['role'] and r['representative_family']==m['representative_family'];assert dict(own)==r['raw_own_family_counts'] and dict(sides[other][pid])==r['cross_capture_family_counts'];assert first[side][pid]==(m['file_index'],m['row_index']);assert own[m['representative_family']]>0
  assert r['mixed_exact_families']==(len(own)>1);binary={0 if k==('BENIGN' if side=='target' else 'Benign') else 1 for k in own};assert r['binary_conflict']==(len(binary)>1)
 for role,rec in saved['by_role'].items():
  rr=[r for r in patterns if r['role']==role];calc={'selected_signatures':len(rr),'mixed_exact_family_signatures':sum(r['mixed_exact_families'] for r in rr),'binary_conflict_signatures':sum(r['binary_conflict'] for r in rr),'raw_own_occurrences':sum(sum(r['raw_own_family_counts'].values()) for r in rr),'signatures_also_present_in_other_capture':sum(bool(r['cross_capture_family_counts']) for r in rr)};assert calc==rec
 normal=lambda s:''.join(c for c in s.casefold() if c.isalnum())
 for c in saved['candidate_families']:
  definition=next(x for x in context['selected_candidates'] if x['candidate_tool_label_family']==c['candidate']);totals=[0,0]
  for rr in c['by_role']:
   role=rr['role'];aliases={normal(s) for s in definition['aliases']['target' if role=='target' else 'source']};rows=[r for r in patterns if r['role']==role];declared=[r for r in rows if normal(r['representative_family']) in aliases];rest=[r for r in rows if normal(r['representative_family']) not in aliases];n=lambda r:sum(v for k,v in r['raw_own_family_counts'].items() if normal(k) in aliases)
   calc={'role':role,'declared_family_signatures':len(declared),'declared_family_signatures_with_other_family_labels':sum(any(normal(k) not in aliases for k in r['raw_own_family_counts']) for r in declared),'raw_candidate_occurrences_in_declared_signatures':sum(n(r) for r in declared),'raw_other_family_occurrences_in_declared_signatures':sum(sum(r['raw_own_family_counts'].values())-n(r) for r in declared),'retained_other_representative_signatures_with_candidate_occurrences':sum(n(r)>0 for r in rest),'raw_candidate_occurrences_hidden_in_retained_signatures':sum(n(r) for r in rest)};assert calc==rr
   if role in ['train','calibration']:totals[0]+=calc['retained_other_representative_signatures_with_candidate_occurrences'];totals[1]+=calc['raw_candidate_occurrences_hidden_in_retained_signatures']
  assert totals==[c['retained_training_calibration_signatures_with_raw_candidate_occurrences'],c['raw_candidate_occurrences_in_retained_training_calibration_patterns']]
 return {'status':'PASS','recorded_file_hashes':len(manifest),'checkpoint_details':len(markers),'checkpoint_output_hash_checks':output_checks,'selected_patterns_reaggregated':len(patterns),'representative_hit_records_checked':sum(map(len,hits.values())),'raw_file_member_records_checked':18,'source_raw_rows':inputrows['source'],'target_raw_rows':inputrows['target'],'candidate_role_summaries_recomputed':15,'original_raw_files_not_available_for_independent_replay_here':True}
if __name__=='__main__':
 result=verify(sys.argv[1] if len(sys.argv)>1 else Path(__file__).parent);print(json.dumps(result,indent=2));(Path(sys.argv[1])/'LOCAL_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n') if len(sys.argv)>1 else None

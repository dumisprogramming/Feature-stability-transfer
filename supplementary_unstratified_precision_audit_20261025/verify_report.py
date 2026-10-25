"""Check the downloaded audit report without accessing Drive or fitting models."""
from pathlib import Path
import json,hashlib,csv,itertools
b=Path(__file__).resolve().parent/'report'
h=json.loads((b/'report_hashes.json').read_text())
assert len(h)==10
for n,v in h.items():assert hashlib.sha256((b/n).read_bytes()).hexdigest()==v,n
s=json.loads((b/'verification_summary.json').read_text())
assert s['status']=='AUDIT_COMPLETED' and s['checked_panels']==60
rows=list(csv.DictReader((b/'panel_audit_60.csv').open()))
datasets=['NF_UNSW_NB15_v3','NF_BoT_IoT_v3','NF_CSE_CIC_IDS2018_v3','NF_ToN_IoT_v3']
grid={(a,t,str(seed)) for a,t,seed in itertools.product(datasets,datasets,range(2027,2032)) if a!=t}
assert len(rows)==60 and {(r['source_dataset'],r['target_dataset'],r['seed']) for r in rows}==grid
for r in rows:
 assert int(r['panel_rows'])==100000
 assert int(r['source_float32_signature_overlap_rows'])==0
 assert int(r['corrected_panel_replacement_rows'])==0
 assert r['original_panel_row_ids_sha256']==r['corrected_panel_row_ids_sha256']
 assert r['original_attack_fraction']==r['corrected_attack_fraction']
 assert r['corrected_pool_sufficient']==r['original_selection_reproduced']=='True'
 assert r['requires_panel_repair']==r['cross_dataset_row_ids_compared']=='False'
parts=list(csv.DictReader((b/'overlap_by_source_partition.csv').open()))
assert len(parts)==240 and all(int(r['matching_target_row_occurrences'])==0 for r in parts)
assert s['target_row_occurrences']==sum(int(r['panel_rows']) for r in rows)==6000000
for n in ['affected_panels','panels_with_source_overlap','source_overlapping_target_row_occurrences','panels_requiring_expanded_pool','new_model_fits','new_model_predictions','raw_archive_scans']:assert s[n]==0,n
assert s['all_original_float64_selections_reproduced'] is True
e=json.loads((b/'EXPECTED_CONTEXT.json').read_text());inputs=json.loads((b/'input_hashes.json').read_text())
stage='supplementary_natural_prevalence_20261024/run_v1/'
for n,v in e['source_pins'].items():assert inputs[n]==v,n
for n,v in e['job_hashes'].items():assert inputs[stage+n]==v,n
for panel in e['panels'].values():
 for n,v in panel['file_hashes'].items():
  if n.endswith(('features.parquet','labels.parquet')):assert inputs[stage+n]==v,n
print('PASS: 10 report hashes; 60 complete panels; 240 partition checks; zero overlap; unchanged corrected selections.')
print('Saved-report verification only: no raw-capture scan, target probability replay or model fitting.')

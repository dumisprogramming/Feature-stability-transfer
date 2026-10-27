"""Verify CTU archived report chain and saved warning arithmetic; no raw rerun."""
from pathlib import Path
import json,zipfile,hashlib,tempfile,csv,argparse
import numpy as np
from verify_warning_report import verify as verify_warning,rho,close

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def extract(zpath,dest):
 with zipfile.ZipFile(zpath) as z:
  for n in z.namelist():
   if Path(n).is_absolute() or '..' in Path(n).parts:raise ValueError('Unsafe archive path')
  z.extractall(dest)
def hashes(root):
 pins=json.loads((root/'SHA256SUMS.json').read_text())
 for n,h in pins.items():assert sha(root/n)==h,n
 details=included=omitted=0
 for f in root.rglob('*.complete.json'):
  m=json.loads(f.read_text())
  if 'identity' in m:assert m['identity']==json.loads((root/'identity.json').read_text())
  if 'details_sha256' in m:assert digest(m['details'])==m['details_sha256'];details+=1
  for n,h in m.get('outputs',{}).items():
   if (root/n).is_file():assert sha(root/n)==h;included+=1
   else:omitted+=1
 return {'file_hashes':len(pins),'checkpoint_details':details,'included_output_refs':included,'omitted_output_refs':omitted}
def verify(root):
 root=Path(root);manifest=json.loads((root/'SHA256SUMS.json').read_text())
 for n,h in manifest.items():assert sha(root/n)==h,n
 expected={'inventory':'ctu13_inventory_report_20261027_043600_767339.zip','qualification':'ctu13_partition_qualification_report_20261027_051144_540754.zip','pilot':'ctu13_warning_pilot_report_20261027_054252_013157.zip'};result={}
 with tempfile.TemporaryDirectory(prefix='paper3_s19_') as tmp:
  dirs={name:Path(tmp)/name for name in expected}
  for name,z in expected.items():extract(root/'archives'/z,dirs[name])
  inv,qual,pilot=[dirs[n] for n in ['inventory','qualification','pilot']];j=lambda p:json.loads(p.read_text())
  for name in ['inventory','qualification']:result[name]=hashes(dirs[name])
  assert j(inv/'summary.json')['status']=='CTU13_INVENTORY_COMPLETE' and j(qual/'summary.json')['status']=='CTU13_PARTITION_QUALIFICATION_COMPLETE'
  qcontext=j(qual/'context.json');assert qcontext['parent_pins']==j(inv/'SHA256SUMS.json');assert qcontext['parent_report_sha256']==sha(root/'archives'/expected['inventory'])
  context=j(pilot/'context.json');assert context['parent_pins']==j(qual/'SHA256SUMS.json');assert context['parent_report_sha256']==sha(root/'archives'/expected['qualification']);assert j(qual/'identity.json')==context['parent_identity'] and j(inv/'identity.json')==qcontext['parent_identity']
  rows=list(csv.DictReader((inv/'capture_inventory_summary.csv').open()));counts={k:sum(int(r[k]) for r in rows) for k in ['parsed_rows','benign','attack','unknown']};assert counts=={'parsed_rows':19976700,'benign':355078,'attack':444699,'unknown':19176923}
  records=[j(f) for f in qual.glob('scenario_*/seed_*/qualification.json')];assert len(records)==65;cfg=j(qual/'identity.json')['config'];sources=[];targets=[]
  for sid in range(1,14):
   ds=[d for d in records if d['scenario']==sid];assert len(ds)==5
   for d in ds:
    for role in ['train','calibration','test']:
     a=d['roles'][role];n=min(cfg[role+'_cap_per_class'],a['available_benign'],a['available_attack']);passed=n>=cfg[role+'_min_per_class'];assert a['support_pass']==passed and a['selected_per_class']==(n if passed else 0)
    g=sum(v['retained'] for v in d['environment_windows']);assert g==d['environment_groups'];assert d['environment_support_pass']==(g>=cfg['minimum_environment_windows']);assert d['target_support_pass']==d['roles']['test']['support_pass'];assert d['source_support_pass']==(d['initial_source_candidate'] and all(d['roles'][r]['support_pass'] for r in ['train','calibration','test']) and d['environment_support_pass'])
   if all(d['source_support_pass'] for d in ds):sources.append(sid)
   if all(d['target_support_pass'] for d in ds):targets.append(sid)
  assert sources==context['sources']==[1,2,6,9,13] and targets==context['targets']==[1,2,3,6,8,9,12,13];assert context['directions']==[[s,t] for s in sources for t in targets if s!=t]
  assert all(v==0 for r in j(qual/'cross_role_audit.json')['results'] for v in r['cross_role_intersections'].values())
  result['qualification']['records_reconstructed']=len(records);result['inventory']['label_counts_recomputed']=counts;result['pilot']=verify_warning(pilot)
  units=[j(f) for f in pilot.glob('source_*/seed_*/unit.json')];caps=list(csv.DictReader((pilot/'capture_summary.csv').open()))
  for row in caps:
   ds=[u for u in units if u['source']==int(row['source'])]
   for k,col in [('source_ba','mean_source_ba'),('mean_loss','mean_loss'),('sampling_instability','sampling_instability'),('environment_instability','environment_instability'),('gap','gap')]:close(float(row[col]),float(np.mean([u[k] for u in ds])))
  for row in csv.DictReader((pilot/'associations.csv').open()):
   ds=caps if not row['seed'] else [u for u in units if u['seed']==int(row['seed'])];close(float(row['rho']),rho([float(u[row['warning']]) for u in ds],[float(u['mean_loss']) for u in ds]))
  for name in ['source_units.csv','directed_losses.csv','capture_summary.csv','associations.csv','prediction_units.csv','prediction_per_source.csv','prediction_summary.csv']:assert (root/'results'/name).read_bytes()==(pilot/name).read_bytes()
  result['derived_tables_match_archived_report']=True
 result.update(status='PASS',raw_data_recovery_or_feature_predictions_replayed=False,input_array_intersections_independently_replayed=False);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--report',type=Path);a=p.parse_args();v=verify(a.root);print(json.dumps(v,indent=2))
 if a.report:a.report.write_text(json.dumps(v,indent=2)+'\n')

#!/usr/bin/env python3
"""Verify supplied artifact hashes without using historical absolute paths."""
from pathlib import Path
import argparse
import hashlib
import json
import platform

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def verify(root=ROOT):
    root=Path(root)
    checks=[]
    def check(name, ok, detail=''):
        checks.append({'check':name,'pass':bool(ok),'detail':detail})
    prot=root/'Paper3_Protocol'
    for name in ['paper3_protocol_frozen_v1','amendment_A1_duplicate_group_disjoint','amendment_A2_target_evaluation_plan']:
        expected=(prot/(name+'.sha256')).read_text().strip().split()[0]
        check(name,sha(prot/(name+'.json'))==expected,'Raw file SHA-256 compared with supplied sidecar')
    a3=json.loads((prot/'amendment_A3_confirmatory_analysis_plan.json').read_text())
    check('A3 internal seal',canonical({k:v for k,v in a3.items() if k not in {'status','seal_sha256'}})==a3['seal_sha256'])
    final=root/'Paper3_Final_Evaluation'
    seal=json.loads((final/'final_results_seal.json').read_text())
    check('Final-results internal seal',canonical({k:v for k,v in seal.items() if k not in {'status','seal_sha256'}})==seal['seal_sha256'])
    for name,item in seal['artifacts'].items():
        p=final/Path(item['path']).name
        check('Result '+p.name,p.is_file() and sha(p)==item['sha256'])
    stability=root/'Paper3_Source_Stability'
    man=stability/'source_only_stability_seal_manifest.json'
    expected=(stability/'source_only_stability_seal_manifest.sha256').read_text().strip()
    check('Source-stability manifest seal',sha(man)==expected==seal['source_stability_seal_sha256'])
    for item in json.loads(man.read_text())['artifacts']:
        p=stability/item['relative_path']
        check('Stability '+item['relative_path'],p.is_file() and sha(p)==item['sha256'])
    for name,key in [('paper3_final_target_evaluation.py','final_evaluation_script_sha256')]:
        check('Original script '+name,sha(root/'Paper3_Code'/name)==seal[key])
    pred=root/'Paper3_Code/paper3_label_blind_target_predictions.py'
    check('Original prediction script',sha(pred)=='bdf16dab915ea5994d9917b89f6902fcf20905ee8a07e5991eeda45a300265cb')
    check('A3 parent A2 link',a3['parent_amendment_A2_sha256']==seal['parent_amendment_A2_sha256'])
    check('A3 prediction-seal link',a3['pre_label_target_prediction_seal_sha256']==seal['pre_label_target_prediction_seal_sha256'])
    return {'status':'PASS' if all(c['pass'] for c in checks) else 'FAIL',
            'checks_passed':sum(c['pass'] for c in checks),'checks_total':len(checks),
            'scope':'Checks byte integrity and supplied links only; does not independently establish chronology or verify absent raw predictions, labels, models or upstream code.',
            'checks':checks,'python':platform.python_version()}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path)
    args=p.parse_args();result=verify()
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(f"{result['status']}: {result['checks_passed']}/{result['checks_total']} artifact checks")
    raise SystemExit(0 if result['status']=='PASS' else 1)

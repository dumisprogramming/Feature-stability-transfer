#!/usr/bin/env python3
"""Replay the supplied result-level analyses without raw data or label access.

Default: full original 10,000-replicate bootstrap and 13,824 permutations.
--skip-bootstrap: faster check of exact p-values, rho, aggregates and LOSO.
"""
from pathlib import Path
import argparse
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import analysis_core as core
from verify_artifacts import verify

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/reanalysis')
    p.add_argument('--skip-bootstrap',action='store_true')
    args=p.parse_args();out=args.output_dir.resolve()
    if out==ROOT or ROOT/'Paper3_Final_Evaluation'==out or ROOT/'Paper3_Final_Evaluation' in out.parents:
        raise ValueError('Use a separate output directory, not a sealed input folder')
    out.mkdir(parents=True,exist_ok=True)
    integrity=verify();assert integrity['status']=='PASS','Artifact verification failed'
    checks=[]
    def check(name,ok,detail=''):checks.append({'check':name,'pass':bool(ok),'detail':detail})
    def read(name):return pd.read_csv(ROOT/'Paper3_Final_Evaluation'/name,float_precision='round_trip')
    def compare(name,observed,expected,keys,atol=1e-10):
        a=observed.sort_values(keys).reset_index(drop=True)
        b=expected.sort_values(keys).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(a[b.columns],b,check_dtype=False,check_exact=False,rtol=0,atol=atol)
            check(name,True)
        except AssertionError as e:check(name,False,str(e)[:1500])
    transfer=read('transfer_analysis_all_180.csv')
    check('180 unique job keys',len(transfer)==180 and not transfer.duplicated(['source_dataset','target_dataset','seed','model']).any())
    check('All jobs degraded',bool((transfer.balanced_accuracy_loss>0).all()))
    check('BA loss identity',np.allclose(transfer.source_test_balanced_accuracy-transfer.target_balanced_accuracy,transfer.balanced_accuracy_loss,rtol=0,atol=1e-12))
    u60,u12=core.build_primary_units(transfer)
    compare('Rebuilt 60 source-seed-model units',u60,read('source_seed_model_primary_units_60.csv'),['source_dataset','seed','model'])
    compare('Rebuilt 12 confirmatory units',u12,read('source_model_confirmatory_units_12.csv'),['source_dataset','model'])
    # Use saved high-precision units for literal replay of the original methods.
    saved12=read('source_model_confirmatory_units_12.csv');saved60=read('source_seed_model_primary_units_60.csv')
    print('Rebuilt aggregate units; running original blocked permutations and bootstrap...',flush=True)
    if args.skip_bootstrap:
        rows=[]
        for predictor in ['environmental_instability','sampling_environment_gap','sampling_instability']:
            rho,pvalue,count,total=core.exact_blocked_permutation_p(saved12,predictor,'mean_balanced_accuracy_loss')
            rows.append(dict(predictor=predictor,spearman_rho=rho,exact_one_sided_p=pvalue,permutation_exceedances=count,permutations=total))
        assoc=pd.DataFrame(rows)
        expected=read('confirmatory_association_results.csv')[assoc.columns]
    else:
        assoc=core.run_confirmatory_associations(saved12)
        expected=read('confirmatory_association_results.csv')
    compare('Confirmatory analysis replay',assoc,expected,['predictor'])
    assoc.to_csv(out/'recomputed_associations.csv',index=False)
    print('Association replay completed; running original LOSO models...',flush=True)
    preds,scores=core.run_loso_validation(saved60)
    compare('LOSO predictions',preds,read('loso_oof_predictions.csv'),['specification','source_dataset','seed','model'])
    compare('LOSO scores',scores,read('loso_model_scores.csv'),['specification'])
    preds.to_csv(out/'recomputed_loso_predictions.csv',index=False);scores.to_csv(out/'recomputed_loso_scores.csv',index=False)
    u60.to_csv(out/'recomputed_units_60.csv',index=False);u12.to_csv(out/'recomputed_units_12.csv',index=False)
    summary=json.loads((ROOT/'Paper3_Final_Evaluation/final_analysis_summary.json').read_text())
    for col,stats in summary['global_descriptive_summary'].items():
        for name,value in [('mean',transfer[col].mean()),('sd',transfer[col].std()),('min',transfer[col].min()),('max',transfer[col].max())]:
            check(col+' '+name,np.isclose(value,stats[name],rtol=0,atol=1e-12))
    report={'status':'PASS' if all(c['pass'] for c in checks) else 'FAIL','scope':'Result-level replay only; not raw-data/model/prediction reproduction.','bootstrap_recomputed':not args.skip_bootstrap,
            'versions':{s:importlib.metadata.version(s) for s in ['numpy','pandas','scipy','scikit-learn']},'python':platform.python_version(),'checks':checks}
    (out/'reproduction_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],sum(c['pass'] for c in checks),'/',len(checks),'checks',flush=True)
    for c in checks:
        if not c['pass']:print(c['check'],c['detail'],flush=True)
    raise SystemExit(0 if report['status']=='PASS' else 1)

if __name__=='__main__':main()

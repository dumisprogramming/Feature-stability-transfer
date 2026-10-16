#!/usr/bin/env python3
"""Replay recovered pairwise stability aggregation on saved importance tables."""
import argparse
import ast
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def verify():
    notebook = ROOT / 'notebooks/Paper3_CompleteCode.ipynb'
    mapping = json.loads((ROOT / 'reports/notebook_source_map.json').read_text())
    checks = []

    def check(name, passed, detail=''):
        checks.append({'check': name, 'pass': bool(passed), 'detail': detail})

    check('Original notebook bytes', hashlib.sha256(notebook.read_bytes()).hexdigest()
          == mapping['uploaded_notebook_sha256'])
    nb = json.loads(notebook.read_text())
    for stage in mapping['stages']:
        source = ''.join(nb['cells'][stage['cell_index_zero_based']]['source']).encode()
        check(stage['script'], (ROOT / stage['script']).read_bytes() == source
              and hashlib.sha256(source).hexdigest() == stage['source_sha256'])

    tree = ast.parse(''.join(nb['cells'][8]['source']))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'calculate_pairwise_stability')
    namespace = {'np': np, 'pd': pd, 'combinations': combinations, 'TOP_K': 10}
    # Only this pure function is loaded; no notebook top-level code is executed.
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(notebook), 'exec'), namespace)
    calculate = namespace['calculate_pairwise_stability']
    files = sorted((ROOT / 'Paper3_Source_Stability').rglob('stability_complete.json'))
    check('60 supplied model/seed records', len(files) == 60)
    for completion in files:
        record = json.loads(completion.read_text())
        importance = pd.read_parquet(completion.parent / 'permutation_importance.parquet')
        features = sorted(importance['feature'].unique())
        check(str(completion.relative_to(ROOT)) + ' 49 features', len(features) == 49)
        for group, key in [('environmental', 'environmental_summary'), ('sampling', 'sampling_summary')]:
            _, summary = calculate(importance, features, group)
            for metric, observed in summary.items():
                expected = record[key][metric]
                passed = np.isclose(observed, expected, rtol=0, atol=1e-12)
                check(f"{record['dataset']}/{record['seed']}/{record['model']}/{group}/{metric}",
                      passed, f'observed={observed}; expected={expected}')
    return {'status': 'PASS' if all(x['pass'] for x in checks) else 'FAIL',
            'checks_passed': sum(x['pass'] for x in checks), 'checks_total': len(checks),
            'scope': 'Exact extraction and pairwise aggregation of saved importance values only; no model fitting or raw-data replay.',
            'checks': checks}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / 'outputs/recovered_stability_verification.json')
    args = parser.parse_args()
    result = verify()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(f"{result['status']}: {result['checks_passed']}/{result['checks_total']} recovered-source checks")
    raise SystemExit(0 if result['status'] == 'PASS' else 1)

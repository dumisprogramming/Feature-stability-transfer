"""Post-hoc source-only disagreement/heterogeneity comparison; see PLAN.md."""
import argparse
import hashlib
import itertools
import json
import platform
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression

MODELS = ['LogisticRegression', 'RandomForest', 'XGBoost']
KEY = ['source_dataset', 'seed', 'model']
PRIMARY = ['environmental_instability', 'sampling_environment_gap']
NEW = ['source_model_disagreement', 'source_feature_heterogeneity']
CATEGORICAL = {'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
               'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS',
               'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID',
               'DNS_QUERY_TYPE', 'FTP_COMMAND_RET_CODE'}
SPECS = {'model_only': [], 'model_plus_primary_warnings': PRIMARY,
         'model_plus_disagreement': NEW[:1],
         'model_plus_heterogeneity': NEW[1:],
         'model_plus_both_additional_warnings': NEW,
         'model_plus_all_four_warnings': NEW + PRIMARY}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def aggregate(frame):
    return frame.groupby(KEY, as_index=False).agg(
        loss=('balanced_accuracy_loss', 'mean'),
        **{c: (c, 'first') for c in NEW + PRIMARY})


def design(train, test, columns):
    aa, bb = [np.ones(len(train))], [np.ones(len(test))]
    for model in MODELS[1:]:
        aa.append(train.model.eq(model).to_numpy(float))
        bb.append(test.model.eq(model).to_numpy(float))
    for col in columns:
        mu, sd = train[col].mean(), train[col].std(ddof=0)
        sd = sd if sd > 0 else 1.
        aa.append((train[col].to_numpy() - mu) / sd)
        bb.append((test[col].to_numpy() - mu) / sd)
    return np.column_stack(aa), np.column_stack(bb)


def heterogeneity(frame, predictors):
    group_codes, groups = pd.factorize(frame.actual_environment_id, sort=True)
    counts = np.bincount(group_codes)
    assert len(groups) >= 2 and np.all(counts == 4000)
    rows = []
    for feature in predictors:
        values = frame[feature].to_numpy(float)
        finite = np.isfinite(values)
        if feature in CATEGORICAL:
            codes = np.full(len(values), -1, dtype=int)
            codes[finite] = pd.factorize(values[finite], sort=True)[0]
            codes += 1  # bucket zero is missing/nonfinite
            edges = []
        else:
            edges = np.unique(np.quantile(values[finite], np.arange(1, 10)/10)) if finite.any() else np.array([])
            codes = np.zeros(len(values), dtype=int)
            codes[finite] = np.searchsorted(edges, values[finite], side='right') + 1
        bucket_count = int(codes.max()) + 1
        table = np.zeros((len(groups), bucket_count))
        np.add.at(table, (group_codes, codes), 1)
        table /= counts[:, None]
        distances = [.5 * np.abs(table[a] - table[b]).sum()
                     for a, b in itertools.combinations(range(len(groups)), 2)]
        value = float(np.mean(distances))
        assert 0 <= value <= 1 + 1e-12
        rows.append({'feature': feature, 'distance': value,
                     'coding': 'categorical' if feature in CATEGORICAL else 'pooled_source_deciles',
                     'finite_rows': int(finite.sum()), 'groups': len(groups),
                     'pairs': len(distances), 'internal_edges': json.dumps(list(map(float, edges)))})
    return float(np.mean([r['distance'] for r in rows])), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, repo, out = args.data_root, args.repo, args.output
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'post_hoc_after_original_outcomes_known',
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'script_sha256': sha(__file__),
                'plan_sha256': sha(Path(__file__).with_name('PLAN.md')),
                'software': {'python': platform.python_version(), 'numpy': np.__version__,
                             'pandas': pd.__version__, 'sklearn': sklearn.__version__},
                'input_hashes': {}}
    save(out/'manifest.json', manifest)

    def record(path, expected=None):
        actual = sha(path)
        if expected is not None:
            assert actual == expected, f'Hash mismatch: {path}'
        label = ('data/' + str(path.relative_to(root))) if path.is_relative_to(root) else ('repo/' + str(path.relative_to(repo)))
        manifest['input_hashes'][label] = actual

    schema = root/'Paper3_Materialized/feature_schema_v1.json'
    record(schema)
    predictors = json.loads(schema.read_text())['predictors_in_order']
    assert len(predictors) == 49 and len(set(predictors)) == 49
    records, feature_rows, alignment = [], [], []
    source_records = sorted((root/'Paper3_Source_Models').rglob('source_seed_complete.json'))
    assert len(source_records) == 20
    prediction_count = 0
    for path in source_records:
        record(path)
        metadata = json.loads(path.read_text())
        dataset, seed = metadata['dataset'], int(metadata['seed'])
        frames = {}
        for model in MODELS:
            for partition in ['calibration', 'source_test']:
                info = metadata['models'][model]['files'][partition + '_predictions']
                file = path.parent/Path(info['path']).name
                record(file, info['sha256'])
                f = pd.read_parquet(file).sort_values('row_id').reset_index(drop=True)
                assert f.row_id.is_unique and f.y_true.isin([0, 1]).all() and f.y_pred.isin([0, 1]).all()
                assert np.isfinite(f.probability_attack).all() and f.probability_attack.between(0, 1).all()
                frames[(model, partition)] = f
                prediction_count += 1
        for partition in ['calibration', 'source_test']:
            first = frames[(MODELS[0], partition)]
            for model in MODELS[1:]:
                pd.testing.assert_frame_equal(first[['row_id', 'y_true']], frames[(model, partition)][['row_id', 'y_true']])
            alignment.append({'source_dataset': dataset, 'seed': seed,
                              'partition': partition, 'rows': len(first), 'aligned': True})
        env = root/'Paper3_Sanitized'/dataset/f'seed_{seed}'/'environment.parquet'
        audit_path = env.with_name('sanitization_manifest.json')
        record(audit_path, metadata['input_sanitization_manifest_sha256'])
        audit = json.loads(audit_path.read_text())
        record(schema, metadata['feature_schema_sha256'])
        record(env, audit['output_files']['environment']['sha256'])
        panel = pd.read_parquet(env)
        assert panel.row_id.is_unique and len(panel) == audit['output_files']['environment']['rows']
        h, detailed = heterogeneity(panel, predictors)
        for row in detailed:
            feature_rows.append({'source_dataset': dataset, 'seed': seed, **row})
        for model in MODELS:
            pred = frames[(model, 'source_test')].y_pred.to_numpy()
            disagreement = float(np.mean([np.mean(pred != frames[(other, 'source_test')].y_pred.to_numpy())
                                          for other in MODELS if other != model]))
            records.append({'source_dataset': dataset, 'seed': seed, 'model': model,
                            NEW[0]: disagreement, NEW[1]: h})
        print('Verified source inputs:', dataset, seed, flush=True)
    indicators = pd.DataFrame(records)
    assert len(indicators) == 60 and not indicators.duplicated(KEY).any()
    source = repo/'Paper3_Final_Evaluation/transfer_analysis_all_180.csv'
    record(source)
    f = pd.read_csv(source).merge(indicators, on=KEY, validate='many_to_one')
    assert len(f) == 180 and np.isfinite(f[NEW+PRIMARY+['balanced_accuracy_loss']]).all().all()
    indicators.to_csv(out/'additional_source_indicators.csv', index=False)
    pd.DataFrame(feature_rows).to_csv(out/'feature_heterogeneity_details.csv', index=False)
    pd.DataFrame(alignment).to_csv(out/'prediction_alignment.csv', index=False)
    predictions, diagnostics = [], []
    for omitted in [None] + sorted(f.seed.unique()):
        sub = f if omitted is None else f[f.seed != omitted]
        run = 'all_seeds' if omitted is None else 'omit_seed:' + str(omitted)
        for held in sorted(sub.source_dataset.unique()):
            training = sub[(sub.source_dataset != held) & (sub.target_dataset != held)]
            testing = sub[sub.source_dataset == held]
            assert held not in set(training.source_dataset) | set(training.target_dataset)
            assert training.groupby(KEY).target_dataset.nunique().eq(2).all()
            assert testing.groupby(KEY).target_dataset.nunique().eq(3).all()
            t, e = aggregate(training), aggregate(testing)
            mutated = sub.copy()
            mutated.loc[mutated.source_dataset == held, 'balanced_accuracy_loss'] += 100
            pd.testing.assert_frame_equal(t, aggregate(mutated[(mutated.source_dataset != held) & (mutated.target_dataset != held)]))
            for spec, cols in SPECS.items():
                a, b = design(t, e, cols)
                changed = e.copy()
                for col in cols:
                    changed[col] += 100
                assert np.array_equal(a, design(t, changed, cols)[0])
                rank = int(np.linalg.matrix_rank(a))
                assert rank == a.shape[1], (run, held, spec, 'rank deficient')
                pred = LinearRegression(fit_intercept=False).fit(a, t.loss).predict(b)
                diff = float(np.max(abs(pred - b @ np.linalg.lstsq(a, t.loss, rcond=None)[0])))
                assert diff < 1e-10
                diagnostics.append({'run': run, 'held_out': held, 'specification': spec,
                                    'rank': rank, 'columns': a.shape[1], 'training_units': len(t),
                                    'test_units': len(e), 'condition_number': float(np.linalg.cond(a)),
                                    'independent_difference': diff})
                for (_, row), value in zip(e.iterrows(), pred):
                    predictions.append({'run': run, 'held_out': held, 'seed': int(row.seed),
                                        'model': row.model, 'specification': spec,
                                        'actual_loss': float(row.loss), 'predicted_loss': float(value)})
    pred = pd.DataFrame(predictions)
    score_rows = []
    for level, keys in [('overall', ['run', 'specification']), ('fold', ['run', 'held_out', 'specification'])]:
        for key, group in pred.groupby(keys):
            error = group.actual_loss - group.predicted_loss
            score_rows.append({**dict(zip(keys, key)), 'level': level, 'units': len(group),
                               'mae': float(abs(error).mean()), 'rmse': float(np.sqrt((error**2).mean()))})
    scores = pd.DataFrame(score_rows).fillna('ALL')
    scope = ['run', 'level', 'held_out']
    baseline = scores[scores.specification == 'model_only'].set_index(scope).mae
    scores['mae_improvement_vs_model_only'] = [float(baseline.loc[tuple(row[k] for k in scope)]) - row.mae for _, row in scores.iterrows()]
    pred.to_csv(out/'predictions.csv', index=False)
    scores.to_csv(out/'scores.csv', index=False)
    save(out/'design_diagnostics.json', diagnostics)
    assert len(pred) == 1800 and len(diagnostics) == 144
    manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(), checks='PASS',
                    source_prediction_files_verified=prediction_count, environment_panels_verified=20,
                    aligned_prediction_panels=len(alignment), source_units=len(indicators),
                    fits=len(diagnostics), predictions=len(pred),
                    maximum_independent_difference=max(d['independent_difference'] for d in diagnostics),
                    output_hashes={p.name: sha(p) for p in out.iterdir() if p.name != 'manifest.json'})
    save(out/'manifest.json', manifest)
    print(scores[(scores.run == 'all_seeds') & (scores.level == 'overall')].to_string(index=False))
    print('PASS: 120 prediction files; 20 environmental panels; 144 full-rank fits; 1800 predictions.')


if __name__ == '__main__':
    main()

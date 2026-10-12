"""Independently validate saved three-domain experiment outputs.

This validator reads original NetCDF values, saved memberships and predictions.
It deliberately does not import the experiment implementation or refit models.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.analysis_deps'))
import numpy as np
import pandas as pd
import xarray as xr

OUT = ROOT / 'results' / 'multiregion'
REGIONS = ['Aral', 'Atyrau', 'Bozaigyr']
MODELS = ['training_mean', 'ridge_env', 'svr_env', 'rf_geo', 'rf_env',
          'rf_env_geo', 'gbr_env', 'gbr_env_geo']
ENV = ['stl1_c', 'stl2_c', 'u10', 'lai_lv']
METRICS = ['rmse', 'mae', 'bias', 'r2']
CHECKS = []


def check(name, condition, details=None):
    CHECKS.append({'check': name, 'passed': bool(condition), 'details': details})


def close(a, b, atol=1e-10, rtol=1e-9):
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                           atol=atol, rtol=rtol, equal_nan=True))


def area(frame):
    return np.cos(np.deg2rad(frame.latitude.to_numpy()))


def compute_metrics(frame):
    y = frame.swvl1.to_numpy(dtype=float)
    residual = frame.prediction.to_numpy(dtype=float) - y
    w = frame.area_weight.to_numpy(dtype=float)
    mean_y = (w*y).sum()/w.sum()
    mse = (w*residual**2).sum()/w.sum()
    variance = (w*(y-mean_y)**2).sum()/w.sum()
    return dict(rmse=np.sqrt(mse), mae=(w*np.abs(residual)).sum()/w.sum(),
                bias=(w*residual).sum()/w.sum(),
                r2=1-mse/variance if variance > 0 else np.nan)


def outside(frame, rectangle):
    # Independent form of the stated coordinate rectangle exclusion.
    eps = 0.3 + 1e-8
    return ((frame.latitude < rectangle.latitude.min()-eps) |
            (frame.latitude > rectangle.latitude.max()+eps) |
            (frame.longitude < rectangle.longitude.min()-eps) |
            (frame.longitude > rectangle.longitude.max()+eps))


def source_expected(df, region):
    target = df[df.region.eq(region)]
    candidates = df[~df.region.eq(region)]
    purged = candidates[outside(candidates, target)]
    retained = purged.sort_values(['region', 'sample_id']).drop_duplicates(
        ['site_id', 'valid_time'], keep='first')
    return retained, (len(candidates), len(purged), len(purged)-len(retained))


def quadrant_expected(target, fold):
    north = target.latitude >= (target.latitude.min()+target.latitude.max())/2
    east = target.longitude >= (target.longitude.min()+target.longitude.max())/2
    code = north.astype(int)*2 + east.astype(int)
    test = target[code.eq(fold)]
    train = target[~code.eq(fold) & outside(target, test)]
    return train, test


def expected_train_weights(frame, target_region=None):
    shares = {r: 1/frame.region.nunique() for r in frame.region.unique()}
    if target_region is not None:
        shares = {r: .5 if r == target_region else .25 for r in frame.region.unique()}
    answer = np.zeros(len(frame))
    for r in frame.region.unique():
        mask = frame.region.eq(r).to_numpy()
        w = area(frame.loc[mask])
        answer[mask] = shares[r] * w/w.sum()
    return answer * len(frame)/answer.sum()


def read_inputs():
    raw = pd.read_csv(OUT/'raw_grid.csv.gz')
    df = pd.read_csv(OUT/'analysis_grid.csv.gz')
    hashes = json.loads((OUT/'input_hashes.json').read_text(encoding='utf-8'))
    audit = json.loads((ROOT/'analysis/multiregion_audit.json').read_text(encoding='utf-8'))
    current_hashes = {x['file'].replace('\\', '/'): x['sha256'] for x in hashes}
    audit_hashes = {x['file'].replace('\\', '/'): x['sha256'] for x in audit['files']}
    paths = [p for r in REGIONS for p in sorted((ROOT/r).glob('*.nc'))]
    check('Exactly 24 original NetCDF files and complete hash coverage',
          len(paths) == len(hashes) == 24 and set(current_hashes) ==
          {p.relative_to(ROOT).as_posix() for p in paths})
    check('Run hashes match the independent pre-run audit', current_hashes == audit_hashes)
    check('Raw and analysis rows have identical sample identity',
          raw[['sample_id', 'region', 'valid_time', 'site_id']].equals(
              df[['sample_id', 'region', 'valid_time', 'site_id']]))
    check('Raw sample IDs are unique and exhaustive',
          len(raw) == 15740 and set(raw.sample_id) == set(range(15740)))
    keys = ['valid_time', 'latitude', 'longitude']
    for p in paths:
        relative = p.relative_to(ROOT).as_posix()
        check(f'Unchanged input: {relative}',
              hashlib.sha256(p.read_bytes()).hexdigest() == current_hashes[relative])
        with xr.open_dataset(p, engine='netcdf4') as ds:
            var = next(iter(ds.data_vars))
            actual = ds[[var]].to_dataframe().reset_index()
        actual['valid_time'] = actual.valid_time.dt.strftime('%Y-%m-%d')
        actual['latitude'] = actual.latitude.round(6)
        actual['longitude'] = actual.longitude.round(6)
        stored = raw[raw.region.eq(p.parent.name)][keys+[var]]
        joined = actual.merge(stored, on=keys, suffixes=('_nc', '_saved'), validate='one_to_one')
        check(f'Full-precision raw values: {relative}',
              len(joined) == len(actual) == len(stored) and
              close(joined[var+'_nc'], joined[var+'_saved'], atol=5e-13, rtol=0))
    for r, count in {'Aral':2193, 'Atyrau':4641, 'Bozaigyr':1036}.items():
        part = df[df.region.eq(r)]
        check(f'Grid counts and dates: {r}', len(part) == 2*count and
              part.site_id.nunique() == count and
              set(part.valid_time) == {'2026-08-10', '2026-08-18'} and
              not part.duplicated(['site_id', 'valid_time']).any() and
              part.groupby('site_id').size().eq(2).all())
    check('Distinct physical sites and site-date records',
          df.site_id.nunique() == 7534 and
          len(df.drop_duplicates(['site_id', 'valid_time'])) == 15068)
    aa = set(df.loc[df.region.eq('Atyrau'), 'site_id'])
    bb = set(df.loc[df.region.eq('Bozaigyr'), 'site_id'])
    ar = set(df.loc[df.region.eq('Aral'), 'site_id'])
    check('Physical overlap is exactly the 336 Atyrau/Bozaigyr sites',
          len(aa & bb) == 336 and not ar & (aa | bb))
    for v in ['swvl1', 'stl1', 'stl2', 'u10', 'lai_lv', 'snowc', 'lblt', 'licd']:
        expected = raw[v].to_numpy(dtype=float).copy()
        if v in ['swvl1', 'licd']:
            expected[(expected < 0) & (expected >= -1e-12)] = 0
        check(f'Only declared numerical normalization: {v}', close(df[v], expected, atol=5e-13, rtol=0))
    for level in [1, 2]:
        check(f'Temperature conversion from raw Kelvin: level {level}',
              close(df[f'stl{level}_c'], raw[f'stl{level}'].astype(float)-273.15, atol=5e-12, rtol=0))
    check('Stored evaluation weights are cosine latitude', close(df.area_weight, area(df)))
    return df.set_index('sample_id', drop=False).rename_axis('row_index')


def check_metric_table(pred, group_keys, table_path, extra=()):
    expected = []
    for key, part in pred.groupby(group_keys):
        key = key if isinstance(key, tuple) else (key,)
        expected.append(dict(zip(group_keys, key), n_test=len(part), **compute_metrics(part)))
    expected = pd.DataFrame(expected).set_index(group_keys).sort_index()
    saved = pd.read_csv(table_path).set_index(group_keys).sort_index()
    check(f'Metric table groups: {table_path.name}', saved.index.equals(expected.index))
    if saved.index.equals(expected.index):
        for m in METRICS:
            check(f'Recomputed {m}: {table_path.name}', close(saved[m], expected[m]))
        if 'n_test' in saved:
            check(f'Recomputed test counts: {table_path.name}', np.array_equal(saved.n_test, expected.n_test))
    return expected.reset_index(), saved.reset_index()


def verify_run(df, adaptation=False):
    prefix = 'adaptation' if adaptation else 'main'
    pred = pd.read_csv(OUT/f'{prefix}_predictions.csv.gz')
    members = pd.read_csv(OUT/f'{prefix}_split_membership.csv.gz')
    definitions = pd.read_csv(OUT/f'{prefix}_split_definitions.csv').set_index('split_id')
    models = ['rf_env', 'gbr_env'] if adaptation else MODELS
    check(f'{prefix}: expected number of splits', len(definitions) == (72 if adaptation else 39))
    check(f'{prefix}: expected total prediction count', len(pred) == (188880 if adaptation else 503680))
    check(f'{prefix}: no repeated predictions or memberships',
          not pred.duplicated(['split_id', 'model', 'sample_id']).any() and
          not members.duplicated(['split_id', 'sample_id']).any())
    check(f'{prefix}: valid membership roles and finite positive weights',
          set(members.role) == {'train', 'test'} and
          np.isfinite(members.weight).all() and members.weight.gt(0).all())
    check(f'{prefix}: complete split coverage',
          set(members.split_id) == set(pred.split_id) == set(definitions.index))
    mapped = df.loc[pred.sample_id]
    check(f'{prefix}: prediction provenance and observed values',
          all(np.array_equal(mapped[k].to_numpy(), pred[k].to_numpy())
              for k in ['region', 'valid_time', 'site_id']) and
          close(mapped.swvl1, pred.swvl1, atol=5e-13, rtol=0) and
          close(mapped.area_weight, pred.area_weight) and np.isfinite(pred.prediction).all())
    members_by_split = {k: v for k, v in members.groupby('split_id')}
    pred_by_split = {k: v for k, v in pred.groupby('split_id')}
    anchors_saved = {}
    for split_id, definition in definitions.iterrows():
        mem = members_by_split[split_id]
        train_mem = mem[mem.role.eq('train')]
        test_mem = mem[mem.role.eq('test')]
        train = df.loc[train_mem.sample_id]
        test = df.loc[test_mem.sample_id]
        region, fold = definition.region, int(definition.fold)
        target = df[df.region.eq(region)]
        ids_train, ids_test = set(train.sample_id), set(test.sample_id)
        source, purge_counts = source_expected(df, region)
        eligible, quad_test = quadrant_expected(target, fold)
        is_mixed = adaptation and definition.setting == 'source_plus_target'
        check(f'{split_id}: stored train/test counts',
              len(train) == definition.n_train and len(test) == definition.n_test)
        check(f'{split_id}: correct training and evaluation weights',
              close(train_mem.weight, expected_train_weights(train, region if is_mixed else None)) and
              close(test_mem.weight, area(test)))
        check(f'{split_id}: target region and no duplicate physical training rows',
              set(test.region) == {region} and
              not train.duplicated(['site_id', 'valid_time']).any() and not ids_train & ids_test)
        protocol = 'adaptation' if adaptation else definition.protocol
        if protocol != 'random_rows':
            check(f'{split_id}: physical train/test site separation', not set(train.site_id) & set(test.site_id))
            check(f'{split_id}: both dates retained per training and test site',
                  train.groupby('site_id').size().eq(2).all() and test.groupby('site_id').size().eq(2).all())
        if not adaptation:
            check(f'{split_id}: stored unique site counts',
                  train.site_id.nunique() == definition.train_sites and test.site_id.nunique() == definition.test_sites)
            if protocol in ['random_rows', 'random_sites']:
                check(f'{split_id}: exact within-region partition',
                      ids_train | ids_test == set(target.sample_id) and set(train.region) == {region})
            elif protocol == 'buffered_quadrants':
                check(f'{split_id}: independently reconstructed buffered quadrant',
                      ids_train == set(eligible.sample_id) and ids_test == set(quad_test.sample_id))
            elif protocol == 'leave_region_out':
                check(f'{split_id}: full target-rectangle purge and source deduplication',
                      ids_train == set(source.sample_id) and ids_test == set(target.sample_id))
                check(f'{split_id}: source purge counters',
                      [definition.source_rows_before_purge, definition.source_rows_after_purge,
                       definition.source_duplicate_rows_removed, definition.source_rows_final] ==
                      [*purge_counts, len(source)])
            else:
                check(f'{split_id}: known protocol', False)
        else:
            selected = train[train.region.eq(region)]
            sites = np.array(sorted(eligible.site_id.unique()))
            ordered = np.random.default_rng(20261005+REGIONS.index(region)*100+fold).permutation(sites)
            count = max(1, int(np.ceil(len(sites)*definition.fraction)))
            selected_expected = eligible[eligible.site_id.isin(ordered[:count])]
            check(f'{split_id}: exact nested target-label selection',
                  set(selected.sample_id) == set(selected_expected.sample_id) and
                  ids_test == set(quad_test.sample_id) and
                  definition.eligible_target_sites == len(sites) and
                  definition.selected_target_sites == count)
            expected_ids = set(selected_expected.sample_id) | (set(source.sample_id) if is_mixed else set())
            check(f'{split_id}: full target-rectangle source exclusion', ids_train == expected_ids)
            anchor_key = (region, fold, float(definition.fraction))
            selected_ids = set(selected.site_id)
            if anchor_key in anchors_saved:
                check(f'{split_id}: identical anchors across learning settings', anchors_saved[anchor_key] == selected_ids)
            anchors_saved[anchor_key] = selected_ids
        part = pred_by_split[split_id]
        check(f'{split_id}: all configurations predict precisely the test membership',
              set(part.model) == set(models) and
              all(set(g.sample_id) == ids_test for _, g in part.groupby('model')))
        if not adaptation:
            baseline = part[part.model.eq('training_mean')]
            reference_mean = np.average(train.swvl1, weights=train_mem.weight)
            check(f'{split_id}: baseline uses training labels and declared weights only',
                  close(baseline.prediction, np.full(len(baseline), reference_mean)))
    coverage_keys = ['region', 'model', 'fraction', 'setting'] if adaptation else ['protocol', 'region', 'model']
    for key, part in pred.groupby(coverage_keys):
        region = key[0] if adaptation else key[1]
        check(f'{prefix}: exhaustive out-of-fold coverage {key}',
              len(part) == len(df[df.region.eq(region)]) and
              set(part.sample_id) == set(df.loc[df.region.eq(region), 'sample_id']))
    check_metric_table(pred, ['split_id', 'model'], OUT/f'{prefix}_fold_metrics.csv')
    expected, saved = check_metric_table(pred, coverage_keys, OUT/f'{prefix}_region_metrics.csv')
    if not adaptation:
        baseline = expected[expected.model.eq('training_mean')].set_index(['protocol', 'region']).rmse
        required = []
        for row in saved.itertuples():
            required.append(1-(row.rmse/baseline.loc[(row.protocol, row.region)])**2)
        check('main: relative MSE skill against the matched training-mean baseline', close(saved.mse_skill, required))
    macro_keys = ['model', 'fraction', 'setting'] if adaptation else ['protocol', 'model']
    columns = METRICS if adaptation else METRICS+['mse_skill']
    expected_macro = saved.groupby(macro_keys)[columns].mean().sort_index()
    macro = pd.read_csv(OUT/f'{prefix}_macro_metrics.csv').set_index(macro_keys).sort_index()
    check(f'{prefix}: macro metrics equally average regional metrics',
          expected_macro.index.equals(macro.index) and close(expected_macro[columns], macro[columns]))
    if adaptation:
        for r in REGIONS:
            for f in range(4):
                a, b, c = [anchors_saved[(r, f, frac)] for frac in [.05, .2, 1.0]]
                check(f'adaptation:{r}:{f}: target label budgets are nested', a <= b <= c)
    return pred


def verify_diagnostics(df, main_predictions):
    shift = pd.read_csv(OUT/'feature_shift.csv').set_index(['region', 'feature'])
    conditional = pd.read_csv(OUT/'transfer_support_metrics.csv').set_index(['region', 'model', 'support'])
    conditional_keys = set()
    for r in REGIONS:
        source, _ = source_expected(df, r)
        target = df[df.region.eq(r)]
        w = expected_train_weights(source)
        flags = []
        for v in ENV:
            lo, hi = source[v].min(), source[v].max()
            flag = target[v].lt(lo) | target[v].gt(hi)
            flags.append(flag.to_numpy())
            sm = np.average(source[v], weights=w)
            sd = np.sqrt(np.average((source[v]-sm)**2, weights=w))
            tm = np.average(target[v], weights=area(target))
            row = shift.loc[(r, v)]
            check(f'Domain shift diagnostic: {r}/{v}',
                  close([row.source_min, row.source_max, row.source_mean, row.target_mean,
                         row.standardized_mean_difference, row.target_fraction_outside],
                        [lo, hi, sm, tm, (tm-sm)/sd if sd > 0 else np.nan,
                         np.average(flag, weights=area(target))]))
        outside_flags = np.any(flags, axis=0)
        check(f'Any-feature support diagnostic: {r}',
              close(shift.loc[(r, 'any_environment_feature')].target_fraction_outside,
                    np.average(outside_flags, weights=area(target))))
        outside_ids = set(target.loc[outside_flags, 'sample_id'])
        predictions = main_predictions[main_predictions.region.eq(r) &
                                       main_predictions.protocol.eq('leave_region_out')]
        for m, part in predictions.groupby('model'):
            for label in ['inside', 'outside']:
                mask = part.sample_id.isin(outside_ids)
                selected = part[mask if label == 'outside' else ~mask]
                if not len(selected):
                    continue
                key = (r, m, label)
                conditional_keys.add(key)
                result = compute_metrics(selected)
                row = conditional.loc[key]
                check(f'Conditional transfer metrics: {r}/{m}/{label}',
                      row.n == len(selected) and close([row[k] for k in METRICS],
                                                     [result[k] for k in METRICS]))
    check('All nonempty transfer support groups reported', set(conditional.index) == conditional_keys)
    runtime = pd.read_csv(OUT/'runtime.csv')
    check('Main runtime records are complete and nonnegative',
          len(runtime) == 312 and not runtime.duplicated(['split_id', 'model']).any() and
          np.isfinite(runtime[['fit_seconds', 'predict_seconds']]).all().all() and
          runtime[['fit_seconds', 'predict_seconds']].ge(0).all().all())


def main():
    df = read_inputs()
    pred = verify_run(df, adaptation=False)
    verify_diagnostics(df, pred)
    del pred
    verify_run(df, adaptation=True)
    summary = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    plan_bytes = (ROOT/'analysis/multiregion_experiment_plan.json').read_bytes()
    check('Summary records correct fixed plan and counts',
          summary['plan_sha256'] == hashlib.sha256(plan_bytes).hexdigest() and
          summary['seed'] == 20261005 and summary['n_region_labelled_rows'] == 15740 and
          summary['n_unique_site_date_rows'] == 15068 and summary['n_unique_sites'] == 7534 and
          summary['main_fit_count'] == 312 and summary['adaptation_fit_count'] == 144 and
          summary['main_prediction_count'] == 503680 and summary['adaptation_prediction_count'] == 188880)
    result = {'method': 'Independent recomputation from original NetCDF data, saved split memberships and predictions; experiment code is not imported.',
              'check_count': len(CHECKS), 'passed': sum(c['passed'] for c in CHECKS),
              'failed': sum(not c['passed'] for c in CHECKS), 'checks': CHECKS}
    (OUT/'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2))
    for failed in [c for c in CHECKS if not c['passed']]:
        print('FAIL:', failed['check'], failed['details'])
    if result['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

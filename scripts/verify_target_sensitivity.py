"""Verify target-subset sensitivity without importing either experiment runner."""
from __future__ import annotations

import hashlib
import json
import sys

import verify_multiregion as v

np, pd = v.np, v.pd
OUT = v.OUT/'sensitivity'
SNAPSHOT = OUT/'verification_primary_hashes.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_primary():
    hashes = {p.name: digest(p) for p in sorted(v.OUT.iterdir()) if p.is_file()}
    SNAPSHOT.write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    print(f'Captured {len(hashes)} primary output hashes without modifying those files.')


def main():
    plan_path = v.ROOT/'analysis/target_subset_sensitivity_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    record = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    v.check('Sensitivity plan is explicitly post hoc and uses the declared seeds and budgets',
            plan['stage'].startswith('Post hoc') and
            plan['subset_seed_bases'] == [20271005, 20281005] and
            plan['primary_subset_seed_base'] == 20261005 and
            plan['model_random_seed'] == 20261005 and plan['fractions'] == [.05, .2])
    v.check('Recorded sensitivity plan hash', record['plan_sha256'] == digest(plan_path))
    v.check('Primary outputs unchanged against sensitivity-run starting hashes',
            record['primary_unchanged'] and
            all(digest(v.OUT/name) == sha for name, sha in record['primary_hashes'].items()))
    if SNAPSHOT.exists():
        before = json.loads(SNAPSHOT.read_text(encoding='utf-8'))
        v.check('All primary outputs unchanged against independent verification snapshot',
                all(digest(v.OUT/name) == sha for name, sha in before.items()),
                {'file_count': len(before)})
    df = v.read_inputs()
    predictions = pd.read_csv(OUT/'predictions.csv.gz', float_precision='round_trip')
    members = pd.read_csv(OUT/'split_membership.csv.gz', float_precision='round_trip')
    definitions = pd.read_csv(OUT/'split_definitions.csv').set_index('split_id')
    v.check('Sensitivity output counts', len(definitions) == 96 and len(predictions) == 251840 and
            record['additional_splits'] == 96 and record['additional_fits'] == 192 and
            record['additional_predictions'] == 251840 and
            predictions.groupby(['split_id', 'model']).ngroups == 192)
    v.check('Sensitivity split IDs are exhaustive across output files',
            set(definitions.index) == set(members.split_id) == set(predictions.split_id))
    v.check('Sensitivity memberships and predictions have no duplicates',
            not members.duplicated(['split_id', 'sample_id']).any() and
            not predictions.duplicated(['split_id', 'model', 'sample_id']).any())
    v.check('Sensitivity models, seeds, settings and budgets match the plan',
            set(predictions.model) == {'rf_env', 'gbr_env'} and
            set(predictions.seed) == set(plan['subset_seed_bases']) and
            set(predictions.setting) == {'target_only', 'source_plus_target'} and
            set(predictions.fraction) == set(plan['fractions']))
    mapped = df.loc[predictions.sample_id]
    v.check('Sensitivity prediction targets and provenance match unchanged analysis records',
            all(np.array_equal(mapped[k].to_numpy(), predictions[k].to_numpy())
                for k in ['region', 'site_id', 'valid_time']) and
            v.close(mapped.swvl1, predictions.swvl1, atol=5e-13, rtol=0) and
            v.close(mapped.area_weight, predictions.area_weight) and
            np.isfinite(predictions.prediction).all())
    mem_groups = dict(tuple(members.groupby('split_id')))
    pred_groups = dict(tuple(predictions.groupby('split_id')))
    anchors_saved = {}
    for split_id, definition in definitions.iterrows():
        mem = mem_groups[split_id]
        a, b = mem[mem.role.eq('train')], mem[mem.role.eq('test')]
        train, test = df.loc[a.sample_id], df.loc[b.sample_id]
        region, fold = definition.region, int(definition.fold)
        target = df[df.region.eq(region)]
        eligible, expected_test = v.quadrant_expected(target, fold)
        source, _ = v.source_expected(df, region)
        sites = np.array(sorted(eligible.site_id.unique()))
        ordered = np.random.default_rng(int(definition.seed)+v.REGIONS.index(region)*100+fold).permutation(sites)
        count = max(1, int(np.ceil(len(sites)*definition.fraction)))
        expected_anchors = eligible[eligible.site_id.isin(ordered[:count])]
        anchors = train[train.region.eq(region)]
        mixed = definition.setting == 'source_plus_target'
        ids = set(expected_anchors.sample_id) | (set(source.sample_id) if mixed else set())
        v.check(f'{split_id}: exact seeded target anchors and buffer eligibility',
                set(anchors.sample_id) == set(expected_anchors.sample_id) and
                definition.selected_target_sites == count and definition.eligible_target_sites == len(sites))
        v.check(f'{split_id}: exact target test block and complete purged source pool',
                set(test.sample_id) == set(expected_test.sample_id) and set(train.sample_id) == ids and
                len(train) == definition.n_train and len(test) == definition.n_test)
        v.check(f'{split_id}: no physical-site leakage or duplicate training observations',
                not set(train.site_id) & set(test.site_id) and
                not train.duplicated(['site_id', 'valid_time']).any() and
                train.groupby('site_id').size().eq(2).all() and test.groupby('site_id').size().eq(2).all())
        v.check(f'{split_id}: declared regional training weights and latitude test weights',
                v.close(a.weight, v.expected_train_weights(train, region if mixed else None)) and
                v.close(b.weight, v.area(test)))
        key = (int(definition.seed), region, fold, float(definition.fraction))
        if key in anchors_saved:
            v.check(f'{split_id}: identical anchors across training settings',
                    anchors_saved[key] == set(anchors.site_id))
        anchors_saved[key] = set(anchors.site_id)
        part = pred_groups[split_id]
        v.check(f'{split_id}: both models predict precisely the test membership',
                set(part.model) == {'rf_env', 'gbr_env'} and
                all(set(g.sample_id) == set(test.sample_id) for _, g in part.groupby('model')))
        v.check(f'{split_id}: prediction metadata matches membership definition',
                all(part[k].eq(definition[k]).all() for k in ['seed', 'region', 'fold', 'fraction', 'setting']))
    for seed in plan['subset_seed_bases']:
        for region in v.REGIONS:
            for fold in range(4):
                v.check(f'{seed}/{region}/{fold}: 5% anchors are nested within 20% anchors',
                        anchors_saved[(seed, region, fold, .05)] <= anchors_saved[(seed, region, fold, .2)])
    keys = ['seed', 'region', 'model', 'fraction', 'setting']
    for key, part in predictions.groupby(keys):
        target = df[df.region.eq(key[1])]
        v.check(f'Sensitivity exhaustive out-of-fold coverage {key}',
                len(part) == len(target) and set(part.sample_id) == set(target.sample_id))
    _, saved = v.check_metric_table(predictions, keys, OUT/'region_metrics.csv')
    v.check('Exactly 48 additional regional metric rows', len(saved) == 48)
    primary = pd.read_csv(v.OUT/'adaptation_region_metrics.csv')
    primary = primary[primary.fraction.isin(plan['fractions'])].copy()
    primary['seed'] = plan['primary_subset_seed_base']
    expected = pd.concat([primary, saved], ignore_index=True).set_index(keys).sort_index()
    combined = pd.read_csv(OUT/'all_three_seed_region_metrics.csv').set_index(keys).sort_index()
    cols = v.METRICS+['n_test']
    v.check('Combined regional metrics retain the primary seed and all added seeds unchanged',
            len(combined) == 72 and expected.index.equals(combined.index) and
            v.close(expected[cols], combined[cols]))
    macro_keys = ['seed', 'model', 'fraction', 'setting']
    expected_macro = expected.reset_index().groupby(macro_keys)[v.METRICS].mean().sort_index()
    macro = pd.read_csv(OUT/'all_three_seed_macro_metrics.csv').set_index(macro_keys).sort_index()
    v.check('Three-seed macro metrics equally weight all three regional metrics',
            len(macro) == 24 and expected_macro.index.equals(macro.index) and
            v.close(expected_macro[v.METRICS], macro[v.METRICS]))
    range_keys = ['model', 'fraction', 'setting']
    expected_ranges = expected_macro.reset_index().groupby(range_keys).agg(
        rmse_min=('rmse', 'min'), rmse_max=('rmse', 'max'), rmse_mean=('rmse', 'mean')).sort_index()
    ranges = pd.read_csv(OUT/'macro_seed_ranges.csv').set_index(range_keys).sort_index()
    cols = ['rmse_min', 'rmse_max', 'rmse_mean']
    v.check('Three-seed RMSE range and mean independently recomputed',
            len(ranges) == 8 and expected_ranges.index.equals(ranges.index) and
            v.close(expected_ranges[cols], ranges[cols]))
    result = {'method': 'Independent verification using original NetCDF inputs, membership reconstruction and saved predictions; neither experiment runner is imported.',
              'check_count': len(v.CHECKS), 'passed': sum(x['passed'] for x in v.CHECKS),
              'failed': sum(not x['passed'] for x in v.CHECKS), 'checks': v.CHECKS}
    (OUT/'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: val for k, val in result.items() if k != 'checks'}, indent=2))
    for failed in [x for x in v.CHECKS if not x['passed']]:
        print('FAIL:', failed['check'], failed['details'])
    if result['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    if '--capture-primary' in sys.argv:
        capture_primary()
    else:
        main()

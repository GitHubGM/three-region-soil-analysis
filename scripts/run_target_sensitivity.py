"""Post hoc target-site selection sensitivity; leaves primary outputs unchanged."""
from __future__ import annotations
import gzip
import hashlib
import json
import time
import run_multiregion as exp
np=exp.np


def main():
    out=exp.OUT/'sensitivity'
    out.mkdir(exist_ok=True)
    plan_path=exp.ROOT/'analysis/target_subset_sensitivity_plan.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    primary_files=['summary.json','main_predictions.csv.gz','adaptation_predictions.csv.gz','adaptation_region_metrics.csv']
    before={p:hashlib.sha256((exp.OUT/p).read_bytes()).hexdigest() for p in primary_files}
    df=exp.pd.read_csv(exp.OUT/'analysis_grid.csv.gz',float_precision='round_trip')
    predictions,definitions=[],[]
    first=True
    start=time.perf_counter()
    with gzip.open(out/'split_membership.csv.gz','wt',encoding='utf-8',newline='') as membership:
        for seed in plan['subset_seed_bases']:
            for ri,region in enumerate(exp.REGIONS):
                source,_=exp.source_pool(df,region)
                target=df.loc[df.region==region]
                for fold,eligible,test in exp.spatial_folds(target):
                    ordered=np.random.default_rng(seed+ri*100+fold).permutation(np.array(sorted(eligible.site_id.unique())))
                    for fraction in plan['fractions']:
                        n=max(1,int(np.ceil(len(ordered)*fraction)))
                        anchors=eligible.loc[eligible.site_id.isin(ordered[:n])]
                        for setting in ['target_only','source_plus_target']:
                            train=anchors if setting=='target_only' else exp.pd.concat([source,anchors])
                            shares=None if setting=='target_only' else {region:.5,**{r:.25 for r in source.region.unique()}}
                            w=exp.weights(train,shares)
                            assert not set(train.site_id)&set(test.site_id)
                            split_id=f'sensitivity:{seed}:{region}:{fold}:{fraction}:{setting}'
                            exp.write_membership(membership,split_id,train,test,w,first)
                            first=False
                            definitions.append(dict(seed=seed,split_id=split_id,region=region,fold=fold,
                                                    fraction=fraction,setting=setting,eligible_target_sites=len(ordered),
                                                    selected_target_sites=n,n_train=len(train),n_test=len(test)))
                            for model in ['rf_env','gbr_env']:
                                pred,_,_=exp.fit_predict(model,train,test,w)
                                meta=dict(seed=seed,model=model,fold=fold,fraction=fraction,setting=setting,split_id=split_id)
                                predictions.append(exp.predict_frame(test,pred,meta))
                print(f'Completed target subset sensitivity {seed} {region}',flush=True)
    allp=exp.pd.concat(predictions,ignore_index=True)
    allp.to_csv(out/'predictions.csv.gz',index=False,compression='gzip')
    regional=exp.summarize_predictions(predictions,['seed','region','model','fraction','setting'])
    regional.to_csv(out/'region_metrics.csv',index=False)
    exp.pd.DataFrame(definitions).to_csv(out/'split_definitions.csv',index=False)
    primary=exp.pd.read_csv(exp.OUT/'adaptation_region_metrics.csv')
    primary=primary.loc[primary.fraction.isin(plan['fractions'])].copy()
    primary['seed']=plan['primary_subset_seed_base']
    combined=exp.pd.concat([primary,regional],ignore_index=True)
    combined.to_csv(out/'all_three_seed_region_metrics.csv',index=False)
    macro=combined.groupby(['seed','model','fraction','setting'])[['rmse','mae','bias','r2']].mean().reset_index()
    macro.to_csv(out/'all_three_seed_macro_metrics.csv',index=False)
    ranges=macro.groupby(['model','fraction','setting']).agg(rmse_min=('rmse','min'),rmse_max=('rmse','max'),rmse_mean=('rmse','mean')).reset_index()
    ranges.to_csv(out/'macro_seed_ranges.csv',index=False)
    assert all(hashlib.sha256((exp.OUT/p).read_bytes()).hexdigest()==h for p,h in before.items())
    record=dict(plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                additional_fits=len(predictions),additional_predictions=len(allp),additional_splits=len(definitions),
                primary_unchanged=True,primary_hashes=before,runtime_seconds=time.perf_counter()-start)
    (out/'summary.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)
    print(ranges.to_string(index=False),flush=True)


if __name__=='__main__':main()

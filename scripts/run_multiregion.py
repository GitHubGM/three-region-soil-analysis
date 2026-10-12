"""Fixed-design ML benchmark with physical-site overlap exclusion.

Inputs and previous Aral outputs are never modified. See the pre-score plan in
analysis/multiregion_experiment_plan.json for the design and its audit amendment.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.analysis_deps'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

OUT = ROOT / 'results' / 'multiregion'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261005
REGIONS = ['Aral', 'Atyrau', 'Bozaigyr']
LABELS = {'Aral':'Aral Sea domain', 'Atyrau':'Atyrau domain', 'Bozaigyr':'Northwestern domain'}
ENV = ['stl1_c','stl2_c','u10','lai_lv']
GEO = ['latitude','longitude']
MODELS = ['training_mean','ridge_env','svr_env','rf_geo','rf_env','rf_env_geo','gbr_env','gbr_env_geo']
VARIABLES = ['swvl1','stl1','stl2','u10','lai_lv','snowc','lblt','licd']
BUFFER = .3


def save_json(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def metric(y, pred, w):
    y, pred, w = map(np.asarray, (y,pred,w))
    mse = np.average((pred-y)**2, weights=w)
    var = np.average((y-np.average(y,weights=w))**2,weights=w)
    return dict(rmse=float(np.sqrt(mse)), mae=float(np.average(np.abs(pred-y),weights=w)),
                bias=float(np.average(pred-y,weights=w)), r2=float(1-mse/var) if var>0 else None)


def weights(frame, shares=None):
    """Regional balance, within-region cell-area weights, global mean one."""
    w = np.zeros(len(frame), dtype=float)
    groups = frame.region.to_numpy()
    shares = shares or {k:1/frame.region.nunique() for k in frame.region.unique()}
    for name in frame.region.unique():
        mask = groups==name
        a = np.cos(np.deg2rad(frame.loc[mask,'latitude'].to_numpy()))
        w[mask] = a/a.sum()*shares[name]
    return w/w.sum()*len(frame)


def test_weights(frame):
    return np.cos(np.deg2rad(frame.latitude.to_numpy()))


def outside_rectangle(frame, target, buffer=BUFFER):
    lat, lon = frame.latitude.to_numpy(), frame.longitude.to_numpy()
    dlat = np.maximum.reduce([target.latitude.min()-lat,lat-target.latitude.max(),np.zeros(len(frame))])
    dlon = np.maximum.reduce([target.longitude.min()-lon,lon-target.longitude.max(),np.zeros(len(frame))])
    return np.maximum(dlat,dlon)>buffer+1e-8


def source_pool(df, target_region):
    target = df.loc[df.region==target_region]
    source = df.loc[df.region!=target_region]
    purged = source.loc[outside_rectangle(source,target)]
    # Alphabetical input order selects a deterministic representative for any
    # duplicated source physical site-date; raw copies remain in raw_grid.csv.gz.
    unique = purged.drop_duplicates(['site_id','valid_time'],keep='first')
    return unique, dict(source_rows_before_purge=len(source),source_rows_after_purge=len(purged),
                        source_duplicate_rows_removed=len(purged)-len(unique),source_rows_final=len(unique))


def load():
    rows, records = [], []
    for region in REGIONS:
        datasets=[]
        for path in sorted((ROOT/region).glob('*.nc')):
            records.append(dict(file=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            with xr.open_dataset(path,engine='netcdf4') as data:
                datasets.append(data.load())
        assert len(datasets)==8
        ds=xr.merge(xr.align(*datasets,join='exact'),compat='no_conflicts',combine_attrs='drop_conflicts')
        assert set(ds.data_vars)==set(VARIABLES)
        frame=ds[VARIABLES].to_dataframe().reset_index()
        frame=frame[['valid_time','latitude','longitude']+VARIABLES].copy()
        frame[VARIABLES]=frame[VARIABLES].astype(float)
        frame[VARIABLES]=frame[VARIABLES].astype(float)
        assert np.isfinite(frame[VARIABLES]).all().all()
        assert list(frame.valid_time.dt.strftime('%Y-%m-%d').unique())==['2026-08-10','2026-08-18']
        frame['valid_time']=frame.valid_time.dt.strftime('%Y-%m-%d')
        frame['latitude']=frame.latitude.round(6)
        frame['longitude']=frame.longitude.round(6)
        frame['site_id']=[f'{a:.6f}_{b:.6f}' for a,b in zip(frame.latitude,frame.longitude)]
        frame['region']=region
        rows.append(frame)
    raw=pd.concat(rows,ignore_index=True)
    raw.insert(0,'sample_id',np.arange(len(raw)))
    raw.to_csv(OUT/'raw_grid.csv.gz',index=False,compression='gzip')
    df=raw.copy()
    roundoff={}
    for var in ['swvl1','licd']:
        mask=df[var].between(-1e-12,0,inclusive='left')
        roundoff[var]=int(mask.sum())
        df.loc[mask,var]=0.0
        assert (df[var]>=0).all()
    df['stl1_c']=df.stl1.astype(float)-273.15
    df['stl2_c']=df.stl2.astype(float)-273.15
    df['area_weight']=test_weights(df)
    df.to_csv(OUT/'analysis_grid.csv.gz',index=False,compression='gzip')
    save_json('input_hashes.json',records)
    return df,records,roundoff


def fit_predict(name, train, test, w):
    t0=time.perf_counter()
    if name=='training_mean':
        center=np.average(train.swvl1,weights=w)
        fit_time=time.perf_counter()-t0
        t1=time.perf_counter()
        pred=np.full(len(test),center)
    else:
        features=GEO if name=='rf_geo' else ENV+GEO if name.endswith('_env_geo') else ENV
        a,b=train[features].to_numpy(),test[features].to_numpy()
        if name in ['ridge_env','svr_env']:
            scaler=StandardScaler().fit(a,sample_weight=w)
            a,b=scaler.transform(a),scaler.transform(b)
        if name=='ridge_env':
            model=Ridge(alpha=10)
        elif name=='svr_env':
            model=SVR(kernel='rbf',C=1,epsilon=.005,gamma='scale',cache_size=256)
        elif name.startswith('rf_'):
            model=RandomForestRegressor(n_estimators=200,min_samples_leaf=10,max_features=1.0,
                                        random_state=SEED,n_jobs=2)
        else:
            model=GradientBoostingRegressor(n_estimators=150,learning_rate=.03,max_depth=2,
                                            min_samples_leaf=10,random_state=SEED,loss='squared_error')
        model.fit(a,train.swvl1.to_numpy(),sample_weight=w)
        fit_time=time.perf_counter()-t0
        t1=time.perf_counter()
        pred=model.predict(b)
    predict_time=time.perf_counter()-t1
    assert np.isfinite(pred).all()
    return pred,fit_time,predict_time


def spatial_folds(region_df):
    midlat=(region_df.latitude.min()+region_df.latitude.max())/2
    midlon=(region_df.longitude.min()+region_df.longitude.max())/2
    groups=2*(region_df.latitude>=midlat).astype(int)+(region_df.longitude>=midlon).astype(int)
    for fold in range(4):
        test=region_df.loc[groups==fold]
        train=region_df.loc[(groups!=fold)&outside_rectangle(region_df,test)]
        yield fold,train,test


def main_splits(df):
    for region in REGIONS:
        subset=df.loc[df.region==region]
        sites=np.array(sorted(subset.site_id.unique()))
        for fold,(a,b) in enumerate(KFold(4,shuffle=True,random_state=SEED).split(subset)):
            yield 'random_rows',region,fold,subset.iloc[a],subset.iloc[b],{}
        for fold,(a,b) in enumerate(KFold(4,shuffle=True,random_state=SEED).split(sites)):
            yield 'random_sites',region,fold,subset.loc[subset.site_id.isin(sites[a])],subset.loc[subset.site_id.isin(sites[b])],{}
        for fold,train,test in spatial_folds(subset):
            yield 'buffered_quadrants',region,fold,train,test,{}
        train,purge=source_pool(df,region)
        yield 'leave_region_out',region,0,train,subset,purge


def write_membership(handle,split_id,train,test,w,first):
    a=pd.DataFrame(dict(split_id=split_id,sample_id=train.sample_id,role='train',weight=w))
    b=pd.DataFrame(dict(split_id=split_id,sample_id=test.sample_id,role='test',weight=test_weights(test)))
    pd.concat([a,b]).to_csv(handle,index=False,header=first)


def predict_frame(test,pred,meta):
    p=test[['sample_id','region','valid_time','site_id','swvl1','area_weight']].copy()
    p['prediction']=pred
    for key,val in meta.items():p[key]=val
    return p


def summarize_predictions(frames,keys):
    allp=pd.concat(frames,ignore_index=True)
    result=[]
    for key,part in allp.groupby(keys,sort=False):
        if not isinstance(key,tuple):key=(key,)
        result.append(dict(zip(keys,key),n_test=len(part),**metric(part.swvl1,part.prediction,part.area_weight)))
    return pd.DataFrame(result)


def descriptive(df):
    rows=[]
    for (region,date),part in df.groupby(['region','valid_time']):
        for var in ['swvl1']+ENV:
            values=part[var].to_numpy()
            w=test_weights(part)
            mu=np.average(values,weights=w)
            rows.append(dict(region=region,date=date,variable=var,n=len(part),mean=float(mu),
                             sd=float(np.sqrt(np.average((values-mu)**2,weights=w))),
                             min=float(values.min()),q25=float(np.quantile(values,.25)),
                             median=float(np.median(values)),q75=float(np.quantile(values,.75)),max=float(values.max())))
    pd.DataFrame(rows).to_csv(OUT/'regional_distributions.csv',index=False)
    shifts=[]
    for region in REGIONS:
        source,_=source_pool(df,region)
        target=df.loc[df.region==region]
        w=weights(source)
        outside=[]
        for var in ENV:
            low,high=float(source[var].min()),float(source[var].max())
            mask=(target[var].to_numpy()<low)|(target[var].to_numpy()>high)
            outside.append(mask)
            sm=np.average(source[var],weights=w)
            sd=np.sqrt(np.average((source[var]-sm)**2,weights=w))
            tm=np.average(target[var],weights=test_weights(target))
            shifts.append(dict(region=region,feature=var,source_min=low,source_max=high,
                               source_mean=float(sm),target_mean=float(tm),
                               standardized_mean_difference=float((tm-sm)/sd) if sd>0 else None,
                               target_fraction_outside=float(np.average(mask,weights=test_weights(target)))))
        shifts.append(dict(region=region,feature='any_environment_feature',
                           target_fraction_outside=float(np.average(np.any(outside,axis=0),weights=test_weights(target)))))
    pd.DataFrame(shifts).to_csv(OUT/'feature_shift.csv',index=False)


def run_main(df):
    predictions,folds,timings,definitions=[],[],[],[]
    with gzip.open(OUT/'main_split_membership.csv.gz','wt',encoding='utf-8',newline='') as membership:
        for i,(protocol,region,fold,train,test,extra) in enumerate(main_splits(df)):
            split_id=f'{protocol}:{region}:{fold}'
            w=weights(train)
            assert len(train)>30 and len(test)>20
            if protocol!='random_rows':
                assert not set(train.site_id)&set(test.site_id)
            assert not train.duplicated(['site_id','valid_time']).any()
            write_membership(membership,split_id,train,test,w,i==0)
            definitions.append(dict(split_id=split_id,protocol=protocol,region=region,fold=fold,
                                    n_train=len(train),n_test=len(test),train_sites=train.site_id.nunique(),
                                    test_sites=test.site_id.nunique(),**extra))
            for name in MODELS:
                pred,fitsec,predsec=fit_predict(name,train,test,w)
                meta=dict(protocol=protocol,model=name,fold=fold,split_id=split_id)
                predictions.append(predict_frame(test,pred,meta))
                folds.append(dict(region=region,**meta,n_train=len(train),n_test=len(test),
                                  **metric(test.swvl1,pred,test_weights(test))))
                timings.append(dict(region=region,**meta,fit_seconds=fitsec,predict_seconds=predsec))
            print(f'Completed {split_id} train={len(train)} test={len(test)}',flush=True)
    allp=pd.concat(predictions,ignore_index=True)
    allp.to_csv(OUT/'main_predictions.csv.gz',index=False,compression='gzip')
    regional=summarize_predictions(predictions,['protocol','region','model'])
    baseline=regional.loc[regional.model=='training_mean',['protocol','region','rmse']].rename(columns={'rmse':'baseline_rmse'})
    regional=regional.merge(baseline,on=['protocol','region'])
    regional['mse_skill']=1-(regional.rmse/regional.baseline_rmse)**2
    regional.to_csv(OUT/'main_region_metrics.csv',index=False)
    macro=regional.groupby(['protocol','model'],sort=False)[['rmse','mae','bias','r2','mse_skill']].mean().reset_index()
    macro.to_csv(OUT/'main_macro_metrics.csv',index=False)
    pd.DataFrame(folds).to_csv(OUT/'main_fold_metrics.csv',index=False)
    pd.DataFrame(timings).to_csv(OUT/'runtime.csv',index=False)
    pd.DataFrame(definitions).to_csv(OUT/'main_split_definitions.csv',index=False)
    # Conditional error is descriptive; envelope is fitted solely on source X.
    support=[]
    for region in REGIONS:
        source,_=source_pool(df,region)
        target=df.loc[df.region==region]
        outside=((target[ENV]<source[ENV].min())|(target[ENV]>source[ENV].max())).any(axis=1)
        ids=set(target.loc[outside,'sample_id'])
        part=allp.loc[(allp.protocol=='leave_region_out')&(allp.region==region)]
        for name,group in part.groupby('model'):
            for label,mask in [('outside',group.sample_id.isin(ids)),('inside',~group.sample_id.isin(ids))]:
                a=group.loc[mask]
                if len(a):support.append(dict(region=region,model=name,support=label,n=len(a),
                                              **metric(a.swvl1,a.prediction,a.area_weight)))
    pd.DataFrame(support).to_csv(OUT/'transfer_support_metrics.csv',index=False)
    return regional,macro,len(allp),len(folds)


def run_adaptation(df):
    predictions,folds,definitions=[],[],[]
    first=True
    with gzip.open(OUT/'adaptation_split_membership.csv.gz','wt',encoding='utf-8',newline='') as membership:
        for ri,region in enumerate(REGIONS):
            target=df.loc[df.region==region]
            source,_=source_pool(df,region)
            for fold,eligible,test in spatial_folds(target):
                sites=np.array(sorted(eligible.site_id.unique()))
                rng=np.random.default_rng(SEED+ri*100+fold)
                ordered=rng.permutation(sites)
                for fraction in [.05,.2,1.0]:
                    n=max(1,int(np.ceil(len(sites)*fraction)))
                    anchors=eligible.loc[eligible.site_id.isin(ordered[:n])]
                    for setting in ['target_only','source_plus_target']:
                        train=anchors if setting=='target_only' else pd.concat([source,anchors])
                        shares=None if setting=='target_only' else {region:.5,**{r:.25 for r in source.region.unique()}}
                        w=weights(train,shares)
                        assert not set(train.site_id)&set(test.site_id)
                        assert not train.duplicated(['site_id','valid_time']).any()
                        split_id=f'adaptation:{region}:{fold}:{fraction}:{setting}'
                        write_membership(membership,split_id,train,test,w,first)
                        first=False
                        definitions.append(dict(split_id=split_id,region=region,fold=fold,fraction=fraction,
                                                setting=setting,eligible_target_sites=len(sites),selected_target_sites=n,
                                                n_train=len(train),n_test=len(test)))
                        for name in ['rf_env','gbr_env']:
                            pred,fitsec,predsec=fit_predict(name,train,test,w)
                            meta=dict(model=name,fold=fold,fraction=fraction,setting=setting,split_id=split_id)
                            predictions.append(predict_frame(test,pred,meta))
                            folds.append(dict(region=region,**meta,n_train=len(train),target_sites=n,
                                              fit_seconds=fitsec,predict_seconds=predsec,
                                              **metric(test.swvl1,pred,test_weights(test))))
                print(f'Completed adaptation {region}:{fold}',flush=True)
    allp=pd.concat(predictions,ignore_index=True)
    allp.to_csv(OUT/'adaptation_predictions.csv.gz',index=False,compression='gzip')
    regional=summarize_predictions(predictions,['region','model','fraction','setting'])
    regional.to_csv(OUT/'adaptation_region_metrics.csv',index=False)
    regional.groupby(['model','fraction','setting'],sort=False)[['rmse','mae','bias','r2']].mean().reset_index().to_csv(OUT/'adaptation_macro_metrics.csv',index=False)
    pd.DataFrame(folds).to_csv(OUT/'adaptation_fold_metrics.csv',index=False)
    pd.DataFrame(definitions).to_csv(OUT/'adaptation_split_definitions.csv',index=False)
    return len(allp),len(folds)


def main():
    start=time.perf_counter()
    plan_path=ROOT/'analysis/multiregion_experiment_plan.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    assert set(plan['models'])==set(MODELS)
    df,hashes,roundoff=load()
    descriptive(df)
    regional,macro,npred,nfit=run_main(df)
    nadapt,nfitadapt=run_adaptation(df)
    assert all(hashlib.sha256((ROOT/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in hashes)
    info={
        'seed':SEED,'labels':LABELS,'dates':sorted(df.valid_time.unique()),
        'n_region_labelled_rows':len(df),'n_unique_site_date_rows':len(df.drop_duplicates(['site_id','valid_time'])),
        'n_unique_sites':df.site_id.nunique(),'main_fit_count':nfit,'adaptation_fit_count':nfitadapt,
        'main_prediction_count':npred,'adaptation_prediction_count':nadapt,
        'roundoff_normalized':roundoff,'all_inputs_unchanged':True,
        'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'runtime_seconds':time.perf_counter()-start,'platform':platform.platform(),'processor':platform.processor(),
        'python':sys.version,'packages':{k:importlib.metadata.version(k) for k in ['numpy','pandas','xarray','netCDF4','scikit-learn','scipy']},
        'regions':[],
    }
    for region,part in df.groupby('region',sort=False):
        info['regions'].append(dict(folder=region,label=LABELS[region],n_sites=part.site_id.nunique(),n_rows=len(part),
                                    lat_min=float(part.latitude.min()),lat_max=float(part.latitude.max()),
                                    lon_min=float(part.longitude.min()),lon_max=float(part.longitude.max())))
    save_json('summary.json',info)
    print(json.dumps(info,indent=2),flush=True)
    print(macro.to_string(index=False),flush=True)


if __name__=='__main__':main()

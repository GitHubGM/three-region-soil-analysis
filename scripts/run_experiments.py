"""Reproducible native-grid audit and spatial soil-moisture reconstruction pilot.

This experiment does not estimate salinity, CSI, SDRI, or future degradation.
Inputs are never modified. Model settings are fixed before inspecting scores.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.analysis_deps'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'analysis' / 'mplconfig'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from audit_inputs import input_paths, safe

OUT = ROOT / 'results'
FIG = OUT / 'figures'
SEED = 20261002
ENV = ['stl1_c', 'stl2_c', 'u10', 'lai_lv']
GEO = ['latitude', 'longitude']


def weighted_mean(a, w):
    ok = np.isfinite(a) & np.isfinite(w)
    return float(np.average(a[ok], weights=w[ok]))


def scores(y, p, w):
    mse = np.average((p - y)**2, weights=w)
    v = np.average((y - np.average(y, weights=w))**2, weights=w)
    return {'rmse': float(np.sqrt(mse)),
            'mae': float(np.average(np.abs(p - y), weights=w)),
            'bias': float(np.average(p - y, weights=w)),
            'r2': float(1 - mse / v) if v > 0 else None}


def load_inputs():
    datasets = []
    for p in input_paths():
        with xr.open_dataset(p, engine='netcdf4') as ds:
            datasets.append(ds.load())
    # Exact coordinate agreement is a requirement, not an interpolation choice.
    aligned = xr.align(*datasets, join='exact')
    ds = xr.merge(aligned, compat='no_conflicts', combine_attrs='drop_conflicts')
    return ds


def new_models():
    rf = dict(n_estimators=200, min_samples_leaf=10, max_features=1.0,
              random_state=SEED, n_jobs=2)
    return {
        'training_mean': (None, []),
        'ridge_environment': (make_pipeline(StandardScaler(), Ridge(alpha=10.0)), ENV),
        'rf_geography': (RandomForestRegressor(**rf), GEO),
        'rf_environment': (RandomForestRegressor(**rf), ENV),
        'rf_combined': (RandomForestRegressor(**rf), ENV + GEO),
        'gbr_combined': (GradientBoostingRegressor(n_estimators=150,
                         learning_rate=0.03, max_depth=2, min_samples_leaf=10,
                         loss='squared_error', random_state=SEED), ENV + GEO),
    }


def spatial_experiments(df, lat, lon):
    """Held-out geographic rectangles, with training cells near each removed.

    Three fixed designs test dependence on geometry and separation. No test fold
    is used to choose features or settings. Both dates of a site stay together.
    """
    records, predictions, folds = [], [], []
    midlat, midlon = (lat.min()+lat.max())/2, (lon.min()+lon.max())/2
    lonedges = np.linspace(lon.min(), lon.max(), 5)
    for design, buffer_deg in [('quadrants_buffer_0.3', 0.3),
                                ('quadrants_buffer_0.6', 0.6),
                                ('longitude_strips_buffer_0.3', 0.3)]:
        if design.startswith('quadrants'):
            groups = (df.latitude.ge(midlat).astype(int)*2 +
                      df.longitude.ge(midlon).astype(int)).to_numpy()
        else:
            groups = np.clip(np.searchsorted(lonedges[1:], df.longitude,
                                             side='right'), 0, 3)
        for fold in range(4):
            test = groups == fold
            held = df.loc[test]
            if held.empty:
                raise ValueError(f'Empty test block: {design}/{fold}')
            # Distance to the held rectangle in degrees. This conservative box
            # also excludes training points beside missing test observations.
            dlat = np.maximum.reduce([held.latitude.min()-df.latitude.to_numpy(),
                    df.latitude.to_numpy()-held.latitude.max(), np.zeros(len(df))])
            dlon = np.maximum.reduce([held.longitude.min()-df.longitude.to_numpy(),
                    df.longitude.to_numpy()-held.longitude.max(), np.zeros(len(df))])
            train = (~test) & (np.maximum(dlat, dlon) > buffer_deg + 1e-8)
            if train.sum() < 30 or test.sum() < 20:
                raise ValueError(f'Insufficient samples in {design}/{fold}')
            a, b = df.loc[train], df.loc[test]
            wtrain, wtest = np.cos(np.deg2rad(a.latitude)), np.cos(np.deg2rad(b.latitude))
            ytrain, ytest = a.swvl1.to_numpy(), b.swvl1.to_numpy()
            folds.append({'design': design, 'fold': fold, 'buffer_degrees': buffer_deg,
                          'n_train': int(train.sum()), 'n_test': int(test.sum()),
                          'n_buffer_excluded': int((~train & ~test).sum()),
                          'test_lat_min': float(b.latitude.min()),
                          'test_lat_max': float(b.latitude.max()),
                          'test_lon_min': float(b.longitude.min()),
                          'test_lon_max': float(b.longitude.max())})
            for name, (model, features) in new_models().items():
                if model is None:
                    pred = np.full(len(b), np.average(ytrain, weights=wtrain))
                else:
                    if name.startswith('ridge'):
                        model.fit(a[features], ytrain, ridge__sample_weight=wtrain)
                    else:
                        model.fit(a[features], ytrain, sample_weight=wtrain)
                    pred = model.predict(b[features])
                # No clipping or post-hoc calibration of predictions.
                records.append({'design': design, 'fold': fold, 'model': name,
                                'n_train': len(a), 'n_test': len(b),
                                **scores(ytest, pred, wtest)})
                frame = b[['valid_time','latitude','longitude','swvl1']].copy()
                frame['prediction'] = pred
                frame['model'], frame['design'], frame['fold'] = name, design, fold
                frame['area_weight'] = wtest
                predictions.append(frame)
        print(f'Completed {design}', flush=True)
    met = pd.DataFrame(records)
    pred = pd.concat(predictions, ignore_index=True)
    overall = []
    for (design, name), g in pred.groupby(['design','model']):
        sc = scores(g.swvl1.to_numpy(), g.prediction.to_numpy(), g.area_weight.to_numpy())
        fold_sc = met[(met.design == design) & (met.model == name)]
        overall.append({'design': design, 'model': name, 'n_test': len(g), **sc,
                        'fold_rmse_min': fold_sc.rmse.min(),
                        'fold_rmse_max': fold_sc.rmse.max(),
                        'fold_r2_min': fold_sc.r2.min(), 'fold_r2_max': fold_sc.r2.max()})
    overall = pd.DataFrame(overall)
    for design, g in overall.groupby('design'):
        base = float(g.loc[g.model == 'training_mean', 'rmse'].iloc[0])
        overall.loc[g.index, 'mse_skill_vs_training_mean'] = 1-(g.rmse/base)**2 if base > 0 else np.nan
    met.to_csv(OUT / 'spatial_fold_metrics.csv', index=False)
    overall.to_csv(OUT / 'spatial_summary.csv', index=False)
    pd.DataFrame(folds).to_csv(OUT / 'spatial_fold_definitions.csv', index=False)
    pred.to_csv(OUT / 'spatial_predictions.csv', index=False)
    return overall, pred


def save_figures(ds, df, summary_metrics, pred):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 220})
    fields = [('swvl1','Soil water at 0–7 cm','m³/m³','YlGnBu'),
              ('stl1_c','Soil temperature at 0–7 cm','°C','inferno'),
              ('stl2_c','Soil temperature at 7–28 cm','°C','inferno'),
              ('lai_lv','Low vegetation LAI climatology','m²/m²','YlGn'),
              ('u10','Eastward wind component at 10 m','m/s','RdBu_r'),
              ('soil_temperature_difference_c','Topsoil minus deeper soil temperature','°C','coolwarm')]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    temp_min = float(min(ds.stl1_c.min(), ds.stl2_c.min()))
    temp_max = float(max(ds.stl1_c.max(), ds.stl2_c.max()))
    for ax, (v, title, unit, cmap) in zip(axes.flat, fields):
        da = ds[v].transpose('latitude', 'longitude')
        limits = {}
        if v in {'stl1_c','stl2_c'}:
            limits = {'vmin':temp_min,'vmax':temp_max}
        elif v in {'u10','soil_temperature_difference_c'}:
            lim = float(np.nanmax(np.abs(da.values))) or 1.0
            limits = {'vmin':-lim,'vmax':lim}
        im = ax.pcolormesh(ds.longitude, ds.latitude, da, cmap=cmap, shading='nearest', **limits)
        ax.set(title=title, xlabel='Longitude °E', ylabel='Latitude °N')
        ax.set_aspect(1/np.cos(np.deg2rad(float(ds.latitude.mean()))))
        fig.colorbar(im, ax=ax, label=unit, shrink=.78)
    fig.suptitle('Supplied native grid • daily mean • '+str(ds.attrs['analysis_date'])+
                 '\nGrid coverage only; no contemporary shoreline or land mask supplied', fontsize=14)
    fig.savefig(FIG/'native_grid_maps.png')
    plt.close(fig)

    corr = df[['swvl1'] + ENV].corr(method='spearman')
    labels = ['Soil water', 'Topsoil temperature', 'Deeper soil temperature',
              'Eastward wind', 'LAI climatology']
    fig, ax = plt.subplots(figsize=(7,6), constrained_layout=True)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center',
                    color='white' if abs(corr.iloc[i,j])>.65 else '#111')
    ax.set_xticks(range(5), labels, rotation=40, ha='right')
    ax.set_yticks(range(5), labels)
    ax.set_title('Spearman correlations pooled across two snapshots\nRepeated sites and spatial dependence remain')
    fig.colorbar(im, ax=ax, label='Spearman correlation', shrink=.75)
    fig.savefig(FIG/'spatial_correlations.png')
    plt.close(fig)

    designs = ['quadrants_buffer_0.3','quadrants_buffer_0.6','longitude_strips_buffer_0.3']
    design_titles = ['Quadrants • 0.3° buffer (primary)', 'Quadrants • 0.6° buffer', 'Longitude strips • 0.3° buffer']
    fig, axes = plt.subplots(1, len(designs), figsize=(14,5), constrained_layout=True)
    names = ['training_mean','ridge_environment','rf_geography','rf_environment','rf_combined','gbr_combined']
    labels = ['Training mean','Ridge environmental','RF coordinates','RF environmental','RF combined','GBR combined']
    for ax, design, title in zip(axes, designs, design_titles):
        m = summary_metrics[summary_metrics.design == design].set_index('model').loc[names]
        ax.barh(labels, m.rmse, color=['#9a9a9a','#8ab8c5','#d9ad70','#6a9a83','#29677c','#735f8a'])
        ax.invert_yaxis()
        ax.set(xlabel='Pooled RMSE (m³/m³)', title=title, xlim=(0,float(summary_metrics.rmse.max())*1.08))
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Spatial reconstruction of same-day ERA5-family soil water\nFixed settings; contiguous test blocks with buffers', fontsize=14)
    fig.savefig(FIG/'spatial_model_comparison.png')
    plt.close(fig)

    g = pred[(pred.design == 'quadrants_buffer_0.3') & (pred.model == 'rf_combined') &
             (pred.valid_time == pd.Timestamp(ds.attrs['analysis_date']))]
    fig, axes = plt.subplots(1,2,figsize=(11,5), constrained_layout=True)
    grid_fold = g.pivot(index='latitude',columns='longitude',values='fold').sort_index()
    grid_error = g.assign(error=g.prediction-g.swvl1).pivot(index='latitude',columns='longitude',values='error').sort_index()
    axes[0].pcolormesh(grid_fold.columns,grid_fold.index,grid_fold,cmap=plt.get_cmap('tab10',4),vmin=-.5,vmax=3.5,shading='nearest')
    for fold, block in g.groupby('fold'):
        axes[0].text(block.longitude.mean(),block.latitude.mean(),str(int(fold)+1),ha='center',va='center',fontsize=15,
                     bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'boxstyle':'round,pad=0.3'})
    axes[0].set(title='Four held-out geographic quadrants',xlabel='Longitude °E',ylabel='Latitude °N')
    im = axes[1].pcolormesh(grid_error.columns,grid_error.index,grid_error,cmap='RdBu_r',
                          vmin=-.15,vmax=.15,shading='nearest')
    axes[1].set(title='RF combined out-of-fold residuals',xlabel='Longitude °E',ylabel='Latitude °N')
    fig.colorbar(im, ax=axes[1], label='Prediction minus reanalysis (m³/m³)', extend='both')
    for ax in axes:
        ax.set_aspect(1/np.cos(np.deg2rad(df.latitude.mean())))
    fig.suptitle('Spatial test on 18 August 2026\nEach location was held out on both dates',fontsize=13)
    fig.savefig(FIG/'spatial_validation.png')
    plt.close(fig)


def paired_analysis(ds):
    rows, changes = [], []
    weights = np.broadcast_to(np.cos(np.deg2rad(ds.latitude.values))[:,None],
                              (ds.sizes['latitude'],ds.sizes['longitude']))
    for date in ds.valid_time.values:
        snap = ds.sel(valid_time=date)
        for v in ['swvl1','stl1_c','stl2_c','u10','lai_lv']:
            a = snap[v].transpose('latitude','longitude').values
            rows.append({'date':str(pd.Timestamp(date).date()),'variable':v,
                         'area_weighted_mean':weighted_mean(a,weights),
                         'min':float(np.nanmin(a)),'max':float(np.nanmax(a))})
    first, last = ds.isel(valid_time=0,drop=True), ds.isel(valid_time=-1,drop=True)
    paired = first[['swvl1']].to_dataframe().reset_index()[['latitude','longitude']]
    for v in ['swvl1','stl1_c','stl2_c','u10','lai_lv']:
        a, b = [x[v].transpose('latitude','longitude').values for x in [first,last]]
        delta = b-a
        valid = np.isfinite(a)&np.isfinite(b)
        tolerance = 1e-12
        changes.append({'variable':v,'n_paired':int(valid.sum()),
                        'mean_first':weighted_mean(a,weights),
                        'mean_last':weighted_mean(b,weights),
                        'mean_change':weighted_mean(delta,weights),
                        'median_change':float(np.nanmedian(delta)),
                        'decrease_percent':100*float(np.average((delta[valid]<-tolerance),weights=weights[valid])),
                        'increase_percent':100*float(np.average((delta[valid]>tolerance),weights=weights[valid]))})
        # Match xarray dataframe coordinate ordering explicitly.
        change_df = xr.Dataset({v+'_first':first[v],v+'_last':last[v],v+'_change':last[v]-first[v]}).to_dataframe().reset_index()
        paired = paired.merge(change_df[['latitude','longitude',v+'_first',v+'_last',v+'_change']],on=['latitude','longitude'],validate='one_to_one')
    paired.to_csv(OUT/'paired_changes.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'daily_statistics.csv',index=False)
    pd.DataFrame(changes).to_csv(OUT/'change_summary.csv',index=False)
    a, b = [x.swvl1.transpose('latitude','longitude').values for x in [first,last]]
    persistence = scores(b.ravel(),a.ravel(),weights.ravel())
    persistence['n_paired'] = int(np.isfinite(a+b).sum())
    fig, axes = plt.subplots(1,3,figsize=(13,4.8),constrained_layout=True)
    vmax = float(max(np.nanmax(a),np.nanmax(b)))
    for ax,values,date in zip(axes[:2],[a,b],ds.valid_time.values):
        im=ax.pcolormesh(ds.longitude,ds.latitude,values,vmin=0,vmax=vmax,cmap='YlGnBu',shading='nearest')
        ax.set_title(str(pd.Timestamp(date).date()))
        fig.colorbar(im,ax=ax,label='Soil water (m³/m³)',shrink=.75)
    delta=b-a
    limit=float(np.nanmax(abs(delta))) or 1
    im=axes[2].pcolormesh(ds.longitude,ds.latitude,delta,vmin=-limit,vmax=limit,cmap='BrBG',shading='nearest')
    axes[2].set_title('18 August minus 10 August')
    fig.colorbar(im,ax=axes[2],label='Soil-water change (m³/m³)',shrink=.75)
    for ax in axes:
        ax.set(xlabel='Longitude °E',ylabel='Latitude °N')
        ax.set_aspect(1/np.cos(np.deg2rad(float(ds.latitude.mean()))))
    fig.suptitle('Two reanalysis snapshots eight days apart\nShort-interval change does not establish a long-term trend',fontsize=14)
    fig.savefig(FIG/'moisture_change.png',dpi=220)
    plt.close(fig)
    return changes,persistence


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    ds = load_inputs()
    times = pd.DatetimeIndex(ds.valid_time.values)
    if list(times.strftime('%Y-%m-%d')) != ['2026-08-10','2026-08-18']:
        raise ValueError('This experiment is designed for the two supplied dates, 10 and 18 August 2026. Reassess new inputs.')
    snap = ds.squeeze(drop=True)
    if set(snap.dims) != {'valid_time','latitude', 'longitude'}:
        raise ValueError(f'Unexpected dimensions: {dict(snap.sizes)}')
    for coord in ('latitude', 'longitude'):
        a = snap[coord].values
        if a.ndim != 1:
            raise ValueError(f'{coord} must be one-dimensional')
        if len(np.unique(a)) != len(a) or not (np.all(np.diff(a)>0) or np.all(np.diff(a)<0)):
            raise ValueError(f'{coord} must be unique and monotonic')
    required = {'swvl1','stl1','stl2','u10','lai_lv'}
    if not required <= set(snap.data_vars):
        raise ValueError(f'Missing expected variables {required-set(snap.data_vars)}')
    if any(set(snap[v].dims) != {'valid_time','latitude','longitude'} for v in required):
        raise ValueError('Every model variable must occupy exactly the native spatial grid')
    snap['stl1_c'], snap['stl2_c'] = snap.stl1-273.15, snap.stl2-273.15
    snap['soil_temperature_difference_c'] = snap.stl1-snap.stl2
    snap.attrs['analysis_date'] = str(times[-1].date())
    lat, lon = snap.latitude.values, snap.longitude.values
    df = snap.to_dataframe().reset_index()
    if df.duplicated(['valid_time','latitude','longitude']).any():
        raise ValueError('Duplicate native cell-date rows in analysis table')
    df.to_csv(OUT/'native_grid_values.csv',index=False)
    roundoff = {}
    for v in ['swvl1','licd']:
        tiny = (snap[v]<0)&(snap[v]>=-1e-12)
        roundoff[v] = int(tiny.sum())
        snap[v] = snap[v].where(~tiny,0)
    df = snap.to_dataframe().reset_index()
    weights = np.cos(np.deg2rad(df.latitude.to_numpy()))
    stats = []
    for name in snap.data_vars:
        a = df[name].to_numpy()
        if not np.issubdtype(a.dtype, np.number):
            continue
        ok = np.isfinite(a)
        val = a[ok]
        if not len(val):
            continue
        stats.append({'variable': name, 'units': snap[name].attrs.get('units', 'degC' if name.endswith('_c') else ''),
                      'valid_cells': int(ok.sum()), 'total_cells': len(a),
                      'missing_percent': 100*(1-ok.mean()),
                      'zero_cells': int((val==0).sum()), 'min': float(val.min()),
                      'p05': float(np.quantile(val,.05)), 'median': float(np.median(val)),
                      'p95': float(np.quantile(val,.95)), 'max': float(val.max()),
                      'unweighted_mean': float(val.mean()),
                      'area_weighted_mean': weighted_mean(a, weights),
                      'spatial_std': float(val.std()), 'unique_count': int(np.unique(val).size)})
    stats = pd.DataFrame(stats)
    stats.to_csv(OUT/'spatial_descriptive_statistics.csv',index=False)
    complete = df.loc[np.isfinite(df[['swvl1']+ENV+GEO]).all(axis=1)].copy()
    if not complete.swvl1.between(0,1).all():
        raise ValueError('Soil-water values outside [0,1] require investigation')
    if len(complete) < 200:
        raise ValueError('Insufficient native spatial cells for declared block experiment')
    if complete.swvl1.nunique() < 2:
        raise ValueError('Constant soil-water target cannot support reconstruction assessment')
    corrs = []
    for v in ENV+['soil_temperature_difference_c']:
        a = complete[['swvl1',v]].dropna()
        corrs.append({'target':'swvl1','predictor':v,'n_pairs':len(a),
                      'pearson_r':a.corr().iloc[0,1],
                      'spearman_rho':float(spearmanr(a.swvl1,a[v]).statistic)})
    pd.DataFrame(corrs).to_csv(OUT/'moisture_associations.csv',index=False)
    metrics, predictions = spatial_experiments(complete,lat,lon)
    latest = snap.isel(valid_time=-1,drop=True)
    save_figures(latest,complete,metrics,predictions)
    changes,persistence = paired_analysis(snap)
    summary = {
        'analysis_type':'Two-date native-grid spatial reconstruction and paired-change pilot',
        'analysis_date':str(times[-1].date()), 'analysis_dates':list(times.strftime('%Y-%m-%d')),
        'n_dates':len(times), 'time_gap_days':int((times[-1]-times[0]).days),
        'grid_shape':[len(lat),len(lon)], 'total_grid_cells':len(lat)*len(lon),
        'complete_model_cells':len(complete[GEO].drop_duplicates()),
        'complete_model_rows':len(complete), 'roundoff_values_set_to_zero':roundoff,
        'change_summary':changes,'persistence_metrics':persistence,
        'latitude_range':[float(lat.min()),float(lat.max())],
        'longitude_range':[float(lon.min()),float(lon.max())],
        'latitude_step_degrees':float(np.median(np.abs(np.diff(lat)))),
        'longitude_step_degrees':float(np.median(np.abs(np.diff(lon)))),
        'time_shift':str(ds.valid_time.attrs.get('time_shift','unknown')),
        'seed':SEED, 'environment_features':ENV, 'coordinate_features':GEO,
        'descriptive_statistics':stats.to_dict('records'),
        'moisture_associations':corrs, 'model_metrics':metrics.to_dict('records'),
        'limitations':[
            'Two dates eight days apart cannot establish climate trends, climatological anomalies, or validated early-warning skill.',
            'No Sentinel-2 reflectance, field labels, other regions, land mask or lake fraction supplied.',
            'Predictors and soil-moisture target share a reanalysis model; reconstruction is not independent validation.',
            'Both dates of each cell are held out together. Training uses both dates; reconstruction is spatial, not prospective forecasting.',
            'Held-out blocks are adjacent within one region. Buffers reduce nearby overlap but do not eliminate spatial dependence.',
            'Lake fields are excluded from models because lake fraction is absent. LAI is prescribed climatology.',
            'Product lineage (ERA5 vs ERA5-Land) and hourly sampling need the original CDS request to confirm.',
            'All model settings are fixed; design comparisons are sensitivity checks, not selection of a deployment model.'
        ],
        'software':{name:importlib.metadata.version(name) for name in
                    ['numpy','pandas','xarray','netCDF4','scipy','scikit-learn','matplotlib']}}
    (OUT/'experiment_summary.json').write_text(json.dumps(safe(summary),indent=2,allow_nan=False),encoding='utf-8')
    (ROOT/'requirements-analysis.txt').write_text('\n'.join(f'{k}=={v}' for k,v in summary['software'].items())+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in
                     ['descriptive_statistics','model_metrics','moisture_associations','limitations']},indent=2))
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    if '--figures-only' in sys.argv:
        data = load_inputs().squeeze(drop=True)
        data['stl1_c'],data['stl2_c'] = data.stl1-273.15,data.stl2-273.15
        data['soil_temperature_difference_c'] = data.stl1-data.stl2
        for variable in ['swvl1','licd']:
            data[variable] = data[variable].where(~((data[variable]<0)&(data[variable]>=-1e-12)),0)
        data.attrs['analysis_date'] = str(pd.Timestamp(data.valid_time.values[-1]).date())
        frame = data.to_dataframe().reset_index()
        frame = frame.loc[np.isfinite(frame[['swvl1']+ENV+GEO]).all(axis=1)]
        save_figures(data.isel(valid_time=-1,drop=True),frame,pd.read_csv(OUT/'spatial_summary.csv'),
                     pd.read_csv(OUT/'spatial_predictions.csv',parse_dates=['valid_time']))
    else:
        main()

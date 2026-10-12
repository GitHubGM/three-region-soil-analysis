"""Build a local HTML report from verified experiment outputs using only stdlib.

Run after audit_inputs.py and run_experiments.py. No input data are modified.
The report uses local PNG figures and contains no remote assets or JavaScript.
"""
from __future__ import annotations

import csv
from html import escape
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
PRIMARY = 'quadrants_buffer_0.3'
MODELS = ['training_mean', 'ridge_environment', 'rf_geography',
          'rf_environment', 'rf_combined', 'gbr_combined']
MODEL_NAMES = {
    'training_mean': 'Training mean',
    'ridge_environment': 'Ridge environmental',
    'rf_geography': 'RF coordinates',
    'rf_environment': 'RF environmental',
    'rf_combined': 'RF combined',
    'gbr_combined': 'Gradient boosting combined',
}
DESIGNS = {
    PRIMARY: 'Four quadrants with 0.3° buffer',
    'quadrants_buffer_0.6': 'Four quadrants with 0.6° buffer',
    'longitude_strips_buffer_0.3': 'Four longitude strips with 0.3° buffer',
}
VARIABLE_NAMES = {
    'swvl1': ('Topsoil water, 0–7 cm', 'm³/m³'),
    'stl1': ('Soil temperature, 0–7 cm', 'K'),
    'stl2': ('Soil temperature, 7–28 cm', 'K'),
    'stl1_c': ('Soil temperature, 0–7 cm', '°C'),
    'stl2_c': ('Soil temperature, 7–28 cm', '°C'),
    'u10': ('Eastward wind at 10 m', 'm/s'),
    'lai_lv': ('Low vegetation LAI climatology', 'm²/m²'),
    'sd': ('Snow cover', 'as supplied'),
    'snowc': ('Snow cover', 'as supplied'),
    'lblt': ('Lake bottom temperature', 'K'),
    'licd': ('Lake ice depth', 'm'),
    'soil_temperature_difference_c': ('Topsoil minus deeper soil temperature', '°C'),
}
SOURCE_NAMES = {
    'era5_land_documentation': 'ECMWF ERA5-Land data documentation',
    'era5_land_paper': 'Muñoz-Sabater et al. (2021), ERA5-Land description',
    'era5_daily': 'Copernicus ERA5 daily statistics catalogue',
    'era5_land_daily': 'Copernicus ERA5-Land daily statistics catalogue',
    'daily_statistics_documentation': 'ECMWF daily statistics documentation',
    'wind_components': 'ECMWF wind component interpretation',
    'physiographic_fields_paper': 'Updated inland-water and physiographic fields study',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(name):
    with (OUT / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def h(value):
    return escape(str(value), quote=True)


def f(value, digits=3):
    if value is None or value == '':
        return '—'
    value = float(value)
    return f'{value:,.{digits}f}' if math.isfinite(value) else '—'


def integer(value):
    return f'{int(float(value)):,}'


def table(headers, rows, numeric_from=None):
    head = ''.join(f'<th scope="col">{h(x)}</th>' for x in headers)
    body = []
    for row in rows:
        cells = ''.join(
            f'<td{chr(32) + "class=\"num\"" if numeric_from is not None and i >= numeric_from else ""}>{h(cell)}</td>'
            for i, cell in enumerate(row)
        )
        body.append(f'<tr>{cells}</tr>')
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def link(href, text):
    return f'<a href="{h(href)}">{h(text)}</a>'


def figure(filename, caption):
    path = OUT / 'figures' / filename
    if not path.is_file():
        raise FileNotFoundError(f'Required figure not found: {path}')
    return (f'<figure><img src="figures/{h(filename)}" alt="{h(caption)}" loading="lazy">'
            f'<figcaption>{h(caption)}</figcaption></figure>')


def gain_text(model_rmse, baseline_rmse):
    difference = 100 * (1 - float(model_rmse) / float(baseline_rmse))
    direction = 'lower' if difference >= 0 else 'higher'
    return f'{abs(difference):.1f}% {direction} RMSE than the training-mean baseline'


def main():
    summary = read_json(OUT / 'experiment_summary.json')
    metadata = read_json(OUT / 'input_metadata.json')
    audit = read_json(ROOT / 'analysis' / 'manuscript_audit.json')
    semantics = read_json(ROOT / 'analysis' / 'variable_semantics.json')
    folds = read_csv('spatial_fold_metrics.csv')
    definitions = read_csv('spatial_fold_definitions.csv')
    inventory = read_csv('input_inventory.csv')
    metrics = summary['model_metrics']
    by_key = {(x['design'], x['model']): x for x in metrics}
    for design in DESIGNS:
        for name in MODELS:
            if (design, name) not in by_key:
                raise ValueError(f'Missing model result: {design}/{name}')
    if summary['n_dates'] != 2:
        raise ValueError('This report describes two supplied dates; reassess for different coverage.')
    rf = by_key[(PRIMARY, 'rf_combined')]
    gbr = by_key[(PRIMARY, 'gbr_combined')]
    baseline = by_key[(PRIMARY, 'training_mean')]
    stat = {x['variable']: x for x in summary['descriptive_statistics']}
    moisture = stat['swvl1']
    date = summary['analysis_date']
    first_date, last_date = summary['analysis_dates']
    date_label = f'{first_date} and {last_date}'
    n = summary['complete_model_cells']
    n_rows = summary['complete_model_rows']
    gap = summary['time_gap_days']
    changes = {x['variable']: x for x in summary['change_summary']}
    water_change = changes['swvl1']
    persistence = summary['persistence_metrics']
    roundoff = summary['roundoff_values_set_to_zero']
    lat = summary['latitude_range']
    lon = summary['longitude_range']
    fold_primary = [x for x in folds if x['design'] == PRIMARY and x['model'] == 'rf_combined']
    time_shift = summary.get('time_shift', 'unknown')
    shift_display = f'UTC offset recorded as {time_shift}' if time_shift != 'unknown' else 'UTC offset is not recorded in the summary'
    histories = [str(x.get('attributes', {}).get('history', '')) for x in metadata]
    history_note = ('The file history includes daily mean aggregation with a +5-hour shift.'
                    if any("'hours': 5" in x or '"hours": 5' in x for x in histories)
                    else shift_display + '.')
    result_paragraph = (
        f'The eight supplied NetCDF files contained two daily snapshots dated {date_label} on a '
        f'{summary["grid_shape"][0]} × {summary["grid_shape"][1]} grid '
        f'({summary["latitude_step_degrees"]:.2f}° latitude × '
        f'{summary["longitude_step_degrees"]:.2f}° longitude), comprising '
        f'{summary["total_grid_cells"]:,} spatial cells and {n_rows:,} complete cell-date records. '
        f'Area-weighted mean topsoil volumetric water content changed from '
        f'{water_change["mean_first"]:.4f} to {water_change["mean_last"]:.4f} m³/m³ '
        f'over the {gap}-day interval, a mean difference of {water_change["mean_change"]:.4f} m³/m³. '
        'In four spatially contiguous held-out quadrants separated from training cells by a '
        '0.3° rectangular buffer, with both observations at each site assigned to the same block, '
        'the combined random forest obtained pooled '
        f'RMSE {rf["rmse"]:.4f} m³/m³, MAE {rf["mae"]:.4f} m³/m³ and '
        f'R² {rf["r2"]:.3f}, compared with training-mean RMSE '
        f'{baseline["rmse"]:.4f} m³/m³ '
        f'({gain_text(rf["rmse"], baseline["rmse"])}). '
        f'Its block-specific RMSE ranged from {rf["fold_rmse_min"]:.4f} '
        f'to {rf["fold_rmse_max"]:.4f} m³/m³. '
        f'The lowest primary-design RMSE among the six fixed configurations was obtained by '
        f'combined gradient boosting ({gbr["rmse"]:.4f} m³/m³; R² {gbr["r2"]:.3f}); '
        f'its block R² ranged from {gbr["fold_r2_min"]:.3f} to {gbr["fold_r2_max"]:.3f}. '
        f'An untrained persistence benchmark carried the first soil-moisture field forward '
        f'{gap} days and obtained RMSE {persistence["rmse"]:.4f} m³/m³ on the second date. '
        'This single temporal pair cannot establish general forecast skill. '
        'The learned-model results measure same-day spatial reconstruction of soil moisture within the '
        'supplied reanalysis. The data do not support multi-year drying trends, independent '
        'field validation, salinity estimation or validation of the proposed CSI and SDRI.'
    )
    sections = []
    sections.append(f'''
      <section id="scope"><h2>What the experiment establishes</h2>
      <p>The local files contain two daily snapshots, <strong>{h(date_label)}</strong>, separated by {gap} days.
      They support a spatial reconstruction experiment and a paired comparison of topsoil water.
      The target and environmental predictors come from the same reanalysis system. Held-out grid cells measure
      how well this internal spatial pattern can be reconstructed from temperature, vegetation climatology,
      wind and location.</p>
      <div class="finding"><strong>Primary comparison</strong><p>The combined random forest obtained
      RMSE <strong>{f(rf['rmse'],4)} m³/m³</strong> and R² <strong>{f(rf['r2'])}</strong>, with
      {h(gain_text(rf['rmse'],baseline['rmse']))}. Individual quadrant RMSE ranged from
      {f(rf['fold_rmse_min'],4)} to {f(rf['fold_rmse_max'],4)} m³/m³.</p></div>
      <p>Combined gradient boosting had the lowest primary-design RMSE among the six fixed configurations:
      <strong>{f(gbr['rmse'],4)} m³/m³</strong>, with R² <strong>{f(gbr['r2'])}</strong> and
      {h(gain_text(gbr['rmse'],baseline['rmse']))}. This is the observed ranking within this experiment.
      Its quadrant R² ranged from {f(gbr['fold_r2_min'])} to {f(gbr['fold_r2_max'])}, so performance varied by location.</p>
      <p>Area-weighted mean soil water increased from {f(water_change['mean_first'],4)} to
      {f(water_change['mean_last'],4)} m³/m³ over the observed interval. That short comparison does not establish
      a long-term wetting or drying trend.</p>
      <p>Two snapshots cannot establish multi-year drying trends or robust early-warning skill. The folder contains no
      Sentinel-2 reflectance bands, independent field observations, or the other two regional datasets
      required by the manuscript. The full salinity, profile and climate-risk fusion remains untested.</p>
      </section>''')
    scope_rows = [
        ['Temporal coverage', f'{date_label}; {summary["n_dates"]} daily records, {gap} days apart'],
        ['Native grid', f'{summary["grid_shape"][0]} × {summary["grid_shape"][1]}; {summary["total_grid_cells"]:,} cells'],
        ['Grid spacing', f'{summary["latitude_step_degrees"]:.2f}° latitude × {summary["longitude_step_degrees"]:.2f}° longitude'],
        ['Extent', f'{lat[0]:.2f}–{lat[1]:.2f}° N; {lon[0]:.2f}–{lon[1]:.2f}° E'],
        ['Complete model sites', integer(n)],
        ['Complete model records', f'{n_rows:,} cell-date records; repeated sites stay in the same spatial block'],
        ['Daily aggregation', history_note],
        ['Product identity', 'ERA5 family; exact ERA5 versus ERA5-Land lineage needs the original CDS request'],
    ]
    inventory_rows = []
    for row in inventory:
        variable = row['variable']
        label = VARIABLE_NAMES.get(variable, (variable, row.get('units', '')))[0]
        inventory_rows.append([variable, label, row.get('units', ''), integer(row['n_valid']),
                               f(row['missing_percent'],2), integer(row['unique_count'])])
    sections.append('<section id="data"><h2>Data coverage and quality</h2>' + table(['Property', 'Verified coverage'], scope_rows)
                    + '<p>Coordinates are aligned exactly across files. No interpolation or synthetic observations are used. '
                    'The grid covers the supplied rectangle; there is no contemporary shoreline, land mask or lake fraction '
                    'to isolate exposed seabed, farmland or current water.</p>'
                    + table(['Variable', 'Interpretation', 'Stored units', 'Valid cell-date records', 'Missing %', 'Unique values'], inventory_rows, 3)
                    + f'<p>Quality control identified {roundoff.get("swvl1",0):,} tiny negative soil-moisture values '
                    f'and {roundoff.get("licd",0):,} tiny negative lake-ice values '
                    '(approximately 10⁻²⁰ and 10⁻¹⁹). Values between −10⁻¹² and zero are set to zero only for the analysis. '
                    'The original NetCDF files and <code>native_grid_values.csv</code> preserve the raw values. '
                    'This correction does not create observations or interpolate missing data.</p>'
                    + '<p class="small">Whole-grid lake temperature and ice summaries in the CSVs are numerical diagnostics only. '
                    'Without lake fraction, they cannot be interpreted as lake-only averages or current Aral Sea conditions. '
                    'The nominal grid spacing does not alone prove product lineage. Daily sampling frequency '
                    'and the exact product request require retrieval provenance. '
                    + link('input_metadata.json', 'Full input metadata and SHA-256 hashes') + ' · '
                    + link('input_inventory.csv', 'Input inventory CSV') + '</p></section>')
    descript_rows = []
    selected = ['swvl1','stl1_c','stl2_c','u10','lai_lv','soil_temperature_difference_c']
    for variable in selected:
        row = stat[variable]
        label, unit = VARIABLE_NAMES[variable]
        descript_rows.append([label, unit, f(row['area_weighted_mean'],4), f(row['median'],4),
                              f(row['min'],4), f(row['max'],4), f(row['p05'],4), f(row['p95'],4)])
    association_rows = [[VARIABLE_NAMES.get(x['predictor'],(x['predictor'],''))[0],
                         integer(x['n_pairs']),f(x['pearson_r']),f(x['spearman_rho'])]
                        for x in summary['moisture_associations']]
    sections.append('<section id="patterns"><h2>Patterns across the two supplied dates</h2>'
                    + table(['Variable', 'Unit', 'Area-weighted mean', 'Median', 'Minimum', 'Maximum', '5th percentile', '95th percentile'],descript_rows,2)
                    + '<p class="small">Mean values use cell-area weights proportional to cos(latitude). Medians, percentiles '
                    'and correlation coefficients are unweighted descriptions. This table pools both dates; the maps below '
                    'show only the latest date. Spatial variation is distinct from change between dates.</p>'
                    + figure('native_grid_maps.png',f'Figure 1. Native-grid fields for {date}. All map cells retain the original spatial support; no shoreline classification is inferred.')
                    + table(['Predictor associated with topsoil water','Pairs','Pearson r','Spearman rho'],association_rows,1)
                    + '<p>These associations are descriptive. Nearby cells are dependent and the variables share a model origin; '
                    'correlations do not establish environmental causes. No correlation significance tests or confidence intervals are reported.</p>'
                    + figure('spatial_correlations.png','Figure 2. Descriptive Spearman correlations pooled across complete native cell-date records. Both temporal repetition and spatial dependence remain.')
                    + '<p class="small">'+link('spatial_descriptive_statistics.csv','All descriptive statistics')+' · '
                    +link('moisture_associations.csv','Moisture associations CSV')+'</p></section>')
    change_rows = []
    for variable in ['swvl1','stl1_c','stl2_c','u10','lai_lv']:
        if variable not in changes:
            continue
        row = changes[variable]
        label, unit = VARIABLE_NAMES[variable]
        change_rows.append([label,unit,f(row['mean_first'],4),f(row['mean_last'],4),
                            f(row['mean_change'],4),f(row['median_change'],4),
                            f(row['decrease_percent'],1),f(row['increase_percent'],1)])
    persistence_rows = [['Carry first-date soil water forward',f(persistence['rmse'],4),
                         f(persistence['mae'],4),f(persistence['bias'],4),f(persistence['r2'])]]
    sections.append('<section id="change"><h2>Change over the observed interval</h2>'
                    + f'<p>Cells are paired at the same coordinates between {h(first_date)} and {h(last_date)}. '
                    f'All differences are second date minus first date. This is one {gap}-day comparison, '
                    'not an estimated long-term trend or an anomaly against a climatological baseline.</p>'
                    + table(['Variable','Unit',f'Mean {first_date}',f'Mean {last_date}',
                             'Mean change','Median change','Decreasing area %','Increasing area %'],change_rows,2)
                    + '<p class="small">Means, mean changes and increasing/decreasing percentages use cos(latitude) '
                    'area weights. Median changes are unweighted. Changes with magnitude at most 10⁻¹² are counted '
                    'as unchanged; this accounts for any remaining percentage. LAI differences reflect a prescribed climatology.</p>'
                    + figure('moisture_change.png',f'Figure 3. Topsoil water on {first_date} and {last_date}, with their paired difference. The maps show one observed interval in the supplied reanalysis.')
                    + '<h3>Persistence over one temporal pair</h3>'
                    + table(['Benchmark','RMSE m³/m³','MAE m³/m³','Bias m³/m³','R²'],persistence_rows,1)
                    + f'<p>This benchmark uses the {h(first_date)} soil-water value at each cell as its prediction for '
                    f'{h(last_date)}, without fitting a model. It quantifies the error of no change over this one '
                    f'{gap}-day interval. It is a different task from the spatial reconstruction models, which use '
                    'contemporaneous covariates. A single date pair cannot verify a learned temporal forecast or a '
                    'degradation early-warning system. High persistence R² mainly indicates that the spatial soil-water '
                    'pattern changes little across this pair; it cannot be compared directly with spatial holdout R².</p>'
                    + '<p class="small">'+link('daily_statistics.csv','Daily statistics')+' · '
                    +link('paired_changes.csv','Paired changes')+' · '+link('change_summary.csv','Change summary')+'</p></section>')
    model_rows = [
        ['Training mean','No predictors','Area-weighted training target mean'],
        ['Ridge environmental','Two soil temperatures, u10, LAI','StandardScaler then Ridge; alpha 10'],
        ['RF coordinates','Latitude and longitude','200 trees; minimum leaf 10; all features considered'],
        ['RF environmental','Two soil temperatures, u10, LAI','200 trees; minimum leaf 10; all features considered'],
        ['RF combined','Environmental features and coordinates','200 trees; minimum leaf 10; all features considered'],
        ['Gradient boosting combined','Environmental features and coordinates','150 stages; learning rate 0.03; depth 2; minimum leaf 10; squared error'],
    ]
    definition_rows = [[DESIGNS[x['design']],int(x['fold'])+1,integer(x['n_train']),integer(x['n_test']),integer(x['n_buffer_excluded'])]
                       for x in definitions]
    sections.append('<section id="methods"><h2>Experimental design</h2>'
                    + '<p><strong>Target:</strong> same-day volumetric soil water in the 0–7 cm layer. '
                    '<strong>Environmental predictors:</strong> soil temperature at 0–7 and 7–28 cm, signed eastward '
                    'wind at 10 m, and low vegetation leaf area index (LAI) climatology. Temperatures are converted '
                    'from kelvin to degrees Celsius. Latitude and longitude form the geographic feature set. '
                    'Both dates are pooled for fitting and scoring. A site’s two observations are always assigned '
                    'to the same held-out block, so a location cannot appear in both training and testing.</p>'
                    + '<p>Six model configurations are evaluated without tuning on the held-out cells. Geographic-only and '
                    'environmental-only random forests test the contribution of each feature group. Lake temperature and ice '
                    'are excluded because a lake fraction mask is missing. Snow is excluded from the declared feature set. '
                    'The temperature difference is described in correlations but is not an additional model predictor.</p>'
                    + table(['Model','Predictors','Fixed settings'],model_rows)
                    + f'<p>The seed is <code>{summary["seed"]}</code>. Training and metric weights are proportional to '
                    'cos(latitude). The ridge scaler is fitted only on its training fold; ridge regression receives '
                    'area weights. Tree fits receive area weights directly. Predictions are not clipped or calibrated afterward.</p>'
                    + '<h3>Separation and sensitivity checks</h3><p>The primary design partitions cells into four geographic '
                    'quadrants using the grid extent midpoints. Each block is held out once. Training cells within 0.3° '
                    'of the held rectangle are excluded. The separation uses max(distance in latitude, distance in longitude) '
                    'in degrees; it is a rectangular coordinate buffer, not a geodesic distance in kilometres. Two fixed '
                    'sensitivity designs widen the quadrant buffer to 0.6° or replace quadrants with four longitude strips '
                    'and a 0.3° buffer. The designs test sensitivity to separation and split geometry.</p>'
                    + table(['Design','Test block','Training records','Test records','Excluded records'],definition_rows,1)
                    + '<p>Counts in this table are cell-date records. Every complete cell-date record receives one '
                    'held-out prediction per model and design. Pooled RMSE, MAE, bias '
                    'and R² are recomputed over those predictions using area weights; they are not averages of fold scores. '
                    'Bias is prediction minus target. MSE skill is 1 − model MSE / training-mean baseline MSE; negative '
                    'values mean worse performance. Pooled R² uses the held-out target mean as its reference. Fold ranges '
                    'describe observed variation across these four blocks; they are not confidence intervals.</p>'
                    + '<p class="small">'+link('spatial_fold_definitions.csv','Exact fold definitions')+'</p></section>')
    metric_headers = ['Model','RMSE m³/m³','MAE m³/m³','Bias m³/m³','Pooled R²','MSE skill %','Block RMSE range']
    model_tables = []
    for design, label in DESIGNS.items():
        rows = []
        for name in MODELS:
            x = by_key[(design,name)]
            rows.append([MODEL_NAMES[name],f(x['rmse'],4),f(x['mae'],4),f(x['bias'],4),f(x['r2']),
                         f(100*x['mse_skill_vs_training_mean'],1),
                         f'{f(x["fold_rmse_min"],4)}–{f(x["fold_rmse_max"],4)}'])
        model_tables.append(f'<h3>{h(label)}'+(' · primary' if design == PRIMARY else ' · sensitivity')+'</h3>'+table(metric_headers,rows,1))
    rf_sensitivity = []
    for design,label in DESIGNS.items():
        x = by_key[(design,'rf_combined')]
        base = by_key[(design,'training_mean')]
        rf_sensitivity.append(f'<li>{h(label)}: RMSE {f(x["rmse"],4)} m³/m³; '
                              f'R² {f(x["r2"])}; {h(gain_text(x["rmse"],base["rmse"]))}.</li>')
    fold_rows = [[int(x['fold'])+1,integer(x['n_train']),integer(x['n_test']),f(x['rmse'],4),
                  f(x['mae'],4),f(x['bias'],4),f(x['r2'])] for x in fold_primary]
    sections.append('<section id="models"><h2>Reconstruction results</h2>'+''.join(model_tables)
                    + figure('spatial_model_comparison.png','Figure 4. All six fixed model configurations under the primary spatial partition and two sensitivity designs. Scores pool both dates while keeping repeated sites in the same block. Lower pooled RMSE indicates better reconstruction of the supplied reanalysis target.')
                    + '<h3>Combined random forest across validation designs</h3><ul>'+''.join(rf_sensitivity)+'</ul>'
                    + '<p>Differences between designs show dependence on geographic split and separation. They do not '
                    'select a deployment model or estimate transfer to a new region or year. The coordinate-only forest '
                    'quantifies how much spatial pattern can be reproduced from location alone; environmental-model gains '
                    'should be read alongside this reference.</p>'
                    + '<h3>Primary-design combined random forest by test block</h3>'
                    + table(['Test block','Training records','Test records','RMSE m³/m³','MAE m³/m³','Bias m³/m³','R²'],fold_rows,0)
                    + figure('spatial_validation.png',f'Figure 5. Primary quadrant assignments and held-out combined random forest residuals on {date} only. Residual colour limits are ±0.15 m³/m³; values outside this range share the endpoint colours. Tables pool both dates.')
                    + '<p class="small">'+link('spatial_summary.csv','Pooled model scores')+' · '
                    +link('spatial_fold_metrics.csv','All fold scores')+' · '+link('spatial_predictions.csv','All held-out predictions')+'</p></section>')
    module_rows = []
    for module in audit['modules']:
        module_rows.append([module['name'],'; '.join(module['required_inputs']),
                            '; '.join(module['unavailable_outputs'])])
    claim_rows = [[x['section'],x['claim'],x['status']] for x in audit['unverified_new_numerical_claims']]
    fixes = ''.join('<li>'+h(x['fix'])+'</li>' for x in audit['main_methodological_fixes'])
    sections.append('<section id="paper"><h2>What the manuscript still requires</h2>'
                    + '<p>The manuscript describes a 2019–2024 study in three regions. Its new numerical claims are '
                    'not reproduced by this two-date experiment. Previously published performance values are attributed '
                    'to source studies in the draft; they are not new measurements made here. The original Word file is unchanged.</p>'
                    + table(['Module','Required evidence','Outputs not established'],module_rows)
                    + '<details><summary>Audit of numerical claims already present in the draft</summary>'
                    + table(['Location','Draft claim','Evidence status'],claim_rows)
                    + '<p>'+h(audit['prior_study_claims']['internal_inconsistency'])+'</p></details>'
                    + '<details><summary>Methodological fixes before a full study</summary><ol>'+fixes+'</ol></details>'
                    + '<p class="small">'+link('../analysis/manuscript_audit.json','Complete machine-readable manuscript audit')+'</p></section>')
    limitations = ''.join('<li>'+h(x)+'</li>' for x in summary['limitations'])
    sections.append('<section id="interpretation"><h2>Limits of interpretation</h2><ul>'+limitations+'</ul>'
                    + '<p>The eastward wind component can be negative and is not scalar wind speed. A second component '
                    'would be required to reconstruct wind magnitude. LAI represents prescribed climatology, so this '
                    'pair of snapshots cannot identify recent vegetation recovery. Lake variables describe modelled lake physics '
                    'and cannot establish changing lake area. Zero ice or snow values are observations of those model '
                    'fields, not evidence of soil degradation.</p></section>'
                    + '<section id="paragraph"><h2>Results paragraph for the manuscript</h2>'
                    + '<p class="small">Suggested text for a clearly labelled preliminary experiment. Its scope should also '
                    'be reflected in the abstract, methods and conclusions before insertion.</p>'
                    + '<blockquote>'+h(result_paragraph)+'</blockquote></section>')
    python_path = str(Path(sys.executable)).replace('\\','/')
    command = (f"& '{python_path}' -m pip install --target .analysis_deps -r requirements-analysis.txt\n"
               f"& '{python_path}' scripts/audit_inputs.py\n"
               f"& '{python_path}' scripts/run_experiments.py\n"
               f"& '{python_path}' scripts/build_report.py")
    software = [[k,v] for k,v in summary['software'].items()]
    sources = ''.join('<li>'+link(url,SOURCE_NAMES.get(key,key))+'</li>'
                      for key,url in semantics['sources'].items())
    sections.append('<section id="reproduce"><h2>Reproduce the experiment</h2>'
                    + '<p>Run these PowerShell commands from the project folder. The installation step is needed only '
                    'when the local analysis dependencies are absent. Input files remain unchanged. Outputs are rebuilt '
                    'in <code>results/</code>, and dependencies are isolated in <code>.analysis_deps/</code>.</p>'
                    + '<pre>'+h(command)+'</pre>'
                    + '<p>'+link('../scripts/audit_inputs.py','Data audit script')+' · '
                    +link('../scripts/run_experiments.py','Experiment script')+' · '
                    +link('../scripts/build_report.py','Report generator')+' · '
                    +link('../requirements-analysis.txt','Pinned package versions')+' · '
                    +link('experiment_summary.json','Experiment summary JSON')+'</p>'
                    + '<p>Independent checks recomputed the reported errors, paired changes, spatial splits and '
                    'prediction coverage from the exported data, and confirmed the eight input file hashes. '
                    +link('verification.json','Verification results')+' · '
                    +link('../scripts/verify_results.py','Rerunnable verification script')+'</p>'
                    + table(['Package','Version'],software)
                    + '<p class="small">The input metadata file records SHA-256 hashes to identify the exact eight files. '
                    'No extra environmental dataset was downloaded for these experiments.</p></section>'
                    + '<section id="sources"><h2>Sources for interpreting the variables</h2><p>These primary sources '
                    'support variable meanings and product limitations. They do not verify the manuscript’s new numerical '
                    'results or replace the missing original CDS request.</p><ul>'+sources+'</ul>'
                    + '<p class="small">'+link('../analysis/variable_semantics.json','Source notes and variable semantics')+'</p></section>')
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Aral Sea soil moisture experiments</title>
    <style>
    :root{--ink:#18313b;--muted:#53666f;--line:#dbe4e7;--accent:#24677b;--paper:#fff;--wash:#f1f5f5}
    *{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font:16px/1.62 system-ui,-apple-system,Segoe UI,sans-serif}
    a{color:#185d79;text-underline-offset:3px}a:hover{color:#103948}main{max-width:1180px;margin:auto;padding:36px 34px 64px;background:var(--paper)}
    header{padding:14px 0 24px;border-bottom:2px solid var(--ink)}.eyebrow{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:var(--accent);font-weight:750}
    h1{font:600 clamp(30px,4vw,46px)/1.16 Georgia,serif;max-width:900px;margin:12px 0 16px}header .lead{font-size:19px;max-width:880px;color:var(--muted)}
    h2{font:600 28px/1.25 Georgia,serif;margin:0 0 18px}h3{font-size:18px;line-height:1.4;margin:26px 0 12px}
    p{margin:12px 0 18px}section{padding:34px 0;border-bottom:1px solid var(--line)}.small,figcaption{font-size:13px;color:var(--muted)}
    .facts{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0 10px}.fact{background:var(--wash);padding:12px 18px;border-radius:4px;min-width:175px}.fact strong{display:block;font-size:23px;color:var(--ink)}.fact span{font-size:12px;color:var(--muted)}
    nav{display:flex;flex-wrap:wrap;gap:9px 21px;padding-top:14px;font-size:13px}.finding{background:#eaf3f4;border-left:4px solid var(--accent);padding:17px 23px;margin:20px 0}.finding p{margin:7px 0 0}
    .table-wrap{overflow-x:auto;margin:18px 0 22px}table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.45}th{background:#eaf0f2;text-align:left;font-weight:650;padding:11px 12px;vertical-align:bottom}td{padding:10px 12px;vertical-align:top;border-bottom:1px solid var(--line)}tbody tr:nth-child(even){background:#fafcfc}.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
    figure{margin:28px 0}figure img{display:block;width:100%;height:auto;border:1px solid var(--line)}figcaption{padding:9px 2px}
    li{margin:9px 0}ul,ol{padding-left:24px}code{font-size:13px;background:#eef2f3;padding:2px 4px;border-radius:3px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f3;padding:20px;font-size:12px;line-height:1.8}
    blockquote{margin:20px 0;padding:20px 26px;background:#f5f7f8;border-left:3px solid #8499a2;font-family:Georgia,serif;line-height:1.8}details{border:1px solid var(--line);padding:15px 20px;margin:18px 0}summary{cursor:pointer;font-weight:650}footer{padding-top:26px;color:var(--muted);font-size:12px}
    @media(max-width:700px){main{padding:22px 18px}h2{font-size:25px}section{padding:28px 0}th,td{padding:9px}header .lead{font-size:17px}.fact{min-width:140px}.facts{gap:8px}}
    @media print{body{background:white}main{max-width:none;padding:0}nav,details{display:none}section{break-inside:auto}h2,h3{break-after:avoid}figure,tr{break-inside:avoid}a{color:inherit}pre{font-size:9px}}
    </style></head><body><main>'''
    html += (f'<header><div class="eyebrow">Experiment report · supplied Aral Sea data</div>'
             '<h1>Aral Sea soil moisture experiments</h1>'
             '<p class="lead">A reproducible spatial analysis of the supplied daily reanalysis files, '
             'with an audit of the evidence needed for the proposed soil degradation framework.</p>'
             f'<div class="facts"><div class="fact"><strong>2 dates</strong><span>{h(date_label)}</span></div>'
             f'<div class="fact"><strong>{n:,} sites</strong><span>{n_rows:,} cell-date records</span></div>'
             '<div class="fact"><strong>6 models</strong><span>Three spatial designs</span></div>'
             '<div class="fact"><strong>8 files</strong><span>Original inputs preserved</span></div></div>'
             '<nav aria-label="Report sections"><a href="#data">Data</a><a href="#patterns">Patterns</a><a href="#change">Observed change</a>'
             '<a href="#methods">Methods</a><a href="#models">Results</a><a href="#paper">Manuscript audit</a>'
             '<a href="#paragraph">Results paragraph</a><a href="#reproduce">Reproduce</a></nav></header>')
    html += ''.join(sections)
    html += '<footer>Generated from local experiment outputs. All model scores concern the supplied reanalysis target. Original manuscript and NetCDF inputs are preserved.</footer></main></body></html>'
    target = OUT / 'experiment_report.html'
    target.write_text(html, encoding='utf-8')
    print(f'Wrote {target} ({len(html):,} characters)')


if __name__ == '__main__':
    main()

# Machine learning across three regional domains

The current study compares soil-moisture regression models under random record
splits, random site splits, buffered spatial blocks and regional holdouts. It also
measures the effect of adding labelled target-region sites to source training data.


## Reproduce the three-domain study

Using the recorded Python environment and dependencies described below, run:

```powershell
python scripts/audit_multiregion.py
python scripts/run_multiregion.py
python scripts/verify_multiregion.py
python scripts/run_target_sensitivity.py
python scripts/verify_target_sensitivity.py
python scripts/make_multiregion_figures.py
python scripts/build_multiregion_manuscript.py
```

The main experiment evaluates eight fixed configurations on 39 splits. The
target-supervision experiment fits two models at three nested site budgets under
target-only and source-plus-target training on 12 buffered test quadrants.
All target budgets include both dates at each selected training site. Models
predict contemporaneous reanalysis soil water; this is a spatial reconstruction
task. Macro RMSE is the arithmetic mean of three regional RMSEs. It is not an
error pooled over all grid cells. Each regional score pools its held-out predictions
before computing RMSE, MAE, bias and R².

`main_predictions.csv.gz` and `adaptation_predictions.csv.gz` contain all
692,560 held-out predictions. The corresponding split membership files identify
training and test sample IDs and weights. `analysis_grid.csv.gz` resolves those
IDs to physical coordinates, dates and features; `raw_grid.csv.gz` preserves the
decoded input values. `verification.json` records independent checks without
importing the experiment implementation or refitting its estimators.

The post hoc sensitivity in `results/multiregion/sensitivity/` repeats the 5% and
20% target-site selections with two additional seeds, preserving the primary
outputs. It adds 192 fits and 251,840 predictions. The three-seed ranges describe
sample-selection sensitivity; they are not confidence intervals. Its separate
plan records why this analysis was added after inspecting the primary curves.
The primary experiment passed 1,258 independent checks and this sensitivity
analysis passed another 783, with no failed checks.

The experiment scripts overwrite their own generated outputs on rerun and never
edit the NetCDF files. Original and Aral-only manuscripts are retained separately.

## Earlier Aral-only study

This project reads the supplied manuscript and NetCDF files and runs a native-grid
soil-moisture reconstruction pilot for 10 and 18 August 2026. The original inputs
are unchanged.

## Results

Open `results/experiment_report.html` for the findings, figures, methods, and
limitations. The report includes a paragraph suitable for adapting into the paper.
Machine-readable results and complete held-out predictions are in `results/`.

The experiment predicts **same-day reanalysis soil moisture at withheld spatial
blocks**, pooling the two snapshots while holding both observations at a location
out together. Its scores do not validate measured soil salinity, the proposed Climate
Stress Index (CSI), the Soil Degradation Risk Index (SDRI), or future degradation.
The manuscript's previously stated numerical claims are audited separately in
`analysis/manuscript_audit.json`.

## Reproduce

Use Python 3.12. From this directory, install the recorded package versions into
the project-local analysis folder, then run:

```powershell
python -m pip install --target .analysis_deps -r requirements-analysis.txt
python scripts/audit_inputs.py
python scripts/run_experiments.py
python scripts/verify_results.py
python scripts/build_report.py
```

The scripts add `.analysis_deps` to their import path. On the original machine,
the interpreter used was:

```text
C:\Users\qorqy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

`run_experiments.py` deliberately stops if the files contain dates other than the
two supplied snapshots, unexpected dimensions, or too few valid cells. New data
need a fresh experimental design; these two dates are not a continuous time series.
Existing generated results are overwritten when the scripts are rerun.

## Experimental design

- Exact coordinate alignment, without interpolation or downscaling.
- Two snapshots eight days apart: 2,193 native sites and 4,386 site-date records.
- Observed target: supplied `swvl1`, in m³/m³.
- Environmental predictors: soil temperature at two depths, signed eastward wind,
  and prescribed low-vegetation leaf area index.
- Additional predictors in selected models: latitude and longitude.
- Six fixed models: training mean, environmental ridge regression, three random
  forest feature sets, and gradient boosting with all predictors.
- Four contiguous test blocks; neighboring training cells are excluded using a
  rectangular buffer in coordinate degrees. Both dates at a site remain in the
  same block. Spatial model training includes both dates.
- Primary design: four quadrants with 0.3° buffer. Sensitivity checks: quadrants
  with 0.6° buffer, and longitude strips with 0.3° buffer.
- Feature scaling uses training cells only. Model fitting and evaluation use
  cosine-latitude area weights. No test-based parameter tuning or output clipping.
- Reported metrics: RMSE, MAE, bias, pooled R², and MSE skill relative to the
  fold-specific training-mean baseline. Fold metrics show geographic variability.
- No p-values or confidence intervals are inferred from dependent grid cells.
- Paired changes and a persistence forecast (10 August moisture carried forward
  to 18 August) are reported separately. A single eight-day pair cannot establish
  climatological trends or reliable early-warning skill.
- Tiny negative soil-water and lake-ice values between -1e-12 and zero are treated
  as numerical roundoff and set to zero only in analysis. Raw values remain in the
  original NetCDF files and `native_grid_values.csv`.

## Output files

| File | Contents |
| --- | --- |
| `results/input_metadata.json` | Native metadata, coordinates, and input SHA-256 hashes |
| `results/input_inventory.csv` | Variables, units, counts, and basic quality checks |
| `results/native_grid_values.csv` | Original values and derived Celsius temperatures |
| `results/spatial_descriptive_statistics.csv` | Spatial summaries; latitude-weighted means |
| `results/daily_statistics.csv` | Separate spatial summaries for each date |
| `results/change_summary.csv` | Paired eight-day changes and area fractions changing |
| `results/paired_changes.csv` | Initial, final and difference values at every native site |
| `results/moisture_associations.csv` | Descriptive spatial correlations |
| `results/spatial_fold_definitions.csv` | Block bounds and train/test/buffer counts |
| `results/spatial_fold_metrics.csv` | Every model's result in every geographic fold |
| `results/spatial_summary.csv` | Pooled metrics and sensitivity comparisons |
| `results/spatial_predictions.csv` | All held-out predictions with native coordinates |
| `results/experiment_summary.json` | Results, limitations, settings, and package versions |
| `results/verification.json` | Independent checks of metrics, split geometry and unchanged inputs |
| `results/figures/` | Scientific figures at the supplied grid resolution |
| `analysis/manuscript_audit.json` | Requirements and claims that need supporting experiments |
| `analysis/variable_semantics.json` | Variable interpretation and primary documentation |

The supplied files do not include a contemporary land/water mask. Lake-variable
summaries are input diagnostics only. Neither nonmissing lake fields nor a
nonmissing soil-water value establishes current land/water cover. Original CDS
request metadata are needed to fully establish product lineage and hourly sampling.

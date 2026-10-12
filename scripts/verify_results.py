"""Verify exported experiment results independently using only Python's stdlib.

Run with any Python 3 interpreter from any working directory. This script reads
the supplied NetCDF hashes and exported CSVs; it does not import the experiment
code or its dependencies, retrain estimators, or independently decode NetCDF.
It writes results/verification.json and exits nonzero if any check fails.
"""
from __future__ import annotations

import bisect
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
TOLERANCE = 1e-7


def read_rows(name):
    with (OUT / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def float32(value):
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def normalized(variable, value):
    value = float32(value)
    return 0.0 if variable in ("swvl1", "licd") and -1e-12 <= value < 0 else value


def area_weight(latitude):
    return math.cos(math.radians(latitude))


def weighted_mean(values, weights):
    return math.fsum(value * weight for value, weight in zip(values, weights)) / math.fsum(weights)


def independent_scores(target, prediction, weights):
    center = weighted_mean(target, weights)
    errors = [pred - actual for actual, pred in zip(target, prediction)]
    mse = weighted_mean([error ** 2 for error in errors], weights)
    variance = weighted_mean([(actual - center) ** 2 for actual in target], weights)
    return {
        "rmse": math.sqrt(mse),
        "mae": weighted_mean([abs(error) for error in errors], weights),
        "bias": weighted_mean(errors, weights),
        "r2": 1 - mse / variance if variance else None,
    }


def close(actual, expected, tolerance=TOLERANCE):
    if actual is None or expected is None:
        return actual is None and expected is None
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def optional_number(value):
    return None if value == "" else float(value)


def cell_date_key(row):
    return row["valid_time"], float(row["latitude"]), float(row["longitude"])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_input_path(recorded_file):
    """Resolve both new root-relative paths and original basename-only records."""
    path = ROOT / recorded_file
    if path.is_file():
        return path
    if Path(recorded_file).name == recorded_file:
        relocated = ROOT / 'Aral' / recorded_file
        if relocated.is_file():
            return relocated
    raise FileNotFoundError(f'Audited input not found: {recorded_file}')


def main():
    checks = []

    def check(name, passed, details=None):
        checks.append({"check": name, "passed": bool(passed), "details": details})

    metadata = json.loads((OUT / "input_metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "experiment_summary.json").read_text(encoding="utf-8"))
    hashes = []
    for record in metadata:
        current = sha256(resolve_input_path(record["file"]))
        hashes.append({"file": record["file"], "recorded_sha256": record["sha256"],
                       "current_sha256": current, "unchanged": current == record["sha256"]})
    check("All eight supplied NetCDF SHA-256 hashes unchanged",
          len(hashes) == 8 and all(item["unchanged"] for item in hashes), hashes)

    raw = read_rows("native_grid_values.csv")
    raw_keys = [cell_date_key(row) for row in raw]
    raw_by_key = {cell_date_key(row): row for row in raw}
    dates = sorted({key[0] for key in raw_keys})
    latitudes = sorted({key[1] for key in raw_keys})
    longitudes = sorted({key[2] for key in raw_keys})
    sites = {(key[1], key[2]) for key in raw_keys}
    expected_keys = {(date, lat, lon) for date in dates for lat in latitudes for lon in longitudes}
    check("Native CSV covers both dates and every native site exactly once",
          len(raw) == len(raw_by_key) == 4386 and len(sites) == 2193
          and len(latitudes) == 43 and len(longitudes) == 51
          and set(raw_keys) == expected_keys and dates == ["2026-08-10", "2026-08-18"],
          {"rows": len(raw), "unique_sites": len(sites),
           "shape": [len(latitudes), len(longitudes)], "dates": dates})
    check("Native grid is uniformly spaced at 0.1 degrees",
          all(abs((values[index + 1] - values[index]) - 0.1) < 1e-10
              for values in (latitudes, longitudes) for index in range(len(values) - 1)))
    native_variables = ["u10", "lblt", "licd", "lai_lv", "snowc", "stl1", "stl2", "swvl1"]
    check("Raw native variables are finite",
          all(math.isfinite(float(row[variable])) for row in raw for variable in native_variables))
    roundoff_counts = {variable: sum(-1e-12 <= float32(row[variable]) < 0 for row in raw)
                      for variable in ("swvl1", "licd")}
    check("Raw CSV preserves tiny negative values and summary records normalization",
          roundoff_counts == summary["roundoff_values_set_to_zero"], roundoff_counts)

    models = {"training_mean", "ridge_environment", "rf_geography", "rf_environment",
              "rf_combined", "gbr_combined"}
    designs = {"quadrants_buffer_0.3": 0.3, "quadrants_buffer_0.6": 0.6,
               "longitude_strips_buffer_0.3": 0.3}
    predictions = read_rows("spatial_predictions.csv")
    fold_metrics = read_rows("spatial_fold_metrics.csv")
    summary_metrics = read_rows("spatial_summary.csv")
    definitions = read_rows("spatial_fold_definitions.csv")
    prediction_keys = [(row["design"], row["model"], cell_date_key(row)) for row in predictions]
    check("Prediction coverage has 78948 unique model-design-cell-date rows",
          len(predictions) == len(set(prediction_keys)) == 78948
          and {row["model"] for row in predictions} == models
          and {row["design"] for row in predictions} == set(designs),
          {"rows": len(predictions), "expected_rows": 6 * 3 * 2 * 2193,
           "duplicate_keys": len(prediction_keys) - len(set(prediction_keys))})
    by_group, by_fold = collections.defaultdict(list), collections.defaultdict(list)
    for row in predictions:
        by_group[row["design"], row["model"]].append(row)
        by_fold[row["design"], row["model"], int(row["fold"])].append(row)
    check("Every model-design predicts every native cell-date exactly once",
          len(by_group) == 18 and all(len(group) == 4386
          and {cell_date_key(row) for row in group} == expected_keys for group in by_group.values()))
    target_error = max(abs(float(row["swvl1"])
                       - normalized("swvl1", raw_by_key[cell_date_key(row)]["swvl1"])) for row in predictions)
    weight_error = max(abs(float(row["area_weight"])
                       - area_weight(float(row["latitude"]))) for row in predictions)
    check("Prediction targets match normalized raw CSV", target_error < 2e-8,
          {"max_absolute_csv_rounding_error": target_error})
    check("Prediction weights equal cosine latitude", weight_error < 1e-14,
          {"max_absolute_error": weight_error})

    latitude_midpoint = (latitudes[0] + latitudes[-1]) / 2
    longitude_midpoint = (longitudes[0] + longitudes[-1]) / 2
    strip_edges = [longitudes[0] + index * (longitudes[-1] - longitudes[0]) / 4
                   for index in range(1, 5)]

    def geographic_group(design, latitude, longitude):
        if design.startswith("quadrants"):
            return 2 * int(latitude >= latitude_midpoint) + int(longitude >= longitude_midpoint)
        return min(bisect.bisect_right(strip_edges, longitude), 3)

    mismatches = sum(int(row["fold"]) != geographic_group(row["design"], float(row["latitude"]),
                     float(row["longitude"])) for row in predictions)
    check("Every test prediction belongs to the declared geographic block", mismatches == 0,
          {"mismatches": mismatches})
    site_folds = collections.defaultdict(set)
    for row in predictions:
        site_folds[row["design"], row["model"], float(row["latitude"]), float(row["longitude"])].add(int(row["fold"]))
    check("Both dates of each site are held out together in every model-design",
          all(len(folds) == 1 for folds in site_folds.values()))

    definitions_by_key = {(row["design"], int(row["fold"])): row for row in definitions}
    split_results, baseline_error = [], 0.0
    for design, buffer in designs.items():
        for fold in range(4):
            test_sites = {site for site in sites if geographic_group(design, *site) == fold}
            lat_min, lat_max = min(site[0] for site in test_sites), max(site[0] for site in test_sites)
            lon_min, lon_max = min(site[1] for site in test_sites), max(site[1] for site in test_sites)

            def distance_to_test_rectangle(site):
                latitude, longitude = site
                return max(lat_min - latitude, latitude - lat_max,
                           lon_min - longitude, longitude - lon_max, 0)

            train_sites = {site for site in sites if site not in test_sites
                           and distance_to_test_rectangle(site) > buffer + 1e-8}
            train_rows = [row for row in raw if (float(row["latitude"]), float(row["longitude"])) in train_sites]
            recorded = definitions_by_key[design, fold]
            weights = [area_weight(float(row["latitude"])) for row in train_rows]
            target = [normalized("swvl1", row["swvl1"]) for row in train_rows]
            expected_baseline = weighted_mean(target, weights)
            error = max(abs(float(row["prediction"]) - expected_baseline)
                        for row in by_fold[design, "training_mean", fold])
            baseline_error = max(baseline_error, error)
            result = {"design": design, "fold": fold, "test_sites": len(test_sites),
                      "training_sites": len(train_sites), "test_rows": 2 * len(test_sites),
                      "training_rows": 2 * len(train_sites),
                      "excluded_buffer_rows": len(raw) - 2 * len(test_sites) - 2 * len(train_sites),
                      "minimum_training_distance_to_test_rectangle_degrees": min(map(distance_to_test_rectangle, train_sites)),
                      "same_site_train_test_overlap": len(test_sites & train_sites),
                      "baseline_max_error": error}
            result["counts_match"] = (int(recorded["n_test"]) == result["test_rows"]
                and int(recorded["n_train"]) == result["training_rows"]
                and int(recorded["n_buffer_excluded"]) == result["excluded_buffer_rows"])
            result["bounds_match"] = all(abs(float(recorded[key]) - value) < 1e-10 for key, value in
                [("test_lat_min", lat_min), ("test_lat_max", lat_max), ("test_lon_min", lon_min), ("test_lon_max", lon_max)])
            split_results.append(result)
    check("All twelve reconstructed splits have disjoint sites, correct buffers and recorded counts",
          len(definitions_by_key) == 12 and all(result["counts_match"] and result["bounds_match"]
          and result["same_site_train_test_overlap"] == 0
          and result["minimum_training_distance_to_test_rectangle_degrees"] > designs[result["design"]]
          for result in split_results), split_results)
    check("Training-mean predictions reproduce independently reconstructed buffered training sets",
          baseline_error < 1e-12, {"maximum_absolute_error": baseline_error})

    def prediction_scores(group):
        return independent_scores(
            [normalized("swvl1", raw_by_key[cell_date_key(row)]["swvl1"]) for row in group],
            [float(row["prediction"]) for row in group],
            [area_weight(float(row["latitude"])) for row in group])

    fold_differences = dict.fromkeys(["rmse", "mae", "bias", "r2"], 0.0)
    fold_pass = len(fold_metrics) == 72
    recomputed_folds = {}
    for row in fold_metrics:
        key = row["design"], row["model"], int(row["fold"])
        recomputed = prediction_scores(by_fold[key])
        recomputed_folds[key] = recomputed
        fold_pass &= (int(row["n_test"]) == len(by_fold[key])
                      and int(row["n_train"]) == int(definitions_by_key[key[0], key[2]]["n_train"]))
        for statistic, value in recomputed.items():
            recorded = optional_number(row[statistic])
            fold_pass &= close(value, recorded)
            if value is not None and recorded is not None:
                fold_differences[statistic] = max(fold_differences[statistic], abs(value - recorded))
    check("All 72 fold scores independently recomputed from raw targets and predictions", fold_pass,
          {"max_absolute_differences": fold_differences, "absolute_and_relative_tolerance": TOLERANCE})

    summary_differences = dict.fromkeys(["rmse", "mae", "bias", "r2", "mse_skill_vs_training_mean"], 0.0)
    summary_pass = len(summary_metrics) == 18
    for row in summary_metrics:
        key = row["design"], row["model"]
        recomputed = prediction_scores(by_group[key])
        baseline = prediction_scores(by_group[row["design"], "training_mean"])["rmse"]
        recomputed["mse_skill_vs_training_mean"] = 1 - (recomputed["rmse"] / baseline) ** 2
        summary_pass &= int(row["n_test"]) == len(by_group[key])
        for statistic, value in recomputed.items():
            recorded = optional_number(row[statistic])
            summary_pass &= close(value, recorded)
            if value is not None and recorded is not None:
                summary_differences[statistic] = max(summary_differences[statistic], abs(value - recorded))
        for statistic in ["rmse", "r2"]:
            values = [recomputed_folds[key[0], key[1], fold][statistic] for fold in range(4)]
            summary_pass &= (close(min(values), optional_number(row["fold_" + statistic + "_min"]))
                             and close(max(values), optional_number(row["fold_" + statistic + "_max"])))
    check("All 18 pooled scores, fold ranges and MSE skill independently recomputed", summary_pass,
          {"max_absolute_differences": summary_differences, "absolute_and_relative_tolerance": TOLERANCE})

    paired = read_rows("paired_changes.csv")
    pair_keys = [(float(row["latitude"]), float(row["longitude"])) for row in paired]
    pair_variables = ["swvl1", "stl1_c", "stl2_c", "u10", "lai_lv"]
    pair_difference = 0.0
    for row in paired:
        latitude, longitude = float(row["latitude"]), float(row["longitude"])
        first, last = raw_by_key[dates[0], latitude, longitude], raw_by_key[dates[1], latitude, longitude]
        for variable in pair_variables:
            start, end = normalized(variable, first[variable]), normalized(variable, last[variable])
            pair_difference = max(pair_difference, abs(float(row[variable + "_first"]) - start),
                                  abs(float(row[variable + "_last"]) - end),
                                  abs(float(row[variable + "_change"]) - float32(end - start)))
    check("Paired change CSV covers every site once and preserves date ordering",
          len(paired) == len(set(pair_keys)) == 2193 and set(pair_keys) == sites and pair_difference < 5e-6,
          {"paired_sites": len(paired), "max_absolute_value_error_due_to_csv_rounding": pair_difference,
           "tolerance": 5e-6})

    changes = read_rows("change_summary.csv")
    change_difference, change_pass = 0.0, len(changes) == 5
    ordered_sites = sorted(sites)
    weights = [area_weight(latitude) for latitude, longitude in ordered_sites]
    for row in changes:
        variable = row["variable"]
        starts = [normalized(variable, raw_by_key[dates[0], latitude, longitude][variable])
                  for latitude, longitude in ordered_sites]
        ends = [normalized(variable, raw_by_key[dates[1], latitude, longitude][variable])
                for latitude, longitude in ordered_sites]
        differences = [float32(end - start) for start, end in zip(starts, ends)]
        recomputed = {"mean_first": weighted_mean(starts, weights), "mean_last": weighted_mean(ends, weights),
                      "mean_change": weighted_mean(differences, weights), "median_change": statistics.median(differences),
                      "decrease_percent": 100 * weighted_mean([float(value < -1e-12) for value in differences], weights),
                      "increase_percent": 100 * weighted_mean([float(value > 1e-12) for value in differences], weights)}
        change_pass &= int(row["n_paired"]) == 2193
        for statistic, value in recomputed.items():
            difference = abs(float(row[statistic]) - value)
            change_difference = max(change_difference, difference)
            change_pass &= difference < TOLERANCE
    check("Paired change summaries independently recomputed with area weights", change_pass,
          {"max_absolute_difference": change_difference, "tolerance": TOLERANCE})

    target = [normalized("swvl1", raw_by_key[dates[1], latitude, longitude]["swvl1"])
              for latitude, longitude in ordered_sites]
    persistence_prediction = [normalized("swvl1", raw_by_key[dates[0], latitude, longitude]["swvl1"])
                              for latitude, longitude in ordered_sites]
    persistence = independent_scores(target, persistence_prediction, weights)
    persistence_differences = {key: abs(value - summary["persistence_metrics"][key])
                               for key, value in persistence.items()}
    check("Eight-day persistence benchmark independently recomputed",
          summary["persistence_metrics"]["n_paired"] == 2193
          and all(value < TOLERANCE for value in persistence_differences.values()),
          {"independent_metrics": persistence, "max_absolute_differences": persistence_differences})

    daily_rows = read_rows("daily_statistics.csv")
    daily_difference, daily_pass = 0.0, len(daily_rows) == 10
    for row in daily_rows:
        matching = [native_row for native_row in raw if native_row["valid_time"] == row["date"]]
        values = [normalized(row["variable"], native_row[row["variable"]]) for native_row in matching]
        weights = [area_weight(float(native_row["latitude"])) for native_row in matching]
        for statistic, value in {"area_weighted_mean": weighted_mean(values, weights),
                                 "min": min(values), "max": max(values)}.items():
            difference = abs(float(row[statistic]) - value)
            daily_difference = max(daily_difference, difference)
            daily_pass &= difference < TOLERANCE
    check("Ten daily variable summaries independently recomputed", daily_pass,
          {"max_absolute_difference": daily_difference, "tolerance": TOLERANCE})

    raw_audit = []
    for record in metadata:
        for variable, stats in record["variables"].items():
            values = [float32(row[variable]) for row in raw]
            raw_audit.append({"variable": variable, "min_matches": min(values) == stats["min"],
                              "max_matches": max(values) == stats["max"],
                              "unique_count_matches": len(set(values)) == stats["unique_count"],
                              "n_matches": len(values) == stats["n_total"]})
    check("Raw CSV extrema, unique counts and sample counts agree with NetCDF audit",
          all(all(item[key] for key in ["min_matches", "max_matches", "unique_count_matches", "n_matches"])
              for item in raw_audit), raw_audit)

    report = {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "method": "Independent Python standard-library recomputation from exported CSVs, plus SHA-256 checks against the pre-experiment NetCDF audit. No NumPy, pandas, xarray or scikit-learn used.",
        "interpretation": "Spatial validation pools two dates while holding each site out on both dates. Persistence is a separate descriptive eight-day benchmark.",
        "limits": [
            "Training membership is reconstructed from the audited script and corroborated by fold counts and exact training-mean predictions; estimators were not retrained.",
            "Individual raw NetCDF array elements were not independently decoded; raw CSV extrema/counts were checked against input_metadata.json and input file hashes were rechecked.",
            "Float32 CSV roundtrips were reconstructed using struct; numerical tolerances reflect CSV formatting and float32 arithmetic.",
            "Figure hashes identify the current files; this script performs no visual assessment of them.",
        ],
        "checks_passed": sum(item["passed"] for item in checks), "checks_total": len(checks),
        "script_sha256": sha256(ROOT / "scripts" / "run_experiments.py"),
        "verification_script_sha256": sha256(Path(__file__).resolve()),
        "figure_sha256": {path.name: sha256(path) for path in sorted((OUT / "figures").glob("*.png"))},
        "checks": checks,
    }
    (OUT / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["status", "checks_passed", "checks_total"]}, indent=2))
    if report["status"] != "passed":
        print(json.dumps([item for item in checks if not item["passed"]], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

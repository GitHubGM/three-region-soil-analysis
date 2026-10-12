"""Audit the supplied regional NetCDF files without changing inputs or old results.

Run with the bundled Python runtime. The project-local scientific dependencies
are imported from .analysis_deps. Output: analysis/multiregion_audit.json.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".analysis_deps"))

import numpy as np
import xarray as xr

REGIONS = ("Aral", "Atyrau", "Bozaigyr")
COORDS = ("valid_time", "latitude", "longitude")
# Decimal rounding only removes sub-nanodegree floating representation noise.
# Strict original-value equality is separately reported for every overlap.
KEY_DECIMALS = 8


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return safe(value[()])
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe(v) for v in value]
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit="ns")
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values):
    array = np.asarray(values)
    finite = array[np.isfinite(array)]
    unique = np.unique(finite)
    return {
        "n_total": int(array.size),
        "n_finite": int(finite.size),
        "n_missing_or_nonfinite": int(array.size - finite.size),
        "missing_percent": float(100 * (array.size - finite.size) / array.size),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "std_population": float(finite.std()) if finite.size else None,
        "finite_unique_count": int(unique.size),
        "constant_among_finite_values": bool(unique.size == 1),
        "constant_value": float(unique[0]) if unique.size == 1 else None,
        "all_missing": bool(not finite.size),
    }


def coord_record(coord):
    values = coord.values
    record = {
        "dimensions": list(coord.dims),
        "dtype": str(values.dtype),
        "attributes": safe(coord.attrs),
        "values": safe(values),
        "count": int(values.size),
        "unique_count": int(np.unique(values).size),
    }
    if values.ndim == 1 and np.issubdtype(values.dtype, np.number):
        delta = np.diff(values.astype(float))
        record.update({
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "median_signed_step": float(np.median(delta)) if delta.size else None,
            "minimum_signed_step": float(delta.min()) if delta.size else None,
            "maximum_signed_step": float(delta.max()) if delta.size else None,
            "regular_within_1e_9_degree": bool(
                not delta.size or np.allclose(delta, delta[0], rtol=0, atol=1e-9)
            ),
        })
    return record


def make_keys(coords):
    times = [np.datetime_as_string(x, unit="ns") for x in coords["valid_time"]]
    latitudes = [float(x) for x in coords["latitude"]]
    longitudes = [float(x) for x in coords["longitude"]]
    raw_sites = [(lat, lon) for lat in latitudes for lon in longitudes]
    sites = [(round(lat, KEY_DECIMALS), round(lon, KEY_DECIMALS)) for lat, lon in raw_sites]
    rows = [(date, lat, lon) for date in times for lat, lon in sites]
    return times, raw_sites, sites, rows


def main():
    output = ROOT / "analysis" / "multiregion_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    old_path = ROOT / "results" / "input_metadata.json"
    old_records = json.loads(old_path.read_text(encoding="utf-8"))
    old_hashes = {Path(entry["file"]).name: entry["sha256"] for entry in old_records}
    file_records = []
    region_records = {}
    caches = {}
    duplicate_file_hashes = defaultdict(list)
    all_site_members = defaultdict(set)
    all_row_members = defaultdict(set)

    for region in REGIONS:
        paths = sorted((ROOT / region).glob("*.nc"))
        if not paths:
            raise FileNotFoundError(f"No NetCDF files in {ROOT / region}")
        reference_coords = None
        variable_arrays = {}
        variable_summaries = {}
        alignment_records = []
        hashes_unchanged = []
        for path in paths:
            file_hash = sha256(path)
            relative = path.relative_to(ROOT).as_posix()
            duplicate_file_hashes[file_hash].append(relative)
            with xr.open_dataset(path, engine="netcdf4") as dataset:
                dataset.load()
                missing_coords = [name for name in COORDS if name not in dataset.coords]
                if missing_coords:
                    raise ValueError(f"{relative}: missing coordinates {missing_coords}")
                coords = {name: dataset[name].values.copy() for name in COORDS}
                if reference_coords is None:
                    reference_coords = coords
                    reference_file = relative
                coord_alignment = {
                    name: bool(np.array_equal(coords[name], reference_coords[name]))
                    for name in COORDS
                }
                aligned = all(coord_alignment.values())
                alignment_records.append({
                    "file": relative,
                    "reference_file": reference_file,
                    "exact_values_equal": coord_alignment,
                    "all_three_coordinates_exactly_aligned": aligned,
                })
                record = {
                    "region": region,
                    "file": relative,
                    "bytes": path.stat().st_size,
                    "sha256": file_hash,
                    "dimensions": dict(dataset.sizes),
                    "attributes": safe(dataset.attrs),
                    "coordinates": {name: coord_record(c) for name, c in dataset.coords.items()},
                    "variables": {},
                }
                for name, data in dataset.data_vars.items():
                    if name in variable_arrays:
                        raise ValueError(f"Duplicate variable {name} inside {region}")
                    summary = {
                        "file": relative,
                        "dimensions": list(data.dims),
                        "shape": list(data.shape),
                        "dtype": str(data.dtype),
                        "units": data.attrs.get("units"),
                        "long_name": data.attrs.get("long_name"),
                        "attributes": safe(data.attrs),
                        **stats(data.values),
                    }
                    if set(data.dims) == set(COORDS):
                        array = data.transpose(*COORDS).values.copy()
                        summary["per_date"] = [
                            {"date": np.datetime_as_string(date, unit="ns"), **stats(array[i])}
                            for i, date in enumerate(coords["valid_time"])
                        ]
                        if array.shape[0] >= 2:
                            change = array[-1] - array[0]
                            valid = np.isfinite(array[-1]) & np.isfinite(array[0])
                            summary["last_minus_first_date"] = {
                                **stats(change),
                                "finite_paired_sites": int(valid.sum()),
                                "sites_with_exactly_equal_values": int((valid & (array[-1] == array[0])).sum()),
                                "sites_with_changes_above_1e_12": int((valid & (np.abs(change) > 1e-12)).sum()),
                            }
                        if aligned:
                            variable_arrays[name] = array
                    record["variables"][name] = summary
                    variable_summaries[name] = summary
                if region == "Aral":
                    old_hash = old_hashes.get(path.name)
                    match = old_hash == file_hash
                    hashes_unchanged.append(match)
                    record["previous_Aral_audit"] = {
                        "metadata_file": old_path.relative_to(ROOT).as_posix(),
                        "previous_sha256": old_hash,
                        "identical_to_previous_file": match,
                    }
                file_records.append(record)

        dates, raw_sites, sites, rows = make_keys(reference_coords)
        for site in sites:
            all_site_members[site].add(region)
        for row in rows:
            all_row_members[row].add(region)
        grid = {
            "latitude_count": len(reference_coords["latitude"]),
            "longitude_count": len(reference_coords["longitude"]),
            "site_count": len(sites),
            "unique_nominal_site_count": len(set(sites)),
            "latitude_min": float(reference_coords["latitude"].min()),
            "latitude_max": float(reference_coords["latitude"].max()),
            "longitude_min": float(reference_coords["longitude"].min()),
            "longitude_max": float(reference_coords["longitude"].max()),
            "latitude_signed_step": float(np.median(np.diff(reference_coords["latitude"]))),
            "longitude_signed_step": float(np.median(np.diff(reference_coords["longitude"]))),
        }
        region_records[region] = {
            "folder": region,
            "region_name_provenance": "User-supplied folder label; geographic identity is defined by the coordinate bounds below",
            "file_count": len(paths),
            "dates": dates,
            "grid": grid,
            "site_date_rows": len(rows),
            "unique_site_date_rows": len(set(rows)),
            "duplicate_site_date_rows_within_region": len(rows) - len(set(rows)),
            "all_files_exactly_coordinate_aligned": all(x["all_three_coordinates_exactly_aligned"] for x in alignment_records),
            "coordinate_alignment": alignment_records,
            "constant_variables": [name for name, item in variable_summaries.items() if item["constant_among_finite_values"]],
            "variables_with_nonfinite_values": [name for name, item in variable_summaries.items() if item["n_missing_or_nonfinite"]],
            "variables": variable_summaries,
        }
        if region == "Aral":
            region_records[region]["all_files_identical_to_previous_audit"] = all(hashes_unchanged)
            region_records[region]["previous_file_count"] = len(old_records)
        caches[region] = {
            "coords": reference_coords, "dates": dates, "raw_sites": raw_sites,
            "sites": sites, "rows": rows, "arrays": variable_arrays,
            "row_index": {key: index for index, key in enumerate(rows)},
            "site_raw": dict(zip(sites, raw_sites)),
        }

    overlaps = []
    for left, right in combinations(REGIONS, 2):
        a, b = caches[left], caches[right]
        shared_sites = sorted(set(a["sites"]) & set(b["sites"]))
        shared_rows = sorted(set(a["rows"]) & set(b["rows"]))
        raw_shared_sites = set(a["raw_sites"]) & set(b["raw_sites"])
        shared_dates = sorted(set(a["dates"]) & set(b["dates"]))
        differences = [
            max(abs(x - y) for x, y in zip(a["site_raw"][key], b["site_raw"][key]))
            for key in shared_sites
        ]
        entry = {
            "regions": [left, right],
            "shared_dates": shared_dates,
            "strict_original_float_site_matches": len(raw_shared_sites),
            "nominal_shared_sites": len(shared_sites),
            "nominal_shared_site_date_rows": len(shared_rows),
            "maximum_coordinate_discrepancy_degrees_for_nominal_matches": max(differences) if differences else None,
            "shared_site_coordinates_lat_lon": shared_sites,
            "shared_bounds": {
                "latitude_min": min(x[0] for x in shared_sites),
                "latitude_max": max(x[0] for x in shared_sites),
                "longitude_min": min(x[1] for x in shared_sites),
                "longitude_max": max(x[1] for x in shared_sites),
            } if shared_sites else None,
            "overlapping_variables": {},
        }
        for name in sorted(set(a["arrays"]) & set(b["arrays"])):
            if not shared_rows:
                continue
            av = a["arrays"][name].ravel()[[a["row_index"][key] for key in shared_rows]]
            bv = b["arrays"][name].ravel()[[b["row_index"][key] for key in shared_rows]]
            finite = np.isfinite(av) & np.isfinite(bv)
            equal = (av == bv) | (np.isnan(av) & np.isnan(bv))
            delta = np.abs(av[finite] - bv[finite])
            mismatches = np.flatnonzero(~equal)
            entry["overlapping_variables"][name] = {
                "matched_site_date_count": len(shared_rows),
                "finite_pair_count": int(finite.sum()),
                "exact_equal_including_paired_nan_count": int(equal.sum()),
                "all_values_exactly_identical": bool(equal.all()),
                "maximum_absolute_difference": float(delta.max()) if delta.size else None,
                "mean_absolute_difference": float(delta.mean()) if delta.size else None,
                "mismatch_examples": [{
                    "site_date_key": shared_rows[i],
                    "left_value": safe(av[i]), "right_value": safe(bv[i]),
                } for i in mismatches[:5]],
            }
        overlaps.append(entry)

    total_rows = sum(len(caches[name]["rows"]) for name in REGIONS)
    total_sites = sum(len(caches[name]["sites"]) for name in REGIONS)
    checks = {
        "expected_24_files": len(file_records) == 24,
        "eight_files_per_region": all(x["file_count"] == 8 for x in region_records.values()),
        "same_variable_names_across_regions": len({tuple(sorted(x["variables"])) for x in region_records.values()}) == 1,
        "all_within_region_coordinates_exactly_aligned": all(x["all_files_exactly_coordinate_aligned"] for x in region_records.values()),
        "Aral_input_hashes_unchanged": region_records["Aral"]["all_files_identical_to_previous_audit"],
        "no_within_region_duplicate_site_date_rows": all(x["duplicate_site_date_rows_within_region"] == 0 for x in region_records.values()),
        "all_nominal_shared_coordinates_within_1e_9_degree": all(x["maximum_coordinate_discrepancy_degrees_for_nominal_matches"] is None or x["maximum_coordinate_discrepancy_degrees_for_nominal_matches"] < 1e-9 for x in overlaps),
    }
    audit = {
        "audit_generated_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(ROOT),
        "read_only_inputs": True,
        "software": {"numpy": np.__version__, "xarray": xr.__version__},
        "coordinate_matching": {
            "within_region": "Exact array equality on original latitude, longitude and valid_time coordinate values",
            "across_regions": "Both original-float equality and nominal grid matching are reported",
            "nominal_decimal_places": KEY_DECIMALS,
            "site_date_key_order": ["valid_time", "latitude", "longitude"],
            "warning": "Coordinates that differ by roundoff can still represent the same physical grid site; do not rely on raw-float tuple equality for train/test separation",
        },
        "totals": {
            "file_count": len(file_records),
            "region_count": len(REGIONS),
            "sum_of_region_sites": total_sites,
            "unique_nominal_sites": len(all_site_members),
            "redundant_site_memberships": total_sites - len(all_site_members),
            "sum_of_region_site_date_rows": total_rows,
            "unique_nominal_site_date_rows": len(all_row_members),
            "redundant_site_date_memberships": total_rows - len(all_row_members),
            "sites_present_in_multiple_regions": sum(len(v) > 1 for v in all_site_members.values()),
            "site_date_rows_present_in_multiple_regions": sum(len(v) > 1 for v in all_row_members.values()),
        },
        "checks": checks,
        "regions": region_records,
        "cross_region_overlaps": overlaps,
        "byte_identical_file_groups": [items for items in duplicate_file_hashes.values() if len(items) > 1],
        "files": file_records,
        "model_design_implications": [
            "Keep both dates at the same physical grid site in the same spatial validation group.",
            "Remove overlapping physical sites from training whenever a region is held out, before applying any additional spatial buffer.",
            "Use normalized coordinate keys only for identity matching; retain original coordinate values for analysis.",
            "Treat folder labels as supplied names and report the actual coordinate bounds.",
            "File-level constant variables provide no within-region predictor variation; decide feature eligibility using training data only.",
            "These files supply two dates, so temporal comparisons represent one eight-day interval.",
        ],
    }
    output.write_text(json.dumps(safe(audit), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    for name, record in region_records.items():
        grid = record["grid"]
        print(f"{name}: {grid['site_count']:,} sites; {record['site_date_rows']:,} rows; "
              f"{grid['latitude_min']:.1f}..{grid['latitude_max']:.1f} N; "
              f"{grid['longitude_min']:.1f}..{grid['longitude_max']:.1f} E; "
              f"dates {', '.join(x[:10] for x in record['dates'])}; "
              f"constants {','.join(record['constant_variables'])}; "
              f"nonfinite {','.join(record['variables_with_nonfinite_values']) or 'none'}")
    print(f"Unique sites {len(all_site_members):,}; unique site-date rows {len(all_row_members):,}; "
          f"redundant memberships {total_rows-len(all_row_members):,}")
    for overlap in overlaps:
        if overlap["nominal_shared_sites"]:
            print(f"Overlap {' / '.join(overlap['regions'])}: "
                  f"{overlap['nominal_shared_sites']} sites, {overlap['nominal_shared_site_date_rows']} rows; "
                  f"max coordinate noise {overlap['maximum_coordinate_discrepancy_degrees_for_nominal_matches']:.3g} deg")
            print("Exact overlapping value equality: " + ", ".join(
                f"{name}={item['all_values_exactly_identical']}"
                for name, item in overlap["overlapping_variables"].items()
            ))
    print("Checks: " + json.dumps(checks))
    if not all(checks.values()):
        raise SystemExit("One or more audit checks failed; inspect the JSON before modeling")


if __name__ == "__main__":
    main()

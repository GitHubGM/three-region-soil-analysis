"""Audit the supplied NetCDF inputs without changing them."""
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


def input_paths():
    """Find the original Aral files before or after their folder reorganization."""
    paths = sorted(ROOT.glob('*.nc'))
    if not paths:
        paths = sorted((ROOT / 'Aral').glob('*.nc'))
    if not paths:
        raise FileNotFoundError(f'No Aral NetCDF inputs found in {ROOT} or {ROOT / "Aral"}')
    return paths


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main():
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    records, rows = [], []
    for path in input_paths():
        relative_name = path.relative_to(ROOT).as_posix()
        with xr.open_dataset(path, engine='netcdf4') as ds:
            record = {'file': relative_name, 'bytes': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'dimensions': dict(ds.sizes), 'attributes': safe(ds.attrs),
                      'coordinates': {}, 'variables': {}}
            for name, coord in ds.coords.items():
                record['coordinates'][name] = {
                    'dimensions': list(coord.dims), 'attributes': safe(coord.attrs),
                    'values': [str(v) for v in coord.values.ravel()]}
            for name, da in ds.data_vars.items():
                a = da.values
                valid = np.isfinite(a)
                finite = a[valid]
                stats = {'dimensions': list(da.dims), 'shape': list(a.shape),
                         'attributes': safe(da.attrs), 'dtype': str(a.dtype),
                         'n_total': int(a.size), 'n_valid': int(valid.sum()),
                         'missing_percent': 100 * (1 - float(valid.mean())),
                         'min': float(finite.min()) if finite.size else None,
                         'max': float(finite.max()) if finite.size else None,
                         'mean': float(finite.mean()) if finite.size else None,
                         'std': float(finite.std()) if finite.size else None,
                         'unique_count': int(np.unique(finite).size)}
                record['variables'][name] = stats
                rows.append({'file': relative_name, 'variable': name,
                             'units': da.attrs.get('units', ''),
                             **{k: v for k, v in stats.items()
                                if k not in {'dimensions', 'attributes', 'shape'}}})
            records.append(record)
            print(relative_name)
            print(ds)
            print(json.dumps(record['variables'], indent=2))
    (out / 'input_metadata.json').write_text(json.dumps(safe(records), indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(out / 'input_inventory.csv', index=False)


if __name__ == '__main__':
    main()

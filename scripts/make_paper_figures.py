"""Render manuscript figures from the verified experiment tables.

The script does not refit models or change the input tables. Figures are drawn
at their intended six-inch print width, with labels at 9--10 points.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.analysis_deps'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'analysis' / 'mplconfig'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

OUT = ROOT / 'results' / 'manuscript_figures'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.linewidth': 0.7,
    'savefig.dpi': 300,
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def moisture_maps() -> None:
    data = pd.read_csv(ROOT / 'results' / 'paired_changes.csv')
    lat = np.sort(data.latitude.unique())
    lon = np.sort(data.longitude.unique())
    fields = {
        name: data.pivot(index='latitude', columns='longitude', values=name)
                  .reindex(index=lat, columns=lon).to_numpy()
        for name in ['swvl1_first', 'swvl1_last', 'swvl1_change']
    }
    assert len(lat) * len(lon) == len(data) == 2193
    vmax = np.ceil(max(fields['swvl1_first'].max(),
                       fields['swvl1_last'].max()) * 100) / 100
    change_max = np.ceil(np.abs(fields['swvl1_change']).max() * 1000) / 1000
    aspect = 1 / np.cos(np.deg2rad(lat.mean()))
    fig = plt.figure(figsize=(6, 6.4))
    fig.text(.5, .986, 'Topsoil moisture in the Aral Sea region',
             ha='center', va='top', fontsize=11, weight='bold')
    fig.text(.5, .953, 'Native grid · 2,193 locations · August 2026',
             ha='center', va='top', fontsize=9)
    positions = [[.115, .554, .31, .326], [.610, .554, .31, .326],
                 [.320, .070, .31, .295]]
    titles = ['a  10 August 2026', 'b  18 August 2026',
              'c  Change: 18 minus 10 August']
    meshes = []
    for position, (name, values), title in zip(positions, fields.items(), titles):
        ax = fig.add_axes(position)
        kwargs = ({'norm': TwoSlopeNorm(vcenter=0, vmin=-change_max,
                                        vmax=change_max), 'cmap': 'BrBG'}
                  if name == 'swvl1_change'
                  else {'vmin': 0, 'vmax': vmax, 'cmap': 'YlGnBu'})
        meshes.append(ax.pcolormesh(lon, lat, values, shading='nearest', **kwargs))
        ax.set_aspect(aspect)
        ax.set_title(title, pad=5, fontsize=9)
        ax.set_xlabel('Longitude (°E)', labelpad=3)
        ax.set_ylabel('Latitude (°N)', labelpad=3)
        ax.set_xticks([58, 60, 62])
        ax.set_yticks([44, 45, 46, 47])
        ax.tick_params(length=3, pad=3)
    cb = fig.colorbar(meshes[0], cax=fig.add_axes([.16, .465, .70, .018]),
                      orientation='horizontal', ticks=[0, .05, .10, .15, .20, .25])
    cb.set_label('Volumetric soil water, layer 1 (m³ m⁻³)', labelpad=3)
    cb.outline.set_linewidth(.5)
    cb = fig.colorbar(meshes[2], cax=fig.add_axes([.742, .066, .022, .285]))
    cb.set_label('Change in soil water (m³ m⁻³)', labelpad=7)
    cb.outline.set_linewidth(.5)
    fig.savefig(OUT / 'figure_1_moisture_maps.png')
    plt.close(fig)


def model_comparison() -> None:
    data = pd.read_csv(ROOT / 'results' / 'spatial_summary.csv')
    designs = [
        ('quadrants_buffer_0.3', 'Quadrants, 0.3° buffer (primary)', '#006b70', 'o'),
        ('quadrants_buffer_0.6', 'Quadrants, 0.6° buffer', '#cc7032', 's'),
        ('longitude_strips_buffer_0.3', 'Longitude strips, 0.3° buffer', '#575e86', '^'),
    ]
    models = ['training_mean', 'ridge_environment', 'rf_environment',
              'rf_geography', 'rf_combined', 'gbr_combined']
    labels = ['Training mean', 'Ridge (environment)', 'RF (environment)',
              'RF (coordinates)', 'RF (combined)', 'Gradient boosting\n(combined)']
    fig = plt.figure(figsize=(6, 5))
    fig.text(.5, .976, 'Same-day spatial reconstruction of topsoil moisture',
             ha='center', va='top', fontsize=10.5, weight='bold')
    fig.text(.5, .926, 'Two dates: 10 and 18 August 2026 · 4,386 held-out records',
             ha='center', va='top', fontsize=9)
    ax = fig.add_axes([.335, .15, .635, .56])
    handles = []
    for offset, (design, label, color, marker) in zip([-.20, 0, .20], designs):
        selected = data[data.design.eq(design)].set_index('model').loc[models]
        assert len(selected) == 6 and (selected.n_test == 4386).all()
        handle = ax.scatter(selected.rmse, np.arange(6) + offset, s=35,
                            marker=marker, color=color, label=label, zorder=3,
                            edgecolor='white', linewidth=.4)
        handles.append(handle)
    ax.set_yticks(np.arange(6), labels)
    ax.set_ylim(5.55, -.55)
    ax.set_xlim(.03, .076)
    ax.set_xticks([.03, .04, .05, .06, .07])
    ax.set_xlabel('Area-weighted pooled RMSE (m³ m⁻³)', labelpad=7)
    ax.grid(axis='x', color='#d8dce0', linewidth=.6)
    ax.tick_params(axis='y', length=0, pad=7)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.57, .88),
               frameon=False, ncol=1, labelspacing=.45, handletextpad=.7)
    fig.text(.5, .030, 'RF = random forest. Combined = environment + coordinates.',
             ha='center', fontsize=9)
    fig.savefig(OUT / 'figure_2_model_comparison.png')
    plt.close(fig)


if __name__ == '__main__':
    moisture_maps()
    model_comparison()
    for path in sorted(OUT.glob('*.png')):
        print(path)

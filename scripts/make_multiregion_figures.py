"""Create publication figures directly from the multiregion experiment CSVs.

Use --footprint-only or --main-only while model outputs are still being generated. The default
requires complete main and adaptation metric tables. Existing Aral figures and
all NetCDF inputs are left unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".analysis_deps"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "analysis" / "mplconfig_multiregion"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd

OUT = ROOT / "results" / "multiregion"
FIG = OUT / "figures"
REGIONS = ["Aral", "Atyrau", "Bozaigyr"]
LABELS = {"Aral": "Aral Sea", "Atyrau": "Atyrau", "Bozaigyr": "Northwestern"}
COLORS = {"Aral": "#2878A7", "Atyrau": "#C76F32", "Bozaigyr": "#27866D"}
MODELS = ["training_mean", "ridge_env", "svr_env", "rf_geo", "rf_env", "rf_env_geo", "gbr_env", "gbr_env_geo"]
MODEL_LABELS = {
    "training_mean": "Training mean", "ridge_env": "Ridge · E", "svr_env": "SVR · E",
    "rf_geo": "RF · G", "rf_env": "RF · E", "rf_env_geo": "RF · E + G",
    "gbr_env": "GBR · E", "gbr_env_geo": "GBR · E + G",
}
PROTOCOLS = ["random_rows", "random_sites", "buffered_quadrants", "leave_region_out"]
PROTOCOL_LABELS = ["Random\nrows", "Random\nsites", "Buffered\nquadrants", "Region\nheld out"]
UNITS = "m³/m³"
BUFFER = 0.3


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5, "figure.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "grid.linewidth": 0.5,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, name, sources, description):
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": digest(path), "dpi": 300,
        "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in sources},
        "description": description,
    }


def footprint(grid):
    sites = grid.drop_duplicates(["region", "site_id"])
    extents = {}
    for region in REGIONS:
        part = sites.loc[sites.region == region]
        extents[region] = dict(
            xmin=float(part.longitude.min()), xmax=float(part.longitude.max()),
            ymin=float(part.latitude.min()), ymax=float(part.latitude.max()), n=len(part),
        )
    a = sites.loc[sites.region == "Atyrau"].set_index("site_id")
    b = sites.loc[sites.region == "Bozaigyr"].set_index("site_id")
    shared = a.loc[a.index.intersection(b.index)]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.subplots_adjust(left=.12, right=.98, top=.89, bottom=.31)
    for region, box in extents.items():
        x, y = box["xmin"], box["ymin"]
        width, height = box["xmax"]-x, box["ymax"]-y
        ax.add_patch(Rectangle((x-BUFFER, y-BUFFER), width+2*BUFFER, height+2*BUFFER,
                               fill=False, edgecolor="#777777", linewidth=.85, linestyle=(0,(3,2))))
        ax.add_patch(Rectangle((x,y), width,height, facecolor=COLORS[region], alpha=.14,
                               edgecolor="none"))
        ax.add_patch(Rectangle((x,y), width,height, fill=False, edgecolor=COLORS[region], linewidth=1.4))
        center_y = (box["ymin"]+box["ymax"])/2
        if region == "Atyrau":
            center_y -= .40
        label=f"{LABELS[region]}\n{box['n']:,} sites"
        if region == "Bozaigyr":
            ax.annotate(label,xy=(box["xmax"],50.9),xytext=(54.1,51.1),
                        ha="left",va="center",fontsize=8.5,color="#16222A",
                        linespacing=1.3,arrowprops=dict(arrowstyle="-",color=COLORS[region],lw=.8))
        else:
            ax.text((box["xmin"]+box["xmax"])/2, center_y,label,
                    ha="center", va="center", fontsize=8.5,color="#16222A",linespacing=1.3)
    if len(shared):
        ax.add_patch(Rectangle((shared.longitude.min(),shared.latitude.min()),
                               shared.longitude.max()-shared.longitude.min(),
                               shared.latitude.max()-shared.latitude.min(),
                               fill=False,hatch="////",edgecolor="#4B4B4B",linewidth=.8))
    ax.set_xlim(46.2,64.0)
    ax.set_ylim(42.5,52.8)
    ax.set_xlabel("Longitude (°E)", labelpad=3)
    ax.set_ylabel("Latitude (°N)", labelpad=5)
    ax.set_xticks([48,52,56,60,64])
    ax.set_yticks([44,46,48,50,52])
    ax.grid(alpha=.25,zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Regional extraction domains", pad=8, fontweight="bold")
    handles = [
        Patch(facecolor="white",edgecolor="#4B4B4B",hatch="////",label=f"Shared grid: {len(shared):,} sites"),
        Line2D([0],[0],color="#777777",linestyle=(0,(3,2)),label="0.3° exclusion buffer"),
    ]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.55,.082),ncol=2,
               frameon=False,columnspacing=1.4,handlelength=2)
    fig.text(.55,.025,"Rectangles use grid-center coordinate bounds.",ha="center",fontsize=8.5)
    return save(fig,"figure_1_domains",[OUT/"analysis_grid.csv.gz"],
                "Exact coordinate extents of the three supplied grids; shaded overlap and example exclusion rectangles expanded by 0.3 degrees. No administrative boundary data used.")


def macro_heatmap(macro, regional):
    # Check that the supplied macro statistic has the explicitly stated meaning.
    means = regional.groupby(["protocol","model"]).rmse.mean()
    supplied = macro.set_index(["protocol","model"]).rmse
    if not np.allclose(supplied.sort_index(),means.sort_index(),rtol=1e-12,atol=1e-12):
        raise ValueError("Macro RMSE does not match arithmetic mean of regional RMSE")
    values = macro.pivot(index="model",columns="protocol",values="rmse").loc[MODELS,PROTOCOLS].to_numpy()
    fig, ax = plt.subplots(figsize=(6,4))
    fig.subplots_adjust(left=.255,right=.82,top=.83,bottom=.22)
    image = ax.imshow(values,cmap="YlGnBu",aspect="auto",vmin=float(values.min()),vmax=float(values.max()))
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            normalized = float(image.norm(values[y,x]))
            ax.text(x,y,f"{values[y,x]:.4f}",ha="center",va="center",fontsize=9,
                    color="white" if normalized>.57 else "#111111")
    ax.set_xticks(range(4),PROTOCOL_LABELS)
    ax.set_yticks(range(8),[MODEL_LABELS[m] for m in MODELS])
    ax.tick_params(axis="both",length=0,pad=6)
    ax.set_xticks(np.arange(-.5,4,1),minor=True)
    ax.set_yticks(np.arange(-.5,8,1),minor=True)
    ax.grid(which="minor",color="white",linewidth=1)
    ax.tick_params(which="minor",bottom=False,left=False)
    for spine in ax.spines.values():spine.set_visible(False)
    cax=fig.add_axes([.847,.24,.023,.57])
    cb=fig.colorbar(image,cax=cax)
    cb.ax.tick_params(labelsize=8.5)
    cb.set_label(f"RMSE ({UNITS})",fontsize=9,labelpad=5)
    fig.suptitle("Validation protocol and reconstruction error",y=.975,fontweight="bold")
    fig.text(.53,.91,"Arithmetic mean of three regional RMSE values",ha="center",fontsize=8.5)
    fig.text(.04,.065,"E: soil temperatures, wind and low-vegetation LAI.",fontsize=8.5)
    fig.text(.04,.022,"G: latitude and longitude. Smaller RMSE indicates lower error.",fontsize=8.5)
    return save(fig,"figure_2_macro_rmse",[OUT/"main_macro_metrics.csv",OUT/"main_region_metrics.csv"],
                "Eight configurations by four validation protocols. Every annotation is macro RMSE: the arithmetic mean of the three regional RMSEs, checked against the regional CSV.")


def transfer_dotplot(regional):
    transfer=regional.loc[regional.protocol=="leave_region_out"]
    fig,ax=plt.subplots(figsize=(6,4))
    fig.subplots_adjust(left=.255,right=.975,top=.76,bottom=.19)
    for index,region in enumerate(REGIONS):
        vals=transfer.loc[transfer.region==region].set_index("model").loc[MODELS,"r2"]
        ax.scatter(vals,np.arange(len(MODELS))+(index-1)*.20,s=28,color=COLORS[region],
                   label=LABELS[region],edgecolor="white",linewidth=.4,zorder=3)
    ax.set_yticks(np.arange(len(MODELS)),[MODEL_LABELS[m] for m in MODELS])
    ax.set_ylim(7.55,-.55)
    low,high=float(transfer.r2.min()),float(transfer.r2.max())
    span=high-low
    ax.set_xlim(low-.06*span,high+.10*span)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel("Regional R²",labelpad=6)
    ax.axvline(0,color="#454545",linestyle=(0,(3,2)),linewidth=.9,zorder=1)
    ax.grid(axis="x",color="#D9DEE3",alpha=.85)
    for row in np.arange(.5,7.5,1):ax.axhline(row,color="#EEF0F2",linewidth=.7,zorder=0)
    ax.tick_params(axis="y",length=0,pad=6)
    ax.spines["left"].set_visible(False)
    fig.suptitle("Transfer to a held-out region",y=.975,fontweight="bold")
    fig.text(.56,.915,"Overlapping sites and a 0.3° coordinate buffer excluded",ha="center",fontsize=8.5)
    fig.legend(*ax.get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.58,.89),
               ncol=3,frameon=False,columnspacing=1.1,handletextpad=.3)
    fig.text(.04,.035,"R² = 0 matches prediction by the target-region mean.",fontsize=8.5)
    return save(fig,"figure_3_region_transfer",[OUT/"main_region_metrics.csv"],
                "Eight leave-region-out configurations, each with separate cosine-latitude-weighted R squared for the three target regions. R squared uses the weighted target mean as its reference; the zero reference line is not the source-training-mean model. Source records inside target bounds plus the 0.3 degree buffer are excluded.")


def adaptation_curves(adaptation,regional):
    models=["rf_env","gbr_env"]
    fractions=[.05,.2,1.0]
    colors={"target_only":"#2878A7","source_plus_target":"#C76F32"}
    markers={"target_only":"o","source_plus_target":"s"}
    fig,axes=plt.subplots(2,3,figsize=(6,5.5),sharex=True,sharey="col")
    fig.subplots_adjust(left=.115,right=.98,top=.78,bottom=.16,wspace=.28,hspace=.27)
    transfer=regional.loc[regional.protocol=="leave_region_out"].set_index(["region","model"])
    for column,region in enumerate(REGIONS):
        area=adaptation.loc[adaptation.region==region]
        refvalues=[float(transfer.loc[(region,model),"rmse"]) for model in models]
        ymax=max(float(area.rmse.max()),max(refvalues))*1.10
        for row,model in enumerate(models):
            ax=axes[row,column]
            for setting in ["target_only","source_plus_target"]:
                values=area.loc[(area.model==model)&(area.setting==setting)].set_index("fraction").loc[fractions,"rmse"]
                ax.plot([5,20,100],values,color=colors[setting],marker=markers[setting],
                        markersize=4,linewidth=1.35)
            ax.axhline(refvalues[row],color="#545454",linestyle=(0,(4,2)),linewidth=1.15,zorder=1)
            ax.set_xlim(-2,107)
            ax.set_ylim(0,ymax)
            ax.set_xticks([5,20,100],["5","20","100"])
            ax.yaxis.set_major_locator(MaxNLocator(4,steps=[1,2,5,10]))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.02f"))
            ax.grid(axis="y",color="#D9DEE3",alpha=.75)
            ax.text(.035,.08,"RF · E" if row==0 else "GBR · E",transform=ax.transAxes,
                    ha="left",va="bottom",fontsize=8.5,fontweight="bold")
            if row==0:ax.set_title(LABELS[region],fontweight="bold",fontsize=9.5,pad=9)
            if column==0:ax.set_ylabel(f"RMSE ({UNITS})",fontsize=9)
    handles=[
        Line2D([0],[0],color=colors["target_only"],marker="o",markersize=4,label="Target only"),
        Line2D([0],[0],color=colors["source_plus_target"],marker="s",markersize=4,label="Source + target"),
        Line2D([0],[0],color="#545454",linestyle=(0,(4,2)),label="Source only: 0% target"),
    ]
    fig.suptitle("Target labels and regional transfer",y=.985,fontweight="bold")
    fig.text(.53,.939,"Environmental predictors with buffered target quadrants",ha="center",fontsize=8.5)
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.53,.912),ncol=2,
               frameon=False,columnspacing=1.2,handlelength=2.3)
    fig.text(.54,.077,"Eligible target training sites (%)",ha="center",fontsize=9)
    fig.text(.54,.029,"Shared y scales within each region; dashed lines show source-only transfer.",
             ha="center",fontsize=8.5)
    return save(fig,"figure_4_target_adaptation",[OUT/"adaptation_region_metrics.csv",OUT/"main_region_metrics.csv"],
                "RF and gradient boosting with environmental predictors. Curves use 5%, 20% and 100% of eligible target training sites under buffered quadrant evaluation; source-only dashed references use the corresponding leave-region-out model at zero target labels. Both model rows share each region's y-axis scale.")


def require_complete_tables(include_adaptation=True):
    macro=pd.read_csv(OUT/"main_macro_metrics.csv")
    regional=pd.read_csv(OUT/"main_region_metrics.csv")
    adaptation=pd.read_csv(OUT/"adaptation_region_metrics.csv") if include_adaptation else None
    if len(macro)!=32 or macro.duplicated(["protocol","model"]).any():
        raise ValueError("Expected 32 unique main macro metric rows")
    if len(regional)!=96 or regional.duplicated(["protocol","region","model"]).any():
        raise ValueError("Expected 96 unique main regional metric rows")
    if include_adaptation and (len(adaptation)!=36 or adaptation.duplicated(["region","model","fraction","setting"]).any()):
        raise ValueError("Expected 36 unique adaptation metric rows")
    for frame in [macro,regional]+([adaptation] if include_adaptation else []):
        if not np.isfinite(frame.rmse).all():raise ValueError("Nonfinite RMSE in figure inputs")
    return macro,regional,adaptation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--footprint-only",action="store_true")
    mode.add_argument("--main-only",action="store_true")
    args=parser.parse_args()
    configure()
    FIG.mkdir(parents=True,exist_ok=True)
    grid=pd.read_csv(OUT/"analysis_grid.csv.gz")
    if set(grid.region)!=set(REGIONS):raise ValueError("Unexpected region labels")
    records=[footprint(grid)]
    if not args.footprint_only:
        macro,regional,adaptation=require_complete_tables(include_adaptation=not args.main_only)
        records.extend([macro_heatmap(macro,regional),transfer_dotplot(regional)])
        if not args.main_only:records.append(adaptation_curves(adaptation,regional))
    manifest={"script_sha256":digest(Path(__file__)),"minimum_font_size_pt":8.5,
              "figures":records,"labels":LABELS}
    (FIG/"figure_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    for record in records:print(record["file"])


if __name__=="__main__":main()

#!/usr/bin/env python
"""
Plot Z-depth sweep waveforms from output/z_sweep.json.

Color axis: distance from anode (0 mm = anode face, 5 mm = cathode face),
jet colormap, shown as a colorbar in the 4th panel.

Produces two files:
  z_sweep.png       — full 0–10 µs window
  z_sweep_zoom.png  — zoomed 0–4 µs window

Usage:  python scripts/plot_z_sweep.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

REPO       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INJSON     = os.path.join(REPO, "output", "z_sweep.json")
DISPLAY_US = 8.0
PRE_US     = 2.0

# Detector geometry: anode at z = +2.5 mm, cathode at z = -2.5 mm
ANODE_Z_MM = 2.5
def depth_from_anode(z_mm):
    return ANODE_Z_MM - z_mm   # 0 = anode face, 5 = cathode face

plt.rcParams.update({
    "font.family":           "sans-serif",
    "font.size":             20,
    "axes.labelsize":        20,
    "axes.titlesize":        18,
    "xtick.labelsize":       17,
    "ytick.labelsize":       17,
    "legend.fontsize":       16,
    "legend.title_fontsize": 17,
    "axes.linewidth":        1.2,
    "xtick.major.width":     1.1,
    "ytick.major.width":     1.1,
})

with open(INJSON) as f:
    data = json.load(f)

events     = data["events"]
z_vals     = [e["z_mm"] for e in events]
depth_vals = [depth_from_anode(z) for z in z_vals]

cmap   = cm.jet
norm   = mcolors.Normalize(vmin=0, vmax=5)   # full 0–5 mm detector range
colors = [cmap(norm(d)) for d in depth_vals]

peak = max(
    abs(np.asarray(e["channels"]["anode"]["signal"])).max()
    for e in events if "anode" in e["channels"]
)

def get_trace(ev, key):
    ch = ev["channels"].get(key)
    if ch is None:
        return None, None
    t_us = np.asarray(ch["time_ns"]) / 1000.0
    q    = np.asarray(ch["signal"]) / peak
    if t_us[-1] < DISPLAY_US:
        t_us = np.append(t_us, DISPLAY_US)
        q    = np.append(q,    q[-1])
    t_plot = np.concatenate([[0.0], t_us + PRE_US])
    q_plot = np.concatenate([[0.0], q])
    return t_plot, q_plot

panels = [
    ("anode",    "Anode (0 V)"),
    ("cathode",  "Cathode (−600 V)"),
    ("steering", "Steering Electrode (−80 V)"),
]

def make_figure(xlim, outfile):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axs = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for ax, (key, title) in zip(axs[:3], panels):
        for iz, ev in enumerate(events):
            t_plot, q_plot = get_trace(ev, key)
            if t_plot is None:
                continue
            ax.plot(t_plot, q_plot, color=colors[iz], lw=2.0, alpha=0.88)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Charge (norm.)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.axvline(PRE_US, color="gray", lw=0.9, ls=":", alpha=0.6)

    # Bottom-right: colorbar from 0 to 5 mm
    ax_leg = axs[3]
    ax_leg.set_axis_off()

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_ax = inset_axes(ax_leg, width="35%", height="90%", loc="center")
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, 1, 2, 3, 4, 5])
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Interaction depth (mm)\n0 = anode,  5 = cathode",
                   fontsize=16, labelpad=12)

    # Mark the actual simulated depths as white tick lines on the colorbar
    for d in depth_vals:
        cbar.ax.axhline(d, color="white", lw=1.8, alpha=0.85)

    fig.suptitle("Z-Depth Sweep — 662 keV  |  SolidStateDetectors.jl", fontsize=22, y=1.01)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outfile}")

make_figure((0, PRE_US + DISPLAY_US), os.path.join(REPO, "output", "z_sweep.png"))
make_figure((0, 4.0),                  os.path.join(REPO, "output", "z_sweep_zoom.png"))

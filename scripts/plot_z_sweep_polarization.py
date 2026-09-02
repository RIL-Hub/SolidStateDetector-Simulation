#!/usr/bin/env python
"""
Electron vs hole contributions (free vs trapped) — all depths, first 500 ns after hit.

Layout: 4 signal panels + colorbar
  Panel 1: Electrons (free = dashed, trapped = solid)
  Panel 2: Holes     (free = dashed, trapped = solid)
  Panel 3: Hole trapping loss  (free - trapped, shaded)
  Panel 4: Total     (free = dashed, trapped = solid)
  Panel 5: Colorbar

Usage:  python scripts/plot_z_sweep_polarization.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

REPO    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INJSON  = os.path.join(REPO, "output", "eh_polarization.json")
OUTFILE = os.path.join(REPO, "output", "eh_polarization.png")

PRE_US  = 2.0
ZOOM_NS = 500.0

plt.rcParams.update({
    "font.size": 16, "axes.labelsize": 16, "axes.titlesize": 14,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.linewidth": 1.2,
})

with open(INJSON) as f:
    data = json.load(f)

depths_data = data["depths"]
energy      = data["energy_keV"]
depth_vals  = [d["depth_from_anode_mm"] for d in depths_data]
tau_h_us    = depths_data[0].get("tau_h_us", 3.0)
tau_e_us    = depths_data[0].get("tau_e_us", 5.0)

cmap = cm.jet
norm = mcolors.Normalize(vmin=0, vmax=5)
colors = [cmap(norm(dv)) for dv in depth_vals]

# Shared normalization: peak free total across all depths
peak = max(
    np.array(d["eh"]["q_total_free"]).max()
    for d in depths_data
)

x0 = PRE_US
x1 = PRE_US + ZOOM_NS / 1000.0

fig, axes = plt.subplots(1, 5, figsize=(22, 5.5), dpi=150,
                         gridspec_kw={"width_ratios": [3, 3, 3, 3, 1]})
ax_e, ax_h, ax_hl, ax_t, ax_cb = axes

def make_t_us(t_ns):
    return np.concatenate([[0.0], np.array(t_ns) / 1000.0 + PRE_US])

def make_q(q_arr):
    return np.concatenate([[0.0], np.array(q_arr) / peak])

for d, col in zip(depths_data, colors):
    eh = d["eh"]
    t_us = make_t_us(eh["time_ns"])

    q_ef = make_q(eh["q_electron_free"])
    q_et = make_q(eh["q_electron_trap"])
    q_hf = make_q(eh["q_hole_free"])
    q_ht = make_q(eh["q_hole_trap"])
    q_tf = make_q(eh["q_total_free"])
    q_tt = make_q(eh["q_total_trap"])

    # electrons: free=dashed, trapped=solid
    ax_e.plot(t_us, q_ef, color=col, lw=1.5, ls="--", alpha=0.7)
    ax_e.plot(t_us, q_et, color=col, lw=2.0, ls="-",  alpha=0.95)

    # holes: free=dashed, trapped=solid
    ax_h.plot(t_us, q_hf, color=col, lw=1.5, ls="--", alpha=0.7)
    ax_h.plot(t_us, q_ht, color=col, lw=2.0, ls="-",  alpha=0.95)

    # trapping loss (free - trapped), shaded
    loss = q_hf - q_ht
    ax_hl.plot(t_us, loss, color=col, lw=2.0, alpha=0.9)
    ax_hl.fill_between(t_us, 0, loss, color=col, alpha=0.15)

    # total: free=dashed, trapped=solid
    ax_t.plot(t_us, q_tf, color=col, lw=1.5, ls="--", alpha=0.7)
    ax_t.plot(t_us, q_tt, color=col, lw=2.0, ls="-",  alpha=0.95)

panel_configs = [
    (ax_e,  "Electrons",          False),
    (ax_h,  "Holes",              False),
    (ax_hl, "Hole trapping loss", True),
    (ax_t,  "Total  (e⁻ + h⁺)",  False),
]

for i, (ax, title, is_loss) in enumerate(panel_configs):
    ax.axvline(PRE_US, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax.axhline(0, color="black", lw=0.6, alpha=0.2)
    ax.set_xlim(x0, x1)
    if is_loss:
        ax.set_ylim(-0.005, 0.25)
    else:
        ax.set_ylim(-0.02, 1.08)
    ax.set_title(title)
    ax.set_xlabel("Time after hit (ns)")
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda x, _: f"{(x - PRE_US)*1000:.0f}"))
    ax.grid(True, alpha=0.2)
    if i > 0:
        ax.set_yticklabels([])

ax_e.set_ylabel("Induced charge (norm.)")

# Legend: dashed=free / solid=trapped
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], color="gray", lw=1.5, ls="--", label="Free (no trapping)"),
    Line2D([0],[0], color="gray", lw=2.0, ls="-",  label=f"Trapped (τ_h={tau_h_us:.0f} µs)"),
]
ax_e.legend(handles=legend_elements, fontsize=10, loc="lower right",
            framealpha=0.85, handlelength=2.0)

# Colorbar panel
ax_cb.set_axis_off()
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = inset_axes(ax_cb, width="60%", height="88%", loc="center")
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_ticks([0, 1, 2, 3, 4, 5])
cbar.ax.tick_params(labelsize=13)
cbar.set_label("Depth from anode (mm)", fontsize=13, labelpad=10)
for dv in depth_vals:
    cbar.ax.axhline(dv, color="white", lw=1.6, alpha=0.85)

fig.suptitle(
    f"e⁻ / h⁺ Contributions — First 500 ns  |  {energy:.0f} keV  |  τ_e={tau_e_us:.0f} µs, τ_h={tau_h_us:.0f} µs",
    fontsize=15, y=1.02)

fig.tight_layout()
fig.savefig(OUTFILE, dpi=150, bbox_inches="tight")
print(f"Saved -> {OUTFILE}")

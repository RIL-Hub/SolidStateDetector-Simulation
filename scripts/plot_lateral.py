#!/usr/bin/env python
"""
Plot the lateral charge-sharing scan past anode_20 (scripts/lateral_sweep.jl).

Two figures, both in volts (C_f = 1 pF, 662 keV -> 22.86 mV):
  lateral_waveforms_tiled.png : per-anode tiles (anode_18..22 + steering),
    each showing the 11 lateral positions overlaid, colored by interaction x.
  lateral_chargesharing.png   : plateau preamp output vs interaction x for each
    captured anode — the canonical charge-sharing S-curves. Vertical guides at
    the anode centers (x = -2,-1,0,+1,+2 mm).

Usage:  python scripts/plot_lateral.py [output/lateral/lateral_sweep.json]
"""
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

SIGN = -1.0
C_FEEDBACK_PF, W_PAIR_EV, E_KEV, E_CHG = 1.0, 4.64, 662.0, 1.602176634e-19
PRIMARY = "anode_20"


def anode_x(name):
    """anode_i lives at x = i - 20 (mm)."""
    m = re.match(r"anode_(\d+)", name)
    return int(m.group(1)) - 20 if m else None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "output/lateral/lateral_sweep.json"
    data = json.load(open(path))
    outdir = os.path.dirname(os.path.abspath(path))
    points = data["points"]
    captured = data["captured_anodes"]                  # e.g. ["anode_18", ..., "anode_22"]
    has_steer = "steering" in points[0]["waveforms"]
    channels = captured + (["steering"] if has_steer else [])

    # common preamp time grid (assume identical across points)
    p0 = points[0]["waveforms"][captured[0]]
    t_ns = np.asarray(p0["preamp_time_ns"], float); t_us = t_ns / 1000.0

    xs = np.array([p["x_mm"] for p in points])
    norm = Normalize(vmin=xs.min(), vmax=xs.max()); cmap = cm.coolwarm
    xcolor = [cmap(norm(x)) for x in xs]

    # per-point per-channel preamp waveform (SIGN inverted -> rising) on common grid
    def wf(p, name):
        w = p["waveforms"].get(name)
        if w is None:
            return None
        return SIGN * np.interp(t_ns,
            np.asarray(w["preamp_time_ns"], float),
            np.asarray(w["preamp_signal"], float))

    sig = {name: np.stack([wf(p, name) if wf(p, name) is not None else np.zeros_like(t_ns)
                           for p in points]) for name in channels}

    # volts calibration from PRIMARY plateau at its central position (x=0)
    tail = slice(int(0.9 * len(t_ns)), None)
    central_idx = int(np.argmin(np.abs(xs - 0.0)))
    Q = (E_KEV * 1e3 / W_PAIR_EV) * E_CHG
    V_full_mV = Q / (C_FEEDBACK_PF * 1e-12) * 1e3
    scale = V_full_mV / np.mean(sig[PRIMARY][central_idx, tail])

    # ── tiled waveforms: anodes stacked vertically on left, steering full-height
    #    on right, colorbar in its own thin column on the far right ──
    anode_channels = [c for c in channels if c.startswith("anode")]
    steer_channels = [c for c in channels if c == "steering"]
    n_an = len(anode_channels)

    figW = plt.figure(figsize=(13, 2.7 * max(n_an, 3) + 0.6))
    # width_ratios: [anodes col] [steering col] [colorbar col]
    if steer_channels:
        gs = figW.add_gridspec(n_an, 3, width_ratios=[1.0, 1.0, 0.05],
                               wspace=0.18, hspace=0.18,
                               left=0.07, right=0.93, top=0.93, bottom=0.08)
    else:
        gs = figW.add_gridspec(n_an, 2, width_ratios=[1.0, 0.05],
                               wspace=0.06, hspace=0.18,
                               left=0.07, right=0.93, top=0.93, bottom=0.08)
    ax_anodes = [figW.add_subplot(gs[i, 0]) for i in range(n_an)]
    for i in range(1, n_an):
        ax_anodes[i].sharex(ax_anodes[0])
        ax_anodes[i].sharey(ax_anodes[0])
    for name, ax in zip(anode_channels, ax_anodes):
        for j, x in enumerate(xs):
            ax.plot(t_us, sig[name][j] * scale, color=xcolor[j], lw=1.5, alpha=0.9)
        title = name + (f"  (x = {anode_x(name):+.0f} mm)" if anode_x(name) is not None else "")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("preamp (mV)")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1.5)
    ax_anodes[-1].set_xlabel("time (µs)")
    for ax in ax_anodes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    if steer_channels:
        ax_steer = figW.add_subplot(gs[:, 1], sharex=ax_anodes[0])
        for j, x in enumerate(xs):
            ax_steer.plot(t_us, sig["steering"][j] * scale, color=xcolor[j], lw=1.5, alpha=0.9)
        ax_steer.set_title("steering", fontsize=11)
        ax_steer.set_xlabel("time (µs)")
        ax_steer.set_ylabel("preamp (mV)")
        ax_steer.grid(alpha=0.3)
        cax = figW.add_subplot(gs[:, 2])
    else:
        cax = figW.add_subplot(gs[:, 1])

    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = figW.colorbar(sm, cax=cax)
    cb.set_label(f"interaction x (mm)   [primary {PRIMARY} at x = 0]")
    figW.suptitle(f"Lateral scan past {PRIMARY} (y=+2.5, z=0, r_cloud={data['cloud_radius_mm']} mm) — "
                  f"volts, C_f={C_FEEDBACK_PF} pF, 662 keV→{V_full_mV:.0f} mV", y=0.985, fontsize=12)
    out_tiled = os.path.join(outdir, "lateral_waveforms_tiled.png")
    figW.savefig(out_tiled, dpi=120); plt.close(figW)

    # ── charge-sharing summary: plateau vs x, with anode + steering geometry overlay ──
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MultipleLocator
    figS, axS = plt.subplots(figsize=(11, 6.5))
    palette = {"anode_18": "tab:purple", "anode_19": "tab:blue", "anode_20": "k",
               "anode_21": "tab:red", "anode_22": "tab:orange", "steering": "tab:green"}
    for name in channels:
        plateaus = np.mean(sig[name][:, tail], axis=1) * scale
        ls = "-" if name.startswith("anode") else ":"
        axS.plot(xs, plateaus, ls + "o", color=palette.get(name, "0.4"),
                 lw=2.0 if name == PRIMARY else 1.5, label=name)
    axS.set_xlabel("interaction x (mm)")
    axS.set_ylabel("preamp plateau (mV)")
    axS.set_title("Charge sharing past anode_20 — plateau vs lateral position\n"
                  "(anodes 100 µm gold · steering 400 µm grey)", fontsize=11)
    axS.grid(which="major", alpha=0.35, zorder=0)
    axS.grid(which="minor", alpha=0.15, zorder=0)
    axS.xaxis.set_major_locator(MultipleLocator(0.2))
    axS.xaxis.set_minor_locator(MultipleLocator(0.1))
    axS.legend(fontsize=9, loc="upper right")

    # Geometry overlay: thin gold rectangles for anodes (100 µm), grey for steering (400 µm).
    # Placed in a strip just below the data so labels stay legible. Drawn in data
    # coords so they sit at the correct x.
    ymin, ymax = axS.get_ylim()
    span = ymax - ymin
    strip_y, strip_h = ymin - 0.13 * span, 0.06 * span
    label_y = strip_y - 0.04 * span
    for nm, x in [("A19", -1.0), ("A20", 0.0), ("A21", +1.0)]:
        axS.add_patch(Rectangle((x - 0.05, strip_y), 0.10, strip_h,
                                facecolor="goldenrod", edgecolor="k", lw=0.5, zorder=4, clip_on=False))
        axS.text(x, label_y, nm, ha="center", va="top", fontsize=9, fontweight="bold")
    for x in [-0.5, +0.5]:
        axS.add_patch(Rectangle((x - 0.2, strip_y), 0.4, strip_h,
                                facecolor="0.6", edgecolor="k", lw=0.5, zorder=3, clip_on=False))
        axS.text(x, label_y, "S", ha="center", va="top", fontsize=8, color="0.3")

    axS.set_xlim(-1.1, +1.1)
    axS.set_ylim(strip_y - 0.10 * span, ymax)
    figS.tight_layout()
    out_share = os.path.join(outdir, "lateral_chargesharing.png")
    figS.savefig(out_share, dpi=130); plt.close(figS)

    print(f"Wrote {out_tiled}\nWrote {out_share}")


if __name__ == "__main__":
    main()

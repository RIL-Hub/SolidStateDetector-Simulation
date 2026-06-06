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
    captured = data["captured_anodes"]                  # ["anode_18", ..., "anode_22"]
    channels = captured + ["steering"]

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

    # ── tiled waveforms (one panel per channel) ──
    n_ax = len(channels)
    ncols = 3
    nrows = int(np.ceil(n_ax / ncols))
    figW, axsW = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows),
                              sharex=True, sharey=True)
    axsW = axsW.ravel()
    for ax, name in zip(axsW, channels):
        for i, x in enumerate(xs):
            ax.plot(t_us, sig[name][i] * scale, color=xcolor[i], lw=1.6, alpha=0.9)
        ax.set_title(name + (f"  (x = {anode_x(name):+.0f} mm)" if anode_x(name) is not None else ""),
                     fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1.5)
    for ax in axsW[n_ax:]:
        ax.set_visible(False)
    for ax in axsW[-ncols:]:
        ax.set_xlabel("time (µs)")
    for r in range(nrows):
        axsW[r * ncols].set_ylabel("preamp (mV)")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = figW.colorbar(sm, ax=axsW.tolist(), shrink=0.75)
    cb.set_label(f"interaction x (mm)   [primary {PRIMARY} at x = 0]")
    figW.suptitle(f"Lateral scan past {PRIMARY} (y=+2.5, z=0, r_cloud={data['cloud_radius_mm']} mm) — "
                  f"volts, C_f={C_FEEDBACK_PF} pF, 662 keV→{V_full_mV:.0f} mV", y=0.995)
    figW.tight_layout(rect=(0, 0, 0.92, 0.96))
    out_tiled = os.path.join(outdir, "lateral_waveforms_tiled.png")
    figW.savefig(out_tiled, dpi=120); plt.close(figW)

    # ── charge-sharing summary: plateau vs x ──
    figS, axS = plt.subplots(figsize=(11, 6))
    palette = {"anode_18": "tab:purple", "anode_19": "tab:blue", "anode_20": "k",
               "anode_21": "tab:red", "anode_22": "tab:orange", "steering": "tab:green"}
    for name in channels:
        plateaus = np.mean(sig[name][:, tail], axis=1) * scale
        ls = "-" if name.startswith("anode") else ":"
        axS.plot(xs, plateaus, ls + "o", color=palette.get(name, "0.4"),
                 lw=1.8 if name == PRIMARY else 1.4, label=name)
    for ai in range(-2, 3):
        axS.axvline(ai, color="0.7", lw=0.6, ls="--", zorder=0)
    axS.set_xlabel("interaction x (mm)")
    axS.set_ylabel("preamp plateau (mV)")
    axS.set_title("Charge sharing past anode_20 — plateau vs lateral position")
    axS.grid(alpha=0.3)
    axS.legend(fontsize=9, loc="best")
    figS.tight_layout()
    out_share = os.path.join(outdir, "lateral_chargesharing.png")
    figS.savefig(out_share, dpi=130); plt.close(figS)

    print(f"Wrote {out_tiled}\nWrote {out_share}")


if __name__ == "__main__":
    main()

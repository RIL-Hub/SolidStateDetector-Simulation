#!/usr/bin/env python
"""
Plot the collimator-sweep waveform results (output/sweep/gate_waveforms_*.json).

Produces 5 figures:
  A  sweep_overlay_tiled.png      3x3 tiles; per file, 5 primary waveforms overlaid (anode/cathode/steering)
  B  sweep_allcolor_nosteer.png   all events, anode+cathode, colored by collimator-x (no steering)
  C  sweep_allcolor_withsteer.png  same as B but including steering
  D  sweep_average_tiled.png       3x3 tiles; per file, averaged primary anode/cathode/steering
  E  sweep_average_allcolor.png    all 9 file-averages on one plot, color-scaled to collimator-x

All signals inverted (SIGN=-1): anode rising, cathode/steering falling.
Collimator-x maps to drift depth (ssd_z).

Usage:  python scripts/plot_sweep.py [output/sweep]
"""
import glob
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
from matplotlib.lines import Line2D

SIGN = -1.0
KIND_COLOR = {"anode": "tab:blue", "cathode": "tab:orange", "steering": "tab:green"}
KIND_LS = {"anode": "-", "cathode": "--", "steering": ":"}


def xval(tag):
    m = re.search(r"x([+-][\d.]+)mm", tag)
    return float(m.group(1)) if m else 0.0


def primary(ev, kind):
    name = {"anode": ev.get("collecting_anode"), "cathode": ev.get("collecting_cathode"),
            "steering": "steering"}[kind]
    return ev["waveforms"].get(name)


def load_files(d):
    files = sorted(glob.glob(os.path.join(d, "gate_waveforms_*.json")), key=lambda f: xval(f))
    data = []
    for f in files:
        with open(f) as fh:
            j = json.load(fh)
        data.append((xval(j.get("tag", os.path.basename(f))), j))
    return data


def preamp(ev, kind, t_grid):
    w = primary(ev, kind)
    if w is None:
        return None
    return SIGN * np.interp(t_grid, np.asarray(w["preamp_time_ns"], float),
                            np.asarray(w["preamp_signal"], float))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "output/sweep"
    data = load_files(d)
    if not data:
        print(f"No gate_waveforms_*.json in {d}", file=sys.stderr); sys.exit(1)
    kinds = ["anode", "cathode", "steering"]

    # common preamp time grid (µs) from the first available waveform
    ev0 = data[0][1]["events"][0]
    t_ns = np.asarray(primary(ev0, "anode")["preamp_time_ns"], float)
    t_us = t_ns / 1000.0

    xs = [x for x, _ in data]
    norm = Normalize(vmin=min(xs), vmax=max(xs))
    cmap = cm.coolwarm
    xcolor = {x: cmap(norm(x)) for x in xs}

    # ── A: overlay tiled (3x3) ──
    figA, axsA = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    for ax, (x, j) in zip(axsA.ravel(), data):
        for ev in j["events"]:
            for k in kinds:
                s = preamp(ev, k, t_ns)
                if s is not None:
                    ax.plot(t_us, s, color=KIND_COLOR[k], alpha=0.55, lw=1.3)
        ax.set_title(f"collimator x = {x:+.1f} mm", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axsA[-1]:
        ax.set_xlabel("time (µs)")
    for ax in axsA[:, 0]:
        ax.set_ylabel("preamp (a.u.)")
    figA.legend(handles=[Line2D([], [], color=KIND_COLOR[k], label=f"primary {k}") for k in kinds],
                loc="upper center", ncol=3, fontsize=10)
    figA.suptitle("Primary-channel waveforms overlaid — per collimator-x (5 events each)", y=0.995)
    figA.tight_layout(rect=(0, 0, 1, 0.97))
    figA.savefig(os.path.join(d, "sweep_overlay_tiled.png"), dpi=110); plt.close(figA)

    # ── B & C: all events, colored by collimator-x ──
    def all_color(kinds_use, fname, title):
        fig, ax = plt.subplots(figsize=(11, 6))
        for x, j in data:
            for ev in j["events"]:
                for k in kinds_use:
                    s = preamp(ev, k, t_ns)
                    if s is not None:
                        ax.plot(t_us, s, color=xcolor[x], ls=KIND_LS[k], alpha=0.45, lw=1.1)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax); cb.set_label("collimator x (mm)  → drift depth ssd_z")
        ax.legend(handles=[Line2D([], [], color="0.3", ls=KIND_LS[k], label=k) for k in kinds_use],
                  fontsize=9, loc="best")
        ax.set_xlabel("time (µs)"); ax.set_ylabel("preamp signal (a.u.)")
        ax.set_title(title); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(d, fname), dpi=120); plt.close(fig)

    all_color(["anode", "cathode"], "sweep_allcolor_nosteer.png",
              "All events colored by collimator-x — anode (solid) + cathode (dashed)")
    all_color(["anode", "cathode", "steering"], "sweep_allcolor_withsteer.png",
              "All events colored by collimator-x — anode + cathode + steering (dotted)")

    # ── per-file averages ──
    avg = {}  # x -> {kind: mean preamp}
    for x, j in data:
        avg[x] = {}
        for k in kinds:
            stack = [s for ev in j["events"] if (s := preamp(ev, k, t_ns)) is not None]
            avg[x][k] = np.mean(np.vstack(stack), axis=0) if stack else None

    # ── D: average tiled (3x3) ──
    figD, axsD = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    for ax, (x, _) in zip(axsD.ravel(), data):
        for k in kinds:
            if avg[x][k] is not None:
                ax.plot(t_us, avg[x][k], color=KIND_COLOR[k], lw=2.0, label=f"avg {k}")
        ax.set_title(f"collimator x = {x:+.1f} mm", fontsize=10); ax.grid(alpha=0.3)
    for ax in axsD[-1]:
        ax.set_xlabel("time (µs)")
    for ax in axsD[:, 0]:
        ax.set_ylabel("preamp (a.u.)")
    figD.legend(handles=[Line2D([], [], color=KIND_COLOR[k], label=f"avg {k}") for k in kinds],
                loc="upper center", ncol=3, fontsize=10)
    figD.suptitle("Average primary waveforms — per collimator-x", y=0.995)
    figD.tight_layout(rect=(0, 0, 1, 0.97))
    figD.savefig(os.path.join(d, "sweep_average_tiled.png"), dpi=110); plt.close(figD)

    # ── E: all averages, color-scaled by collimator-x (anode + cathode, no steering) ──
    kinds_E = ["anode", "cathode"]
    figE, axE = plt.subplots(figsize=(11, 6))
    for x in xs:
        for k in kinds_E:
            if avg[x][k] is not None:
                axE.plot(t_us, avg[x][k], color=xcolor[x], ls=KIND_LS[k], lw=1.8, alpha=0.9)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = figE.colorbar(sm, ax=axE); cb.set_label("collimator x (mm)  → drift depth ssd_z")
    axE.legend(handles=[Line2D([], [], color="0.3", ls=KIND_LS[k], label=k) for k in kinds_E],
               fontsize=9, loc="best")
    axE.set_xlabel("time (µs)")
    axE.set_ylabel("charge-sensitive preamp output (a.u. ∝ collected charge)")
    axE.set_title("File-averages, color-scaled to collimator-x — anode (solid) + cathode (dashed)")
    axE.grid(alpha=0.3)
    figE.tight_layout(); figE.savefig(os.path.join(d, "sweep_average_allcolor.png"), dpi=120); plt.close(figE)

    print("Wrote:")
    for f in ["sweep_overlay_tiled.png", "sweep_allcolor_nosteer.png", "sweep_allcolor_withsteer.png",
              "sweep_average_tiled.png", "sweep_average_allcolor.png"]:
        print(f"  {os.path.join(d, f)}")


if __name__ == "__main__":
    main()

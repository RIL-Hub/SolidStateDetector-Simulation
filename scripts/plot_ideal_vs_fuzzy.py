#!/usr/bin/env python
"""
Overlay the IDEAL (output/sweep_ideal) and FUZZY/real (output/sweep) sweeps on one
axis: ideal bold, fuzzy faint, anode solid / cathode dashed, colored by depth
(collimator-x), each aligned at its own anode 10% rising-edge crossing. Shared
volts scale (from the ideal anode plateau), so amplitudes are directly comparable.

Usage:  python scripts/plot_ideal_vs_fuzzy.py
"""
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FUZZY_DIR = os.path.join(REPO, "output", "sweep")
IDEAL_DIR = os.path.join(REPO, "output", "sweep_ideal")
OUT = os.path.join(FUZZY_DIR, "ideal_vs_fuzzy.png")

SIGN = -1.0
C_FEEDBACK_PF, W_PAIR_EV, E_KEV, E_CHG = 1.0, 4.64, 662.0, 1.602176634e-19
KIND_LS = {"anode": "-", "cathode": "--"}


def xval(tag):
    m = re.search(r"x([+-][\d.]+)mm", tag)
    return float(m.group(1)) if m else 0.0


def primary(ev, kind):
    name = ev.get("collecting_anode") if kind == "anode" else ev.get("collecting_cathode")
    return ev["waveforms"].get(name)


def load_avgs(d, t_ns):
    """tag-x -> {anode, cathode} averaged, inverted, on t_ns."""
    out = {}
    for f in glob.glob(os.path.join(d, "gate_waveforms_*.json")):
        j = json.load(open(f))
        x = xval(j.get("tag", f))
        out[x] = {}
        for k in ("anode", "cathode"):
            stk = [SIGN * np.interp(t_ns, np.asarray(primary(ev, k)["preamp_time_ns"], float),
                                    np.asarray(primary(ev, k)["preamp_signal"], float))
                   for ev in j["events"] if primary(ev, k) is not None]
            out[x][k] = np.mean(np.vstack(stk), axis=0) if stk else None
    return out


def t10(sig, t_us):
    peak = np.max(sig); thr = 0.10 * peak
    i = int(np.argmax(sig >= thr))
    if i == 0:
        return t_us[0]
    t0, t1, y0, y1 = t_us[i-1], t_us[i], sig[i-1], sig[i]
    return t0 + (thr - y0) * (t1 - t0) / (y1 - y0) if y1 != y0 else t1


def main():
    # common preamp time grid from an ideal file
    ev0 = json.load(open(sorted(glob.glob(os.path.join(IDEAL_DIR, "gate_waveforms_*.json")))[0]))["events"][0]
    t_ns = np.asarray(primary(ev0, "anode")["preamp_time_ns"], float)
    t_us = t_ns / 1000.0

    ideal = load_avgs(IDEAL_DIR, t_ns)
    fuzzy = load_avgs(FUZZY_DIR, t_ns)
    xs = sorted(set(ideal) & set(fuzzy))

    # shared volts scale from ideal anode plateau
    Q = (E_KEV * 1e3 / W_PAIR_EV) * E_CHG
    V_full_mV = Q / (C_FEEDBACK_PF * 1e-12) * 1e3
    tail = slice(int(0.9 * len(t_ns)), None)
    A = np.mean([np.mean(ideal[x]["anode"][tail]) for x in xs])
    scale = V_full_mV / A

    norm = Normalize(vmin=min(xs), vmax=max(xs)); cmap = cm.coolwarm
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for x in xs:
        col = cmap(norm(x))
        for src, lw, alpha in ((ideal, 2.6, 1.0), (fuzzy, 1.2, 0.45)):
            a = src[x]["anode"] * scale
            c = src[x]["cathode"] * scale
            dt = t10(a, t_us)
            ax.plot(t_us - dt, a, color=col, ls=KIND_LS["anode"], lw=lw, alpha=alpha)
            ax.plot(t_us - dt, c, color=col, ls=KIND_LS["cathode"], lw=lw, alpha=alpha)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlim(-0.3, 1.0)
    ax.set_xlabel("time after anode 10% crossing (µs)")
    ax.set_ylabel(f"preamp output (mV)   [C_f={C_FEEDBACK_PF} pF, 662 keV→{V_full_mV:.0f} mV]")
    ax.set_title("Ideal (bold) vs fuzzy/real (faint) — anode solid, cathode dashed, color = depth")
    ax.grid(alpha=0.3)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label("distance from slit center (mm)\n−2 mm = anode side   ·   +2 mm = cathode side")
    ax.legend(handles=[
        Line2D([], [], color="0.3", lw=2.6, label="ideal (point-like depth)"),
        Line2D([], [], color="0.3", lw=1.2, alpha=0.5, label="fuzzy (real GATE, 5 events)"),
        Line2D([], [], color="0.3", ls="-", label="anode"),
        Line2D([], [], color="0.3", ls="--", label="cathode"),
    ], fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"Wrote {OUT}  (V_full={V_full_mV:.2f} mV, scale={scale:.3e} mV/a.u.)")


if __name__ == "__main__":
    main()

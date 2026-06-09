#!/usr/bin/env python
"""
Per-position variation: 3x3 tiles, one per collimator slit position. Each panel
overlays the individual events' primary anode (solid) and cathode (dashed) in
volts, plus the bold per-position average, so the event-to-event spread at each
position is visible.

Usage:  python scripts/plot_sweep_variation.py [output/sweep]
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
from matplotlib.lines import Line2D

SIGN = -1.0
C_FEEDBACK_PF, W_PAIR_EV, E_KEV, E_CHG = 1.0, 4.64, 662.0, 1.602176634e-19
A_COL, C_COL = "tab:blue", "tab:orange"
XMAX_US = 2.0


def xval(tag):
    m = re.search(r"x([+-][\d.]+)mm", tag)
    return float(m.group(1)) if m else 0.0


def primary(ev, kind):
    name = ev.get("collecting_anode") if kind == "anode" else ev.get("collecting_cathode")
    return ev["waveforms"].get(name)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "output/sweep"
    files = sorted(glob.glob(os.path.join(d, "gate_waveforms_*.json")), key=xval)
    data = [(xval(json.load(open(f)).get("tag", f)), json.load(open(f))) for f in files]

    ev0 = data[0][1]["events"][0]
    t_ns = np.asarray(primary(ev0, "anode")["preamp_time_ns"], float)
    t_us = t_ns / 1000.0

    def wf(ev, k):
        w = primary(ev, k)
        return None if w is None else SIGN * np.interp(
            t_ns, np.asarray(w["preamp_time_ns"], float), np.asarray(w["preamp_signal"], float))

    # shared volts scale from anode plateau across all events
    Q = (E_KEV * 1e3 / W_PAIR_EV) * E_CHG
    V_full_mV = Q / (C_FEEDBACK_PF * 1e-12) * 1e3
    tail = slice(int(0.9 * len(t_ns)), None)
    plateaus = [np.mean(wf(ev, "anode")[tail]) for _, j in data for ev in j["events"] if wf(ev, "anode") is not None]
    scale = V_full_mV / np.mean(plateaus)

    fig, axs = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    for ax, (x, j) in zip(axs.ravel(), data):
        for k, col in (("anode", A_COL), ("cathode", C_COL)):
            stk = []
            for ev in j["events"]:
                s = wf(ev, k)
                if s is None:
                    continue
                stk.append(s)
                ax.plot(t_us, s * scale, color=col, lw=1.0, alpha=0.45,
                        ls="-" if k == "anode" else "--")
            if stk:
                ax.plot(t_us, np.mean(np.vstack(stk), axis=0) * scale, color=col, lw=2.6,
                        ls="-" if k == "anode" else "--")
        ax.set_title(f"slit x = {x:+.1f} mm  (depth {-x:+.1f} mm)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, XMAX_US)
    for ax in axs[-1]:
        ax.set_xlabel("time (µs)")
    for ax in axs[:, 0]:
        ax.set_ylabel("preamp (mV)")
    fig.legend(handles=[
        Line2D([], [], color=A_COL, lw=2.6, label="anode (bold=avg, faint=events)"),
        Line2D([], [], color=C_COL, lw=2.6, ls="--", label="cathode (bold=avg, faint=events)"),
    ], loc="upper center", ncol=2, fontsize=10)
    fig.suptitle(f"Per-position event variation — anode + cathode, volts "
                 f"[C_f={C_FEEDBACK_PF} pF, 662 keV→{V_full_mV:.0f} mV]", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(d, "sweep_variation_tiled.png")
    fig.savefig(out, dpi=110)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

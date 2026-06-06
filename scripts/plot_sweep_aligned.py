#!/usr/bin/env python
"""
Volts-calibrated version of the sweep average-all-color plot (anode + cathode,
no steering), in two forms:
  1) sweep_average_volts.png          — as-is, y in mV
  2) sweep_average_aligned_volts.png  — each file's average anode shifted so its
                                        10%-of-max rising-edge crossing lands at
                                        t=0; the paired cathode is shifted by the
                                        SAME Δt (preserving anode↔cathode timing).

Volts calibration (charge-sensitive preamp, V = Q_collected / C_f):
  full-energy 662 keV -> N = E/w_pair e-h pairs -> Q = N*e -> V_full = Q/C_f.
  The preamp output is linear, so we scale the (arbitrary-unit) waveforms by
  V_full / (mean anode full-collection plateau). Edit C_FEEDBACK_PF to recalibrate.

Usage:  python scripts/plot_sweep_aligned.py [output/sweep]
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
KIND_LS = {"anode": "-", "cathode": "--"}

# ── Volts calibration (edit these) ──
C_FEEDBACK_PF = 1.0     # charge-sensitive preamp feedback capacitance (pF)
W_PAIR_EV = 4.64        # CdZnTe electron-hole pair creation energy (eV)
E_PHOTOPEAK_KEV = 662.0
ELEM_CHARGE = 1.602176634e-19


def xval(tag):
    m = re.search(r"x([+-][\d.]+)mm", tag)
    return float(m.group(1)) if m else 0.0


def primary(ev, kind):
    name = ev.get("collecting_anode") if kind == "anode" else ev.get("collecting_cathode")
    return ev["waveforms"].get(name)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "output/sweep"
    files = sorted(glob.glob(os.path.join(d, "gate_waveforms_*.json")), key=xval)
    data = []
    for f in files:
        with open(f) as fh:
            j = json.load(fh)
        data.append((xval(j.get("tag", f)), j))

    kinds = ["anode", "cathode"]
    ev0 = data[0][1]["events"][0]
    t_ns = np.asarray(primary(ev0, "anode")["preamp_time_ns"], float)
    t_us = t_ns / 1000.0

    # per-file averages in arbitrary units (inverted: anode rising)
    avg = {}
    for x, j in data:
        avg[x] = {}
        for k in kinds:
            stack = [SIGN * np.interp(t_ns, np.asarray(primary(ev, k)["preamp_time_ns"], float),
                                      np.asarray(primary(ev, k)["preamp_signal"], float))
                     for ev in j["events"] if primary(ev, k) is not None]
            avg[x][k] = np.mean(np.vstack(stack), axis=0) if stack else None

    # ── Volts calibration ──
    Q_full = (E_PHOTOPEAK_KEV * 1e3 / W_PAIR_EV) * ELEM_CHARGE   # Coulombs
    V_full_mV = Q_full / (C_FEEDBACK_PF * 1e-12) * 1e3           # mV
    # reference full-collection anode plateau (mean of last 10% of samples, across files)
    tail = slice(int(0.9 * len(t_ns)), None)
    A = np.mean([np.mean(avg[x]["anode"][tail]) for x in avg])
    scale = V_full_mV / A                                       # mV per a.u.
    print(f"Q(662 keV) = {Q_full*1e15:.2f} fC;  C_f = {C_FEEDBACK_PF} pF;  "
          f"V_full = {V_full_mV:.2f} mV;  anode-plateau a.u. = {A:.3e};  scale = {scale:.3e} mV/a.u.")

    xs = [x for x, _ in data]
    norm = Normalize(vmin=min(xs), vmax=max(xs))
    cmap = cm.coolwarm
    xcolor = {x: cmap(norm(x)) for x in xs}

    def finish(ax, fig, title):
        sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax)
        cb.set_label("distance from slit center (mm)\n−2 mm = anode side   ·   +2 mm = cathode side")
        ax.legend(handles=[Line2D([], [], color="0.3", ls=KIND_LS[k], label=k) for k in kinds],
                  fontsize=9, loc="best")
        ax.set_ylabel(f"preamp output (mV)   [C_f={C_FEEDBACK_PF} pF, 662 keV→{V_full_mV:.0f} mV]")
        ax.set_title(title); ax.grid(alpha=0.3)
        fig.tight_layout()

    # ── 1) volts, not aligned ──
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    for x in xs:
        for k in kinds:
            ax1.plot(t_us, avg[x][k] * scale, color=xcolor[x], ls=KIND_LS[k], lw=1.8, alpha=0.9)
    ax1.set_xlabel("time (µs)")
    finish(ax1, fig1, "File-averages in volts — anode (solid) + cathode (dashed)")
    fig1.savefig(os.path.join(d, "sweep_average_volts.png"), dpi=120); plt.close(fig1)

    # ── 2) volts, anode 10%-aligned (cathode shifted by same Δt) ──
    def t10_crossing(sig_v):
        peak = np.max(sig_v)
        thr = 0.10 * peak
        i = np.argmax(sig_v >= thr)               # first index at/above threshold
        if i == 0:
            return t_us[0]
        # linear interp between i-1 and i
        t0, t1 = t_us[i - 1], t_us[i]
        y0, y1 = sig_v[i - 1], sig_v[i]
        return t0 + (thr - y0) * (t1 - t0) / (y1 - y0) if y1 != y0 else t1

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    for x in xs:
        a_v = avg[x]["anode"] * scale
        c_v = avg[x]["cathode"] * scale
        dt = t10_crossing(a_v)                    # shift so anode 10% crossing -> t=0
        ax2.plot(t_us - dt, a_v, color=xcolor[x], ls=KIND_LS["anode"], lw=1.8, alpha=0.9)
        ax2.plot(t_us - dt, c_v, color=xcolor[x], ls=KIND_LS["cathode"], lw=1.8, alpha=0.9)
    ax2.axvline(0, color="k", ls=":", lw=0.8)
    ax2.set_xlabel("time after anode 10% crossing (µs)")
    finish(ax2, fig2, "Anode-aligned at 10% rising edge (cathode shifted with its anode)")
    ax2.set_xlim(-0.3, 4.5)
    fig2.savefig(os.path.join(d, "sweep_average_aligned_volts.png"), dpi=120)
    # zoomed: -0.5 µs to +1 µs
    ax2.set_xlim(-0.5, 1.0)
    ax2.set_title("Anode-aligned at 10% rising edge — zoom (cathode shifted with its anode)")
    fig2.savefig(os.path.join(d, "sweep_average_aligned_volts_zoom1us.png"), dpi=120)
    plt.close(fig2)

    print("Wrote sweep_average_volts.png, sweep_average_aligned_volts.png, "
          "sweep_average_aligned_volts_zoom1us.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Average the PRIMARY (collecting) anode, primary cathode, and steering waveforms
across all events in a gate_waveforms_*.json file.

For each event we take:
  - the collecting anode   (ev["collecting_anode"])
  - the collecting cathode (ev["collecting_cathode"])
  - the steering electrode ("steering")
resample each onto a common time base, and average element-wise across events.

Outputs:
  output/gate_average_waveform.png   (preamp-shaped + raw current, averaged)
  output/gate_average_waveform.csv   (averaged preamp signals)

Usage:
    python scripts/average_waveform.py output/gate_waveforms_x+0.0mm.json
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_MAX_NS = 1500.0  # common grid extent for the (variable-length) raw current
SIGN = -1.0          # invert all signals: anode -> rising, cathode/steering -> falling


def primary(ev, kind):
    name = {"anode": ev.get("collecting_anode"),
            "cathode": ev.get("collecting_cathode"),
            "steering": "steering"}[kind]
    return name, ev["waveforms"].get(name)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    with open(path) as f:
        data = json.load(f)
    events = data["events"]
    outdir = os.path.dirname(os.path.abspath(path))

    kinds = ["anode", "cathode", "steering"]
    colors = {"anode": "tab:blue", "cathode": "tab:orange", "steering": "tab:green"}

    # Common time bases
    t_pre = np.asarray(events[0]["waveforms"][events[0]["collecting_anode"]]["preamp_time_ns"], float)
    t_raw = np.arange(0.0, RAW_MAX_NS, 1.0)

    pre_stack = {k: [] for k in kinds}   # per-event preamp signals on t_pre
    raw_stack = {k: [] for k in kinds}   # per-event raw current on t_raw

    for ev in events:
        for k in kinds:
            name, w = primary(ev, k)
            if w is None:
                continue
            pre_stack[k].append(SIGN * np.interp(t_pre, np.asarray(w["preamp_time_ns"], float),
                                                 np.asarray(w["preamp_signal"], float)))
            raw_stack[k].append(SIGN * np.interp(t_raw, np.asarray(w["raw_time_ns"], float),
                                                 np.asarray(w["raw_current"], float),
                                                 left=0.0, right=0.0))

    pre_avg = {k: np.mean(np.vstack(pre_stack[k]), axis=0) for k in kinds}
    raw_avg = {k: np.mean(np.vstack(raw_stack[k]), axis=0) for k in kinds}
    n = {k: len(pre_stack[k]) for k in kinds}

    # ── Plot (preamp-shaped output only) ──
    fig, axL = plt.subplots(1, 1, figsize=(9, 5.5))
    for k in kinds:
        # faint individual events
        for s in pre_stack[k]:
            axL.plot(t_pre / 1000.0, s, color=colors[k], alpha=0.18, lw=0.8)
        axL.plot(t_pre / 1000.0, pre_avg[k], color=colors[k], lw=2.6,
                 label=f"avg primary {k} (n={n[k]})")
    axL.set_title(f"Average preamp-shaped waveform (primary channels) — {len(events)} events")
    axL.set_xlabel("time (µs)")
    axL.set_ylabel("preamp signal (a.u.)")
    axL.grid(alpha=0.3)
    axL.legend(fontsize=10)

    fig.tight_layout()
    out_png = os.path.join(outdir, "gate_average_waveform.png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    # ── Overlay plot: every individual primary waveform, colored by type ──
    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 5.5))
    for k in kinds:
        for i, s in enumerate(pre_stack[k]):
            ax2.plot(t_pre / 1000.0, s, color=colors[k], alpha=0.55, lw=1.4,
                     label=f"primary {k}" if i == 0 else None)
    ax2.set_title(f"Primary-channel waveforms overlaid ({len(events)} events)")
    ax2.set_xlabel("time (µs)")
    ax2.set_ylabel("preamp signal (a.u.)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=10)
    fig2.tight_layout()
    out_overlay = os.path.join(outdir, "gate_overlay_waveform.png")
    fig2.savefig(out_overlay, dpi=120)
    plt.close(fig2)

    # ── CSV of averaged preamp signals ──
    out_csv = os.path.join(outdir, "gate_average_waveform.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_us", "avg_anode", "avg_cathode", "avg_steering"])
        for i in range(len(t_pre)):
            w.writerow([f"{t_pre[i]/1000:.4f}", f"{pre_avg['anode'][i]:.6e}",
                        f"{pre_avg['cathode'][i]:.6e}", f"{pre_avg['steering'][i]:.6e}"])

    # ── Report ──
    print(f"Averaged {len(events)} events from {os.path.basename(path)}")
    print(f"  primary channels per event: anode={n['anode']}, cathode={n['cathode']}, steering={n['steering']}")
    print("\nPlateau (final preamp) values, averaged:")
    for k in kinds:
        print(f"  {k:9s}: {pre_avg[k][-1]:+.3e}")
    print("\nPer-event primary channels & positions:")
    print(f"  {'event_id':>10s} {'pos (x,y,z) mm':>22s} {'E_keV':>6s}  anode      cathode")
    for ev in events:
        p = ev["position_mm"]
        print(f"  {int(ev['event_id']):>10d} ({p['x']:6.2f},{p['y']:6.2f},{p['z']:6.2f})    "
              f"{ev['energy_keV']:6.0f}  {ev.get('collecting_anode',''):<8s} {ev.get('collecting_cathode','')}")
    print(f"\nWrote {out_png}")
    print(f"Wrote {out_overlay}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Plot CZT waveforms produced by scripts/simulate_gate_events.jl.

For each event, draws a 2-panel figure:
  (left)  preamp-shaped signals for anodes + cathodes (charge-sensitive output)
  (right) raw induced current on the collecting anode and its neighbors

Saves one PNG per event to output/ and prints the anode/cathode each event maps to.

Usage:
    python scripts/plot_gate_waveforms.py output/gate_waveforms_x+0.0mm.json
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def style(kind):
    return "-" if kind == "anode" else ("--" if kind == "cathode" else ":")


def plot_event(ev, meta, outdir, sig_frac=0.02):
    """sig_frac: only draw contacts whose peak |preamp| exceeds this fraction of the max."""
    eid = int(ev["event_id"])
    pos = ev["position_mm"]
    wfs = ev["waveforms"]
    coll_a = ev.get("collecting_anode", "")
    coll_c = ev.get("collecting_cathode", "")

    # Rank contacts by preamp amplitude; keep the significant ones (+ always the collectors).
    peak = {n: max((abs(v) for v in w["preamp_signal"]), default=0.0) for n, w in wfs.items()}
    pmax = max(peak.values(), default=1.0) or 1.0
    keep = {n for n, p in peak.items() if p >= sig_frac * pmax}
    keep |= {coll_a, coll_c}
    keep &= set(wfs.keys())
    n_hidden = len(wfs) - len(keep)

    # Drift window: last time any raw current is still appreciable (for the zoom).
    t_end = 0.0
    for w in wfs.values():
        cmax = max((abs(c) for c in w["raw_current"]), default=0.0)
        if cmax <= 0:
            continue
        thr = 0.01 * cmax
        for t, c in zip(w["raw_time_ns"], w["raw_current"]):
            if abs(c) >= thr:
                t_end = max(t_end, t)
    t_end = max(t_end * 1.3, 50.0)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.2))

    def ordered(names):
        # collectors first (bold), then by descending peak
        return sorted(names, key=lambda n: (n not in (coll_a, coll_c), -peak.get(n, 0.0)))

    # Left: preamp-shaped output (the digitized waveform)
    for name in ordered(keep):
        w = wfs[name]
        t_us = [x / 1000.0 for x in w["preamp_time_ns"]]
        lw = 2.4 if name in (coll_a, coll_c) else 1.0
        axL.plot(t_us, w["preamp_signal"], style(w["contact_type"]), lw=lw, label=name)
    axL.set_title("Preamp-shaped output (digitized)")
    axL.set_xlabel("time (µs)")
    axL.set_ylabel("preamp signal (a.u.)")
    axL.grid(alpha=0.3)
    axL.legend(fontsize=7, ncol=2, loc="best")

    # Right: raw induced current, zoomed to the drift window
    for name in ordered(keep):
        w = wfs[name]
        lw = 2.4 if name in (coll_a, coll_c) else 1.0
        axR.plot(w["raw_time_ns"], w["raw_current"], style(w["contact_type"]), lw=lw, label=name)
    axR.set_title("Raw induced current (drift window)")
    axR.set_xlabel("time (ns)")
    axR.set_ylabel("dQ/dt (a.u.)")
    axR.set_xlim(0, t_end)
    axR.grid(alpha=0.3)
    axR.legend(fontsize=7, ncol=2, loc="best")

    hidden_note = f"  (+{n_hidden} contacts < {sig_frac:.0%} peak, hidden)" if n_hidden else ""
    fig.suptitle(
        f"Event {eid}  •  pos=({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f}) mm  •  "
        f"{ev['energy_keV']:.0f} keV  •  collecting {coll_a} / {coll_c}{hidden_note}",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = os.path.join(outdir, f"gate_event_{eid}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    with open(path) as f:
        data = json.load(f)

    outdir = os.path.dirname(os.path.abspath(path))
    events = data["events"]
    print(f"{len(events)} events from {os.path.basename(path)} "
          f"(geometry={data.get('geometry')}, preamp τ={data.get('preamp_tau_us')} µs)")
    print(f"  {'event_id':>10s} {'pos (x,y,z) mm':>22s} {'E_keV':>6s}  collecting    n_signals")
    for ev in events:
        p = ev["position_mm"]
        out = plot_event(ev, data, outdir)
        print(f"  {int(ev['event_id']):>10d} "
              f"({p['x']:6.2f},{p['y']:6.2f},{p['z']:6.2f})    "
              f"{ev['energy_keV']:6.0f}  {ev.get('collecting_anode',''):>8s}/"
              f"{ev.get('collecting_cathode',''):<10s} {len(ev['waveforms'])}  -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Load GATE photon-interaction positions from a `blurred` ROOT RNTuple and map
them into the SolidStateDetectors.jl (SSD) CZT detector coordinate frame.

GATE crystal frame is 5(X) x 40(Y) x 40(Z) mm with Z centered at 120 mm; the SSD
crystal is 40(x) x 40(y) x 5(z, drift) mm centered at the origin. The same crystal
with axes permuted, so the mapping is forced for the 5 mm drift axis:

    SSD_x = GATE_Z - GATE_Z_CENTER   (anode-segmentation axis, 39 strips)
    SSD_y = GATE_Y                    (cathode-segmentation axis, 8 strips)
    SSD_z = DRIFT_SIGN * GATE_X       (drift, 5 mm; anode at z=+2.5, cathode at z=-2.5)

Energy is converted MeV -> keV. The interaction point is PostPosition (the
energy-depositing electron tracks have Pre approx Post).

Run with the gate_env interpreter (has uproot + awkward):
    /opt/homebrew/Caskroom/miniconda/base/envs/gate_env/bin/python \
        scripts/load_gate_hits.py <file.root> [--n 5] [...]

Writes output/gate_events_<tag>.csv and prints a span/mean summary.
"""
import argparse
import csv
import os
import re
import sys

import numpy as np

# ── Coordinate mapping constants (edit here to change the GATE->SSD convention) ──
GATE_Z_CENTER = 120.0  # mm; GATE Z spans [100,140], crystal center at 120
DRIFT_SIGN = +1.0      # flip to -1.0 if rise-time-vs-depth comes out inverted

# Crystal half-extents (mm) and an inward margin so the charge cloud stays inside.
CRYSTAL_HALF = (20.0, 20.0, 2.5)  # (x, y, z)
EDGE_MARGIN = 0.2                 # mm; events kept only this far inside each face


def map_gate_to_ssd(gx, gy, gz, x_offset=0.0):
    """GATE lab-frame (mm) -> SSD crystal-frame (mm).

    The collimator sweep shifts the crystal in lab-X, so gate_X is a lab
    coordinate; the depth inside the crystal is gate_X minus the collimator
    offset (the file's nominal x). gate_Z / gate_Y are unaffected by the sweep.
    """
    ssd_x = gz - GATE_Z_CENTER
    ssd_y = gy
    ssd_z = DRIFT_SIGN * (gx - x_offset)
    return ssd_x, ssd_y, ssd_z


def offset_from_tag(tag):
    """Parse the collimator x offset (mm) from a tag like 'x+2.0mm' -> 2.0."""
    m = re.search(r"x([+-][\d.]+)mm", tag)
    return float(m.group(1)) if m else 0.0


def file_tag(path):
    """Derive a short tag, e.g. blurred_merged_x+0.0mm.root -> x+0.0mm."""
    base = os.path.basename(path)
    m = re.search(r"(x[+-][\d.]+mm)", base)
    if m:
        return m.group(1)
    return os.path.splitext(base)[0]


def summarize(name, arr):
    return f"  {name:10s} n={len(arr):5d}  min={np.min(arr):8.3f}  max={np.max(arr):8.3f}  mean={np.mean(arr):8.3f}  std={np.std(arr):7.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rootfile", help="path to a blurred_merged_*.root file")
    ap.add_argument("--n", type=int, default=5,
                    help="number of events to select (default 5)")
    ap.add_argument("--emin", type=float, default=650.0,
                    help="min energy keV for photopeak window (default 650)")
    ap.add_argument("--emax", type=float, default=680.0,
                    help="max energy keV for photopeak window (default 680)")
    ap.add_argument("--all-energy", action="store_true",
                    help="ignore the energy window; sample any energy > 0")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for selection")
    ap.add_argument("--random", action="store_true",
                    help="pure random sample (default: stratified across ssd_x)")
    ap.add_argument("--tree", default="blurred", help="RNTuple name (default 'blurred')")
    ap.add_argument("--outdir", default=None,
                    help="output dir (default: <repo>/output)")
    args = ap.parse_args()

    import uproot
    import awkward as ak

    obj = uproot.open(args.rootfile)[args.tree]
    a = obj.arrays()

    def col(name):
        return ak.to_numpy(a[name])

    # Interaction point = PostPosition; energy MeV -> keV.
    gx = col("PostPosition_X")
    gy = col("PostPosition_Y")
    gz = col("PostPosition_Z")
    energy_keV = col("TotalEnergyDeposit") * 1000.0
    event_id = col("EventID")

    x_offset = offset_from_tag(file_tag(args.rootfile))
    ssd_x, ssd_y, ssd_z = map_gate_to_ssd(gx, gy, gz, x_offset=x_offset)
    print(f"Collimator x-offset for this file: {x_offset:+.1f} mm "
          f"(subtracted from gate_X to get drift depth ssd_z)")

    # ── Summary of the full file (the "see the data" deliverable) ──
    print(f"=== {os.path.basename(args.rootfile)} : {obj.num_entries} hits ===")
    print("GATE frame (mm):")
    print(summarize("GATE_X", gx))
    print(summarize("GATE_Y", gy))
    print(summarize("GATE_Z", gz))
    print("SSD frame (mm):")
    print(summarize("SSD_x", ssd_x))
    print(summarize("SSD_y", ssd_y))
    print(summarize("SSD_z", ssd_z))
    print("Energy (keV):")
    print(summarize("E", energy_keV))
    in_pp = np.sum((energy_keV >= args.emin) & (energy_keV <= args.emax))
    print(f"  photopeak [{args.emin:.0f},{args.emax:.0f}] keV: {in_pp} events")

    # ── Selection ──
    valid = energy_keV > 0
    if not args.all_energy:
        valid &= (energy_keV >= args.emin) & (energy_keV <= args.emax)
    # keep events safely inside the crystal so the charge cloud fits
    hx, hy, hz = CRYSTAL_HALF
    valid &= (np.abs(ssd_x) <= hx - EDGE_MARGIN) & (np.abs(ssd_y) <= hy - EDGE_MARGIN) \
        & (np.abs(ssd_z) <= hz - EDGE_MARGIN)
    idx_pool = np.where(valid)[0]
    if len(idx_pool) == 0:
        print("No events match the selection.", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(idx_pool))
    if args.random:
        # pure random sample from the photopeak pool
        sel = rng.choice(idx_pool, size=n, replace=False)
    else:
        # stratified: spread the pick across the SSD_x range (different anodes)
        order = idx_pool[np.argsort(ssd_x[idx_pool])]
        edges = np.linspace(0, len(order), n + 1).astype(int)
        sel = np.array([order[rng.integers(edges[i], max(edges[i] + 1, edges[i + 1]))]
                        for i in range(n)])

    outdir = args.outdir or os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(outdir, exist_ok=True)
    tag = file_tag(args.rootfile)
    outfile = os.path.join(outdir, f"gate_events_{tag}.csv")

    cols = ["event_id", "energy_keV",
            "ssd_x_mm", "ssd_y_mm", "ssd_z_mm",
            "gate_x_mm", "gate_y_mm", "gate_z_mm"]
    with open(outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in sel:
            w.writerow([int(event_id[i]), f"{energy_keV[i]:.3f}",
                        f"{ssd_x[i]:.4f}", f"{ssd_y[i]:.4f}", f"{ssd_z[i]:.4f}",
                        f"{gx[i]:.4f}", f"{gy[i]:.4f}", f"{gz[i]:.4f}"])

    print(f"\nSelected {len(sel)} events "
          f"({'any energy' if args.all_energy else f'{args.emin:.0f}-{args.emax:.0f} keV'}):")
    print(f"  {'event_id':>10s} {'E_keV':>8s} {'ssd_x':>8s} {'ssd_y':>8s} {'ssd_z':>8s}")
    for i in sel:
        print(f"  {int(event_id[i]):>10d} {energy_keV[i]:8.1f} "
              f"{ssd_x[i]:8.3f} {ssd_y[i]:8.3f} {ssd_z[i]:8.3f}")
    print(f"\nWrote {os.path.abspath(outfile)}")


if __name__ == "__main__":
    main()

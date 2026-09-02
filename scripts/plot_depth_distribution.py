#!/usr/bin/env python
"""
Distribution of interaction depth (ssd_z, the 5 mm drift axis) for photopeak
events, one histogram per collimator slit position, colored by slit position.
Shows how the sweep samples depth across the crystal thickness.

ssd_z = gate_X - (collimator offset); anode at +2.5 mm, cathode at -2.5 mm.

Run with gate_env (uproot):
  /opt/homebrew/Caskroom/miniconda/base/envs/gate_env/bin/python \
      scripts/plot_depth_distribution.py
"""
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# Dataset location: override with GATE_DATA_DIR env var (e.g. on the cluster /
# in CI). Falls back to a local-dev default so it still works on a Mac checkout.
DATA_DIR = os.environ.get(
    "GATE_DATA_DIR",
    "/Users/maxteicheira/Documents/RIL/Gate_sim_collimator/blurred_merged_sweep",
)
OUT = os.path.join(os.path.dirname(__file__), "..", "output", "depth_distribution.png")
HZ = 2.5            # crystal half-thickness (mm)
EMIN, EMAX = 650.0, 680.0


def offset(fname):
    m = re.search(r"x([+-][\d.]+)mm", fname)
    return float(m.group(1)) if m else 0.0


def main():
    import uproot, awkward as ak
    files = sorted(glob.glob(os.path.join(DATA_DIR, "blurred_merged_x*.root")), key=offset)
    xs = [offset(f) for f in files]
    norm = Normalize(vmin=min(xs), vmax=max(xs)); cmap = cm.coolwarm
    bins = np.linspace(-HZ, HZ, 61)

    fig, ax = plt.subplots(figsize=(11, 6))
    total = []
    for f in files:
        a = uproot.open(f)["blurred"].arrays()
        gx = ak.to_numpy(a["PostPosition_X"])
        e = ak.to_numpy(a["TotalEnergyDeposit"]) * 1000.0
        off = offset(f)
        ssd_z = gx - off
        pp = (e >= EMIN) & (e <= EMAX)
        z = ssd_z[pp]
        total.append(z)
        ax.hist(z, bins=bins, histtype="step", lw=2, color=cmap(norm(off)))

    # faint combined total (all positions) for overall coverage
    ax.hist(np.concatenate(total), bins=bins, color="0.85", alpha=0.5, zorder=0,
            label=f"all positions ({sum(len(z) for z in total)} photopeak events)")

    ax.axvline(-HZ, color="k", ls=":", lw=0.8); ax.axvline(HZ, color="k", ls=":", lw=0.8)
    ax.text(-HZ, ax.get_ylim()[1]*0.95, " cathode", ha="left", va="top", fontsize=9, color="0.3")
    ax.text(HZ, ax.get_ylim()[1]*0.95, "anode ", ha="right", va="top", fontsize=9, color="0.3")
    ax.set_xlabel("ssd_z  —  interaction depth (mm)   [the 5 mm drift axis]")
    ax.set_ylabel("counts (photopeak events)")
    ax.set_title("Interaction-depth distribution across the 5 mm thickness, by slit position")
    ax.set_xlim(-HZ - 0.1, HZ + 0.1)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper right")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label("collimator slit x (mm)\n−2 mm = anode side   ·   +2 mm = cathode side")
    fig.tight_layout()
    fig.savefig(os.path.abspath(OUT), dpi=130)

    print(f"{'slit':>6s} {'N_pp':>6s} {'mean_z':>7s} {'std':>6s}")
    for f, z in zip(files, total):
        print(f"{offset(f):+6.1f} {len(z):6d} {z.mean():7.2f} {z.std():6.2f}")
    print(f"Wrote {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()

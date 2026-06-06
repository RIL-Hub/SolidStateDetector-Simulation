#!/usr/bin/env python
"""
Density of the drift-depth (ssd_z) interaction positions across the 5 mm crystal,
to show how much the interaction depth varies. Overlays the selected events.

Run with gate_env (uproot):
    /opt/homebrew/Caskroom/miniconda/base/envs/gate_env/bin/python \
        scripts/plot_z_density.py <file.root> [events.csv]
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GATE_Z_CENTER = 120.0
DRIFT_SIGN = +1.0
HZ = 2.5  # crystal half-thickness (mm)


def main():
    rootfile = sys.argv[1]
    csvfile = sys.argv[2] if len(sys.argv) > 2 else None
    import uproot, awkward as ak

    a = uproot.open(rootfile)["blurred"].arrays()
    gx = ak.to_numpy(a["PostPosition_X"])
    e = ak.to_numpy(a["TotalEnergyDeposit"]) * 1000.0
    sz = DRIFT_SIGN * gx
    pp = (e >= 650) & (e <= 680)

    sel_z = None
    if csvfile and os.path.exists(csvfile):
        rows = list(csv.DictReader(open(csvfile)))
        sel_z = np.array([float(r["ssd_z_mm"]) for r in rows])

    bins = np.linspace(-HZ, HZ, 61)
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.hist(sz, bins=bins, density=True, color="0.7", alpha=0.6,
            label=f"all hits (n={len(sz)})")
    ax.hist(sz[pp], bins=bins, density=True, histtype="step", color="tab:blue", lw=2,
            label=f"photopeak (n={pp.sum()})")

    # smooth KDE if scipy is around
    try:
        from scipy.stats import gaussian_kde
        xs = np.linspace(-HZ, HZ, 400)
        ax.plot(xs, gaussian_kde(sz)(xs), color="k", lw=1.5, label="KDE (all)")
    except Exception:
        pass

    if sel_z is not None:
        for z in sel_z:
            ax.axvline(z, color="tab:red", lw=1.5, alpha=0.8)
        ax.plot([], [], color="tab:red", lw=1.5, label="selected events")

    ax.axvline(-HZ, color="k", ls=":", lw=0.8)
    ax.axvline(HZ, color="k", ls=":", lw=0.8)
    ax.set_xlim(-HZ - 0.1, HZ + 0.1)
    ax.set_xlabel("ssd_z  (drift depth, mm)   cathode −2.5 ……… anode +2.5")
    ax.set_ylabel("probability density (1/mm)")
    ax.set_title(f"Interaction-depth density — {os.path.basename(rootfile)}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(csvfile or rootfile)), "gate_z_density.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)

    print(f"ssd_z (drift, mm) over {len(sz)} hits:")
    print(f"  min={sz.min():.3f}  max={sz.max():.3f}  mean={sz.mean():.3f}  std={sz.std():.3f}")
    print(f"  percentiles [1,5,25,50,75,95,99]: "
          f"{np.round(np.percentile(sz, [1,5,25,50,75,95,99]), 3)}")
    print(f"  fraction within ±1 mm of mid-plane: {np.mean(np.abs(sz) <= 1.0)*100:.1f}%")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

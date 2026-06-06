#!/usr/bin/env python
"""
Sanity-check the photon-interaction positions: scatter ALL interactions from a
GATE root file in the SSD detector frame (3 projections), overlay the selected
events from a gate_events_*.csv, and draw the crystal outline.

Run with gate_env (uproot):
    /opt/homebrew/Caskroom/miniconda/base/envs/gate_env/bin/python \
        scripts/plot_positions.py <file.root> [events.csv]
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# GATE -> SSD mapping (must match load_gate_hits.py)
GATE_Z_CENTER = 120.0
DRIFT_SIGN = +1.0

# SSD crystal half-extents (mm): 40 x 40 x 5
HX, HY, HZ = 20.0, 20.0, 2.5


def main():
    rootfile = sys.argv[1]
    csvfile = sys.argv[2] if len(sys.argv) > 2 else None
    import uproot, awkward as ak

    a = uproot.open(rootfile)["blurred"].arrays()
    gx, gy, gz = (ak.to_numpy(a["PostPosition_X"]), ak.to_numpy(a["PostPosition_Y"]),
                  ak.to_numpy(a["PostPosition_Z"]))
    e = ak.to_numpy(a["TotalEnergyDeposit"]) * 1000.0
    sx, sy, sz = gz - GATE_Z_CENTER, gy, DRIFT_SIGN * gx
    pp = (e >= 650) & (e <= 680)  # photopeak

    sel = None
    if csvfile and os.path.exists(csvfile):
        rows = list(csv.DictReader(open(csvfile)))
        sel = {k: np.array([float(r[k]) for r in rows]) for k in
               ("ssd_x_mm", "ssd_y_mm", "ssd_z_mm")}

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    planes = [("ssd_x", sx, "ssd_y", sy, HX, HY),
              ("ssd_x", sx, "ssd_z (drift)", sz, HX, HZ),
              ("ssd_y", sy, "ssd_z (drift)", sz, HY, HZ)]
    selkeys = [("ssd_x_mm", "ssd_y_mm"), ("ssd_x_mm", "ssd_z_mm"), ("ssd_y_mm", "ssd_z_mm")]

    for ax, (xl, X, yl, Y, hx, hy), sk in zip(axs, planes, selkeys):
        ax.scatter(X, Y, s=2, alpha=0.10, color="0.6", label="all hits")
        ax.scatter(X[pp], Y[pp], s=3, alpha=0.30, color="tab:blue", label="photopeak")
        if sel is not None:
            ax.scatter(sel[sk[0]], sel[sk[1]], s=90, marker="*",
                       color="tab:red", edgecolor="k", zorder=5, label="selected")
        ax.add_patch(plt.Rectangle((-hx, -hy), 2 * hx, 2 * hy, fill=False,
                                   edgecolor="k", lw=1.5, ls="--"))
        ax.set_xlabel(f"{xl} (mm)")
        ax.set_ylabel(f"{yl} (mm)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Interaction positions (SSD frame) — {os.path.basename(rootfile)}  "
                 f"[{len(sx)} hits, {pp.sum()} photopeak]  dashed = 40×40×5 mm crystal", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(os.path.dirname(os.path.abspath(csvfile or rootfile)), "gate_positions.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)

    # text summary
    def rng(name, v):
        return f"  {name:14s} min={v.min():8.3f}  max={v.max():8.3f}  mean={v.mean():8.3f}"
    print(f"{os.path.basename(rootfile)}: {len(sx)} hits, {pp.sum()} photopeak (650-680 keV)")
    print("SSD frame (mm), all hits:")
    for nm, v in (("ssd_x", sx), ("ssd_y", sy), ("ssd_z(drift)", sz)):
        print(rng(nm, v))
    print(f"Crystal bounds: x∈[-20,20], y∈[-20,20], z∈[-2.5,2.5]  -> "
          f"in-bounds: {np.mean((np.abs(sx)<=HX)&(np.abs(sy)<=HY)&(np.abs(sz)<=HZ))*100:.1f}%")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

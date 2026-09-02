#!/usr/bin/env python
"""
Cross-section plots for a cross-strip CZT detector.

Produces three files:
  efield_xsection_lines.png    — |E| background + field streamlines
  efield_xsection_contour.png  — |E| background + equipotential contours
  efield_xsection_potential.png — electric potential background + streamlines

Usage:  python scripts/plot_efield_xsection.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from scipy.interpolate import RegularGridInterpolator

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

plt.rcParams.update({
    "font.size": 14, "axes.labelsize": 14, "axes.titlesize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
})

with open(os.path.join(REPO, "output", "efield_xsection.json")) as f:
    data = json.load(f)

x_mm = np.array(data["x_mm"])
z_mm = np.array(data["z_mm"])
nx, nz = len(x_mm), len(z_mm)

pot    = np.array(data["pot_V"]).reshape((nx, nz), order="F")
Ex_raw = np.array(data["Ex_Vm"]).reshape((nx, nz), order="F")
Ez_raw = np.array(data["Ez_Vm"]).reshape((nx, nz), order="F")

# ── Crop to anode region ──────────────────────────────────────────────────────
X0, X1 = -1.8,  1.0
Z0, Z1 = -2.5,  2.5

ix0, ix1 = np.searchsorted(x_mm, X0), min(np.searchsorted(x_mm, X1) + 1, nx)
iz0, iz1 = np.searchsorted(z_mm, Z0), min(np.searchsorted(z_mm, Z1) + 1, nz)

xc = x_mm[ix0:ix1]
zc = z_mm[iz0:iz1]
Pc  = pot[ix0:ix1, iz0:iz1]
Exc = Ex_raw[ix0:ix1, iz0:iz1]
Ezc = Ez_raw[ix0:ix1, iz0:iz1]

# ── Interpolate onto uniform grid ─────────────────────────────────────────────
Nx, Nz = 300, 400
xu = np.linspace(xc[0], xc[-1], Nx)
zu = np.linspace(zc[0], zc[-1], Nz)

kw = dict(method="linear", bounds_error=False)
interp_P  = RegularGridInterpolator((xc, zc), Pc,  fill_value=None, **kw)
interp_Ex = RegularGridInterpolator((xc, zc), Exc, fill_value=0,    **kw)
interp_Ez = RegularGridInterpolator((xc, zc), Ezc, fill_value=0,    **kw)

XX, ZZ = np.meshgrid(xu, zu, indexing="ij")
pts  = np.column_stack([XX.ravel(), ZZ.ravel()])
Pu   = interp_P(pts).reshape(Nx, Nz)
Eu_x = interp_Ex(pts).reshape(Nx, Nz)
Eu_z = interp_Ez(pts).reshape(Nx, Nz)
Emag = np.sqrt(Eu_x**2 + Eu_z**2)   # V/m

# ── Electrode geometry ────────────────────────────────────────────────────────
ANODE_Z   =  2.5
CATHODE_Z = -2.5
ELEC_H    =  0.04
ANODE_HW  =  0.05
STEER_HW  =  0.20

anodes    = [-1.0,  0.0]
steerings = [-1.5, -0.5,  0.5]


def draw_electrodes(ax):
    for x in anodes:
        ax.add_patch(Rectangle(
            (x - ANODE_HW, ANODE_Z), 2 * ANODE_HW, ELEC_H,
            color="#FFD700", zorder=6, clip_on=False))
        ax.text(x, ANODE_Z + ELEC_H + 0.04, "A",
                ha="center", va="bottom", color="#FFD700",
                fontsize=9, fontweight="bold", clip_on=False)

    for x in steerings:
        if X0 <= x <= X1:
            ax.add_patch(Rectangle(
                (x - STEER_HW, ANODE_Z), 2 * STEER_HW, ELEC_H,
                color="#9B59B6", zorder=6, clip_on=False))
            ax.text(x, ANODE_Z + ELEC_H + 0.04, "S",
                    ha="center", va="bottom", color="#9B59B6",
                    fontsize=9, fontweight="bold", clip_on=False)

    ax.add_patch(Rectangle(
        (X0, CATHODE_Z - ELEC_H), X1 - X0, ELEC_H,
        color="#3498DB", zorder=6, clip_on=False))
    ax.text(X1 + 0.08, CATHODE_Z - ELEC_H / 2,
            "Cathode (−600 V)", ha="left", va="center",
            color="#3498DB", fontsize=11, clip_on=False)


def make_figure(mode, outname):
    """
    mode: 'lines'     — |E| bg + streamlines
          'contour'   — |E| bg + equipotential contours
          'potential' — V bg + streamlines
    """
    fig, ax = plt.subplots(figsize=(6, 9), dpi=150)

    if mode == "potential":
        pcm = ax.pcolormesh(xu, zu, Pu.T,
            cmap="jet", shading="gouraud",
            norm=mcolors.Normalize(vmin=pot.min(), vmax=0))
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02, shrink=0.70)
        cbar.set_label("Electric potential (V)", fontsize=13)
        ax.streamplot(xu, zu, Eu_x.T, Eu_z.T,
            color="white", linewidth=0.8, density=1.3,
            arrowsize=0.9, arrowstyle="->")
        title = "Electric Potential — Cross-Strip CZT"
    else:
        pcm = ax.pcolormesh(xu, zu, Emag.T / 1e4,
            cmap="jet", shading="gouraud",
            norm=mcolors.Normalize(vmin=0, vmax=np.percentile(Emag, 97) / 1e4))
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02, shrink=0.70)
        cbar.set_label("|E| (V/cm)", fontsize=13)

        if mode == "lines":
            ax.streamplot(xu, zu, Eu_x.T, Eu_z.T,
                color="white", linewidth=0.8, density=1.3,
                arrowsize=0.9, arrowstyle="->")
            title = "Electric Field Cross-Section — Cross-Strip CZT"
        else:
            lvls = np.linspace(pot.min(), 0, 25)
            ax.contour(xu, zu, Pu.T, levels=lvls,
                       colors="white", linewidths=0.7, alpha=0.7)
            title = "Electric Field (Equipotentials) — Cross-Strip CZT"

    draw_electrodes(ax)

    ax.axhline(ANODE_Z,   color="white", lw=0.7, ls="--", alpha=0.35)
    ax.axhline(CATHODE_Z, color="white", lw=0.7, ls="--", alpha=0.35)

    ax.set_xlim(X0, X1)
    ax.set_ylim(Z0, Z1)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(title, pad=28)

    fig.tight_layout()
    out = os.path.join(REPO, "output", outname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


make_figure("lines",     "efield_xsection_lines.png")
make_figure("contour",   "efield_xsection_contour.png")
make_figure("potential", "efield_xsection_potential.png")

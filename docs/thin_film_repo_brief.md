# Thin-Film Detector Repo — Agent Briefing

## Goal

Build a clean, self-contained boilerplate repository for simulating thin-film
solid-state radiation detectors using **SolidStateDetectors.jl** (SSD.jl).
Thin films are typically 100–500 µm thick (vs 5–10 mm for bulk CZT), single-side
strip or pixel geometries, common in X-ray imaging and synchrotron applications.

The repo should be immediately runnable by someone new to SSD.jl and produce
publication-quality output plots covering the three key visualization categories
below.

---

## Source Repository (do not modify, copy from here)

Working directory with all existing code:
`/Users/maxteicheira/.superset/worktrees/SolidStateDetector-Simulation/iris-archduke/`

Key files to draw from:

| File | Role |
|------|------|
| `geometries/thin_film.yaml` | 300 µm Si thin-film geometry, 2-contact planar, starting point |
| `scripts/single_event_thin_film.jl` | Solve fields + simulate one event + plot waveforms |
| `scripts/plot_efield_xsection.jl` | Export E-field / potential xz slice to JSON (SSD → JSON → Python) |
| `scripts/plot_efield_xsection.py` | Plot 3 figures: E-field+streamlines, E-field+equipotentials, potential |
| `scripts/plot_z_sweep_polarization.jl` | Sweep interaction depths, decompose e⁻/h⁺ contributions with trapping |
| `scripts/plot_z_sweep_polarization.py` | Plot e/h panels with free vs trapped, jet colormap |
| `scripts/plot_z_sweep.jl` + `.py` | Z-depth sweep: anode/cathode/steering waveforms |
| `scripts/export_drift_paths.jl` | Export drift path coordinates to JSON for custom plotting |

---

## Thin-Film Geometry Details

Existing `thin_film.yaml`:
- **Material:** Si (swap to CdZnTe, GaAs, a-Se as needed)
- **Size:** 10 × 10 × 0.3 mm (z = drift/thickness axis)
- **Contacts:** full-area anode (id=1, 0 V, top) + cathode (id=2, −100 V, bottom)
- **Mobilities:** µ_e = 1350, µ_h = 480 cm²/(V·s)  [Si values]
- **Lifetimes:** τ_e = τ_h = 1 ms  [Si, much longer than CZT]
- **Charge cloud:** radius = 0.01 mm (10 µm — thin film deposits are tiny)
- **Energy:** 5.9 keV (Fe-55 source typical for thin-film characterization)

**Extensions to add for the boilerplate:**
1. A second geometry variant with strip anodes (e.g. 5 strips, 2 mm pitch) on the top face
2. Optional: an interdigitated electrode variant

---

## Three Example Scripts to Build / Adapt

### 1. Electric field & weighting potential cross-section

**What it shows:** xz slice at y=0 with:
- Electric field magnitude (jet colormap) + field streamlines
- Weighting potential for the collecting electrode

**Pattern:** Julia script exports JSON (grid + field arrays), Python reads and plots.
This separation is necessary because SSD.jl's built-in Plots recipes fix the axis units.

Key gotcha: SSD grids are in **metres**; multiply × 1000 for mm in plots.
JSON3.jl serializes 2D Julia arrays as flat 1D (column-major) → reshape in Python:
```python
np.array(data["pot_V"]).reshape((nx, nz), order="F")
```

See `scripts/plot_efield_xsection.jl` + `.py` for a working implementation.

---

### 2. Single event: waveforms + drift paths

**What it shows:**
- Induced charge waveforms on anode and cathode vs time
- Electron vs hole contributions (Ramo's theorem, see below)

**Physics note (Ramo's theorem):**
```
ΔQ_e(t) = +ΔΦ_w(r_e(t))   electrons moving toward anode
ΔQ_h(t) = −ΔΦ_w(r_h(t))   holes moving away from anode
```
For a thin planar detector (uniform field, no small-pixel effect), holes and electrons
contribute roughly equally at mid-depth. Near the anode, electrons dominate;
near the cathode, holes dominate.

**Trapping effect** (τ_e, τ_h from YAML):
Each carrier's incremental WP contribution is weighted by survival probability:
```
q_trap(T) = Σ_j exp(−j·dt/τ) · ΔΦ_w(step j)
```
For Si with τ=1 ms and drift time ~10 ns, trapping is negligible.
For CZT with τ_e=5 µs / τ_h=3 µs, it matters for deep events (~5–10% loss).

See `scripts/plot_z_sweep_polarization.jl` for the full working implementation
including the correct incremental trapping formula (avoid reading `q_trap[n]`
from the shared accumulator — use the local `cumulative_trap` scalar for holdout).

---

### 3. Depth sweep: waveform shape vs interaction depth

**What it shows:** 9 depths from near-anode to near-cathode, overlaid,
jet colormap by depth. Shows how cathode/anode signal ratio encodes depth.

See `scripts/plot_z_sweep.jl` + `scripts/plot_z_sweep.py`.

---

## Julia / SSD.jl API Notes (gotchas the agent must know)

- **Weighting potential interpolation:** `wp(pos)` does NOT work.
  Use: `itp = SolidStateDetectors.interpolated_scalarfield(wp)` then `itp(x, y, z)` in metres.

- **Solve order is fixed:** E-potential → E-field → weighting potentials (one per contact).

- **CartesianPoint units:** always metres, even if YAML uses mm.

- **Cloud radius:** default 0.5 mm is too large for thin-film or small-pitch strips.
  Set `radius = [0.01u"mm"]` or similar.

- **Missing waveforms:** check `ismissing(evt.waveforms[id])` before accessing.

- **Caching:** use `Serialization.serialize/deserialize` (not JLD2 — fails on complex geometries).
  Cache path: `output/sim_cache.jls`. See `scripts/build_wp_cache.jl`.

- **Preamp IIR filter** (optional, matches charge-sensitive amp response):
  τ = 140 µs → b0 = 1400, a1 = 0.9999992857 at dt = 0.1 ns.
  If dt = 1 ns: a1_new = a1^10, b0_new = b0 × 10.

---

## Target Repo Structure

```
thin-film-ssd/
├── Project.toml
├── Manifest.toml
├── README.md
├── geometries/
│   ├── thin_film_planar.yaml      # planar 2-contact (from existing thin_film.yaml)
│   └── thin_film_strips.yaml      # 5-strip anode variant (new)
├── scripts/
│   ├── 01_single_event.jl         # solve + simulate + waveforms (Julia)
│   ├── 02_efield_export.jl        # export E-field xz slice to JSON
│   ├── 02_efield_plot.py          # plot from JSON
│   ├── 03_depth_sweep.jl          # 9-depth sweep, export JSON
│   ├── 03_depth_sweep_plot.py     # waveforms vs depth, jet colormap
│   ├── 04_eh_decomposition.jl     # e/h contributions + trapping
│   └── 04_eh_decomposition_plot.py
└── output/                        # gitignored, created at runtime
```

---

## Presentation (optional, after repo is working)

Once the repo scripts run and produce output PNGs, build a slide deck covering:
1. What is a thin-film detector? (geometry, applications)
2. SSD.jl overview (adaptive grid, Ramo theorem)
3. The geometry YAML (annotated example)
4. Electric field + weighting potential plots (output of script 02)
5. Single event waveforms — electrons vs holes (output of script 01 + 04)
6. Depth sweep — cathode/anode ratio encodes depth (output of script 03)
7. Pitfalls reference card

Source for slides: `output/CZT_simulation_tutorial.pptx` and `docs/slides_outline.md`
in the source repo above — adapt those for thin-film context.

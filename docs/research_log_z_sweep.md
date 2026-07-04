# Research Log — Z-Depth Sweep, CZT Cross-Strip Detector
**Repo:** SolidStateDetector-Simulation, branch `iris-archduke`
**Date:** 2026-07-04

---

## What was done

Simulated a 662 keV gamma interaction at 6 depths across the 5 mm CZT crystal thickness and plotted the induced charge waveforms on the collecting anode, cathode, and steering electrode. Purpose: figure for UCSC group presentation / paper draft.

---

## Geometry

File: `geometries/czt_cross_strip.yaml`

Simplified cross-strip geometry, 8 contacts total:
- **Anodes 1–5**: 100 µm wide strips, 1 mm pitch, along x, at z = +2.5 mm (anode face), 0 V
- **Steering (contact 6)**: union of 6 × 400 µm strips between anodes, −80 V
- **Cathodes 7–8**: two half-area strips along y, at z = −2.5 mm (cathode face), −600 V
  - Cathode 7: y < 0, cathode 8: y > 0

Crystal: CdZnTe, 40 × 40 × 5 mm, τ_e = τ_h = 1 µs (good commercial CZT).
Drift model: IsotropicChargeDriftModel, μ_e = 1000 cm²/(V·s), μ_h = 50 cm²/(V·s).

---

## Simulation parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Energy | 662 keV | Cs-137 |
| Interaction x, y | 0 mm, 2.5 mm | center anode strip; inside cathode-8 y-range |
| Z positions | −2.0, −1.2, −0.4, +0.4, +1.2, +2.0 mm | 6 steps |
| Distance from anode | 0.5, 1.3, 2.1, 2.9, 3.7, 4.5 mm | anode face = z+2.5 |
| N_CARRIERS | 50 | macro-charge points |
| N_SHELLS | 2 | shell radii at 0.1 mm and 0.2 mm |
| CLOUD_RADIUS | 0.1 mm | physical charge cloud size |
| DT | 0.1 ns | waveform time step |
| MAX_STEPS | 100,000 | = 10 µs max drift time |
| Refinement limits | [0.2, 0.1, 0.05] | 3-pass adaptive grid |
| Convergence limit | 1e-6 | |
| Depletion handling | true | |

**Why 10 µs:** Hole mobility is 20× lower than electron mobility. At 600 V over 5 mm, holes drift at ~0.6 mm/µs. Events near the anode (depth ~0.5 mm) require ~8 µs for holes to reach the cathode. Using 5 µs cuts off these waveforms mid-slope.

---

## Key physics observations

- **Anode**: Shows classic small-pixel effect — signal is flat while holes drift across the bulk, then rises sharply as electrons approach the 100 µm strip. Near-anode events (blue) rise faster; near-cathode events (red) have a longer, shallower rise.
- **Cathode**: Slower, depth-dependent amplitude. Near-cathode events have larger cathode signal; near-anode events have smaller cathode signal. Cathode/anode ratio → depth reconstruction.
- **Steering electrode**: Induced transient signal from passing carriers; amplitude small relative to anode/cathode.

---

## Output files

All in `output/`:

| File | Description |
|------|-------------|
| `z_sweep.json` | Raw waveform data — time (ns) + signal arrays per channel per depth. Portable; can replot without re-running Julia. |
| `z_sweep.png` | 2×2 panel figure, 0–10 µs, jet colormap, journal font sizes |
| `z_sweep_zoom.png` | Same, zoomed to 0–4 µs |

---

## Plot conventions

- **X-axis**: 2 µs pre-trigger baseline prepended so signal onset is at t = 2 µs; axis starts at 0
- **Y-axis**: Normalized to peak anode signal (0–1, no units, no sign flip)
- **Color**: jet colormap, 0–5 mm range (0 = anode, 5 = cathode); white ticks on colorbar mark the 6 simulated depths
- **Panel labels**: "Anode (0 V)", "Cathode (−600 V)", "Steering Electrode (−80 V)"
- **Title**: "Z-Depth Sweep — 662 keV | SolidStateDetectors.jl"

---

## How to reproduce from scratch

```bash
# 1. Solve fields + run sweep, export z_sweep.json (~35 s on 8 threads)
julia --project=. -t8 scripts/plot_z_sweep.jl

# 2. Generate both PNG figures
python scripts/plot_z_sweep.py
```

To replot from existing data only (no Julia needed):
```bash
python scripts/plot_z_sweep.py
```

---

## Related files

- `scripts/generate_slides_output.jl` — generates E-potential, WP, and single-event waveform PNGs for tutorial slides
- `scripts/make_slides.py` — builds `output/CZT_simulation_tutorial.pptx` (11-slide intro deck)
- `docs/slides_outline.md` — slide content outline
- `geometries/czt_cross_strip_full.yaml` — full 48-contact version (39 anodes, 1 steering, 8 cathodes)

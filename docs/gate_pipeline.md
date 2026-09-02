# GATE → SSD Waveform Pipeline

Converts GATE Monte Carlo photon-interaction data into simulated CZT detector waveforms
using SolidStateDetectors.jl and the full 39-anode cross-strip geometry.

---

## Scripts

| Script | Role |
|--------|------|
| `scripts/load_gate_hits.py` | Reads a GATE ROOT RNTuple, remaps to SSD coordinates, writes a CSV |
| `scripts/simulate_gate_events.jl` | Reads the CSV, simulates charge drift + preamp shaping, writes JSON waveforms |
| `scripts/plot_gate_waveforms.py` | Reads the JSON, saves one PNG per event |
| `scripts/build_wp_cache.jl` | One-time setup: pre-builds E-field + all 48 WPs and saves a `.jls` cache |

---

## Geometry

**Full geometry:** `geometries/czt_cross_strip_full.yaml`
- 39 anode strips, 1 mm pitch, 100 µm wide, at z = +2.5 mm (0 V)
- 1 steering union (contact 40), −80 V
- 8 cathode strips at z = −2.5 mm (−600 V), contacts 41–48
- Crystal: CdZnTe, 40 × 40 × 5 mm, τ = 1 µs, μ_e = 1000 / μ_h = 50 cm²/(V·s)

Contact ID mapping:
- Anodes: id 1–39 → anode_1 … anode_39 (anode 20 = center, x = 0 mm)
- Steering: id 40
- Cathodes: id 41–48 → cathode_1 … cathode_8

---

## Coordinate mapping (GATE → SSD)

GATE crystal frame: 5 mm (X, drift) × 40 mm (Y) × 40 mm (Z), Z center at 120 mm.
SSD crystal frame: 40 mm (x) × 40 mm (y) × 5 mm (z, drift), centered at origin.

```
SSD_x = GATE_Z − 120     (anode-strip axis, 39 strips, 1 mm pitch)
SSD_y = GATE_Y           (cathode-strip axis, 8 strips, 5 mm pitch)
SSD_z = GATE_X − offset  (drift axis; anode at z = +2.5, cathode at z = −2.5)
```

`offset` is the collimator X position (extracted from the filename tag, e.g., `x+2.0mm`).

---

## Pipeline

### Step 0 — Build field cache (one-time, ~7 min on 8 threads)

```bash
julia --project=. -t8 scripts/build_wp_cache.jl
# → output/sim_cache.jls  (~few hundred MB)
```

This pre-solves the electric potential, electric field, and all 48 weighting potentials
and serialises them to `output/sim_cache.jls`. Subsequent simulation runs load the cache
in seconds instead of re-solving.

Skip this if you only need a quick test — the simulation script computes fields on-the-fly
for the contacts it actually needs (slower per run, no disk requirement).

### Step 1 — Extract events from a GATE ROOT file

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/gate_env/bin/python \
    scripts/load_gate_hits.py path/to/blurred_merged_x+0.0mm.root \
    --n 10 --emin 650 --emax 680
# → output/gate_events_x+0.0mm.csv
```

Key options:
| Flag | Default | Meaning |
|------|---------|---------|
| `--n` | 5 | Number of events to select |
| `--emin/--emax` | 650/680 keV | Photopeak energy window |
| `--all-energy` | off | Ignore energy window |
| `--random` | off | Pure random sample (default: stratified across anodes) |
| `--seed` | 0 | RNG seed for reproducibility |

### Step 2 — Simulate waveforms

```bash
julia --project=. -t8 scripts/simulate_gate_events.jl output/gate_events_x+0.0mm.csv
# → output/gate_waveforms_x+0.0mm.json
```

ENV overrides (all optional):

| Variable | Default | Meaning |
|----------|---------|---------|
| `SSD_CACHE` | `output/sim_cache.jls` | Path to pre-built cache; set to a non-existent path to force on-the-fly |
| `SSD_GEOMETRY` | `geometries/czt_cross_strip_full.yaml` | Geometry YAML |
| `SSD_N_CARRIERS` | `50` | Macro-charge points per event |
| `SSD_DT_NS` | `0.1` | Time step (ns) |
| `SSD_MAX_NSTEPS` | `50000` | Max drift steps (= 5 µs at 0.1 ns) |
| `SSD_REFINE` | `0.2,0.1,0.05` | Grid refinement passes |

The script computes WPs only for contacts within ±2 anodes and ±1 cathode of each event's
interaction position, which limits the WP solve to ~5–20 contacts instead of 48.
If a cache is loaded the WPs are already present for all contacts.

### Step 3 — Plot waveforms

```bash
python scripts/plot_gate_waveforms.py output/gate_waveforms_x+0.0mm.json
# → output/gate_event_<id>.png  (one per event)
```

---

## Output formats

### `gate_events_*.csv`

```
event_id, energy_keV, ssd_x_mm, ssd_y_mm, ssd_z_mm, gate_x_mm, gate_y_mm, gate_z_mm
```

### `gate_waveforms_*.json`

```json
{
  "simulator": "SolidStateDetectors.jl",
  "geometry":  "czt_cross_strip_full.yaml",
  "events": [{
    "event_id": 1001,
    "energy_keV": 662.0,
    "position_mm": {"x": 0.0, "y": 2.5, "z": 0.0},
    "collecting_anode":   "anode_20",
    "collecting_cathode": "cathode_4",
    "waveforms": {
      "anode_20": {
        "contact_id": 20, "contact_type": "anode",
        "raw_time_ns":    [...],
        "raw_current":    [...],
        "preamp_time_ns": [...],
        "preamp_signal":  [...]
      },
      ...
    }
  }]
}
```

---

## Synthetic test (no ROOT file needed)

A minimal test CSV is provided at `output/gate_events_test.csv`. Run:

```bash
julia --project=. -t8 scripts/simulate_gate_events.jl output/gate_events_test.csv
python scripts/plot_gate_waveforms.py output/gate_waveforms_test.json
```

The test CSV contains 3 synthetic 662 keV events at:

| Event | SSD x (mm) | SSD y (mm) | SSD z (mm) | Collecting |
|-------|-----------|-----------|-----------|------------|
| 1001  | 0.0       | 2.5       | 0.0       | anode_20 / cathode_4 |
| 1002  | 5.0       | −5.0      | −1.5      | anode_25 / cathode_3 |
| 1003  | −10.0     | 7.5       | 1.5       | anode_10 / cathode_6 |

---

## Caching notes

- Cache format: Julia `Serialization` (not JLD2 — the steering strip CSGUnion is too deeply nested for JLD2)
- The cache is Julia/package-version specific. If you update Julia or SSD.jl, delete the cache and rebuild.
- Default cache path: `output/sim_cache.jls` (override with `SSD_CACHE` env var)
- `build_wp_cache.jl` and `simulate_gate_events.jl` share the same default path

---

## Preamp parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| τ (decay) | 140 µs | Charge-sensitive amplifier decay constant |
| b0 | 1400 | IIR gain at dt = 0.1 ns |
| a1 | 0.9999992857 | IIR pole at dt = 0.1 ns |
| Display window | 5 µs | Preamp output saved for first 5 µs |
| Subsample | 5× | Preamp output exported at 0.5 ns effective step |

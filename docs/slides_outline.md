# CZT Detector Simulation — Slide Outline

## Slide 1: Title
**CZT Strip Detector Simulation with SolidStateDetectors.jl**
Subtitle: From geometry definition to electrode waveforms

---

## Slide 2: What is SolidStateDetectors.jl?
- Julia package for simulating solid-state radiation detectors
- Solves the electric field and weighting potentials on an adaptive 3D grid
- Drifts charge carriers (e⁻/h⁺), models trapping, computes induced signals
- Shockley-Ramo theorem: induced current = q · v_drift · ∇Φ_w
- Geometry defined in a single YAML file — no mesh editor or C++ required

---

## Slide 3: Two Files, One Command
```
geometry.yaml       → defines the detector
single_event.jl     → solves fields, simulates charge, plots waveforms
```
```bash
julia --project=. -t8 scripts/single_event.jl
```

---

## Slide 4: The Geometry File
Key sections: semiconductor (material, dimensions, μ, τ), contacts (id, potential, position/size)

Box geometry defined from centre point:
- origin = centre [x, y, z]
- hX/hY/hZ = half-widths (full width = 2×)
- hZ: 0 = surface contact (no thickness)

CZT example contacts: anodes 0V, steering −80V, cathodes −600V

---

## Slide 5: The Adaptive Grid — Refinement Limits
refinement_limits = [0.2, 0.1, 0.05] runs three passes:
1. Solve on coarse initial grid
2. Find neighbors where ΔΦ > 0.2 × V_bias → insert grid points → re-solve
3. Repeat at 0.1 × V_bias → denser near electrode edges
4. Repeat at 0.05 × V_bias → finest, concentrates cells at strip edges

Grid is adaptive — fine only where field changes rapidly (near contacts).
Finer limits = more accurate, more memory, slower.
convergence_limit = 1e-6: stop SOR iteration when max change < this.
depletion_handling = true: required for biased detectors.

---

## Slide 6: Charge Cloud & Drift Knobs
```julia
evt = Event([pos], [662.0u"keV"], N_CARRIERS;
    radius = [0.1u"mm"],    # physical cloud size
    number_of_shells = 2,
)
simulate!(evt, sim; Δt = 0.1u"ns", max_nsteps = 50_000)
```

| Knob | Effect |
|------|--------|
| radius | Cloud size — default 0.5mm causes staircase artifacts on small-pitch strips |
| N_CARRIERS | Macro-charge sampling points (not number of electrons) |
| number_of_shells | Spatial extent of cloud |
| Δt | Waveform time resolution |
| max_nsteps | Max drift time = Δt × max_nsteps |

Note: position is in metres, not mm.

---

## Slide 7: Built-in Visualizations — Fields & Geometry
```julia
plot(sim.detector)                              # 3D geometry with contacts
plot(sim.electric_potential, y = 0.0u"mm")     # E-potential xz cross-section
plot_electric_fieldlines!(sim, sampling=3u"mm") # field lines
plot(sim.weighting_potentials[id], y=0.0u"mm") # WP for one contact
```
[placeholder: electric_potential_xz.png]
[placeholder: weighting_potential.png]

---

## Slide 8: Built-in Visualizations — Drift & Waveforms
```julia
plot!(evt.drift_paths)   # overlay carrier trajectories on geometry
plot(evt.waveforms)      # all contacts: time vs induced charge
```
[placeholder: drift_paths.png]
[placeholder: single_event.png]

Anode: sharp rise as electrons collect → plateau = energy
Cathode: slower, amplitude = depth
Ratio cathode/anode → depth reconstruction

---

## Slide 9: Pitfalls
- hX/hY/hZ are half-widths — contacts end up twice as large if you forget
- hZ: 0 for surface contacts — any thickness changes the field solve
- Solve order is fixed: E-potential → E-field → weighting potentials
- Default cloud radius 0.5mm is too large for small-pitch strips → staircase artifact; use 0.1–0.2mm
- Position in CartesianPoint is metres, not mm
- τ_e for CZT: 10µs (example files) is research-grade; good commercial = ~1µs
- Weighting potential can be missing — always check before accessing waveforms
- Solve time scales with number of contacts — cache if iterating on events

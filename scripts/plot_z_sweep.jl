"""
Depth (Z) sweep: simulate one event per Z position and export waveform data
as JSON for plotting by scripts/plot_z_sweep.py (Matplotlib).

Interaction point: x=0, y=2.5 mm (cathode-8 strip), z swept in 6 steps.
Max drift time = 10 µs — holes drift slowly (~0.6 mm/µs) and events near
the anode require ~8 µs for full hole collection.

Run:  julia --project=. -t8 scripts/plot_z_sweep.jl
      python scripts/plot_z_sweep.py
"""

using SolidStateDetectors
using Unitful
using JSON3

const REPO     = abspath(joinpath(@__DIR__, ".."))
const GEOMETRY = joinpath(REPO, "geometries", "czt_cross_strip.yaml")
const OUTDIR   = joinpath(REPO, "output")
const REFINE   = [0.2, 0.1, 0.05]

const Z_POSITIONS  = range(-2.0, 2.0; length=6)
const DT_NS        = 0.1
const MAX_STEPS    = 100_000    # 10 µs max

const ANODE_ID   = 3
const CATHODE_ID = 8
const STEER_ID   = 6

mkpath(OUTDIR)

# ── Field solve ───────────────────────────────────────────────────────────────
println("Loading geometry...")
sim = Simulation{Float32}(GEOMETRY)
println("  $(length(sim.detector.contacts)) contacts")

println("Solving electric potential...")
t0 = @elapsed calculate_electric_potential!(sim;
    refinement_limits = REFINE, convergence_limit = 1e-6, depletion_handling = true)
println("  done ($(round(t0; digits=1))s)")
calculate_electric_field!(sim)

println("Solving weighting potentials...")
for contact in sim.detector.contacts
    t1 = @elapsed calculate_weighting_potential!(sim, contact.id;
        refinement_limits = REFINE, convergence_limit = 1e-6)
    println("  contact $(contact.id): $(round(t1; digits=1))s")
end

# ── Sweep and export ──────────────────────────────────────────────────────────
println("\nSweeping $(length(Z_POSITIONS)) Z positions...")

events = []
for z_mm in Z_POSITIONS
    pos = CartesianPoint{Float32}(0f0, Float32(2.5/1000), Float32(z_mm/1000))
    evt = Event([pos], [662f0 * u"keV"], 50;
        number_of_shells = 2, radius = [0.1u"mm"])
    simulate!(evt, sim; Δt = DT_NS * u"ns", max_nsteps = MAX_STEPS)
    println("  z=$(round(z_mm; digits=1)) mm done")

    rec = Dict("z_mm" => Float64(z_mm), "channels" => Dict{String,Any}())
    for (key, cid) in [("anode", ANODE_ID), ("cathode", CATHODE_ID), ("steering", STEER_ID)]
        wf = evt.waveforms[cid]
        ismissing(wf) && continue
        rec["channels"][key] = Dict(
            "time_ns" => Float64.(ustrip.(u"ns", collect(wf.time))),
            "signal"  => Float64.(ustrip.(collect(wf.signal))),
        )
    end
    push!(events, rec)
end

outjson = joinpath(OUTDIR, "z_sweep.json")
open(outjson, "w") do io
    JSON3.write(io, Dict("dt_ns" => DT_NS, "events" => events))
end
println("\nSaved → $outjson  (plot with: python scripts/plot_z_sweep.py)")

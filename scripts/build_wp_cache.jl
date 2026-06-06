#!/usr/bin/env julia
"""
Build (and verify) the reusable solved-simulation cache: electric field + all 48
weighting potentials for the fixed CZT geometry, serialized with Julia's
Serialization so future sweep_pipeline.jl runs skip the ~260 s solve.

Uses Serialization (not JLD2) because the 40-box steering CSGUnion is too deeply
nested for JLD2 to round-trip. Serialization is Julia/package-version specific.

Run:  JULIA_NUM_THREADS=8 julia --project=. -t8 scripts/build_wp_cache.jl
"""

const REPO = abspath(joinpath(@__DIR__, ".."))
const GEOMETRY = joinpath(REPO, "geometries", "czt_cross_strip_full.yaml")
const CACHE_FILE = get(ENV, "SSD_CACHE", joinpath(REPO, "output", "sweep", "sim_cache.jls"))
const REFINE = [0.2, 0.1, 0.05]

using SolidStateDetectors
using Unitful
using Serialization
mkpath(dirname(CACHE_FILE))

print("Parsing geometry … "); flush(stdout)
sim = Simulation{Float32}(GEOMETRY)
println("$(length(sim.detector.contacts)) contacts")

print("Electric potential … "); flush(stdout)
t = @elapsed calculate_electric_potential!(sim; refinement_limits=REFINE, convergence_limit=1e-6, depletion_handling=true)
println("$(round(t;digits=1))s")
calculate_electric_field!(sim)
print("Weighting potentials (all 48) … "); flush(stdout)
t = @elapsed for c in sim.detector.contacts
    print("$(c.id) "); flush(stdout)
    calculate_weighting_potential!(sim, c.id; refinement_limits=REFINE, convergence_limit=1e-6)
end
println("\n  done in $(round(t;digits=1))s")

print("Serializing → $(basename(CACHE_FILE)) … "); flush(stdout)
t = @elapsed serialize(CACHE_FILE, sim)
println("$(round(t;digits=1))s ($(round(filesize(CACHE_FILE)/1e6;digits=0)) MB)")

# ── Verify: reload and run one event ──
println("\nVerifying cache reload …")
t_load = @elapsed sim2 = deserialize(CACHE_FILE)
println("  deserialize: $(round(t_load;digits=1))s  ($(length(sim2.detector.contacts)) contacts, "
        * "$(count(c -> !ismissing(sim2.weighting_potentials[c.id]), sim2.detector.contacts)) WPs present)")
pos = CartesianPoint{Float32}(0.0f0, 0.0025f0, 0.0f0)
evt = Event([pos], [662.0f0 * u"keV"], 50; number_of_shells=2)
t_sim = @elapsed simulate!(evt, sim2; Δt=0.1u"ns", max_nsteps=50000)
nsig = count(w -> !ismissing(w), evt.waveforms)
println("  test simulate!: $(round(t_sim;digits=1))s  ($nsig signals)")
println(nsig > 0 ? "CACHE OK ✓ — future runs load in ~$(round(t_load;digits=0))s instead of ~260s." :
                   "CACHE PRODUCED NO SIGNALS ✗")

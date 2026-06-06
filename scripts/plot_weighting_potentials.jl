#!/usr/bin/env julia
"""
Compute and plot weighting potentials for representative electrodes of the full
CZT geometry, to verify they look physically correct.

Weighting potentials solve Laplace with the target contact at 1 V and all others
at 0 V, so the electric (bias) potential is NOT needed here.

Cross-sections drawn:
  - anode 20 (x=0)      : x–z slice at y=0   (expect sharp focusing to 1 at the 100 µm strip)
  - anode 21 (x=1)      : x–z slice at y=0   (neighbor strip)
  - steering 40         : x–z slice at y=0   (interleaved grid strips -> 1)
  - cathode 45 (y=2.5)  : y–z slice at x=0   (wide strip on bottom face)

Run with threads:  JULIA_NUM_THREADS=8 julia --project=. -t8 scripts/plot_weighting_potentials.jl
"""

using SolidStateDetectors
using Unitful
using Plots
gr()

const REPO = abspath(joinpath(@__DIR__, ".."))
const GEOMETRY = joinpath(REPO, "geometries", "czt_cross_strip_full.yaml")
const REFINE = [0.2, 0.1, 0.05]

print("Parsing geometry … "); flush(stdout)
sim = Simulation{Float32}(GEOMETRY)
println("$(length(sim.detector.contacts)) contacts")

# id => (label, slice-plane kwarg as (axis=>value))
targets = [
    (20, "anode_20 (x=0 mm)",       :y => 0.0u"mm"),
    (21, "anode_21 (x=+1 mm)",      :y => 0.0u"mm"),
    (40, "steering (x=half-int)",   :y => 0.0u"mm"),
    (45, "cathode_5 (y=+2.5 mm)",   :x => 0.0u"mm"),
]

for (id, label, _) in targets
    print("Weighting potential id=$id ($label) … "); flush(stdout)
    t = @elapsed calculate_weighting_potential!(sim, id;
        refinement_limits=REFINE, convergence_limit=1e-6)
    println("$(round(t; digits=1))s")
end

mkpath(joinpath(REPO, "output"))
plts = Plots.Plot[]
for (id, label, slicekw) in targets
    k, v = slicekw
    p = plot(sim.weighting_potentials[id]; (k => v,)...,
        title="W_$(id): $label", clims=(0, 1), colorbar=true)
    push!(plts, p)
    outind = joinpath(REPO, "output", "wp_$(id).png")
    savefig(p, outind)
    println("Wrote $outind")
end

combined = plot(plts...; layout=(2, 2), size=(1400, 1000))
outfile = joinpath(REPO, "output", "weighting_potentials.png")
savefig(combined, outfile)
println("Wrote $outfile")

#!/usr/bin/env julia
"""
Rerun the near-anode event (x=0, y=2.5, z=+2.0 mm) with shrinking charge-cloud
radius. Default Gamma radius is 0.5 mm (shells out to 1.0 mm) -> staircase.
Smaller radius -> cloud fits below the anode gap -> staircase should collapse.

Run:  JULIA_NUM_THREADS=8 julia --project=. -t8 scripts/test_radius.jl
"""

const REPO = abspath(joinpath(@__DIR__, ".."))
const CACHE_FILE = joinpath(REPO, "output", "sweep", "sim_cache.jls")
const ANODE_ID = 20
const DT_NS = 0.1
const MAX_NSTEPS = 50000
const PREAMP_B0 = 1400.0
const PREAMP_A1 = 0.9999992857142857
const XP, YP, ZP = 0.0, 2.5, 2.0

using SolidStateDetectors
using Unitful; using Unitful: ustrip
using Serialization
using Plots; gr()

function apply_preamp(current, dt_ns, b0, a1; display_us=3.0, subsample=5)
    n_pad = round(Int, display_us * 1000 / dt_ns); n_in = length(current); n_total = max(n_in, n_pad)
    out = zeros(Float64, n_total); out[1] = b0 * (n_in >= 1 ? current[1] : 0.0)
    for i in 2:n_total
        out[i] = b0 * (i <= n_in ? current[i] : 0.0) + a1 * out[i-1]
    end
    idx = 1:subsample:n_total
    return Float64.(collect(idx) .- 1) .* dt_ns, out[idx]
end

function rise(sim, radius_mm)
    pos = CartesianPoint{Float32}(Float32(XP/1000), Float32(YP/1000), Float32(ZP/1000))
    evt = Event([pos], [662.0f0 * u"keV"], 50; number_of_shells=2, radius=[radius_mm * u"mm"])
    simulate!(evt, sim; Δt=DT_NS * u"ns", max_nsteps=MAX_NSTEPS)
    wf = evt.waveforms[ANODE_ID]
    t_ns = Float64.(ustrip.(u"ns", collect(wf.time))); sig = Float64.(ustrip.(collect(wf.signal)))
    cur = diff(sig) ./ (t_ns[2]-t_ns[1])
    t_pre, sig_pre = apply_preamp(cur, DT_NS, PREAMP_B0, PREAMP_A1)
    sig_pre .*= -1.0
    plateau = sum(sig_pre[end-200:end]) / 201
    return t_pre ./ 1000, sig_pre ./ plateau
end

print("Loading cached simulation … "); flush(stdout)
sim = deserialize(CACHE_FILE)
println("ok")

configs = [
    (0.5,  "radius=0.5 mm (Gamma default)", :red),
    (0.2,  "radius=0.2 mm",                 :orange),
    (0.1,  "radius=0.1 mm",                 :seagreen),
    (0.05, "radius=0.05 mm",                :black),
]
plt = plot(xlabel="time (µs)", ylabel="anode preamp, normalized to plateau",
           title="z=+2.0 mm: anode rise vs charge-cloud radius (N=50, shells=2)",
           legend=:bottomright, xlims=(0, 0.3), ylims=(-0.05, 1.15))
for (r, lab, col) in configs
    print("  $lab … "); flush(stdout)
    t = @elapsed (tt, vv) = rise(sim, r)
    println("$(round(t;digits=1))s")
    plot!(plt, tt, vv, label=lab, color=col, lw=2)
end
out = joinpath(REPO, "output", "sweep_ideal", "radius_test.png")
savefig(plt, out)
println("Wrote $out")

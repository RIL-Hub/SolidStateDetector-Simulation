"""
X × Z sweep — simulate waveforms at 5 x-positions across the target
anode strip (x = -0.4, -0.2, 0, +0.2, +0.4 mm) for each of 10 z-depths
(0.2 to 5.0 mm from anode).  Purpose: see how much x-position affects
the anode waveform, especially near the anode where the small-pixel-
effect focusing has little time to concentrate the charge on the
target strip.

Total events: 5 x × 10 z = 50.  Field solve done once (~30 s).

Output: output/xz_sweep.json  containing 50 waveforms tagged by
(x_mm, z_from_anode_mm) with anode / cathode / steering channels.
"""

using SolidStateDetectors
using Unitful
using JSON3

const REPO       = abspath(joinpath(@__DIR__, ".."))
const GEOMETRY   = joinpath(REPO, "geometries", "czt_cross_strip.yaml")
const OUTDIR     = joinpath(REPO, "output")
const REFINE     = [0.2, 0.1, 0.05]
const DT_NS      = 0.1
const MAX_STEPS  = 100_000
const ANODE_ID   = 3
const CATHODE_ID = 8
const STEER_ID   = 6

const CLOUD_RADIUS_MM = 0.20
const CLOUD_SHELLS    = 2
const N_CARRIERS      = 50

# Paper depth convention: 0 = anode, 5 = cathode.
# Julia z convention: anode at +2.5 mm, cathode at −2.5 mm.
# So julia_z = 2.5 − paper_depth.
const X_MM     = [-0.4, -0.2, 0.0, +0.2, +0.4]
const DEPTHS_FROM_ANODE_MM = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 4.9]

mkpath(OUTDIR)

println("=== X×Z sweep — $(length(X_MM)) x × $(length(DEPTHS_FROM_ANODE_MM)) z = $(length(X_MM)*length(DEPTHS_FROM_ANODE_MM)) events ===")

# ── Field solve ──
println("Loading geometry ...")
sim = Simulation{Float32}(GEOMETRY)
println("  $(length(sim.detector.contacts)) contacts")

println("Solving electric potential ...")
t0 = @elapsed calculate_electric_potential!(sim;
    refinement_limits = REFINE, convergence_limit = 1e-6, depletion_handling = true)
println("  done ($(round(t0; digits=1))s)")
calculate_electric_field!(sim)

println("Solving weighting potentials ...")
for contact in sim.detector.contacts
    t1 = @elapsed calculate_weighting_potential!(sim, contact.id;
        refinement_limits = REFINE, convergence_limit = 1e-6)
    println("  contact $(contact.id): $(round(t1; digits=1))s")
end

# ── Event sweep ──
println("\nSweeping events ...")
events = []
const N_TOTAL = length(X_MM) * length(DEPTHS_FROM_ANODE_MM)
let n_done = 0, t_sweep = time()
for x_mm in X_MM
    for depth_mm in DEPTHS_FROM_ANODE_MM
        z_julia_mm = 2.5 - depth_mm
        pos = CartesianPoint{Float32}(
            Float32(x_mm/1000),
            Float32(2.5/1000),
            Float32(z_julia_mm/1000))
        evt = Event([pos], [662f0 * u"keV"], N_CARRIERS;
            number_of_shells = CLOUD_SHELLS,
            radius = [Float32(CLOUD_RADIUS_MM) * u"mm"])
        simulate!(evt, sim; Δt = DT_NS * u"ns", max_nsteps = MAX_STEPS)
        n_done += 1
        el = time() - t_sweep
        eta = el * (N_TOTAL - n_done) / n_done
        println("  [$n_done/$N_TOTAL] x=$(x_mm) mm depth=$(depth_mm) mm  ($(round(el;digits=0))s elapsed, ETA $(round(eta;digits=0))s)")

        rec = Dict(
            "x_mm" => Float64(x_mm),
            "depth_from_anode_mm" => Float64(depth_mm),
            "z_julia_mm" => Float64(z_julia_mm),
            "channels" => Dict{String,Any}(),
        )
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
end
end  # let

outjson = joinpath(OUTDIR, "xz_sweep.json")
open(outjson, "w") do io
    JSON3.write(io, Dict(
        "dt_ns" => DT_NS,
        "cloud_radius_mm" => CLOUD_RADIUS_MM,
        "cloud_shells" => CLOUD_SHELLS,
        "n_carriers" => N_CARRIERS,
        "x_positions_mm" => X_MM,
        "depths_from_anode_mm" => DEPTHS_FROM_ANODE_MM,
        "events" => events,
    ))
end
println("\nSaved → $outjson")

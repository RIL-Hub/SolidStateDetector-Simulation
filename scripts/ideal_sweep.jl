#!/usr/bin/env julia
"""
"Ideal-position" counterpart to the fuzzy GATE sweep: one event per slit step at
the EXACT intended position — centered on anode_20 (x=0) and cathode_5 (y=+2.5),
with depth ssd_z = -(collimator x), so depth runs +2 mm (anode side) → -2 mm
(cathode side). Uses the cached solved Simulation (no WP recompute).

Writes output/sweep_ideal/gate_waveforms_x<offset>mm.json in the same schema as
sweep_pipeline.jl, so scripts/plot_sweep_aligned.py plots it identically.

Run:  JULIA_NUM_THREADS=8 julia --project=. -t8 scripts/ideal_sweep.jl
"""

const REPO = abspath(joinpath(@__DIR__, ".."))
const CACHE_FILE = joinpath(REPO, "output", "sweep", "sim_cache.jls")
const OUT_DIR = joinpath(REPO, "output", "sweep_ideal")
const GEOMETRY = joinpath(REPO, "geometries", "czt_cross_strip_full.yaml")
const REFINE = [0.2, 0.1, 0.05]

const SSD_X_IDEAL = 0.0    # mm, centered on anode_20
const SSD_Y_IDEAL = 2.5    # mm, centered on cathode_5
const COLLIMATORS = -2.0:0.5:2.0          # slit positions (mm)
const ENERGY_KEV = 662.0
const N_CARRIERS = 50
const CLOUD_RADIUS_MM = 0.1   # physical charge-cloud radius (default Gamma 0.5mm causes staircasing)
const DT_NS = 0.1
const MAX_NSTEPS = 50000
const N_ANODE = 39
const STEER_ID = 40
const CATHODE_ID0 = STEER_ID
const PREAMP_B0 = 1400.0
const PREAMP_A1 = 0.9999992857142857
const PREAMP_DISPLAY_US = 5.0
const PREAMP_SUBSAMPLE = 5

contact_name(id) = id <= N_ANODE ? "anode_$(id)" : (id == STEER_ID ? "steering" : "cathode_$(id - CATHODE_ID0)")
contact_type(id) = id <= N_ANODE ? "anode" : (id == STEER_ID ? "steering" : "cathode")
anode_id_for_x(x)   = clamp(round(Int, x) + (N_ANODE + 1) ÷ 2, 1, N_ANODE)
cathode_id_for_y(y) = CATHODE_ID0 + clamp(round(Int, y / 5 + (8 + 1) / 2), 1, 8)

to_json(v::AbstractVector{<:Number}) = "[" * join((isnan(x)||isinf(x) ? "null" : string(Float64(x)) for x in v), ",") * "]"
to_json(v::Number) = (isnan(v)||isinf(v)) ? "null" : string(Float64(v))
to_json(v::String) = "\"$(replace(v, "\"" => "\\\""))\""
function to_json(d::AbstractDict)
    io = IOBuffer(); print(io, "{")
    for (i, (k, v)) in enumerate(d)
        i > 1 && print(io, ",")
        print(io, "\"$k\":")
        v isa AbstractDict ? print(io, to_json(v)) : v isa AbstractVector ? print(io, to_json(v)) :
        v isa Number ? print(io, to_json(v)) : v isa String ? print(io, to_json(v)) : print(io, "\"$(v)\"")
    end
    print(io, "}"); String(take!(io))
end
to_json(v::AbstractVector{<:AbstractDict}) = "[" * join(to_json.(v), ",") * "]"

function apply_preamp(current, dt_ns, b0, a1; display_us=5.0, subsample=5)
    n_pad = round(Int, display_us * 1000 / dt_ns); n_in = length(current); n_total = max(n_in, n_pad)
    out = zeros(Float64, n_total); out[1] = b0 * (n_in >= 1 ? current[1] : 0.0)
    for i in 2:n_total
        out[i] = b0 * (i <= n_in ? current[i] : 0.0) + a1 * out[i-1]
    end
    idx = 1:subsample:n_total
    return Float64.(collect(idx) .- 1) .* dt_ns, out[idx]
end

print("Loading SolidStateDetectors … "); flush(stdout)
using SolidStateDetectors
using Unitful; using Unitful: ustrip
using Serialization
println("ok")

local sim
if isfile(CACHE_FILE)
    print("Loading cached solved simulation … "); flush(stdout)
    try
        t = @elapsed (sim = deserialize(CACHE_FILE)); println("$(round(t;digits=1))s")
    catch err
        println("cache failed ($(typeof(err))) — recomputing."); global sim = nothing
    end
end
if !@isdefined(sim) || sim === nothing
    sim = Simulation{Float32}(GEOMETRY)
    calculate_electric_potential!(sim; refinement_limits=REFINE, convergence_limit=1e-6, depletion_handling=true)
    calculate_electric_field!(sim)
    for c in sim.detector.contacts
        calculate_weighting_potential!(sim, c.id; refinement_limits=REFINE, convergence_limit=1e-6)
    end
end

function extract_primaries(evt, aid, cid)
    out = Dict{String,Any}()
    for id in (aid, cid, STEER_ID)
        id <= length(evt.waveforms) || continue
        wf = evt.waveforms[id]; ismissing(wf) && continue
        t_ns = Float64.(ustrip.(u"ns", collect(wf.time))); sig = Float64.(ustrip.(collect(wf.signal)))
        length(t_ns) < 2 && continue
        dt = t_ns[2] - t_ns[1]; cur = diff(sig) ./ dt; t_mid = t_ns[1:end-1] .+ dt/2
        t_pre, sig_pre = apply_preamp(cur, DT_NS, PREAMP_B0, PREAMP_A1; display_us=PREAMP_DISPLAY_US, subsample=PREAMP_SUBSAMPLE)
        raw_idx = 1:10:length(cur)
        out[contact_name(id)] = Dict{String,Any}("contact_id"=>id, "contact_type"=>contact_type(id),
            "raw_time_ns"=>t_mid[raw_idx], "raw_current"=>cur[raw_idx],
            "preamp_time_ns"=>t_pre, "preamp_signal"=>sig_pre)
    end
    return out
end

mkpath(OUT_DIR)
aid = anode_id_for_x(SSD_X_IDEAL); cid = cathode_id_for_y(SSD_Y_IDEAL)
println("Ideal lateral position: $(contact_name(aid)) (x=$SSD_X_IDEAL), $(contact_name(cid)) (y=$SSD_Y_IDEAL)")
println("Simulating $(length(COLLIMATORS)) ideal positions:")
for coll in COLLIMATORS
    z_ideal = -coll                      # ssd_z = -(collimator x)
    tag = "x" * (coll >= 0 ? "+" : "-") * string(abs(coll)) * "mm"
    print("  $tag  (x=$SSD_X_IDEAL, y=$SSD_Y_IDEAL, z=$z_ideal) … "); flush(stdout)
    pos = CartesianPoint{Float32}(Float32(SSD_X_IDEAL/1000), Float32(SSD_Y_IDEAL/1000), Float32(z_ideal/1000))
    evt = Event([pos], [Float32(ENERGY_KEV) * u"keV"], N_CARRIERS; number_of_shells=2, radius=[CLOUD_RADIUS_MM * u"mm"])
    t = @elapsed simulate!(evt, sim; Δt=DT_NS * u"ns", max_nsteps=MAX_NSTEPS)
    rec = Dict{String,Any}("event_id"=>0, "energy_keV"=>ENERGY_KEV,
        "position_mm"=>Dict("x"=>SSD_X_IDEAL, "y"=>SSD_Y_IDEAL, "z"=>z_ideal),
        "collecting_anode"=>contact_name(aid), "collecting_cathode"=>contact_name(cid),
        "waveforms"=>extract_primaries(evt, aid, cid))
    output = Dict{String,Any}("simulator"=>"SolidStateDetectors.jl", "tag"=>tag, "ideal"=>true,
        "n_carriers"=>N_CARRIERS, "dt_ns"=>DT_NS, "preamp_tau_us"=>140.0, "events"=>[rec])
    write(joinpath(OUT_DIR, "gate_waveforms_$(tag).json"), to_json(output) * "\n")
    println("$(round(t;digits=1))s")
end
println("Wrote 9 ideal JSONs to $OUT_DIR")

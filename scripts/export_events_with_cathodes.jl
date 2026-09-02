#!/usr/bin/env julia
"""
Small re-sim to capture cathode waveforms alongside A19/A20/A21 + steering.
Uses the cached full-geometry solve (all 48 WPs already computed) so this
runs in seconds.

Simulates 5 events at (x = -0.4, -0.2, 0.0, +0.2, +0.4 mm), y = 2.5,
z = 0.0 (mid depth), V = -80 V (default geometry).
Records waveforms for A19, A20, A21, steering, and all 8 cathodes.
"""

const REPO = abspath(joinpath(@__DIR__, ".."))
const CACHE_FILE = get(ENV, "SSD_CACHE",
    joinpath(REPO, "output", "sweep", "sim_cache.jls"))
const OUT_JSON = joinpath(REPO, "output", "lateral",
                            "events_with_cathodes_mid_v80.json")

const X_LIST = [-0.4, -0.2, 0.0, +0.2, +0.4]
const Y_MM, Z_MM = 2.5, 0.0
const DEPTH_MM = 2.5 - Z_MM
const ENERGY_KEV = 662.0
const N_CARRIERS = 50
const CLOUD_RADIUS_MM = 0.2
const NSHELLS = 2
const DT_NS = 0.1
const MAX_NSTEPS = 50000
const N_ANODE = 39
const STEER_ID = 40
const CATHODE_ID0 = STEER_ID
const CAPTURE_ANODES = [19, 20, 21]
const CAPTURE_CATHODES = collect(41:48)          # all 8 cathodes
const PREAMP_B0 = 1400.0
const PREAMP_A1 = 0.9999992857142857
const PREAMP_DISPLAY_US = 5.0
const PREAMP_SUBSAMPLE = 5

contact_name(id) = id <= N_ANODE ? "anode_$(id)" :
    (id == STEER_ID ? "steering" : "cathode_$(id - CATHODE_ID0)")
contact_type(id) = id <= N_ANODE ? "anode" :
    (id == STEER_ID ? "steering" : "cathode")

to_json(v::AbstractVector{<:Number}) = "[" * join((isnan(x)||isinf(x) ? "null" : string(Float64(x)) for x in v), ",") * "]"
to_json(v::AbstractVector{<:AbstractString}) = "[" * join((to_json(s) for s in v), ",") * "]"
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

isfile(CACHE_FILE) || error("Cache not found: $CACHE_FILE. Build with build_wp_cache.jl.")
print("Loading cache ($(round(filesize(CACHE_FILE)/1e6;digits=0)) MB) … "); flush(stdout)
t = @elapsed (sim = deserialize(CACHE_FILE)); println("$(round(t;digits=1))s")

function extract(evt, ids)
    out = Dict{String,Any}()
    for id in ids
        id <= length(evt.waveforms) || continue
        wf = evt.waveforms[id]; ismissing(wf) && continue
        t_ns = Float64.(ustrip.(u"ns", collect(wf.time)))
        sig = Float64.(ustrip.(collect(wf.signal)))
        length(t_ns) < 2 && continue
        dt = t_ns[2] - t_ns[1]; cur = diff(sig) ./ dt; t_mid = t_ns[1:end-1] .+ dt/2
        t_pre, sig_pre = apply_preamp(cur, DT_NS, PREAMP_B0, PREAMP_A1;
            display_us=PREAMP_DISPLAY_US, subsample=PREAMP_SUBSAMPLE)
        raw_idx = 1:10:length(cur)
        out[contact_name(id)] = Dict{String,Any}(
            "contact_id"=>id, "contact_type"=>contact_type(id),
            "raw_time_ns"=>t_mid[raw_idx], "raw_current"=>cur[raw_idx],
            "preamp_time_ns"=>t_pre, "preamp_signal"=>sig_pre)
    end
    return out
end

ids = vcat(CAPTURE_ANODES, [STEER_ID], CAPTURE_CATHODES)
records = Dict{String,Any}[]
println("Simulating $(length(X_LIST)) events at mid depth with cathodes:")
for x in X_LIST
    print("  x=$(x) mm … "); flush(stdout)
    pos = CartesianPoint{Float32}(Float32(x/1000), Float32(Y_MM/1000),
                                   Float32(Z_MM/1000))
    evt = Event([pos], [Float32(ENERGY_KEV) * u"keV"], N_CARRIERS;
        number_of_shells=NSHELLS, radius=[CLOUD_RADIUS_MM * u"mm"])
    t = @elapsed simulate!(evt, sim; Δt=DT_NS * u"ns", max_nsteps=MAX_NSTEPS)
    push!(records, Dict{String,Any}(
        "x_mm"=>x, "y_mm"=>Y_MM, "z_mm"=>Z_MM,
        "depth_from_anode_mm"=>DEPTH_MM,
        "energy_keV"=>ENERGY_KEV,
        "waveforms"=>extract(evt, ids)))
    println("$(round(t;digits=1))s")
end

output = Dict{String,Any}(
    "simulator"=>"SolidStateDetectors.jl",
    "y_mm"=>Y_MM, "z_mm"=>Z_MM, "depth_from_anode_mm"=>DEPTH_MM,
    "captured_anodes"=>[contact_name(a) for a in CAPTURE_ANODES],
    "captured_cathodes"=>[contact_name(c) for c in CAPTURE_CATHODES],
    "cloud_radius_mm"=>CLOUD_RADIUS_MM,
    "n_carriers"=>N_CARRIERS,
    "dt_ns"=>DT_NS,
    "points"=>records,
)
mkpath(dirname(OUT_JSON))
write(OUT_JSON, to_json(output) * "\n")
println("Wrote $OUT_JSON")

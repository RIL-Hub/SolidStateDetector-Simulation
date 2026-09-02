#!/usr/bin/env julia
"""
Wide lateral scan (−2 to +2 mm across A17–A23) at 3 depths (paper convention:
depth from anode = 0.5, 2.5, 4.5 mm → SSD z = 2.0, 0.0, -2.0).

Same physics as lateral_sweep_steer_wide.jl, just an outer z-loop.
Field solve is done once (steering voltage stays fixed for one run).

x range: −2.0 to +2.0 mm, 50 µm step (81 points).
Captured: A17..A23 (7 anodes) + steering.

Reads:  ENV["GEOMETRY"]
Writes: ENV["OUTPUT"]
"""

const REPO = abspath(joinpath(@__DIR__, ".."))
const GEOMETRY = get(ENV, "GEOMETRY",
    joinpath(REPO, "geometries", "czt_cross_strip_full.yaml"))
const OUT_JSON = get(ENV, "OUTPUT",
    joinpath(REPO, "output", "lateral",
             "lateral_sweep_wide_xz_" *
             replace(basename(GEOMETRY), ".yaml" => "") * ".json"))

const REFINE = [0.2, 0.1, 0.05]
const X_RANGE = -2.0:0.05:2.0           # 81 lateral positions, 50 µm step
const Z_LIST_MM = [2.0, 0.0, -2.0]      # SSD z; = paper depths 0.5, 2.5, 4.5 mm from anode
const Y_MM = 2.5
const ENERGY_KEV = 662.0
const N_CARRIERS = 50
const CLOUD_RADIUS_MM = 0.2
const NSHELLS = 2
const DT_NS = 0.1
const MAX_NSTEPS = 50000
const PRIMARY_ANODE = 20
const CAPTURE_ANODES = [17, 18, 19, 20, 21, 22, 23]
const STEER_ID = 40
const N_ANODE = 39
const CATHODE_ID0 = STEER_ID
const PREAMP_B0 = 1400.0
const PREAMP_A1 = 0.9999992857142857
const PREAMP_DISPLAY_US = 5.0
const PREAMP_SUBSAMPLE = 5

contact_name(id) = id <= N_ANODE ? "anode_$(id)" : (id == STEER_ID ? "steering" : "cathode_$(id - CATHODE_ID0)")
contact_type(id) = id <= N_ANODE ? "anode" : (id == STEER_ID ? "steering" : "cathode")

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

t_all_start = time()
print("Loading SolidStateDetectors … "); flush(stdout)
using SolidStateDetectors
using Unitful; using Unitful: ustrip
println("ok")

println("Geometry: $GEOMETRY")
print("Parsing … "); flush(stdout)
sim = Simulation{Float32}(GEOMETRY)
println("$(length(sim.detector.contacts)) contacts")

t_field_start = time()
print("Electric potential … "); flush(stdout)
t = @elapsed calculate_electric_potential!(sim; refinement_limits=REFINE,
    convergence_limit=1e-6, depletion_handling=true)
println("$(round(t;digits=1))s")
calculate_electric_field!(sim)
print("Weighting potentials for anodes 17-23 + steering (8 total) … ")
flush(stdout)
t = @elapsed for id in vcat(CAPTURE_ANODES, [STEER_ID])
    print("$id "); flush(stdout)
    calculate_weighting_potential!(sim, id; refinement_limits=REFINE,
        convergence_limit=1e-6)
end
println(" done in $(round(t;digits=1))s")
t_field_solve = time() - t_field_start

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

t_events_start = time()
ids = vcat(CAPTURE_ANODES, [STEER_ID])
records = Dict{String,Any}[]
println("Lateral scan at 3 depths × $(length(X_RANGE)) x-points from "
        * "$(minimum(X_RANGE)) to $(maximum(X_RANGE)) mm:")
for z_mm in Z_LIST_MM
    depth_from_anode = 2.5 - z_mm
    println("─── z = $z_mm mm  (depth $depth_from_anode from anode) ───")
    for x in X_RANGE
        print("  x=$(round(x;digits=3)) mm … "); flush(stdout)
        pos = CartesianPoint{Float32}(Float32(x/1000), Float32(Y_MM/1000),
                                       Float32(z_mm/1000))
        evt = Event([pos], [Float32(ENERGY_KEV) * u"keV"], N_CARRIERS;
            number_of_shells=NSHELLS, radius=[CLOUD_RADIUS_MM * u"mm"])
        t = @elapsed simulate!(evt, sim; Δt=DT_NS * u"ns",
                                 max_nsteps=MAX_NSTEPS)
        push!(records, Dict{String,Any}(
            "x_mm"=>x, "y_mm"=>Y_MM, "z_mm"=>z_mm,
            "depth_from_anode_mm"=>depth_from_anode,
            "energy_keV"=>ENERGY_KEV,
            "primary_anode"=>contact_name(PRIMARY_ANODE),
            "waveforms"=>extract(evt, ids)))
        println("$(round(t;digits=1))s")
    end
end
t_events = time() - t_events_start
t_total = time() - t_all_start

println("─── Timing summary ($(basename(GEOMETRY))) ───")
println("  Field solve (E + $(length(ids)) WPs):  $(round(t_field_solve;digits=1))s")
println("  Events ($(length(records)) × simulate!): $(round(t_events;digits=1))s")
println("  Total wall time:                        $(round(t_total;digits=1))s")

output = Dict{String,Any}(
    "simulator"=>"SolidStateDetectors.jl",
    "geometry_file"=>GEOMETRY,
    "primary_anode"=>contact_name(PRIMARY_ANODE),
    "y_mm"=>Y_MM,
    "z_list_mm"=>collect(Z_LIST_MM),
    "captured_anodes"=>[contact_name(a) for a in CAPTURE_ANODES],
    "cloud_radius_mm"=>CLOUD_RADIUS_MM,
    "n_carriers"=>N_CARRIERS,
    "dt_ns"=>DT_NS,
    "wall_time_s"=>t_total,
    "field_solve_s"=>t_field_solve,
    "events_s"=>t_events,
    "points"=>records,
)
mkpath(dirname(OUT_JSON))
write(OUT_JSON, to_json(output) * "\n")
println("Wrote $OUT_JSON")

#!/usr/bin/env julia
"""
Run the waveform pipeline across the whole collimator sweep.

Computes the electric field + ALL 48 weighting potentials ONCE (the event-
independent, dominant cost), optionally caching the solved Simulation to JLD2,
then loops over every output/sweep/gate_events_*.csv and simulates its events,
exporting only the PRIMARY channels (collecting anode, collecting cathode,
steering) to keep the JSON small.

Run:  JULIA_NUM_THREADS=8 julia --project=. -t8 scripts/sweep_pipeline.jl
ENV:  SWEEP_CACHE=1   (default) save/load output/sweep/sim_cache.jld2
      SWEEP_CACHE=0   recompute, don't save
"""

using Dates

const REPO = abspath(joinpath(@__DIR__, ".."))
const SWEEP_DIR = joinpath(REPO, "output", "sweep")
const GEOMETRY = joinpath(REPO, "geometries", "czt_cross_strip_full.yaml")
const CACHE = get(ENV, "SWEEP_CACHE", "1") == "1"
const CACHE_FILE = get(ENV, "SSD_CACHE", joinpath(SWEEP_DIR, "sim_cache.jls"))
const REFINE = [0.2, 0.1, 0.05]

const N_CARRIERS = 50
const CLOUD_RADIUS_MM = 0.1   # physical charge-cloud radius (default Gamma 0.5mm staircases)
const DT_NS = 0.1
const MAX_NSTEPS = 50000
const N_ANODE = 39
const STEER_ID = 40
const N_CATHODE = 8
const CATHODE_ID0 = STEER_ID

const PREAMP_B0 = 1400.0
const PREAMP_A1 = 0.9999992857142857
const PREAMP_DISPLAY_US = 5.0
const PREAMP_SUBSAMPLE = 5

contact_name(id) = id <= N_ANODE ? "anode_$(id)" : (id == STEER_ID ? "steering" : "cathode_$(id - CATHODE_ID0)")
contact_type(id) = id <= N_ANODE ? "anode" : (id == STEER_ID ? "steering" : "cathode")
anode_id_for_x(x)   = clamp(round(Int, x) + (N_ANODE + 1) ÷ 2, 1, N_ANODE)
cathode_id_for_y(y) = CATHODE_ID0 + clamp(round(Int, y / 5 + (N_CATHODE + 1) / 2), 1, N_CATHODE)

# JSON helpers
to_json(v::AbstractVector{<:Number}) = "[" * join((isnan(x)||isinf(x) ? "null" : string(Float64(x)) for x in v), ",") * "]"
to_json(v::Number) = (isnan(v)||isinf(v)) ? "null" : string(Float64(v))
to_json(v::String) = "\"$(replace(v, "\"" => "\\\""))\""
function to_json(d::AbstractDict)
    io = IOBuffer(); print(io, "{")
    for (i, (k, v)) in enumerate(d)
        i > 1 && print(io, ",")
        print(io, "\"$k\":")
        v isa AbstractDict ? print(io, to_json(v)) :
        v isa AbstractVector ? print(io, to_json(v)) :
        v isa Number ? print(io, to_json(v)) :
        v isa String ? print(io, to_json(v)) : print(io, "\"$(v)\"")
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

function read_events(path)
    lines = readlines(path); header = split(lines[1], ",")
    ci(name) = findfirst(==(name), header)
    cols = (ci("event_id"), ci("energy_keV"), ci("ssd_x_mm"), ci("ssd_y_mm"), ci("ssd_z_mm"))
    [(id=parse(Int, f[cols[1]]), energy_keV=parse(Float64, f[cols[2]]),
      x=parse(Float64, f[cols[3]]), y=parse(Float64, f[cols[4]]), z=parse(Float64, f[cols[5]]))
     for f in split.(filter(!isempty, strip.(lines[2:end])), ",")]
end

# ── Load SSD + solve (once) ──
print("Loading SolidStateDetectors … "); flush(stdout)
t_pkg = @elapsed using SolidStateDetectors
using Unitful; using Unitful: ustrip
using Serialization
println("$(round(t_pkg;digits=1))s")

# The detector geometry is fixed, so the field + all weighting potentials are
# reusable. We cache the whole solved Simulation with Julia's Serialization
# (NOT JLD2 — the 40-box steering CSGUnion is too deeply nested for JLD2 to
# round-trip). Serialization is Julia/package-version specific; safe for local reuse.
function solve_sim()
    print("Parsing geometry … "); flush(stdout)
    t_geom = @elapsed (s = Simulation{Float32}(GEOMETRY))
    println("$(round(t_geom;digits=1))s ($(length(s.detector.contacts)) contacts)")
    print("Electric potential … "); flush(stdout)
    t_pot = @elapsed calculate_electric_potential!(s; refinement_limits=REFINE, convergence_limit=1e-6, depletion_handling=true)
    println("$(round(t_pot;digits=1))s")
    print("Electric field … "); flush(stdout)
    t_field = @elapsed calculate_electric_field!(s); println("$(round(t_field;digits=1))s")
    print("Weighting potentials (all $(length(s.detector.contacts))) … "); flush(stdout)
    t_wp = @elapsed for c in s.detector.contacts
        print("$(c.id) "); flush(stdout)
        calculate_weighting_potential!(s, c.id; refinement_limits=REFINE, convergence_limit=1e-6)
    end
    println("\n  done in $(round(t_wp;digits=1))s")
    return s
end

local sim
loaded = false
if CACHE && isfile(CACHE_FILE)
    print("Loading cached solved simulation ($(round(filesize(CACHE_FILE)/1e6;digits=0)) MB) … "); flush(stdout)
    try
        t = @elapsed (sim = deserialize(CACHE_FILE))
        println("$(round(t;digits=1))s"); loaded = true
    catch err
        println("FAILED ($(typeof(err))) — recomputing."); @warn "cache load failed" exception=err
    end
end
if !loaded
    sim = solve_sim()
    if CACHE
        print("Caching solved simulation → $(basename(CACHE_FILE)) … "); flush(stdout)
        t = @elapsed serialize(CACHE_FILE, sim)
        println("$(round(t;digits=1))s ($(round(filesize(CACHE_FILE)/1e6;digits=0)) MB)")
    end
end

# ── Extract the 3 primary channels for an event ──
function extract_primaries(evt, e)
    ids = [anode_id_for_x(e.x), cathode_id_for_y(e.y), STEER_ID]
    out = Dict{String,Any}()
    for cid in ids
        cid <= length(evt.waveforms) || continue
        wf = evt.waveforms[cid]; ismissing(wf) && continue
        t_ns = Float64.(ustrip.(u"ns", collect(wf.time)))
        sig = Float64.(ustrip.(collect(wf.signal)))
        length(t_ns) < 2 && continue
        dt = t_ns[2] - t_ns[1]; cur = diff(sig) ./ dt; t_mid = t_ns[1:end-1] .+ dt/2
        t_pre, sig_pre = apply_preamp(cur, DT_NS, PREAMP_B0, PREAMP_A1; display_us=PREAMP_DISPLAY_US, subsample=PREAMP_SUBSAMPLE)
        raw_idx = 1:10:length(cur)
        out[contact_name(cid)] = Dict{String,Any}(
            "contact_id"=>cid, "contact_type"=>contact_type(cid),
            "raw_time_ns"=>t_mid[raw_idx], "raw_current"=>cur[raw_idx],
            "preamp_time_ns"=>t_pre, "preamp_signal"=>sig_pre)
    end
    return out
end

# ── Loop over the sweep files ──
csvs = sort(filter(f -> startswith(basename(f), "gate_events_") && endswith(f, ".csv"), readdir(SWEEP_DIR; join=true)))
println("\nProcessing $(length(csvs)) files:")
for csv in csvs
    tag = replace(basename(csv), "gate_events_"=>"", ".csv"=>"")
    events = read_events(csv)
    recs = Dict{String,Any}[]
    print("  $tag ($(length(events)) events): "); flush(stdout)
    for e in events
        pos = CartesianPoint{Float32}(Float32(e.x/1000), Float32(e.y/1000), Float32(e.z/1000))
        evt = Event([pos], [Float32(e.energy_keV) * u"keV"], N_CARRIERS; number_of_shells=2, radius=[CLOUD_RADIUS_MM * u"mm"])
        try
            simulate!(evt, sim; Δt=DT_NS * u"ns", max_nsteps=MAX_NSTEPS)
            push!(recs, Dict{String,Any}(
                "event_id"=>e.id, "energy_keV"=>e.energy_keV,
                "position_mm"=>Dict("x"=>e.x, "y"=>e.y, "z"=>e.z),
                "collecting_anode"=>contact_name(anode_id_for_x(e.x)),
                "collecting_cathode"=>contact_name(cathode_id_for_y(e.y)),
                "waveforms"=>extract_primaries(evt, e)))
            print("."); flush(stdout)
        catch err
            print("x"); flush(stdout)
            @warn "Skipped event $(e.id) at ($(e.x),$(e.y),$(e.z)) mm" exception=err
        end
    end
    output = Dict{String,Any}(
        "simulator"=>"SolidStateDetectors.jl", "tag"=>tag,
        "n_carriers"=>N_CARRIERS, "dt_ns"=>DT_NS, "preamp_tau_us"=>140.0,
        "events"=>recs)
    out = joinpath(SWEEP_DIR, "gate_waveforms_$(tag).json")
    write(out, to_json(output) * "\n")
    println(" → $(basename(out))")
end
println("\nAll done.")

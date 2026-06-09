#!/usr/bin/env julia
"""
Simulate CZT waveforms for GATE interaction events.

Reads an events CSV produced by scripts/load_gate_hits.py (SSD-frame mm positions
+ energy keV), loads the full 39-anode/steering/8-cathode geometry, computes the
electric potential + field once, then computes weighting potentials ONLY for the
contacts each event can induce signal on (anodes within ±ANODE_NEIGHBORS strips,
steering, cathodes within ±CATHODE_NEIGHBORS strips). Each event is drifted and
the raw induced current + preamp-shaped output exported to JSON.

Usage:
    julia --project=. scripts/simulate_gate_events.jl output/gate_events_x+0.0mm.csv

ENV overrides: SSD_N_CARRIERS (50), SSD_DT_NS (0.1), SSD_MAX_NSTEPS (50000),
               SSD_REFINE ("0.2,0.1,0.05"), SSD_GEOMETRY (full yaml path).
"""

using Dates

const REPO = abspath(joinpath(@__DIR__, ".."))
mkpath(joinpath(REPO, "output"))

const CSV_PATH = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(REPO, "output", "gate_events_x+0.0mm.csv")
const GEOMETRY = get(ENV, "SSD_GEOMETRY",
    joinpath(REPO, "geometries", "czt_cross_strip_full.yaml"))

const N_CARRIERS = parse(Int,     get(ENV, "SSD_N_CARRIERS", "50"))
const DT_NS      = parse(Float64, get(ENV, "SSD_DT_NS", "0.1"))
const MAX_NSTEPS = parse(Int,     get(ENV, "SSD_MAX_NSTEPS", "50000"))
const REFINE     = parse.(Float64, split(get(ENV, "SSD_REFINE", "0.2,0.1,0.05"), ","))

# Contacts to evaluate around each event
const ANODE_NEIGHBORS   = 2
const CATHODE_NEIGHBORS = 1

# Detector contact-id layout (must match gen_full_geometry.jl)
const N_ANODE   = 39
const STEER_ID  = 40
const N_CATHODE = 8
const CATHODE_ID0 = STEER_ID  # cathode j (1..8) -> id = STEER_ID + j

# Preamp: b0=1400, τ=140μs at dt=0.1ns
const PREAMP_B0 = 1400.0
const PREAMP_A1 = 0.9999992857142857
const PREAMP_DISPLAY_US = 5.0
const PREAMP_SUBSAMPLE  = 5

# ── id ↔ name/type helpers ──
function contact_name(id)
    id <= N_ANODE      && return "anode_$(id)"
    id == STEER_ID     && return "steering"
    return "cathode_$(id - CATHODE_ID0)"
end
function contact_type(id)
    id <= N_ANODE  && return "anode"
    id == STEER_ID && return "steering"
    return "cathode"
end
# Position → nearest contact id
anode_id_for_x(x)   = clamp(round(Int, x) + (N_ANODE + 1) ÷ 2, 1, N_ANODE)         # x=-19→1, 0→20, 19→39
cathode_id_for_y(y) = CATHODE_ID0 + clamp(round(Int, y / 5 + (N_CATHODE + 1) / 2), 1, N_CATHODE)

# ── JSON writer (no extra deps) ──
to_json(v::AbstractVector{<:Number}) =
    "[" * join((isnan(x) || isinf(x) ? "null" : string(Float64(x)) for x in v), ",") * "]"
to_json(v::Number) = (isnan(v) || isinf(v)) ? "null" : string(Float64(v))
to_json(v::String) = "\"$(replace(v, "\"" => "\\\""))\""
function to_json(d::AbstractDict)
    io = IOBuffer(); print(io, "{")
    for (i, (k, v)) in enumerate(d)
        i > 1 && print(io, ",")
        print(io, "\"$k\":")
        if v isa AbstractDict;            print(io, to_json(v))
        elseif v isa AbstractVector;      print(io, to_json(v))
        elseif v isa Number;              print(io, to_json(v))
        elseif v isa String;              print(io, to_json(v))
        else;                             print(io, "\"$(v)\"") end
    end
    print(io, "}"); return String(take!(io))
end
to_json(v::AbstractVector{<:AbstractDict}) = "[" * join(to_json.(v), ",") * "]"

# ── Preamp (charge-sensitive, exponential decay IIR) ──
function apply_preamp(current, dt_ns, b0, a1; display_us=5.0, subsample=5)
    n_pad = round(Int, display_us * 1000 / dt_ns)
    n_in = length(current)
    n_total = max(n_in, n_pad)
    out = zeros(Float64, n_total)
    out[1] = b0 * (n_in >= 1 ? current[1] : 0.0)
    for i in 2:n_total
        x = i <= n_in ? current[i] : 0.0
        out[i] = b0 * x + a1 * out[i-1]
    end
    idx = 1:subsample:n_total
    return Float64.(collect(idx) .- 1) .* dt_ns, out[idx]
end

# ── Read events CSV (header: event_id,energy_keV,ssd_x_mm,ssd_y_mm,ssd_z_mm,gate_*) ──
function read_events(path)
    lines = readlines(path)
    header = split(lines[1], ",")
    col(name) = findfirst(==(name), header)
    ci_id, ci_e = col("event_id"), col("energy_keV")
    ci_x, ci_y, ci_z = col("ssd_x_mm"), col("ssd_y_mm"), col("ssd_z_mm")
    events = NamedTuple[]
    for ln in lines[2:end]
        isempty(strip(ln)) && continue
        f = split(ln, ",")
        push!(events, (id = parse(Int, f[ci_id]),
                       energy_keV = parse(Float64, f[ci_e]),
                       x = parse(Float64, f[ci_x]),
                       y = parse(Float64, f[ci_y]),
                       z = parse(Float64, f[ci_z])))
    end
    return events
end

events = read_events(CSV_PATH)
println("Loaded $(length(events)) events from $CSV_PATH")

# ── Which contacts do we need weighting potentials for? ──
needed = Set{Int}()
for e in events
    a = anode_id_for_x(e.x)
    for d in -ANODE_NEIGHBORS:ANODE_NEIGHBORS
        (1 <= a + d <= N_ANODE) && push!(needed, a + d)
    end
    push!(needed, STEER_ID)
    c = cathode_id_for_y(e.y)
    for d in -CATHODE_NEIGHBORS:CATHODE_NEIGHBORS
        (CATHODE_ID0 + 1 <= c + d <= CATHODE_ID0 + N_CATHODE) && push!(needed, c + d)
    end
end
needed = sort(collect(needed))
println("Need weighting potentials for $(length(needed)) contacts: ",
        join(["$(id):$(contact_name(id))" for id in needed], ", "))

# ── Load SSD & compute fields ──
print("Loading SolidStateDetectors … "); flush(stdout)
t_pkg = @elapsed using SolidStateDetectors
using Unitful; using Unitful: ustrip
println("$(round(t_pkg; digits=1))s")

print("Parsing geometry … "); flush(stdout)
t_geom = @elapsed (sim = Simulation{Float32}(GEOMETRY))
println("$(round(t_geom; digits=1))s ($(length(sim.detector.contacts)) contacts)")

print("Electric potential (refine=$(REFINE)) … "); flush(stdout)
t_pot = @elapsed calculate_electric_potential!(sim;
    refinement_limits=REFINE, convergence_limit=1e-6, depletion_handling=true)
println("$(round(t_pot; digits=1))s")

print("Electric field … "); flush(stdout)
t_field = @elapsed calculate_electric_field!(sim)
println("$(round(t_field; digits=1))s")

print("Weighting potentials … "); flush(stdout)
t_wp = @elapsed for id in needed
    print("$id "); flush(stdout)
    calculate_weighting_potential!(sim, id; refinement_limits=REFINE, convergence_limit=1e-6)
end
println("\n  done in $(round(t_wp; digits=1))s")

# ── Extract raw + preamp waveforms for the needed contacts ──
function extract_waveforms(evt)
    wfs = Dict{String,Any}[]
    out = Dict{String,Any}()
    for cid in needed
        cid <= length(evt.waveforms) || continue
        wf = evt.waveforms[cid]
        ismissing(wf) && continue
        t_ns = Float64.(ustrip.(u"ns", collect(wf.time)))
        sig  = Float64.(ustrip.(collect(wf.signal)))
        length(t_ns) < 2 && continue
        dt = t_ns[2] - t_ns[1]
        cur = diff(sig) ./ dt
        t_mid = t_ns[1:end-1] .+ dt/2
        maximum(abs, cur; init=0.0) < 1e-15 && continue

        t_pre, sig_pre = apply_preamp(cur, DT_NS, PREAMP_B0, PREAMP_A1;
            display_us=PREAMP_DISPLAY_US, subsample=PREAMP_SUBSAMPLE)
        raw_idx = 1:10:length(cur)   # ~1 ns export spacing

        out[contact_name(cid)] = Dict{String,Any}(
            "contact_id"     => cid,
            "contact_type"   => contact_type(cid),
            "raw_time_ns"    => t_mid[raw_idx],
            "raw_current"    => cur[raw_idx],
            "preamp_time_ns" => t_pre,
            "preamp_signal"  => sig_pre,
        )
    end
    return out
end

# ── Simulate each event ──
event_records = Dict{String,Any}[]
for (k, e) in enumerate(events)
    print("  event $(e.id)  ($(round(e.x;digits=2)),$(round(e.y;digits=2)),$(round(e.z;digits=2))) mm, $(round(e.energy_keV;digits=0)) keV … ")
    flush(stdout)
    pos = CartesianPoint{Float32}(Float32(e.x/1000), Float32(e.y/1000), Float32(e.z/1000))
    evt = Event([pos], [Float32(e.energy_keV) * u"keV"], N_CARRIERS; number_of_shells=2)
    t_sim = @elapsed simulate!(evt, sim; Δt=DT_NS * u"ns", max_nsteps=MAX_NSTEPS)
    wfs = extract_waveforms(evt)
    println("$(round(t_sim;digits=1))s ($(length(wfs)) signals)")
    push!(event_records, Dict{String,Any}(
        "event_id"      => e.id,
        "energy_keV"    => e.energy_keV,
        "position_mm"   => Dict("x"=>e.x, "y"=>e.y, "z"=>e.z),
        "collecting_anode"   => contact_name(anode_id_for_x(e.x)),
        "collecting_cathode" => contact_name(cathode_id_for_y(e.y)),
        "waveforms"     => wfs,
    ))
end

output = Dict{String,Any}(
    "simulator"     => "SolidStateDetectors.jl",
    "geometry"      => basename(GEOMETRY),
    "source_csv"    => basename(CSV_PATH),
    "n_carriers"    => N_CARRIERS,
    "dt_ns"         => DT_NS,
    "preamp_b0"     => PREAMP_B0,
    "preamp_tau_us" => 140.0,
    "events"        => event_records,
)

tag = replace(basename(CSV_PATH), "gate_events_" => "", ".csv" => "")
outfile = joinpath(REPO, "output", "gate_waveforms_$(tag).json")
write(outfile, to_json(output) * "\n")
println("\nWrote $outfile")

#!/usr/bin/env julia
"""
Export SSD waveforms as JSON for cross-simulator comparison.
Runs a single event at a configurable interaction point and writes
raw induced current + preamp-shaped output for each contact.
"""

using Dates
mkpath("output")

# ── Parameters (can be overridden via ENV) ──
const ENERGY_KEV   = parse(Float32, get(ENV, "SSD_ENERGY_KEV", "662"))
const X_MM         = parse(Float64, get(ENV, "SSD_X_MM", "0.0"))
const Y_MM         = parse(Float64, get(ENV, "SSD_Y_MM", "2.5"))
const Z_MM         = parse(Float64, get(ENV, "SSD_Z_MM", "0.0"))      # mid-depth in SSD coords
const N_CARRIERS   = parse(Int,     get(ENV, "SSD_N_CARRIERS", "50"))
const DT_NS        = 0.1
const MAX_NSTEPS   = 50000

# Preamp: b0=1400, τ=140μs at dt=0.1ns
const PREAMP_B0    = 1400.0
const PREAMP_A1    = 0.9999992857142857
const PREAMP_DISPLAY_US = 5.0
const PREAMP_SUBSAMPLE  = 5

# ── Helpers ──
function apply_preamp(current, dt_ns, b0, a1; display_us=5.0, subsample=5)
    n_pad = round(Int, display_us * 1000 / dt_ns)
    n_in = length(current)
    n_total = max(n_in, n_pad)
    out = zeros(Float64, n_total)
    out[1] = b0 * (1 <= n_in ? current[1] : 0.0)
    for i in 2:n_total
        x = i <= n_in ? current[i] : 0.0
        out[i] = b0 * x + a1 * out[i-1]
    end
    idx = 1:subsample:n_total
    t_out = Float64.(collect(idx) .- 1) .* dt_ns
    return t_out, out[idx]
end

# ── Load SSD ──
print("Loading SolidStateDetectors … ")
t_pkg = @elapsed using SolidStateDetectors
println("$(round(t_pkg; digits=2))s")
using Unitful; using Unitful: ustrip

print("Parsing geometry … ")
t_geom = @elapsed begin
    sim = Simulation{Float32}(joinpath(@__DIR__, "..", "geometries", "czt_cross_strip.yaml"))
end
println("$(round(t_geom; digits=2))s ($(length(sim.detector.contacts)) contacts)")

print("Electric potential … ")
t_pot = @elapsed calculate_electric_potential!(sim;
    refinement_limits=[0.2, 0.1, 0.05], convergence_limit=1e-6, depletion_handling=true)
println("$(round(t_pot; digits=2))s")

print("Electric field … ")
t_field = @elapsed calculate_electric_field!(sim)
println("$(round(t_field; digits=2))s")

print("Weighting potentials … ")
t_wp = @elapsed for c in sim.detector.contacts
    print("$(c.id) ")
    calculate_weighting_potential!(sim, c.id;
        refinement_limits=[0.2, 0.1, 0.05], convergence_limit=1e-6)
end
println("$(round(t_wp; digits=2))s")

# ── Simulate event ──
x_m = Float32(X_MM / 1000)
y_m = Float32(Y_MM / 1000)
z_m = Float32(Z_MM / 1000)

println("\nSimulating event: ($X_MM, $Y_MM, $Z_MM) mm, $(ENERGY_KEV) keV, $N_CARRIERS carriers")
pos = CartesianPoint{Float32}(x_m, y_m, z_m)
evt = Event([pos], [ENERGY_KEV * u"keV"], N_CARRIERS; number_of_shells=2)
t_sim = @elapsed simulate!(evt, sim; Δt=DT_NS * u"ns", max_nsteps=MAX_NSTEPS)
println("Simulation: $(round(t_sim; digits=2))s")

# ── Contact metadata ──
contact_names = ["anode_1","anode_2","anode_3","anode_4","anode_5",
    "steering","cathode_1","cathode_2"]
contact_types = ["anode","anode","anode","anode","anode",
    "steering","cathode","cathode"]

# ── Extract and export ──
println("Extracting waveforms …")

waveforms = Dict{String, Any}()
for (cid, wf) in enumerate(evt.waveforms)
    ismissing(wf) && continue
    t_ns = Float64.(ustrip.(u"ns", collect(wf.time)))
    sig  = Float64.(ustrip.(collect(wf.signal)))
    dt = t_ns[2] - t_ns[1]
    cur = diff(sig) ./ dt
    t_mid = t_ns[1:end-1] .+ dt/2

    max_abs = maximum(abs, cur; init=0.0)
    max_abs < 1e-15 && continue

    # Preamp
    t_pre, sig_pre = apply_preamp(cur, DT_NS, PREAMP_B0, PREAMP_A1;
        display_us=PREAMP_DISPLAY_US, subsample=PREAMP_SUBSAMPLE)

    # Subsample raw current for export (every 10th point)
    raw_sub = 10
    raw_idx = 1:raw_sub:length(cur)

    waveforms[contact_names[cid]] = Dict(
        "contact_id" => cid,
        "contact_type" => contact_types[cid],
        "raw_time_ns" => t_mid[raw_idx],
        "raw_current" => cur[raw_idx],
        "preamp_time_ns" => t_pre,
        "preamp_signal" => sig_pre,
        "charge_time_ns" => t_ns,
        "charge_signal" => sig,
    )
end

# ── Write JSON manually (avoid extra deps) ──
function to_json(v::AbstractVector{<:Number})
    io = IOBuffer()
    print(io, "[")
    for (i, x) in enumerate(v)
        i > 1 && print(io, ",")
        (isnan(x) || isinf(x)) ? print(io, "null") : print(io, Float64(x))
    end
    print(io, "]")
    return String(take!(io))
end

function to_json(v::Number)
    (isnan(v) || isinf(v)) ? "null" : string(Float64(v))
end

function to_json(v::String)
    "\"$(replace(v, "\"" => "\\\""))\""
end

function to_json(d::Dict)
    io = IOBuffer()
    print(io, "{")
    for (i, (k, v)) in enumerate(sort(collect(d); by=first))
        i > 1 && print(io, ",")
        print(io, "\"$k\":")
        if v isa Dict
            print(io, to_json(v))
        elseif v isa AbstractVector{<:Number}
            print(io, to_json(v))
        elseif v isa Number
            print(io, to_json(v))
        elseif v isa String
            print(io, to_json(v))
        else
            print(io, "\"$(v)\"")
        end
    end
    print(io, "}")
    return String(take!(io))
end

output = Dict(
    "simulator" => "SolidStateDetectors.jl",
    "energy_keV" => Float64(ENERGY_KEV),
    "position_mm" => Dict("x" => X_MM, "y" => Y_MM, "z" => Z_MM),
    "n_carriers" => N_CARRIERS,
    "dt_ns" => DT_NS,
    "max_nsteps" => MAX_NSTEPS,
    "preamp_b0" => PREAMP_B0,
    "preamp_tau_us" => 140.0,
    "waveforms" => waveforms,
)

outfile = "output/ssd_waveforms.json"
write(outfile, to_json(output) * "\n")
println("Wrote $outfile")

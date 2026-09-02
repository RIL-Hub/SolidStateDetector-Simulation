#!/usr/bin/env julia
"""
Simulate each (x, z) position N_REPEATS times with diffusion AND a small
(~30 µm σ) Gaussian position jitter to produce stochastic waveform spread
suitable for error-bar overlays and beam-width studies.

Configurations: $(length(X_LIST)) x-positions × 3 depths × 10 repeats.
X positions: -0.5 to 0.0 mm across the anode strip in 0.1 mm steps.
Depths (SSD z): 2.0, 0.0, -2.0 mm → 0.5, 2.5, 4.5 mm from anode.

Requires the full-geometry WP cache built by build_wp_cache.jl.
Diffusion coefficients for CdZnTe are patched in at runtime (De=26, Dh=2.6 cm²/s).

Reads:  ENV["SSD_CACHE"] (default = output/sim_cache.jls)
        ENV["OUTPUT"]    (default = output/lateral/stochastic_repeats.json)
Writes: JSON with per-repeat preamp waveforms for anode_19, anode_20, anode_21,
        and steering; tagged by nominal and actual (x, y, z) position.

Run:
  julia --project=. -t8 scripts/simulate_with_diffusion_repeats.jl
"""

using Random

const REPO = abspath(joinpath(@__DIR__, ".."))
const CACHE_FILE = get(ENV, "SSD_CACHE",
    joinpath(REPO, "output", "sim_cache.jls"))
const OUT_JSON = get(ENV, "OUTPUT",
    joinpath(REPO, "output", "lateral", "stochastic_repeats.json"))

const X_LIST = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
const Z_LIST = [2.0, 0.0, -2.0]         # SSD z; depths 0.5, 2.5, 4.5 mm
const Y_MM = 2.5
const N_REPEATS = 10
const POSITION_JITTER_MM = 0.03         # 30 µm σ (approximates beam width)
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
const PREAMP_B0 = 1400.0
const PREAMP_A1 = 0.9999992857142857
const PREAMP_DISPLAY_US = 5.0
const PREAMP_SUBSAMPLE = 5
const RNG_SEED = 42

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

# Add diffusion coefficients to CdZnTe so diffusion=true actually applies.
# (Default package material lacks De/Dh in SSD 0.9+.)
if haskey(SolidStateDetectors.material_properties, :CdZnTe)
    old = SolidStateDetectors.material_properties[:CdZnTe]
    new = merge(NamedTuple(pairs(old)),
                  (; De = 26u"cm^2/s", Dh = 2.6u"cm^2/s"))
    SolidStateDetectors.material_properties[:CdZnTe] = new
    println("CdZnTe: De=26, Dh=2.6 cm²/s")
end

isfile(CACHE_FILE) || error("Cache not found: $CACHE_FILE")
print("Loading cache ($(round(filesize(CACHE_FILE)/1e6;digits=0)) MB) … "); flush(stdout)
tt = @elapsed (sim = deserialize(CACHE_FILE)); println("$(round(tt;digits=1))s")

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
        out[contact_name(id)] = Dict{String,Any}(
            "contact_id"=>id, "contact_type"=>contact_type(id),
            "preamp_time_ns"=>t_pre, "preamp_signal"=>sig_pre)
    end
    return out
end

ids = vcat(CAPTURE_ANODES, [STEER_ID])
records = Dict{String,Any}[]
rng = MersenneTwister(RNG_SEED)

println("Simulating $(length(X_LIST) * length(Z_LIST)) configs × $N_REPEATS repeats:")
for z_mm in Z_LIST
    depth = 2.5 - z_mm
    for x_nominal in X_LIST
        println("─── x=$(x_nominal) mm, z=$(z_mm) mm (depth $(depth) mm) ───")
        for rep in 1:N_REPEATS
            # Jitter the interaction position
            jx = x_nominal + POSITION_JITTER_MM * randn(rng)
            jy = Y_MM + POSITION_JITTER_MM * randn(rng)
            jz = z_mm + POSITION_JITTER_MM * randn(rng)
            print("  rep $rep  jitter (x,y,z)=($(round(jx-x_nominal;digits=4))," *
                    "$(round(jy-Y_MM;digits=4)),$(round(jz-z_mm;digits=4))) mm … ")
            flush(stdout)
            pos = CartesianPoint{Float32}(Float32(jx/1000),
                                           Float32(jy/1000),
                                           Float32(jz/1000))
            evt = Event([pos], [Float32(ENERGY_KEV) * u"keV"], N_CARRIERS;
                number_of_shells=NSHELLS, radius=[CLOUD_RADIUS_MM * u"mm"])
            t = @elapsed simulate!(evt, sim; Δt=DT_NS * u"ns",
                                    max_nsteps=MAX_NSTEPS,
                                    diffusion=true, self_repulsion=true)
            push!(records, Dict{String,Any}(
                "x_nominal_mm"=>x_nominal,
                "z_nominal_mm"=>z_mm,
                "depth_from_anode_mm"=>depth,
                "x_actual_mm"=>jx,
                "y_actual_mm"=>jy,
                "z_actual_mm"=>jz,
                "repeat"=>rep,
                "waveforms"=>extract(evt, ids)))
            println("$(round(t;digits=1))s")
        end
    end
end

output = Dict{String,Any}(
    "simulator"=>"SolidStateDetectors.jl",
    "captured_anodes"=>[contact_name(a) for a in CAPTURE_ANODES],
    "n_repeats"=>N_REPEATS,
    "position_jitter_sigma_mm"=>POSITION_JITTER_MM,
    "diffusion_enabled"=>true,
    "cloud_radius_mm"=>CLOUD_RADIUS_MM,
    "n_carriers"=>N_CARRIERS,
    "dt_ns"=>DT_NS,
    "points"=>records,
)
mkpath(dirname(OUT_JSON))
write(OUT_JSON, to_json(output) * "\n")
println("Wrote $OUT_JSON")

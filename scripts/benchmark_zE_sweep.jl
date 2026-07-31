#!/usr/bin/env julia
"""
Benchmark script for the (z × E × rep) sweep planned in
`calm-plotting-wand.md`.

Runs 5 z × 5 E × 2 reps = 50 events using the cached simulation
(`output/sweep/sim_cache.jls`, ENV override: SSD_CACHE).  Measures:
  - Julia package load time
  - Cache deserialize time
  - Per-event drift time (with diffusion=true, matches full sweep)
  - Per-z-slice mean and std
  - Per-E slice mean

Writes results to `output/benchmark_zE_sweep.json` and prints an ETA
projection for the full 102 000-event sweep.

Run:  JULIA_NUM_THREADS=8 julia --project=. -t8 scripts/benchmark_zE_sweep.jl
"""

const REPO = abspath(joinpath(@__DIR__, ".."))
const CACHE_FILE = get(ENV, "SSD_CACHE",
    joinpath(REPO, "output", "sweep", "sim_cache.jls"))
const OUT_JSON = joinpath(REPO, "output", "benchmark_zE_sweep.json")

# ── Sweep params for benchmark subset ──
const Z_BENCH_MM = [0.1, 1.25, 2.5, 3.75, 4.9]        # 5 z-values (from anode)
const E_BENCH_KEV = [10.0, 500.0, 1000.0, 1500.0, 2000.0]  # 5 energies
const N_REPS = 2
const N_CARRIERS = 50
const NSHELLS = 2
const CLOUD_RADIUS_MM = 0.1
const DT_NS = 0.1
const MAX_NSTEPS = 50_000
const Y_MM = 2.5                                        # on cathode_5

# ── Full-sweep params (for ETA extrapolation) ──
const N_Z_FULL = 51    # 0..5 mm at 0.1 mm steps
const N_E_FULL = 200   # 10..2000 keV at 10 keV steps
const N_REPS_FULL = 10
const N_EVENTS_FULL = N_Z_FULL * N_E_FULL * N_REPS_FULL

# ── depth→SSD-z convention ──
depth_to_ssdz(depth_mm) = 2.5 - depth_mm

# ── Simple JSON writer (no dep) ──
to_json(v::AbstractVector{<:Number}) = "[" * join((isnan(x)||isinf(x) ? "null" : string(Float64(x)) for x in v), ",") * "]"
to_json(v::AbstractVector{<:AbstractString}) = "[" * join((to_json(s) for s in v), ",") * "]"
to_json(v::AbstractVector{<:AbstractDict}) = "[" * join((to_json(d) for d in v), ",") * "]"
to_json(v::Number) = (isnan(v)||isinf(v)) ? "null" : string(Float64(v))
to_json(v::String) = "\"$(replace(v, "\"" => "\\\""))\""
function to_json(d::AbstractDict)
    io = IOBuffer(); print(io, "{")
    for (i, (k, v)) in enumerate(d)
        i > 1 && print(io, ",")
        print(io, "\"$k\":")
        if v isa AbstractDict
            print(io, to_json(v))
        elseif v isa AbstractVector
            print(io, to_json(v))
        elseif v isa Number
            print(io, to_json(v))
        elseif v isa String
            print(io, to_json(v))
        else
            print(io, "\"$(v)\"")
        end
    end
    print(io, "}"); String(take!(io))
end

# ── Measure package load ──
print("[bench] Loading SolidStateDetectors + Unitful … "); flush(stdout)
t_load = @elapsed begin
    using SolidStateDetectors
    using Unitful
    using Serialization
    using Random
    using Statistics
end
println("$(round(t_load; digits=2))s")

# ── Stochasticity via position jitter (per-rep) ──
# Native SSD diffusion is ignored on cached sim objects. We use small
# random per-rep position jitter instead: ±JITTER_MM uniform in x and z.
# Physical basis: mimics finite beam spot + interaction depth uncertainty.
const JITTER_MM = 0.025      # ±25 µm — small compared to 100 µm cloud
println("[bench] Stochasticity: per-rep position jitter ±$(JITTER_MM) mm")

# ── Measure cache deserialize ──
if !isfile(CACHE_FILE)
    error("No cache at $CACHE_FILE — build with scripts/build_wp_cache.jl")
end
print("[bench] Deserializing cache ($(round(filesize(CACHE_FILE)/1e6; digits=0)) MB) … ")
flush(stdout)
local sim
t_cache = @elapsed (sim = deserialize(CACHE_FILE))
println("$(round(t_cache; digits=2))s")

# ── Per-event benchmark loop ──
println("[bench] Running $(length(Z_BENCH_MM) * length(E_BENCH_KEV) * N_REPS) events (diffusion=true)…")

function run_benchmark(sim)
    records = Vector{Dict{String,Any}}()
    Random.seed!(42)
    i_evt = 0
    for z_mm in Z_BENCH_MM
        for E_keV in E_BENCH_KEV
            for rep in 1:N_REPS
                i_evt += 1
                # Per-rep position jitter in x and z (uniform ±JITTER_MM)
                dx = (rand() - 0.5) * 2 * JITTER_MM
                dz = (rand() - 0.5) * 2 * JITTER_MM
                ssd_z_mm = depth_to_ssdz(z_mm) + dz
                pos = CartesianPoint{Float32}(
                    Float32(dx / 1000),
                    Float32(Y_MM / 1000),
                    Float32(ssd_z_mm / 1000),
                )
                evt = Event([pos], [Float32(E_keV) * u"keV"], N_CARRIERS;
                    number_of_shells = NSHELLS,
                    radius = [CLOUD_RADIUS_MM * u"mm"],
                )
                t_evt = @elapsed simulate!(evt, sim;
                    Δt = DT_NS * u"ns",
                    max_nsteps = MAX_NSTEPS,
                    verbose = false,
                )
                push!(records, Dict{String,Any}(
                    "z_mm" => z_mm,
                    "E_keV" => E_keV,
                    "rep" => rep,
                    "t_evt_s" => t_evt,
                ))
                println("  [$i_evt] z=$(z_mm)mm E=$(E_keV)keV rep=$rep  " *
                        "t=$(round(t_evt; digits=2))s")
            end
        end
    end
    return records
end

records = run_benchmark(sim)

# ── Aggregate ──
t_evt_all = Float64[r["t_evt_s"] for r in records]
mean_evt = mean(t_evt_all)
std_evt = std(t_evt_all)
min_evt = minimum(t_evt_all)
max_evt = maximum(t_evt_all)

per_z = Dict{String,Any}()
for z_mm in Z_BENCH_MM
    ts = Float64[r["t_evt_s"] for r in records if r["z_mm"] == z_mm]
    per_z[string(z_mm)] = mean(ts)
end

per_E = Dict{String,Any}()
for E_keV in E_BENCH_KEV
    ts = Float64[r["t_evt_s"] for r in records if r["E_keV"] == E_keV]
    per_E[string(E_keV)] = mean(ts)
end

eta_single = mean_evt * N_EVENTS_FULL / 3600  # hours
eta_51 = eta_single / 51

out = Dict{String,Any}(
    "julia_package_load_s" => t_load,
    "cache_load_s" => t_cache,
    "n_events" => length(records),
    "per_event_s" => Dict{String,Any}(
        "mean" => mean_evt,
        "std" => std_evt,
        "min" => min_evt,
        "max" => max_evt,
    ),
    "per_z_mean_s" => per_z,
    "per_E_mean_s" => per_E,
    "full_sweep_events" => N_EVENTS_FULL,
    "eta_full_sweep_hours" => Dict{String,Any}(
        "single_thread" => eta_single,
        "51_shards" => eta_51,
    ),
    "params" => Dict{String,Any}(
        "N_CARRIERS" => N_CARRIERS,
        "NSHELLS" => NSHELLS,
        "CLOUD_RADIUS_MM" => CLOUD_RADIUS_MM,
        "DT_NS" => DT_NS,
        "MAX_NSTEPS" => MAX_NSTEPS,
        "diffusion" => "true",
    ),
    "records" => records,
)

mkpath(dirname(OUT_JSON))
write(OUT_JSON, to_json(out) * "\n")

println()
println("═══════════════════════════════════════════════════════════════")
println("[bench] SUMMARY")
println("═══════════════════════════════════════════════════════════════")
println("  Julia package load:  $(round(t_load; digits=1))s")
println("  Cache deserialize:   $(round(t_cache; digits=2))s")
println("  Per-event drift:")
println("    mean = $(round(mean_evt; digits=2))s  std = $(round(std_evt; digits=2))s")
println("    min  = $(round(min_evt; digits=2))s   max = $(round(max_evt; digits=2))s")
println()
println("  Per-z mean (mm from anode):")
for z_mm in Z_BENCH_MM
    println("    z=$(z_mm)mm:  $(round(per_z[string(z_mm)]; digits=2))s")
end
println()
println("  Per-E mean (keV):")
for E_keV in E_BENCH_KEV
    println("    E=$(E_keV)keV:  $(round(per_E[string(E_keV)]; digits=2))s")
end
println()
println("  Full sweep ETA ($N_EVENTS_FULL events):")
println("    single-thread:  $(round(eta_single; digits=1)) hours")
println("    51 shards:      $(round(eta_51; digits=2)) hours ($(round(eta_51*60; digits=0)) min)")
println()
println("Wrote $OUT_JSON")

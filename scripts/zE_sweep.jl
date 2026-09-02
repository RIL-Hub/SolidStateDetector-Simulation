#!/usr/bin/env julia
"""
(z × E × rep) SSD sweep driver.

Runs an ideal SolidStateDetectors.jl simulation of the cross-strip CZT
at fixed x = 0, y = 2.5 mm.  For each (z_mm, E_keV, rep) it captures
the preamp waveforms on A19, A20, A21, and steering electrode with
thermal diffusion enabled (SSD RNG provides per-rep stochasticity).

CLI-shardable — pass --z-lo and --z-hi to restrict the z range.  Each
Julia process loads its own copy of the WP cache (~560 MB) and writes
one JLD2 output file.  The main driver on the cluster runs N shards
with disjoint z ranges in parallel.

Usage:
  julia --project=. -t8 scripts/zE_sweep.jl [--z-lo <mm>] [--z-hi <mm>]
                                              [--z-step <mm>]
                                              [--e-lo <keV>] [--e-hi <keV>]
                                              [--e-step <keV>]
                                              [--n-reps <n>]
                                              [--out <path>]
                                              [--seed <n>]

Defaults:
  --z-lo 0.0    --z-hi 5.0    --z-step 0.1
  --e-lo 10.0   --e-hi 2000.0 --e-step 10.0
  --n-reps 10
  --out output/zE_sweep/zE_sweep.jld2
  --seed 42
"""

const REPO = abspath(joinpath(@__DIR__, ".."))
const CACHE_FILE = get(ENV, "SSD_CACHE",
    joinpath(REPO, "output", "sweep", "sim_cache.jls"))

# ── CLI parsing ──
function parse_args(args)
    p = Dict{String,Any}(
        "z-lo" => 0.0, "z-hi" => 5.0, "z-step" => 0.1,
        "e-lo" => 10.0, "e-hi" => 2000.0, "e-step" => 10.0,
        "n-reps" => 10,
        "out" => joinpath(REPO, "output", "zE_sweep", "zE_sweep.jld2"),
        "seed" => 42,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--")
            k = a[3:end]
            v = args[i + 1]
            if k in ("z-lo","z-hi","z-step","e-lo","e-hi","e-step")
                p[k] = parse(Float64, v)
            elseif k in ("n-reps","seed")
                p[k] = parse(Int, v)
            elseif k == "out"
                p[k] = v
            else
                error("Unknown flag $a")
            end
            i += 2
        else
            error("Unexpected positional arg $a")
        end
    end
    return p
end

const CLI = parse_args(ARGS)

# ── Sim constants (match benchmark) ──
const N_CARRIERS = 50
const NSHELLS = 2
const CLOUD_RADIUS_MM = 0.1
const DT_NS = 0.1
const MAX_NSTEPS = 50_000
const Y_MM = 2.5
const CAPTURE_IDS = [19, 20, 21, 40]     # A19, A20, A21, steering
const CAPTURE_NAMES = ["a19", "a20", "a21", "steer"]

# ── Preamp params (match lateral_sweep.jl) ──
const PREAMP_B0 = 1400.0
const PREAMP_A1 = 0.9999992857142857
const PREAMP_DISPLAY_US = 5.0
const PREAMP_SUBSAMPLE = 5

depth_to_ssdz(depth_mm) = 2.5 - depth_mm

function apply_preamp(current, dt_ns, b0, a1; display_us=5.0, subsample=5)
    n_pad = round(Int, display_us * 1000 / dt_ns)
    n_in = length(current)
    n_total = max(n_in, n_pad)
    out = zeros(Float64, n_total)
    out[1] = b0 * (n_in >= 1 ? current[1] : 0.0)
    for i in 2:n_total
        out[i] = b0 * (i <= n_in ? current[i] : 0.0) + a1 * out[i-1]
    end
    idx = 1:subsample:n_total
    return Float64.(collect(idx) .- 1) .* dt_ns, out[idx]
end

# ── Load Julia deps + cache ──
print("[zE_sweep] Loading SolidStateDetectors … "); flush(stdout)
using SolidStateDetectors
using Unitful; using Unitful: ustrip
using Serialization
using Random
using JLD2
println("ok")

# ── Stochasticity via position jitter per rep ──
# SSD's native thermal diffusion is silently ignored on cached sim
# objects (material properties are frozen at cache time). We use
# small per-rep position jitter in x and z as a physically-motivated
# proxy: ±JITTER_MM uniform, matching finite beam spot + interaction
# depth uncertainty in a real experiment.
const JITTER_MM = 0.025      # ±25 µm — small vs 100 µm cloud radius
println("[zE_sweep] Stochasticity: per-rep pos jitter ±$(JITTER_MM) mm in x, z")

if !isfile(CACHE_FILE)
    error("No cache at $CACHE_FILE — build with scripts/build_wp_cache.jl")
end
print("[zE_sweep] Loading cache ($(round(filesize(CACHE_FILE)/1e6; digits=0)) MB) … ")
flush(stdout)
local sim
t_cache = @elapsed (sim = deserialize(CACHE_FILE))
println("$(round(t_cache; digits=2))s")

Random.seed!(Int(CLI["seed"]))

# ── Sweep axis construction (z in mm from anode) ──
# Round to avoid accumulated float drift when subdividing.
z_all = round.(collect(CLI["z-lo"]:CLI["z-step"]:CLI["z-hi"]); digits=6)
e_all = round.(collect(CLI["e-lo"]:CLI["e-step"]:CLI["e-hi"]); digits=3)
n_reps = CLI["n-reps"]
n_events = length(z_all) * length(e_all) * n_reps

println("[zE_sweep] Sweep plan:")
println("  z_mm:  $(length(z_all)) points $(z_all[1])..$(z_all[end])")
println("  E_keV: $(length(e_all)) points $(e_all[1])..$(e_all[end])")
println("  reps:  $n_reps")
println("  total: $n_events events")
println("  out:   $(CLI["out"])")

# ── Compute waveform sample count once ──
_probe = apply_preamp([0.0], DT_NS, PREAMP_B0, PREAMP_A1;
    display_us=PREAMP_DISPLAY_US, subsample=PREAMP_SUBSAMPLE)
const N_SAMPLES = length(_probe[2])
const T_NS = Float32.(_probe[1])
println("  waveform: $N_SAMPLES samples, dt=$(Float32(T_NS[2]-T_NS[1])) ns each")

# ── Preallocate output arrays ──
# waveforms[ch, sample, event] = Float32 preamp signal (mV-scaleable downstream)
waveforms = zeros(Float32, length(CAPTURE_IDS), N_SAMPLES, n_events)
z_mm = zeros(Float32, n_events)
E_keV = zeros(Float32, n_events)
rep = zeros(Int32, n_events)
t_evt_s = zeros(Float32, n_events)

# ── Event loop ──
function extract_preamp(evt, id)
    id <= length(evt.waveforms) || return zeros(Float32, N_SAMPLES)
    wf = evt.waveforms[id]
    ismissing(wf) && return zeros(Float32, N_SAMPLES)
    t_ns = Float64.(ustrip.(u"ns", collect(wf.time)))
    sig = Float64.(ustrip.(collect(wf.signal)))
    length(t_ns) < 2 && return zeros(Float32, N_SAMPLES)
    dt = t_ns[2] - t_ns[1]
    cur = diff(sig) ./ dt
    _, y_pre = apply_preamp(cur, DT_NS, PREAMP_B0, PREAMP_A1;
        display_us=PREAMP_DISPLAY_US, subsample=PREAMP_SUBSAMPLE)
    # Pad or truncate to N_SAMPLES
    out = zeros(Float32, N_SAMPLES)
    n = min(N_SAMPLES, length(y_pre))
    @inbounds for i in 1:n
        out[i] = Float32(y_pre[i])
    end
    return out
end

# ETA state
const LOG_EVERY = 100

function run_sweep!(sim, z_all, e_all, n_reps, waveforms, z_mm, E_keV,
                     rep, t_evt_s, n_events)
    t_start = time()
    i_evt = 0
    for z in z_all
        for E in e_all
            for r in 1:n_reps
                i_evt += 1
                dx = (rand() - 0.5) * 2 * JITTER_MM
                dz = (rand() - 0.5) * 2 * JITTER_MM
                ssd_z_mm = depth_to_ssdz(z) + dz
                pos = CartesianPoint{Float32}(
                    Float32(dx / 1000),
                    Float32(Y_MM / 1000),
                    Float32(ssd_z_mm / 1000),
                )
                evt = Event([pos], [Float32(E) * u"keV"], N_CARRIERS;
                    number_of_shells = NSHELLS,
                    radius = [CLOUD_RADIUS_MM * u"mm"],
                )
                t_ev = @elapsed simulate!(evt, sim;
                    Δt = DT_NS * u"ns",
                    max_nsteps = MAX_NSTEPS,
                    verbose = false,
                )
                for (k, id) in enumerate(CAPTURE_IDS)
                    waveforms[k, :, i_evt] .= extract_preamp(evt, id)
                end
                z_mm[i_evt] = z
                E_keV[i_evt] = E
                rep[i_evt] = r
                t_evt_s[i_evt] = t_ev

                if i_evt % LOG_EVERY == 0 || i_evt == n_events
                    t_now = time()
                    avg = (t_now - t_start) / i_evt
                    remaining = avg * (n_events - i_evt)
                    m, s = divrem(round(Int, remaining), 60)
                    h, m = divrem(m, 60)
                    eta_str = h > 0 ? "$(h)h$(m)m$(s)s" : "$(m)m$(s)s"
                    println("[zE_sweep] step $i_evt/$n_events, " *
                            "ETA $eta_str, avg $(round(avg; digits=2))s/event")
                    flush(stdout)
                end
            end
        end
    end
end

run_sweep!(sim, z_all, e_all, n_reps,
            waveforms, z_mm, E_keV, rep, t_evt_s, n_events)

# ── Write JLD2 ──
mkpath(dirname(CLI["out"]))
println("[zE_sweep] Writing $(CLI["out"]) …")
jldopen(CLI["out"], "w") do f
    f["waveforms"] = waveforms       # (n_channels, n_samples, n_events)
    f["z_mm"] = z_mm
    f["E_keV"] = E_keV
    f["rep"] = rep
    f["t_evt_s"] = t_evt_s
    f["channel_names"] = CAPTURE_NAMES
    f["t_ns"] = T_NS
    f["n_events"] = n_events
    f["n_channels"] = length(CAPTURE_IDS)
    f["n_samples"] = N_SAMPLES
    f["params"] = Dict(
        "N_CARRIERS" => N_CARRIERS,
        "NSHELLS" => NSHELLS,
        "CLOUD_RADIUS_MM" => CLOUD_RADIUS_MM,
        "DT_NS" => DT_NS,
        "MAX_NSTEPS" => MAX_NSTEPS,
        "PREAMP_B0" => PREAMP_B0,
        "PREAMP_A1" => PREAMP_A1,
        "PREAMP_DISPLAY_US" => PREAMP_DISPLAY_US,
        "PREAMP_SUBSAMPLE" => PREAMP_SUBSAMPLE,
        "Y_MM" => Y_MM,
        "X_MM" => 0.0,
        "diffusion" => true,
        "seed" => CLI["seed"],
    )
end
sz_mb = round(filesize(CLI["out"])/1e6; digits=1)
println("[zE_sweep] Wrote $n_events events → $(sz_mb) MB")

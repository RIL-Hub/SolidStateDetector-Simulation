#!/usr/bin/env julia
"""
CZT strip detector simulation: Cs-137 depth scan experiment.

Simulates 662 keV photoelectric interactions at multiple depths (cathode-to-anode),
computes induced current waveforms on all electrodes, applies a charge-sensitive
preamplifier model, and generates an interactive HTML report.

Geometry: 40×40×5 mm CdZnTe, 5 anodes (100μm), steering (-80V), 2 cathodes (-600V)
"""

using Dates

mkpath("output")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

const ENERGY_KEV      = Float32(662)      # Cs-137 gamma
const N_CARRIERS      = 200               # representative charge cloud size
const INTERACTION_X   = 0.1               # mm (near center anode 3)
const INTERACTION_Y   = 2.5               # mm (centered over cathode 2)
const Z_FROM_CATHODE  = collect(0.5:0.5:4.5)  # mm, 9 depths
const Z_SIM           = Z_FROM_CATHODE .- 2.5  # mm, in simulation coords
const DT_NS           = 0.1               # ns — fine resolution near interaction
const MAX_NSTEPS      = 50000             # 5 μs window
const DETAIL_Z_IDX    = 5                 # center depth for detailed event

const PREAMP_B0       = 1400.0            # gain
const PREAMP_A1       = 0.9999992857142857   # pole (τ = 140 μs at dt=0.1ns)
const PREAMP_DISPLAY_US = 5.0             # show 5 μs of preamp output
const PREAMP_SUBSAMPLE = 5                # display every 5th sample (0.5 ns)

# Contacts of interest
const CENTER_ANODE    = 3                 # primary collecting pixel
const ANODE_IDS       = [1, 2, 3, 4, 5]
const CATHODE_IDS     = [7, 8]
const STEERING_ID     = 6

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

function js_array(arr)
    io = IOBuffer()
    print(io, "[")
    for (i, v) in enumerate(arr)
        i > 1 && print(io, ",")
        (isnan(v) || isinf(v)) ? print(io, "null") : print(io, Float64(v))
    end
    print(io, "]")
    return String(take!(io))
end

function js_heatmap_z(mat)
    rows = [js_array(mat[:, j]) for j in axes(mat, 2)]
    return "[" * join(rows, ",\n") * "]"
end

function box_trace(cx, cy, cz, hx, hy, hz; color="blue", opacity=0.3, name="Box")
    x = [cx-hx, cx+hx, cx+hx, cx-hx, cx-hx, cx+hx, cx+hx, cx-hx]
    y = [cy-hy, cy-hy, cy+hy, cy+hy, cy-hy, cy-hy, cy+hy, cy+hy]
    z = [cz-hz, cz-hz, cz-hz, cz-hz, cz+hz, cz+hz, cz+hz, cz+hz]
    ii = [0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1]
    jj = [1, 2, 5, 6, 1, 5, 3, 7, 3, 7, 2, 6]
    kk = [2, 3, 6, 7, 5, 4, 7, 6, 7, 4, 6, 5]
    return """{type:'mesh3d',x:$(x),y:$(y),z:$(z),
      i:$(ii),j:$(jj),k:$(kk),
      color:'$(color)',opacity:$(opacity),name:'$(name)',flatshading:true,showlegend:true}"""
end

function depth_color(t)
    # t ∈ [0,1]: 0=near cathode (blue), 1=near anode (red). Jet-like.
    keys = [(0,0,180), (0,100,220), (0,180,180), (0,200,80),
            (140,210,0), (220,180,0), (240,100,0), (220,30,0), (160,0,0)]
    n = length(keys)
    idx = clamp(t * (n - 1), 0, n - 1 - 1e-10)
    i = floor(Int, idx) + 1
    f = idx - (i - 1)
    c1, c2 = keys[i], keys[min(i+1, n)]
    r = round(Int, c1[1]*(1-f) + c2[1]*f)
    g = round(Int, c1[2]*(1-f) + c2[2]*f)
    b = round(Int, c1[3]*(1-f) + c2[3]*f)
    return "rgb($r,$g,$b)"
end

function apply_preamp(current, dt_ns, b0, a1; display_us=40.0, subsample=20)
    # Apply IIR preamp filter: y[n] = b0*x[n] + a1*y[n-1]
    # Pad input with zeros to show the long exponential decay
    n_pad = round(Int, display_us * 1000 / dt_ns)  # total samples for display window
    n_in = length(current)
    n_total = max(n_in, n_pad)

    out = zeros(Float64, n_total)
    out[1] = b0 * (1 <= n_in ? current[1] : 0.0)
    for i in 2:n_total
        x = i <= n_in ? current[i] : 0.0
        out[i] = b0 * x + a1 * out[i-1]
    end

    # Subsample for display
    idx = 1:subsample:n_total
    t_out = Float64.(collect(idx) .- 1) .* dt_ns
    return t_out, out[idx]
end

function fmt_time(s)
    s < 1  && return "$(round(s*1000; digits=1)) ms"
    s < 60 && return "$(round(s; digits=2)) s"
    return "$(round(s/60; digits=1)) min"
end

function fmt_bytes(b)
    for (u, d) in [("GB", 2^30), ("MB", 2^20), ("KB", 2^10)]
        b >= d && return "$(round(b / d; digits=1)) $u"
    end
    return "$b B"
end

# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM INFO
# ═══════════════════════════════════════════════════════════════════════════════

timestamp = Dates.format(now(UTC), dateformat"yyyy-mm-dd HH:MM:SS") * " UTC"
sys = Dict(
    "hostname"      => gethostname(),
    "julia_version" => string(VERSION),
    "threads"       => Threads.nthreads(),
    "cpus"          => Sys.CPU_THREADS,
    "memory_gb"     => round(Sys.total_memory() / 2^30; digits=1),
)
println("── System ─────────────────────")
for (k, v) in sort(collect(sys); by=first)
    println("  $k: $v")
end

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD & COMPUTE POTENTIALS
# ═══════════════════════════════════════════════════════════════════════════════

print("\nLoading SolidStateDetectors … ")
t_pkg = @elapsed using SolidStateDetectors
println("$(round(t_pkg; digits=2))s")
using Unitful; using Unitful: ustrip

print("Parsing geometry … ")
t_geom = @elapsed begin
    sim = Simulation{Float32}(joinpath(@__DIR__, "..", "geometries", "czt_cross_strip.yaml"))
end
n_contacts = length(sim.detector.contacts)
println("$(round(t_geom; digits=2))s  ($n_contacts contacts)")

print("Electric potential … ")
pot_stats = @timed calculate_electric_potential!(sim;
    refinement_limits=[0.2, 0.1, 0.05], convergence_limit=1e-6, depletion_handling=true)
println("$(round(pot_stats.time; digits=2))s")

print("Electric field … ")
t_field = @elapsed calculate_electric_field!(sim)
println("$(round(t_field; digits=2))s")

print("Weighting potentials ($n_contacts) … ")
t_wp = @elapsed for c in sim.detector.contacts
    print("$(c.id) ")
    calculate_weighting_potential!(sim, c.id;
        refinement_limits=[0.2, 0.1, 0.05], convergence_limit=1e-6)
end
println("$(round(t_wp; digits=2))s")

# ═══════════════════════════════════════════════════════════════════════════════
# Z-SCAN: SIMULATE AT EACH DEPTH
# ═══════════════════════════════════════════════════════════════════════════════

println("\n── Z-Scan: $(length(Z_SIM)) depths, x=$(INTERACTION_X)mm, y=$(INTERACTION_Y)mm ──")

# zscan_preamp[contact_id][z_idx] = (t_ns, shaped) — after preamp filter
zscan_preamp  = Dict{Int, Vector{Tuple{Vector{Float64}, Vector{Float64}}}}()
zscan_events  = Vector{Any}(nothing, length(Z_SIM))

t_zscan = @elapsed for (zi, z_mm) in enumerate(Z_SIM)
    x_m = Float32(INTERACTION_X / 1000)
    y_m = Float32(INTERACTION_Y / 1000)
    z_m = Float32(z_mm / 1000)

    print("  z=$(z_mm)mm ($(Z_FROM_CATHODE[zi])mm from cathode), $N_CARRIERS carriers … ")
    pos = CartesianPoint{Float32}(x_m, y_m, z_m)
    evt = Event([pos], [ENERGY_KEV * u"keV"], N_CARRIERS; number_of_shells=2)
    simulate!(evt, sim; Δt=DT_NS * u"ns", max_nsteps=MAX_NSTEPS)
    zscan_events[zi] = evt
    println("done")

    for (cid, wf) in enumerate(evt.waveforms)
        ismissing(wf) && continue
        t_ns = Float64.(ustrip.(u"ns", collect(wf.time)))
        sig  = Float64.(ustrip.(collect(wf.signal)))
        dt = t_ns[2] - t_ns[1]
        cur = diff(sig) ./ dt
        t_mid = t_ns[1:end-1] .+ dt/2

        # Apply charge-sensitive preamp (IIR filter with long decay)
        t_pre, sig_pre = apply_preamp(cur, DT_NS, PREAMP_B0, PREAMP_A1;
            display_us=PREAMP_DISPLAY_US, subsample=PREAMP_SUBSAMPLE)

        haskey(zscan_preamp, cid)  || (zscan_preamp[cid]  = [])
        push!(zscan_preamp[cid],  (t_pre, sig_pre))
    end
end
println("Z-scan total: $(round(t_zscan; digits=2))s")

t_total = t_pkg + t_geom + pot_stats.time + t_field + t_wp + t_zscan

# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACT DETAILED EVENT (center depth)
# ═══════════════════════════════════════════════════════════════════════════════

detail_evt = zscan_events[DETAIL_Z_IDX]
detail_z_mm = Z_SIM[DETAIL_Z_IDX]
detail_z_cathode = Z_FROM_CATHODE[DETAIL_Z_IDX]

# Extract drift path data for trajectory visualization
e_paths_xz = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()
h_paths_xz = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()
e_z_vs_t   = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()
h_z_vs_t   = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()

const MAX_DISPLAY_PATHS = 20   # cap drift paths shown per carrier type
const PATH_SUBSAMPLE    = 50   # show every 50th point in drift paths

if !ismissing(detail_evt.drift_paths)
    n_paths = length(detail_evt.drift_paths)
    println("\nExtracting drift paths ($n_paths carriers, showing ≤$MAX_DISPLAY_PATHS) …")
    dp1 = detail_evt.drift_paths[1]
    println("  EHDriftPath fields: $(fieldnames(typeof(dp1)))")

    # Evenly sample a subset of paths for display
    display_indices = if n_paths <= MAX_DISPLAY_PATHS
        1:n_paths
    else
        round.(Int, range(1, n_paths; length=MAX_DISPLAY_PATHS))
    end

    for di in display_indices
        dp = detail_evt.drift_paths[di]
        fnames = fieldnames(typeof(dp))
        for (paths_xz, paths_zt, fname_hint) in [
            (e_paths_xz, e_z_vs_t, :e),
            (h_paths_xz, h_z_vs_t, :h)]
            path = nothing
            for fn in fnames
                if startswith(string(fn), string(fname_hint))
                    path = getfield(dp, fn)
                    break
                end
            end
            path === nothing && continue
            try
                # Subsample path points for display
                idx = 1:PATH_SUBSAMPLE:length(path)
                xs = Float64[path[i].x * 1000 for i in idx]
                zs = Float64[path[i].z * 1000 for i in idx]
                ts = Float64.(collect(idx) .- 1) .* DT_NS
                push!(paths_xz, (xs, zs))
                push!(paths_zt, (ts, zs))
            catch e
                println("  Warning: could not extract path: $e")
            end
        end
    end
    println("  Extracted $(length(e_paths_xz)) electron + $(length(h_paths_xz)) hole paths (subsampled)")
else
    println("\nNo drift paths available (drift_paths is missing)")
end

# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACT POTENTIAL SLICE DATA
# ═══════════════════════════════════════════════════════════════════════════════

println("Extracting potential data …")

ep = sim.electric_potential
ep_x_mm = Float64.(ep.grid.x.ticks .* 1000)
ep_y_mm = Float64.(ep.grid.y.ticks .* 1000)
ep_z_mm = Float64.(ep.grid.z.ticks .* 1000)

iy = argmin(abs.(ep.grid.y.ticks .- 0.0025f0))
slice_xz = Float64.(ep.data[:, iy, :])

iz_anode = argmin(abs.(ep.grid.z.ticks .- 0.0023f0))
slice_xy = Float64.(ep.data[:, :, iz_anode])

wp3 = sim.weighting_potentials[CENTER_ANODE]
wp3_x_mm = Float64.(wp3.grid.x.ticks .* 1000)
wp3_z_mm = Float64.(wp3.grid.z.ticks .* 1000)
wp3_iy = argmin(abs.(wp3.grid.y.ticks .- 0.0025f0))
wp3_slice = Float64.(wp3.data[:, wp3_iy, :])

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD PLOTLY TRACES
# ═══════════════════════════════════════════════════════════════════════════════

println("Building plot traces …")

# Contact metadata
contact_names = ["Anode 1 (x=-2)","Anode 2 (x=-1)","Anode 3 (x=0)",
    "Anode 4 (x=+1)","Anode 5 (x=+2)","Steering (-80V)",
    "Cathode 1 (y=-2.5)","Cathode 2 (y=+2.5)"]
contact_colors = ["#27ae60","#2ecc71","#e74c3c","#3498db","#2980b9",
    "#f39c12","#8e44ad","#9b59b6"]

# ── Z-Scan traces (4 panels: anode 2, anode 3, anode 4, cathode 2) ──
zscan_panels = [
    (2, "Anode 2 (neighbor, x=-1mm)"),
    (CENTER_ANODE, "Anode 3 (collecting pixel, x=0)"),
    (4, "Anode 4 (neighbor, x=+1mm)"),
    (8, "Cathode 2 (y=+2.5mm)"),
]

zscan_panel_traces = Dict{Int, String}()
for (cid, _title) in zscan_panels
    traces = String[]
    if haskey(zscan_preamp, cid)
        for (zi, (t, shaped)) in enumerate(zscan_preamp[cid])
            frac = (zi - 1) / max(length(Z_SIM) - 1, 1)
            col = depth_color(frac)
            lbl = "z=$(Z_FROM_CATHODE[zi])mm"
            push!(traces, """{x:$(js_array(t)),y:$(js_array(shaped)),
              type:'scatter',mode:'lines',name:'$lbl',
              line:{color:'$col',width:1.5},showlegend:$(zi==1 ? "true" : "false")}""")
        end
    end
    zscan_panel_traces[cid] = join(traces, ",\n")
end

# ── Single-event waveform traces (detail event, preamp shaped) ──
single_traces = String[]
for (cid, wf) in enumerate(detail_evt.waveforms)
    ismissing(wf) && continue
    t_ns = Float64.(ustrip.(u"ns", collect(wf.time)))
    sig  = Float64.(ustrip.(collect(wf.signal)))
    dt = t_ns[2] - t_ns[1]
    cur = diff(sig) ./ dt
    # Skip contacts with negligible signal
    max_abs = maximum(abs, cur; init=0.0)
    max_abs < 1e-15 && continue
    # Apply preamp
    t_pre, sig_pre = apply_preamp(cur, DT_NS, PREAMP_B0, PREAMP_A1;
        display_us=PREAMP_DISPLAY_US, subsample=PREAMP_SUBSAMPLE)
    dash = cid in CATHODE_IDS ? "dash" : "solid"
    w = cid == CENTER_ANODE ? 3 : 1.5
    push!(single_traces, """{x:$(js_array(t_pre)),y:$(js_array(sig_pre)),
      type:'scatter',mode:'lines',name:'$(contact_names[cid])',
      line:{color:'$(contact_colors[cid])',width:$w,dash:'$dash'}}""")
end

# ── 6-Panel: (D) anode signals, (E) cathode signals (preamp shaped) ──
panel_d_traces = String[]
panel_e_traces = String[]
for (cid, wf) in enumerate(detail_evt.waveforms)
    ismissing(wf) && continue
    t_ns = Float64.(ustrip.(u"ns", collect(wf.time)))
    sig  = Float64.(ustrip.(collect(wf.signal)))
    dt = t_ns[2] - t_ns[1]
    cur = diff(sig) ./ dt
    max_abs = maximum(abs, cur; init=0.0)
    max_abs < 1e-15 && continue
    # Apply preamp
    t_pre, sig_pre = apply_preamp(cur, DT_NS, PREAMP_B0, PREAMP_A1;
        display_us=PREAMP_DISPLAY_US, subsample=PREAMP_SUBSAMPLE)
    w = cid == CENTER_ANODE ? 3 : 1.5
    tr = """{x:$(js_array(t_pre)),y:$(js_array(sig_pre)),
      type:'scatter',mode:'lines',name:'$(contact_names[cid])',
      line:{color:'$(contact_colors[cid])',width:$w}}"""
    if cid in ANODE_IDS || cid == STEERING_ID
        push!(panel_d_traces, tr)
    elseif cid in CATHODE_IDS
        push!(panel_e_traces, tr)
    end
end

# ── 6-Panel: (A) X-Z trajectories, (B) Z vs time ──
traj_traces_xz = String[]
traj_traces_zt = String[]

for (i, (xs, zs)) in enumerate(e_paths_xz)
    push!(traj_traces_xz, """{x:$(js_array(xs)),y:$(js_array(zs)),
      type:'scatter',mode:'lines',name:'e⁻ $i',
      line:{color:'#3498db',width:1},showlegend:$(i==1 ? "true" : "false")}""")
end
for (i, (xs, zs)) in enumerate(h_paths_xz)
    push!(traj_traces_xz, """{x:$(js_array(xs)),y:$(js_array(zs)),
      type:'scatter',mode:'lines',name:'h⁺ $i',
      line:{color:'#e74c3c',width:1,dash:'dash'},showlegend:$(i==1 ? "true" : "false")}""")
end
for (i, (ts, zs)) in enumerate(e_z_vs_t)
    push!(traj_traces_zt, """{x:$(js_array(ts)),y:$(js_array(zs)),
      type:'scatter',mode:'lines',name:'e⁻ $i',
      line:{color:'#3498db',width:1},showlegend:false}""")
end
for (i, (ts, zs)) in enumerate(h_z_vs_t)
    push!(traj_traces_zt, """{x:$(js_array(ts)),y:$(js_array(zs)),
      type:'scatter',mode:'lines',name:'h⁺ $i',
      line:{color:'#e74c3c',width:1,dash:'dash'},showlegend:false}""")
end

# ── 6-Panel: (C) mobile charge vs time (use ALL drift paths, not just displayed subset) ──
mobile_traces = ""
if !ismissing(detail_evt.drift_paths) && !isempty(detail_evt.drift_paths)
    # Compute actual path lengths from all carriers
    all_e_lens = Int[]
    all_h_lens = Int[]
    for dp in detail_evt.drift_paths
        fnames = fieldnames(typeof(dp))
        for fn in fnames
            path = getfield(dp, fn)
            if startswith(string(fn), "e")
                try push!(all_e_lens, length(path)) catch; end
            elseif startswith(string(fn), "h")
                try push!(all_h_lens, length(path)) catch; end
            end
        end
    end
    max_steps = max(maximum(all_e_lens; init=0), maximum(all_h_lens; init=0))
    ne = length(all_e_lens)
    nh = length(all_h_lens)
    # Subsample output for display (every 50th step)
    display_steps = 1:PATH_SUBSAMPLE:max_steps
    t_steps = Float64.(collect(display_steps) .- 1) .* DT_NS
    mobile_e = Float64[count(l -> l > i, all_e_lens) / max(ne, 1) for i in display_steps]
    mobile_h = Float64[count(l -> l > i, all_h_lens) / max(nh, 1) for i in display_steps]
    mobile_traces = """{x:$(js_array(t_steps)),y:$(js_array(mobile_e)),
      type:'scatter',mode:'lines',name:'Electrons',
      line:{color:'#3498db',width:2}},
    {x:$(js_array(t_steps)),y:$(js_array(mobile_h)),
      type:'scatter',mode:'lines',name:'Holes',
      line:{color:'#e74c3c',width:2,dash:'dash'}}"""
end

# ── 3D geometry traces ──
geo_traces = String[]
push!(geo_traces, box_trace(0,0,0,20,20,2.5; color="#4a90d9",opacity=0.08,name="CdZnTe (40×40×5mm)"))
for (i, ax) in enumerate([-2.0,-1.0,0.0,1.0,2.0])
    push!(geo_traces, box_trace(ax,0,2.5,0.05,5,0.08;
        color=(i==3 ? "#e74c3c" : "#27ae60"),opacity=0.9,name="Anode $i ($(ax)mm)"))
end
for (i, sx) in enumerate([-2.5,-1.5,-0.5,0.5,1.5,2.5])
    push!(geo_traces, box_trace(sx,0,2.5,0.2,5,0.06;
        color="#f39c12",opacity=0.7,name=(i==1 ? "Steering (-80V)" : "")))
end
push!(geo_traces, box_trace(0,-2.5,-2.5,20,2.45,0.08; color="#8e44ad",opacity=0.6,name="Cathode 1"))
push!(geo_traces, box_trace(0, 2.5,-2.5,20,2.45,0.08; color="#9b59b6",opacity=0.6,name="Cathode 2"))
# Z-scan positions as markers
zscan_xs = fill(INTERACTION_X, length(Z_SIM))
zscan_ys = fill(INTERACTION_Y, length(Z_SIM))
push!(geo_traces, """{type:'scatter3d',mode:'markers',
  x:$(js_array(Float64.(zscan_xs))),y:$(js_array(Float64.(zscan_ys))),z:$(js_array(Float64.(Z_SIM))),
  marker:{size:5,color:$(js_array(Float64.(Z_FROM_CATHODE))),colorscale:'Jet',showscale:false},
  name:'Z-scan positions'}""")

# ═══════════════════════════════════════════════════════════════════════════════
# WRITE JSON METRICS
# ═══════════════════════════════════════════════════════════════════════════════

metrics_json = """{
  "experiment":"Cs-137 z-scan","energy_keV":662,
  "hostname":"$(sys["hostname"])","julia_version":"$(sys["julia_version"])",
  "threads":$(sys["threads"]),"cpus":$(sys["cpus"]),"memory_gb":$(sys["memory_gb"]),
  "pkg_load_s":$(round(t_pkg;digits=3)),"geometry_s":$(round(t_geom;digits=3)),
  "potential_s":$(round(pot_stats.time;digits=3)),"field_s":$(round(t_field;digits=3)),
  "weighting_s":$(round(t_wp;digits=3)),"zscan_s":$(round(t_zscan;digits=3)),
  "total_s":$(round(t_total;digits=3)),"timestamp":"$(timestamp)"
}"""
write("output/benchmark.json", metrics_json * "\n")
println("Wrote output/benchmark.json")

# ═══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

println("Generating HTML report …")

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CZT Strip Detector &mdash; Cs-137 Depth Scan</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  *{box-sizing:border-box}
  body{font-family:-apple-system,"Segoe UI",Roboto,sans-serif;max-width:1200px;
       margin:0 auto;padding:20px;color:#1a1a1a;background:#f8f9fa}
  h1{font-size:1.5em;border-bottom:3px solid #2c3e50;padding-bottom:8px;margin-bottom:4px}
  .sub{color:#666;margin-bottom:24px}
  h2{font-size:1.15em;margin-top:36px;color:#2c3e50}
  .note{font-size:0.85em;color:#888;margin-top:4px}
  table{border-collapse:collapse;width:100%;margin:8px 0 16px}
  th,td{text-align:left;padding:7px 12px;border-bottom:1px solid #dee2e6}
  th{background:#e9ecef;font-weight:600;width:35%}
  .val{font-family:"SF Mono",Menlo,monospace;font-size:0.95em}
  .time{color:#0366d6;font-weight:600}
  .g2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
  .pb{background:#fff;border:1px solid #dee2e6;border-radius:6px;padding:8px;margin-bottom:16px}
  footer{margin-top:40px;padding-top:12px;border-top:1px solid #dee2e6;font-size:0.85em;color:#888}
  @media(max-width:800px){.g2,.g3{grid-template-columns:1fr}}
</style>
</head>
<body>

<h1>CZT Strip Detector &mdash; Cs-137 Depth Scan</h1>
<p class="sub">662 keV photoelectric interaction &bull; x=$(INTERACTION_X)mm, y=$(INTERACTION_Y)mm &bull;
z-scan: $(Z_FROM_CATHODE[1])–$(Z_FROM_CATHODE[end]) mm from cathode &bull; $(timestamp)</p>

<h2>Experiment Parameters</h2>
<div class="g2">
<table>
<tr><th>Source</th><td class="val">Cs-137, 662 keV</td></tr>
<tr><th>Interaction type</th><td class="val">Photoelectric, $(N_CARRIERS) carriers</td></tr>
<tr><th>Position (x, y)</th><td class="val">$(INTERACTION_X) mm, $(INTERACTION_Y) mm</td></tr>
<tr><th>Z-scan range</th><td class="val">$(Z_FROM_CATHODE[1])–$(Z_FROM_CATHODE[end]) mm from cathode ($(length(Z_SIM)) points)</td></tr>
</table>
<table>
<tr><th>Max simulation steps</th><td class="val">$(MAX_NSTEPS)</td></tr>
<tr><th>Time step</th><td class="val">$(DT_NS) ns</td></tr>
<tr><th>Preamp gain</th><td class="val">$(PREAMP_B0)</td></tr>
<tr><th>Preamp pole</th><td class="val">$(PREAMP_A1)</td></tr>
</table>
</div>

<h2>Z-Scan: Preamp-Shaped Waveforms</h2>
<p class="note">Each panel overlays preamp output for all 9 depths. Blue = near cathode, red = near anode. Charge-sensitive preamp with &tau; &asymp; 70 &mu;s decay.</p>
<div class="g2">
  <div class="pb"><div id="zs0" style="height:350px"></div></div>
  <div class="pb"><div id="zs1" style="height:350px"></div></div>
  <div class="pb"><div id="zs2" style="height:350px"></div></div>
  <div class="pb"><div id="zs3" style="height:350px"></div></div>
</div>

<h2>Single Event Waveform (z = $(detail_z_cathode) mm from cathode)</h2>
<p class="note">Preamp-shaped output. Solid = anodes, dashed = cathodes. Center anode (red) is the primary collecting pixel.</p>
<div class="pb"><div id="single" style="height:420px"></div></div>

<h2>Event Visualization (z = $(detail_z_cathode) mm from cathode)</h2>
<div class="g3">
  <div class="pb"><div id="pa" style="height:320px"></div></div>
  <div class="pb"><div id="pb" style="height:320px"></div></div>
  <div class="pb"><div id="pc" style="height:320px"></div></div>
  <div class="pb"><div id="pd" style="height:320px"></div></div>
  <div class="pb"><div id="pe" style="height:320px"></div></div>
  <div class="pb" style="display:flex;align-items:center;justify-content:center">
    <div style="font-family:monospace;font-size:0.85em;line-height:1.6;padding:12px">
      <strong>Summary</strong><br>
      Crystal: CdZnTe 40&times;40&times;5 mm<br>
      Bias: -600 V cathode, -80 V steering<br>
      Bulk field: ~1200 V/cm<br>
      Interaction: ($(INTERACTION_X), $(INTERACTION_Y), $(detail_z_mm)) mm<br>
      Depth: $(detail_z_cathode) mm from cathode<br>
      Energy: 662 keV (Cs-137)<br>
      Collecting pixel: Anode 3 (x=0)<br>
      e&minus; drift paths: $(length(e_paths_xz))<br>
      h&plus; drift paths: $(length(h_paths_xz))
    </div>
  </div>
</div>

<h2>3D Detector Geometry</h2>
<p class="note">Electrode widths at true proportional scale. Colored markers show z-scan positions.</p>
<div class="pb"><div id="geo3d" style="height:500px"></div></div>

<h2>Electric Potential</h2>
<div class="g2">
  <div class="pb"><div id="pot_xz" style="height:380px"></div></div>
  <div class="pb"><div id="pot_xy" style="height:380px"></div></div>
</div>

<h2>Weighting Potential &mdash; Anode 3 (x=0)</h2>
<div class="pb"><div id="wp3" style="height:380px"></div></div>

<h2>Benchmark</h2>
<table>
<tr><th>Host</th><td class="val">$(sys["hostname"]) &bull; $(sys["julia_version"]) &bull; $(sys["threads"]) threads</td></tr>
<tr><th>Package load</th><td class="val time">$(fmt_time(t_pkg))</td></tr>
<tr><th>Geometry + potentials</th><td class="val time">$(fmt_time(t_geom + pot_stats.time + t_field + t_wp))</td></tr>
<tr><th>Z-scan ($(length(Z_SIM)) events)</th><td class="val time">$(fmt_time(t_zscan))</td></tr>
<tr><th>Total</th><td class="val time" style="font-size:1.1em">$(fmt_time(t_total))</td></tr>
</table>

<footer>Generated $(timestamp) &bull; SolidStateDetectors.jl</footer>

<script>
var C={responsive:true,displaylogo:false};

// ── Z-Scan panels ──
var zsTitles = [$(join(["'$(t)'" for (_,t) in zscan_panels], ","))];
var zsIds = ['zs0','zs1','zs2','zs3'];
var zsData = [
  [$(zscan_panel_traces[zscan_panels[1][1]])],
  [$(zscan_panel_traces[zscan_panels[2][1]])],
  [$(zscan_panel_traces[zscan_panels[3][1]])],
  [$(zscan_panel_traces[zscan_panels[4][1]])]
];
for(var i=0;i<4;i++){
  Plotly.newPlot(zsIds[i],zsData[i],{
    title:zsTitles[i],
    xaxis:{title:'Time (ns)'},yaxis:{title:'Preamp output (a.u.)'},
    margin:{t:40,b:45,l:60,r:10},showlegend:false
  },C);
}

// ── Single event waveform ──
Plotly.newPlot('single',[$(join(single_traces,",\n"))],{
  title:'Preamp Output — z = $(detail_z_cathode) mm from cathode',
  xaxis:{title:'Time (ns)'},yaxis:{title:'Preamp output (a.u.)'},
  margin:{t:40,b:50,l:70,r:20},hovermode:'x unified'
},C);

// ── 6-Panel: (A) X-Z trajectories ──
Plotly.newPlot('pa',[$(join(traj_traces_xz,",\n"))],{
  title:'(A) X-Z Trajectories',
  xaxis:{title:'x (mm)'},yaxis:{title:'z (mm)',range:[-2.7,2.7]},
  margin:{t:35,b:40,l:50,r:10},showlegend:true,
  legend:{x:0.01,y:0.99,font:{size:10}},
  shapes:[{type:'line',x0:-3,x1:3,y0:2.5,y1:2.5,line:{color:'#27ae60',width:1,dash:'dot'}},
          {type:'line',x0:-3,x1:3,y0:-2.5,y1:-2.5,line:{color:'#8e44ad',width:1,dash:'dot'}}]
},C);

// ── 6-Panel: (B) Z vs time ──
Plotly.newPlot('pb',[$(join(traj_traces_zt,",\n"))],{
  title:'(B) Z vs Time',
  xaxis:{title:'Time (ns)'},yaxis:{title:'z (mm)',range:[-2.7,2.7]},
  margin:{t:35,b:40,l:50,r:10},showlegend:false,
  shapes:[{type:'line',x0:0,x1:20000,y0:2.5,y1:2.5,line:{color:'#27ae60',width:1,dash:'dot'}},
          {type:'line',x0:0,x1:20000,y0:-2.5,y1:-2.5,line:{color:'#8e44ad',width:1,dash:'dot'}}]
},C);

// ── 6-Panel: (C) Mobile charge ──
Plotly.newPlot('pc',[$(mobile_traces)],{
  title:'(C) Mobile Charge Fraction',
  xaxis:{title:'Time (ns)'},yaxis:{title:'Fraction mobile',range:[0,1.05]},
  margin:{t:35,b:40,l:50,r:10}
},C);

// ── 6-Panel: (D) Anode signals ──
Plotly.newPlot('pd',[$(join(panel_d_traces,",\n"))],{
  title:'(D) Anode Preamp Output',
  xaxis:{title:'Time (ns)'},yaxis:{title:'Preamp output (a.u.)'},
  margin:{t:35,b:40,l:50,r:10},hovermode:'x unified'
},C);

// ── 6-Panel: (E) Cathode signals ──
Plotly.newPlot('pe',[$(join(panel_e_traces,",\n"))],{
  title:'(E) Cathode Preamp Output',
  xaxis:{title:'Time (ns)'},yaxis:{title:'Preamp output (a.u.)'},
  margin:{t:35,b:40,l:50,r:10},hovermode:'x unified'
},C);

// ── 3D Geometry ──
Plotly.newPlot('geo3d',[$(join(geo_traces,",\n"))],{
  scene:{
    xaxis:{title:'x (mm)',range:[-5,5]},
    yaxis:{title:'y (mm)',range:[-6,6]},
    zaxis:{title:'z (mm)'},
    camera:{eye:{x:1.8,y:0.8,z:0.9}},aspectmode:'data'},
  margin:{l:0,r:0,t:30,b:0},showlegend:true,legend:{x:0.01,y:0.99}
},C);

// ── Electric Potential ──
Plotly.newPlot('pot_xz',[{type:'heatmap',
  x:$(js_array(ep_x_mm)),y:$(js_array(ep_z_mm)),z:$(js_heatmap_z(slice_xz)),
  colorscale:'RdBu',reversescale:true,colorbar:{title:'V'},
  hovertemplate:'x:%{x:.1f}mm z:%{y:.1f}mm V:%{z:.1f}<extra></extra>'}],{
  title:'Electric Potential (y=$(round(ep_y_mm[iy];digits=1))mm)',
  xaxis:{title:'x (mm)'},yaxis:{title:'z (mm)'},margin:{t:40,b:50,l:60,r:20}},C);

Plotly.newPlot('pot_xy',[{type:'heatmap',
  x:$(js_array(ep_x_mm)),y:$(js_array(ep_y_mm)),z:$(js_heatmap_z(slice_xy)),
  colorscale:'RdBu',reversescale:true,colorbar:{title:'V'},
  hovertemplate:'x:%{x:.1f}mm y:%{y:.1f}mm V:%{z:.1f}<extra></extra>'}],{
  title:'Electric Potential (z≈$(round(ep_z_mm[iz_anode];digits=1))mm)',
  xaxis:{title:'x (mm)'},yaxis:{title:'y (mm)'},margin:{t:40,b:50,l:60,r:20}},C);

// ── Weighting Potential ──
Plotly.newPlot('wp3',[{type:'heatmap',
  x:$(js_array(wp3_x_mm)),y:$(js_array(wp3_z_mm)),z:$(js_heatmap_z(wp3_slice)),
  colorscale:'Viridis',colorbar:{title:'W.P.'},
  hovertemplate:'x:%{x:.2f}mm z:%{y:.2f}mm WP:%{z:.3f}<extra></extra>'}],{
  title:'Weighting Potential — Anode 3 (y=$(round(Float64(wp3.grid.y.ticks[wp3_iy])*1000;digits=1))mm)',
  xaxis:{title:'x (mm)'},yaxis:{title:'z (mm)'},margin:{t:40,b:50,l:60,r:20}},C);
</script>
</body>
</html>"""

write("output/benchmark.html", html)
println("Wrote output/benchmark.html")
println("\n══ Done ══ Total: $(fmt_time(t_total)) ══")

#!/usr/bin/env julia
"""
Full CZT cross-strip detector simulation benchmark.
Computes electric potential, field, weighting potentials, and simulates
a photon interaction with charge drift and signal induction.
Outputs an interactive HTML report for GitHub Pages.
"""

using Dates

mkpath("output")

# ── Helpers ──────────────────────────────────────────────────────────────────

function js_array(arr)
    io = IOBuffer()
    print(io, "[")
    for (i, v) in enumerate(arr)
        i > 1 && print(io, ",")
        if isnan(v)
            print(io, "null")
        else
            print(io, Float64(v))
        end
    end
    print(io, "]")
    return String(take!(io))
end

function js_heatmap_z(mat)
    # mat[ix, iz] → Plotly z[iz][ix]
    rows = String[]
    for j in axes(mat, 2)
        push!(rows, js_array(mat[:, j]))
    end
    return "[" * join(rows, ",\n") * "]"
end

function box_trace(cx, cy, cz, hx, hy, hz; color="blue", opacity=0.3, name="Box")
    x = [cx-hx, cx+hx, cx+hx, cx-hx, cx-hx, cx+hx, cx+hx, cx-hx]
    y = [cy-hy, cy-hy, cy+hy, cy+hy, cy-hy, cy-hy, cy+hy, cy+hy]
    z = [cz-hz, cz-hz, cz-hz, cz-hz, cz+hz, cz+hz, cz+hz, cz+hz]
    ii = [0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1]
    jj = [1, 2, 5, 6, 1, 5, 3, 7, 3, 7, 2, 6]
    kk = [2, 3, 6, 7, 5, 4, 7, 6, 7, 4, 6, 5]
    return """{
      type:'mesh3d', x:$(x), y:$(y), z:$(z),
      i:$(ii), j:$(jj), k:$(kk),
      color:'$(color)', opacity:$(opacity), name:'$(name)',
      flatshading:true, showlegend:true
    }"""
end

# ── System info ──────────────────────────────────────────────────────────────

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

# ── Phase 1: Load packages ───────────────────────────────────────────────────

print("\nLoading SolidStateDetectors … ")
t_pkg = @elapsed using SolidStateDetectors
println("$(round(t_pkg; digits=2))s")

using Unitful
using Unitful: ustrip

# ── Phase 2: Parse geometry ──────────────────────────────────────────────────

print("Parsing geometry … ")
t_geom = @elapsed begin
    sim = Simulation{Float32}(joinpath(@__DIR__, "..", "geometries", "czt_cross_strip.yaml"))
end
println("$(round(t_geom; digits=2))s  ($(length(sim.detector.contacts)) contacts)")

# ── Phase 3: Electric potential ──────────────────────────────────────────────

print("Electric potential … ")
pot_stats = @timed calculate_electric_potential!(sim;
    refinement_limits = [0.2, 0.1, 0.05],
    convergence_limit = 1e-6,
    depletion_handling = true,
)
println("$(round(pot_stats.time; digits=2))s")

# ── Phase 4: Electric field ──────────────────────────────────────────────────

print("Electric field … ")
t_field = @elapsed calculate_electric_field!(sim)
println("$(round(t_field; digits=2))s")

# ── Phase 5: Weighting potentials ────────────────────────────────────────────

print("Weighting potentials … ")
t_wp = @elapsed for contact in sim.detector.contacts
    calculate_weighting_potential!(sim, contact.id;
        refinement_limits = [0.2, 0.1, 0.05],
        convergence_limit = 1e-6,
    )
end
println("$(round(t_wp; digits=2))s")

# ── Phase 6: Charge drift ───────────────────────────────────────────────────

# Interaction at (0, 2.5, 5) mm — directly above anode strip 3 (contact 4)
interaction_mm = (0.0, 2.5, 5.0)
interaction_m  = Float32.(interaction_mm ./ 1000)
println("\nSimulating photon at ($(interaction_mm[1]), $(interaction_mm[2]), $(interaction_mm[3])) mm …")

evt = Event(CartesianPoint{Float32}(interaction_m...))
drift_stats = @timed simulate!(evt, sim;
    Δt = 1u"ns",
    max_nsteps = 5000,
)
println("  Drift + signals: $(round(drift_stats.time; digits=2))s")

t_total = t_pkg + t_geom + pot_stats.time + t_field + t_wp + drift_stats.time

# ── Extract electric potential slice data ────────────────────────────────────

println("\nExtracting plot data …")

ep = sim.electric_potential
x_mm = Float64.(ep.grid.x.ticks .* 1000)
y_mm = Float64.(ep.grid.y.ticks .* 1000)
z_mm = Float64.(ep.grid.z.ticks .* 1000)

# XZ slice at y closest to 2.5 mm
iy = argmin(abs.(ep.grid.y.ticks .- 0.0025f0))
slice_xz = Float64.(ep.data[:, iy, :])

# XY slice at z closest to 5 mm
iz = argmin(abs.(ep.grid.z.ticks .- 0.005f0))
slice_xy = Float64.(ep.data[:, :, iz])

# Weighting potential for contact 4 (primary collecting anode)
wp4 = sim.weighting_potentials[4]
wp4_x_mm = Float64.(wp4.grid.x.ticks .* 1000)
wp4_z_mm = Float64.(wp4.grid.z.ticks .* 1000)
wp4_iy = argmin(abs.(wp4.grid.y.ticks .- 0.0025f0))
wp4_slice = Float64.(wp4.data[:, wp4_iy, :])

# ── Extract waveform data ────────────────────────────────────────────────────

contact_names = ["Cathode", "Anode 1 (y=-7.5)", "Anode 2 (y=-2.5)",
                 "Anode 3 (y=2.5)", "Anode 4 (y=7.5)"]
contact_colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]

charge_traces = String[]
current_traces = String[]

for (idx, wf) in enumerate(evt.waveforms)
    ismissing(wf) && continue

    t_ns = Float64.(ustrip.(u"ns", collect(wf.time)))
    sig  = Float64.(ustrip.(collect(wf.signal)))

    # Induced charge trace
    push!(charge_traces, """{
      x: $(js_array(t_ns)), y: $(js_array(sig)),
      type:'scatter', mode:'lines',
      name:'$(contact_names[idx])',
      line:{color:'$(contact_colors[idx])'}
    }""")

    # Induced current (numerical derivative)
    if length(sig) > 1
        dt = t_ns[2] - t_ns[1]
        current = diff(sig) ./ dt
        t_mid = t_ns[1:end-1] .+ dt / 2
        push!(current_traces, """{
          x: $(js_array(t_mid)), y: $(js_array(current)),
          type:'scatter', mode:'lines',
          name:'$(contact_names[idx])',
          line:{color:'$(contact_colors[idx])'}
        }""")
    end
end

# ── Build JSON metrics ───────────────────────────────────────────────────────

metrics_json = """{
  "hostname": "$(sys["hostname"])",
  "julia_version": "$(sys["julia_version"])",
  "threads": $(sys["threads"]),
  "cpus": $(sys["cpus"]),
  "memory_gb": $(sys["memory_gb"]),
  "pkg_load_s": $(round(t_pkg; digits=3)),
  "geometry_parse_s": $(round(t_geom; digits=3)),
  "potential_s": $(round(pot_stats.time; digits=3)),
  "potential_bytes": $(pot_stats.bytes),
  "potential_gc_s": $(round(pot_stats.gctime; digits=3)),
  "field_s": $(round(t_field; digits=3)),
  "weighting_potentials_s": $(round(t_wp; digits=3)),
  "drift_signals_s": $(round(drift_stats.time; digits=3)),
  "total_s": $(round(t_total; digits=3)),
  "timestamp": "$(timestamp)"
}"""

write("output/benchmark.json", metrics_json * "\n")
println("Wrote output/benchmark.json")

# ── Build HTML report ────────────────────────────────────────────────────────

function fmt_time(s)
    s < 1   && return "$(round(s*1000; digits=1)) ms"
    s < 60  && return "$(round(s; digits=2)) s"
    return "$(round(s/60; digits=1)) min"
end

function fmt_bytes(b)
    for (u, d) in [("GB", 2^30), ("MB", 2^20), ("KB", 2^10)]
        b >= d && return "$(round(b / d; digits=1)) $u"
    end
    return "$b B"
end

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CZT Simulation Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; max-width: 1100px;
         margin: 0 auto; padding: 20px; color: #1a1a1a; background: #f8f9fa; }
  h1 { font-size: 1.5em; border-bottom: 3px solid #2c3e50; padding-bottom: 8px; margin-bottom: 4px; }
  .subtitle { color: #666; margin-bottom: 24px; }
  h2 { font-size: 1.15em; margin-top: 32px; color: #2c3e50; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0 16px; }
  th, td { text-align: left; padding: 8px 14px; border-bottom: 1px solid #dee2e6; }
  th { background: #e9ecef; font-weight: 600; width: 40%; }
  .val { font-family: "SF Mono", Menlo, monospace; font-size: 0.95em; }
  .time { color: #0366d6; font-weight: 600; }
  .plot-row { display: flex; gap: 16px; flex-wrap: wrap; }
  .plot-row > div { flex: 1; min-width: 400px; }
  .plot-box { background: #fff; border: 1px solid #dee2e6; border-radius: 6px;
              padding: 8px; margin-bottom: 16px; }
  footer { margin-top: 40px; padding-top: 12px; border-top: 1px solid #dee2e6;
           font-size: 0.85em; color: #888; }
</style>
</head>
<body>

<h1>CZT Cross-Strip Detector &mdash; Simulation Report</h1>
<p class="subtitle">Photon interaction at ($(interaction_mm[1]), $(interaction_mm[2]), $(interaction_mm[3])) mm
&bull; $(timestamp)</p>

<h2>System &amp; Benchmark</h2>
<table>
<tr><th>Hostname</th><td class="val">$(sys["hostname"])</td></tr>
<tr><th>Julia</th><td class="val">$(sys["julia_version"]) &bull; $(sys["threads"]) threads &bull; $(sys["cpus"]) CPUs</td></tr>
<tr><th>Memory</th><td class="val">$(sys["memory_gb"]) GB</td></tr>
</table>
<table>
<tr><th>Package load</th><td class="val time">$(fmt_time(t_pkg))</td></tr>
<tr><th>Geometry parse</th><td class="val time">$(fmt_time(t_geom))</td></tr>
<tr><th>Electric potential</th><td class="val time">$(fmt_time(pot_stats.time)) &bull; $(fmt_bytes(pot_stats.bytes)) alloc &bull; $(fmt_time(pot_stats.gctime)) GC</td></tr>
<tr><th>Electric field</th><td class="val time">$(fmt_time(t_field))</td></tr>
<tr><th>Weighting potentials (5)</th><td class="val time">$(fmt_time(t_wp))</td></tr>
<tr><th>Charge drift + signals</th><td class="val time">$(fmt_time(drift_stats.time))</td></tr>
<tr><th>Total</th><td class="val time" style="font-size:1.1em">$(fmt_time(t_total))</td></tr>
</table>

<h2>3D Detector Geometry</h2>
<div class="plot-box"><div id="geo3d" style="height:500px"></div></div>

<h2>Electric Potential</h2>
<div class="plot-row">
  <div class="plot-box"><div id="pot_xz" style="height:400px"></div></div>
  <div class="plot-box"><div id="pot_xy" style="height:400px"></div></div>
</div>

<h2>Weighting Potential &mdash; Anode 3 (Contact 4)</h2>
<div class="plot-box"><div id="wp4" style="height:400px"></div></div>

<h2>Induced Current</h2>
<div class="plot-box"><div id="current" style="height:400px"></div></div>

<h2>Induced Charge</h2>
<div class="plot-box"><div id="charge" style="height:400px"></div></div>

<footer>Generated $(timestamp) &bull; SolidStateDetectors.jl</footer>

<script>
var cfg = {responsive: true, displaylogo: false};

// ── 3D Geometry ──
Plotly.newPlot('geo3d', [
  $(box_trace(0, 0, 5, 10, 10, 5; color="#4a90d9", opacity=0.12, name="CdZnTe Crystal")),
  $(box_trace(0, 0, 10, 10, 10, 0.15; color="#e74c3c", opacity=0.7, name="Cathode (-1000V)")),
  $(box_trace(0, -7.5, 0, 10, 1.5, 0.15; color="#2ecc71", opacity=0.7, name="Anode 1")),
  $(box_trace(0, -2.5, 0, 10, 1.5, 0.15; color="#3498db", opacity=0.7, name="Anode 2")),
  $(box_trace(0, 2.5, 0, 10, 1.5, 0.15; color="#f39c12", opacity=0.7, name="Anode 3")),
  $(box_trace(0, 7.5, 0, 10, 1.5, 0.15; color="#9b59b6", opacity=0.7, name="Anode 4")),
  {type:'scatter3d', mode:'markers', x:[$(interaction_mm[1])], y:[$(interaction_mm[2])], z:[$(interaction_mm[3])],
   marker:{size:8, color:'red', symbol:'diamond'}, name:'Photon interaction'}
], {
  scene: {
    xaxis:{title:'x (mm)'}, yaxis:{title:'y (mm)'}, zaxis:{title:'z (mm)'},
    camera:{eye:{x:1.5, y:1.5, z:1.0}},
    aspectmode:'data'
  },
  margin:{l:0,r:0,t:30,b:0}, title:'CZT Cross-Strip Detector'
}, cfg);

// ── Electric Potential XZ ──
Plotly.newPlot('pot_xz', [{
  type:'heatmap',
  x: $(js_array(x_mm)),
  y: $(js_array(z_mm)),
  z: $(js_heatmap_z(slice_xz)),
  colorscale:'RdBu', reversescale:true,
  colorbar:{title:'V'}
}], {
  title:'Electric Potential (y = $(round(y_mm[iy]; digits=1)) mm)',
  xaxis:{title:'x (mm)'}, yaxis:{title:'z (mm)'},
  margin:{t:40,b:50,l:60,r:20}
}, cfg);

// ── Electric Potential XY ──
Plotly.newPlot('pot_xy', [{
  type:'heatmap',
  x: $(js_array(x_mm)),
  y: $(js_array(y_mm)),
  z: $(js_heatmap_z(slice_xy)),
  colorscale:'RdBu', reversescale:true,
  colorbar:{title:'V'}
}], {
  title:'Electric Potential (z = $(round(z_mm[iz]; digits=1)) mm)',
  xaxis:{title:'x (mm)'}, yaxis:{title:'y (mm)'},
  margin:{t:40,b:50,l:60,r:20}
}, cfg);

// ── Weighting Potential Contact 4 ──
Plotly.newPlot('wp4', [{
  type:'heatmap',
  x: $(js_array(wp4_x_mm)),
  y: $(js_array(wp4_z_mm)),
  z: $(js_heatmap_z(wp4_slice)),
  colorscale:'Viridis',
  colorbar:{title:'W.P.'}
}], {
  title:'Weighting Potential — Anode 3 (y = $(round(Float64(wp4.grid.y.ticks[wp4_iy]) * 1000; digits=1)) mm)',
  xaxis:{title:'x (mm)'}, yaxis:{title:'z (mm)'},
  margin:{t:40,b:50,l:60,r:20}
}, cfg);

// ── Induced Current ──
Plotly.newPlot('current', [$(join(current_traces, ",\n"))], {
  title:'Induced Current',
  xaxis:{title:'Time (ns)'}, yaxis:{title:'dQ/dt (a.u./ns)'},
  margin:{t:40,b:50,l:70,r:20},
  hovermode:'x unified'
}, cfg);

// ── Induced Charge ──
Plotly.newPlot('charge', [$(join(charge_traces, ",\n"))], {
  title:'Induced Charge',
  xaxis:{title:'Time (ns)'}, yaxis:{title:'Charge (a.u.)'},
  margin:{t:40,b:50,l:70,r:20},
  hovermode:'x unified'
}, cfg);
</script>
</body>
</html>"""

write("output/benchmark.html", html)
println("Wrote output/benchmark.html")
println("\n── Done ── Total: $(fmt_time(t_total)) ──")

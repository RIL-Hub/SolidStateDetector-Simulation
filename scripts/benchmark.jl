#!/usr/bin/env julia
"""
Full CZT strip detector simulation benchmark.
Computes electric potential, field, weighting potentials, and simulates
a photon interaction with charge drift and signal induction.
Outputs an interactive HTML report for GitHub Pages.

Geometry: 40×40×5 mm CdZnTe crystal
  Anode face (z=+2.5): 5 anode strips (100 μm, 0V) + steering electrode (6×400 μm, -80V)
  Cathode face (z=-2.5): 2 cathode strips (4.9 mm, -600V)
"""

using Dates

mkpath("output")

# ── Helpers ──────────────────────────────────────────────────────────────────

function js_array(arr)
    io = IOBuffer()
    print(io, "[")
    for (i, v) in enumerate(arr)
        i > 1 && print(io, ",")
        if isnan(v) || isinf(v)
            print(io, "null")
        else
            print(io, Float64(v))
        end
    end
    print(io, "]")
    return String(take!(io))
end

function js_heatmap_z(mat)
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

n_contacts = length(sim.detector.contacts)
print("Weighting potentials ($n_contacts contacts) … ")
t_wp = @elapsed for contact in sim.detector.contacts
    print("$(contact.id) ")
    calculate_weighting_potential!(sim, contact.id;
        refinement_limits = [0.2, 0.1, 0.05],
        convergence_limit = 1e-6,
    )
end
println("$(round(t_wp; digits=2))s")

# ── Phase 6: Charge drift ───────────────────────────────────────────────────

# Interaction at (0, 0, 0) mm — center of crystal,
# directly below center anode (contact 3, x=0, z=+2.5)
# and above cathodes (z=-2.5)
interaction_mm = (0.0, 0.0, 0.0)
interaction_m  = Float32.(interaction_mm ./ 1000)
println("\nSimulating photon at ($(interaction_mm[1]), $(interaction_mm[2]), $(interaction_mm[3])) mm …")

evt = Event(CartesianPoint{Float32}(interaction_m...))
drift_stats = @timed simulate!(evt, sim;
    Δt = 1u"ns",
    max_nsteps = 10000,
)
println("  Drift + signals: $(round(drift_stats.time; digits=2))s")

t_total = t_pkg + t_geom + pot_stats.time + t_field + t_wp + drift_stats.time

# ── Extract electric potential slice data ────────────────────────────────────

println("\nExtracting plot data …")

ep = sim.electric_potential
x_mm = Float64.(ep.grid.x.ticks .* 1000)
y_mm = Float64.(ep.grid.y.ticks .* 1000)
z_mm = Float64.(ep.grid.z.ticks .* 1000)

# XZ slice at y=0 (through center)
iy = argmin(abs.(ep.grid.y.ticks))
slice_xz = Float64.(ep.data[:, iy, :])

# XY slice near anode face (z ≈ 2.3 mm) — shows lateral field from strip pattern
iz_anode = argmin(abs.(ep.grid.z.ticks .- 0.0023f0))
slice_xy = Float64.(ep.data[:, :, iz_anode])

# Weighting potential for contact 3 (center anode, primary collecting electrode)
wp3 = sim.weighting_potentials[3]
wp3_x_mm = Float64.(wp3.grid.x.ticks .* 1000)
wp3_z_mm = Float64.(wp3.grid.z.ticks .* 1000)
wp3_iy = argmin(abs.(wp3.grid.y.ticks))
wp3_slice = Float64.(wp3.data[:, wp3_iy, :])

# ── Extract waveform data ────────────────────────────────────────────────────

contact_names = [
    "Anode 1 (x=-2)",     # ID 1
    "Anode 2 (x=-1)",     # ID 2
    "Anode 3 (x=0)",      # ID 3  ← primary
    "Anode 4 (x=+1)",     # ID 4
    "Anode 5 (x=+2)",     # ID 5
    "Steering (-80V)",    # ID 6
    "Cathode 1 (y=-2.5)", # ID 7
    "Cathode 2 (y=+2.5)", # ID 8
]
contact_colors = [
    "#27ae60",  # anode 1 - green
    "#2ecc71",  # anode 2 - light green
    "#e74c3c",  # anode 3 - red (primary, highlighted)
    "#3498db",  # anode 4 - blue
    "#2980b9",  # anode 5 - dark blue
    "#f39c12",  # steering - orange
    "#8e44ad",  # cathode 1 - purple
    "#9b59b6",  # cathode 2 - light purple
]

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
      line:{color:'$(contact_colors[idx])', width:$(idx == 3 ? 3 : 1.5)}
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
          line:{color:'$(contact_colors[idx])', width:$(idx == 3 ? 3 : 1.5)}
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

# 3D geometry traces — strip widths exaggerated 5× for visibility
geo_traces = String[]
# Crystal
push!(geo_traces, box_trace(0, 0, 0, 20, 20, 2.5; color="#4a90d9", opacity=0.08, name="CdZnTe Crystal (40×40×5 mm)"))
# Anode strips (100μm → shown as 0.5mm for visibility)
anode_xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
for (i, ax) in enumerate(anode_xs)
    push!(geo_traces, box_trace(ax, 0, 2.5, 0.25, 5, 0.08;
        color=(i == 3 ? "#e74c3c" : "#27ae60"), opacity=0.85,
        name="Anode $i (x=$(ax)mm, 0V)"))
end
# Steering strips (400μm, shown at scale)
steer_xs = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
for (i, sx) in enumerate(steer_xs)
    push!(geo_traces, box_trace(sx, 0, 2.5, 0.2, 5, 0.06;
        color="#f39c12", opacity=0.7,
        name=(i == 1 ? "Steering (-80V)" : "")))
end
# Cathode strips
push!(geo_traces, box_trace(0, -2.5, -2.5, 20, 2.45, 0.08; color="#8e44ad", opacity=0.6, name="Cathode 1 (-600V)"))
push!(geo_traces, box_trace(0,  2.5, -2.5, 20, 2.45, 0.08; color="#9b59b6", opacity=0.6, name="Cathode 2 (-600V)"))
# Interaction point
push!(geo_traces, """{
  type:'scatter3d', mode:'markers',
  x:[$(interaction_mm[1])], y:[$(interaction_mm[2])], z:[$(interaction_mm[3])],
  marker:{size:8, color:'red', symbol:'diamond'},
  name:'Photon (0, 0, 0) mm'
}""")

geo_traces_js = join(geo_traces, ",\n")

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CZT Strip Detector — Simulation Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; max-width: 1100px;
         margin: 0 auto; padding: 20px; color: #1a1a1a; background: #f8f9fa; }
  h1 { font-size: 1.5em; border-bottom: 3px solid #2c3e50; padding-bottom: 8px; margin-bottom: 4px; }
  .subtitle { color: #666; margin-bottom: 24px; }
  h2 { font-size: 1.15em; margin-top: 32px; color: #2c3e50; }
  .note { font-size: 0.85em; color: #888; margin-top: 4px; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0 16px; }
  th, td { text-align: left; padding: 8px 14px; border-bottom: 1px solid #dee2e6; }
  th { background: #e9ecef; font-weight: 600; width: 40%; }
  .val { font-family: "SF Mono", Menlo, monospace; font-size: 0.95em; }
  .time { color: #0366d6; font-weight: 600; }
  .geom-table th { width: 30%; }
  .geom-table td { font-family: "SF Mono", Menlo, monospace; font-size: 0.9em; }
  .plot-row { display: flex; gap: 16px; flex-wrap: wrap; }
  .plot-row > div { flex: 1; min-width: 400px; }
  .plot-box { background: #fff; border: 1px solid #dee2e6; border-radius: 6px;
              padding: 8px; margin-bottom: 16px; }
  footer { margin-top: 40px; padding-top: 12px; border-top: 1px solid #dee2e6;
           font-size: 0.85em; color: #888; }
</style>
</head>
<body>

<h1>CZT Strip Detector &mdash; Simulation Report</h1>
<p class="subtitle">Photon interaction at ($(interaction_mm[1]), $(interaction_mm[2]), $(interaction_mm[3])) mm
&bull; $(timestamp)</p>

<h2>Detector Geometry</h2>
<table class="geom-table">
<tr><th>Crystal</th><td>CdZnTe, 40 &times; 40 &times; 5 mm, 293 K</td></tr>
<tr><th>Anodes (IDs 1–5)</th><td>5 strips, 100 &mu;m wide, 1 mm pitch, x = -2…+2 mm, z = +2.5 mm, 0 V</td></tr>
<tr><th>Steering (ID 6)</th><td>6 strips, 400 &mu;m wide, x = -2.5…+2.5 mm, z = +2.5 mm, -80 V</td></tr>
<tr><th>Cathodes (IDs 7–8)</th><td>2 strips, 40 &times; 4.9 mm, 100 &mu;m gap at y=0, z = -2.5 mm, -600 V</td></tr>
<tr><th>Bulk field</th><td>~1200 V/cm (cathode &rarr; anode)</td></tr>
</table>

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
<tr><th>Weighting potentials ($n_contacts)</th><td class="val time">$(fmt_time(t_wp))</td></tr>
<tr><th>Charge drift + signals</th><td class="val time">$(fmt_time(drift_stats.time))</td></tr>
<tr><th>Total</th><td class="val time" style="font-size:1.1em">$(fmt_time(t_total))</td></tr>
</table>

<h2>3D Detector Geometry</h2>
<p class="note">Anode strip widths exaggerated 5&times; for visibility. Drag to rotate, scroll to zoom.</p>
<div class="plot-box"><div id="geo3d" style="height:550px"></div></div>

<h2>Electric Potential</h2>
<div class="plot-row">
  <div class="plot-box"><div id="pot_xz" style="height:400px"></div></div>
  <div class="plot-box"><div id="pot_xy" style="height:400px"></div></div>
</div>

<h2>Weighting Potential &mdash; Center Anode (Contact 3, x=0)</h2>
<div class="plot-box"><div id="wp3" style="height:400px"></div></div>

<h2>Induced Current</h2>
<p class="note">Numerical derivative of induced charge. Center anode (red) is the primary collecting electrode.</p>
<div class="plot-box"><div id="current" style="height:450px"></div></div>

<h2>Induced Charge</h2>
<div class="plot-box"><div id="charge" style="height:450px"></div></div>

<footer>Generated $(timestamp) &bull; SolidStateDetectors.jl</footer>

<script>
var cfg = {responsive: true, displaylogo: false};

// ── 3D Geometry ──
Plotly.newPlot('geo3d', [$(geo_traces_js)], {
  scene: {
    xaxis:{title:'x (mm)', range:[-5, 5]},
    yaxis:{title:'y (mm)', range:[-6, 6]},
    zaxis:{title:'z (mm)'},
    camera:{eye:{x:1.8, y:0.8, z:0.9}},
    aspectmode:'data'
  },
  margin:{l:0,r:0,t:30,b:0},
  title:'CZT Strip Detector — Electrode Layout',
  showlegend:true, legend:{x:0.01, y:0.99}
}, cfg);

// ── Electric Potential XZ (y=0) ──
Plotly.newPlot('pot_xz', [{
  type:'heatmap',
  x: $(js_array(x_mm)),
  y: $(js_array(z_mm)),
  z: $(js_heatmap_z(slice_xz)),
  colorscale:'RdBu', reversescale:true,
  colorbar:{title:'V'},
  hovertemplate:'x: %{x:.1f} mm<br>z: %{y:.1f} mm<br>V: %{z:.1f}<extra></extra>'
}], {
  title:'Electric Potential (y = 0 mm)',
  xaxis:{title:'x (mm)'}, yaxis:{title:'z (mm)'},
  margin:{t:40,b:50,l:60,r:20}
}, cfg);

// ── Electric Potential XY (near anode face) ──
Plotly.newPlot('pot_xy', [{
  type:'heatmap',
  x: $(js_array(x_mm)),
  y: $(js_array(y_mm)),
  z: $(js_heatmap_z(slice_xy)),
  colorscale:'RdBu', reversescale:true,
  colorbar:{title:'V'},
  hovertemplate:'x: %{x:.1f} mm<br>y: %{y:.1f} mm<br>V: %{z:.1f}<extra></extra>'
}], {
  title:'Electric Potential (z ≈ $(round(z_mm[iz_anode]; digits=1)) mm, near anode face)',
  xaxis:{title:'x (mm)'}, yaxis:{title:'y (mm)'},
  margin:{t:40,b:50,l:60,r:20}
}, cfg);

// ── Weighting Potential — Center Anode ──
Plotly.newPlot('wp3', [{
  type:'heatmap',
  x: $(js_array(wp3_x_mm)),
  y: $(js_array(wp3_z_mm)),
  z: $(js_heatmap_z(wp3_slice)),
  colorscale:'Viridis',
  colorbar:{title:'W.P.'},
  hovertemplate:'x: %{x:.2f} mm<br>z: %{y:.2f} mm<br>WP: %{z:.3f}<extra></extra>'
}], {
  title:'Weighting Potential — Anode 3 (y = 0 mm)',
  xaxis:{title:'x (mm)'}, yaxis:{title:'z (mm)'},
  margin:{t:40,b:50,l:60,r:20}
}, cfg);

// ── Induced Current ──
Plotly.newPlot('current', [$(join(current_traces, ",\n"))], {
  title:'Induced Current (dQ/dt)',
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

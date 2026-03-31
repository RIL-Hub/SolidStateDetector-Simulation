#!/usr/bin/env julia
"""
Lightweight benchmark: loads SolidStateDetectors, runs coarse electric-potential
calculation, and writes compute metrics to output/.
"""

using Dates

mkpath("output")

# ── System info ──────────────────────────────────────────────────────────────
sys = Dict{String,Any}(
    "hostname"        => gethostname(),
    "julia_version"   => string(VERSION),
    "threads"         => Threads.nthreads(),
    "cpus"            => Sys.CPU_THREADS,
    "total_memory_gb" => round(Sys.total_memory() / 2^30; digits=1),
    "timestamp"       => Dates.format(now(UTC), dateformat"yyyy-mm-dd HH:MM:SS \U\T\C"),
)

println("── System ─────────────────────────────────────")
for (k, v) in sort(collect(sys))
    println("  $k: $v")
end

# ── Package load ─────────────────────────────────────────────────────────────
print("\nLoading SolidStateDetectors … ")
pkg_stats = @timed using SolidStateDetectors
println("$(round(pkg_stats.time; digits=2)) s")

# ── Simulation load ──────────────────────────────────────────────────────────
print("Parsing geometry … ")
sim_stats = @timed begin
    sim = Simulation{Float32}(joinpath(@__DIR__, "..", "geometries", "czt_cross_strip.yaml"))
end
println("$(round(sim_stats.time; digits=2)) s")

# ── Electric potential (coarse) ──────────────────────────────────────────────
print("Calculating electric potential (coarse) … ")
pot_stats = @timed begin
    calculate_electric_potential!(sim; refinement_limits=[0.2])
end
println("$(round(pot_stats.time; digits=2)) s")

# ── Collect metrics ──────────────────────────────────────────────────────────
metrics = merge(sys, Dict{String,Any}(
    "pkg_load_time_s"          => round(pkg_stats.time; digits=3),
    "sim_load_time_s"          => round(sim_stats.time; digits=3),
    "potential_time_s"         => round(pot_stats.time; digits=3),
    "potential_bytes_allocated" => pot_stats.bytes,
    "potential_gc_time_s"      => round(pot_stats.gctime; digits=3),
    "total_time_s"             => round(pkg_stats.time + sim_stats.time + pot_stats.time; digits=3),
))

println("\n── Metrics ────────────────────────────────────")
for (k, v) in sort(collect(metrics))
    println("  $k: $v")
end

# ── Write JSON ───────────────────────────────────────────────────────────────
function simple_json(d::Dict)
    entries = String[]
    for (k, v) in sort(collect(d))
        val = v isa AbstractString ? "\"$(escape_string(v))\"" : string(v)
        push!(entries, "  \"$k\": $val")
    end
    return "{\n" * join(entries, ",\n") * "\n}\n"
end

write("output/benchmark.json", simple_json(metrics))
println("\nWrote output/benchmark.json")

# ── Write HTML report ────────────────────────────────────────────────────────
function fmt_bytes(b)
    for (unit, div) in [("GB", 2^30), ("MB", 2^20), ("KB", 2^10)]
        b >= div && return "$(round(b / div; digits=1)) $unit"
    end
    return "$b B"
end

html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SSD Benchmark Report</title>
<style>
  body { font-family: -apple-system, "Segoe UI", Roboto, monospace; max-width: 720px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; background: #fafafa; }
  h1 { font-size: 1.4em; border-bottom: 2px solid #333; padding-bottom: 8px; }
  h2 { font-size: 1.1em; margin-top: 28px; color: #555; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0; }
  th, td { text-align: left; padding: 6px 12px; border-bottom: 1px solid #ddd; }
  th { background: #eee; font-weight: 600; width: 45%; }
  .time { font-weight: bold; color: #0366d6; }
  footer { margin-top: 32px; font-size: 0.85em; color: #888; }
</style>
</head>
<body>
<h1>SolidStateDetector Benchmark Report</h1>
<p>Coarse electric-potential calculation on CZT cross-strip geometry.</p>

<h2>System</h2>
<table>
<tr><th>Hostname</th><td>$(metrics["hostname"])</td></tr>
<tr><th>Julia version</th><td>$(metrics["julia_version"])</td></tr>
<tr><th>Threads</th><td>$(metrics["threads"])</td></tr>
<tr><th>CPUs</th><td>$(metrics["cpus"])</td></tr>
<tr><th>Total memory</th><td>$(metrics["total_memory_gb"]) GB</td></tr>
</table>

<h2>Timing</h2>
<table>
<tr><th>Package load</th><td class="time">$(metrics["pkg_load_time_s"]) s</td></tr>
<tr><th>Geometry parse</th><td class="time">$(metrics["sim_load_time_s"]) s</td></tr>
<tr><th>Electric potential</th><td class="time">$(metrics["potential_time_s"]) s</td></tr>
<tr><th>Total</th><td class="time">$(metrics["total_time_s"]) s</td></tr>
</table>

<h2>Memory</h2>
<table>
<tr><th>Potential allocated</th><td>$(fmt_bytes(metrics["potential_bytes_allocated"]))</td></tr>
<tr><th>Potential GC time</th><td>$(metrics["potential_gc_time_s"]) s</td></tr>
</table>

<footer>Generated $(metrics["timestamp"])</footer>
</body>
</html>
"""

write("output/benchmark.html", html)
println("Wrote output/benchmark.html")

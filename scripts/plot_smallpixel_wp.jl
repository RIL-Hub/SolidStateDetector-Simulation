#!/usr/bin/env julia
"""
Show the small-pixel anode weighting potential vs the broad cathode weighting
potential: zoomed 2D cross-sections plus the 1D Φ_w(z) depth profile (the
Shockley–Ramo curve) sampled along the drift line at (x=0, y=2.5 mm).

Run:  JULIA_NUM_THREADS=8 julia --project=. -t8 scripts/plot_smallpixel_wp.jl
"""

using SolidStateDetectors
using Unitful
using Plots
gr()

const REPO = abspath(joinpath(@__DIR__, ".."))
const GEOMETRY = joinpath(REPO, "geometries", "czt_cross_strip_full.yaml")
const REFINE = [0.2, 0.1, 0.05]
const ANODE_ID   = 20   # x = 0 mm (collecting strip)
const CATHODE_ID = 45   # y = +2.5 mm strip

sim = Simulation{Float32}(GEOMETRY)
for id in (ANODE_ID, CATHODE_ID)
    print("Weighting potential id=$id … "); flush(stdout)
    t = @elapsed calculate_weighting_potential!(sim, id; refinement_limits=REFINE, convergence_limit=1e-6)
    println("$(round(t;digits=1))s")
end

wa = sim.weighting_potentials[ANODE_ID]
wc = sim.weighting_potentials[CATHODE_ID]

mm(t) = t .* 1000  # m -> mm
nearest(ticks, v) = argmin(abs.(ticks .- v))

xa = collect(wa.grid.axes[1].ticks); ya = collect(wa.grid.axes[2].ticks); za = collect(wa.grid.axes[3].ticks)
xc = collect(wc.grid.axes[1].ticks); yc = collect(wc.grid.axes[2].ticks); zc = collect(wc.grid.axes[3].ticks)

# ── 2D zoomed cross-sections ──
# Anode: x–z slice at y=0 (strip spans all y), zoom x to ±3 mm
iya = nearest(ya, 0.0)
Aa = permutedims(wa.data[:, iya, :])           # (nz, nx)
pa = heatmap(mm(xa), mm(za), Aa; clims=(0,1), c=:viridis,
    xlims=(-3, 3), xlabel="x (mm)", ylabel="z (mm)  [cathode −2.5 → anode +2.5]",
    title="Anode W (x=0 strip): small-pixel focusing", colorbar_title="Φ_w")
hline!(pa, [2.5]; c=:white, ls=:dot, lw=1, label="anode face")

# Cathode: y–z slice at x=0 (strip spans all x), zoom y to ±12 mm
ixc = nearest(xc, 0.0)
Ac = permutedims(wc.data[ixc, :, :])           # (nz, ny)
pc = heatmap(mm(yc), mm(zc), Ac; clims=(0,1), c=:viridis,
    xlims=(-12, 12), xlabel="y (mm)", ylabel="z (mm)  [cathode −2.5 → anode +2.5]",
    title="Cathode W (y=+2.5 strip): broad / planar", colorbar_title="Φ_w")
hline!(pc, [-2.5]; c=:white, ls=:dot, lw=1, label="cathode face")

savefig(pa, joinpath(REPO, "output", "wp_anode_smallpixel.png"))
savefig(pc, joinpath(REPO, "output", "wp_cathode.png"))
println("Wrote output/wp_anode_smallpixel.png, output/wp_cathode.png")

# ── 1D depth profiles along (x=0, y=+2.5 mm) ──
xline, yline = 0.0, 0.0025  # m
ia_x, ia_y = nearest(xa, xline), nearest(ya, yline)
ic_x, ic_y = nearest(xc, xline), nearest(yc, yline)
phi_a = wa.data[ia_x, ia_y, :]
phi_c = wc.data[ic_x, ic_y, :]

pp = plot(mm(za), phi_a; lw=2.5, marker=:circle, ms=2, label="anode_20 (small-pixel)",
    xlabel="depth z (mm)   cathode −2.5 ……… anode +2.5", ylabel="weighting potential Φ_w",
    title="Φ_w vs depth at (x=0, y=+2.5 mm)", legend=:left, ylims=(-0.05, 1.05))
plot!(pp, mm(zc), phi_c; lw=2.5, marker=:diamond, ms=2, label="cathode_5 (broad)")
# reference: ideal planar (linear) cathode response
plot!(pp, [-2.5, 2.5], [1.0, 0.0]; ls=:dash, c=:gray, lw=1, label="ideal planar (linear)")
vline!(pp, [-2.5, 2.5]; c=:black, ls=:dot, lw=0.7, label="")
savefig(pp, joinpath(REPO, "output", "wp_depth_profile.png"))
println("Wrote output/wp_depth_profile.png")

# Quick numeric summary
println("\nAnode  Φ_w along z (mm): ",
    join(["$(round(mm(za)[i];digits=2)):$(round(phi_a[i];digits=3))" for i in 1:length(za)], "  "))
println("\nCathode Φ_w along z (mm): ",
    join(["$(round(mm(zc)[i];digits=2)):$(round(phi_c[i];digits=3))" for i in 1:length(zc)], "  "))

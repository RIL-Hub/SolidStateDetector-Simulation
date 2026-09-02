"""
Extract the anode-strip-3 weighting potential W_A(z) along the drift
axis (x=0, y=2.5 mm, z varying) from the base geometry.  Also export a
2D W_A(x, z) slice at y=2.5 mm and a 1D V(z) potential (for
sanity check that the bulk field is right).

Used to directly test the small-pixel-effect prediction:
   ideal induced signal for an electron drifting z_0 → anode
   = 1 - W_A(z_0)

Output: output/wp_anode_axis.json
"""

using SolidStateDetectors
using Unitful
using JSON3

const REPO      = abspath(joinpath(@__DIR__, ".."))
const GEOMETRY  = joinpath(REPO, "geometries", "czt_cross_strip.yaml")
const OUTDIR    = joinpath(REPO, "output")
const REFINE    = [0.2, 0.1, 0.05]
const ANODE_ID  = 3

mkpath(OUTDIR)

println("Loading geometry ...")
sim = Simulation{Float32}(GEOMETRY)
println("  $(length(sim.detector.contacts)) contacts")

println("Solving electric potential ...")
t0 = @elapsed calculate_electric_potential!(sim;
    refinement_limits = REFINE, convergence_limit = 1e-6, depletion_handling = true)
println("  done ($(round(t0; digits=1))s)")

println("Solving weighting potential for anode strip $(ANODE_ID) ...")
t1 = @elapsed calculate_weighting_potential!(sim, ANODE_ID;
    refinement_limits = REFINE, convergence_limit = 1e-6)
println("  done ($(round(t1; digits=1))s)")

ep = sim.electric_potential
wp = sim.weighting_potentials[ANODE_ID]

# WP grid and potential grid have separate refinement; use each grid's
# own ticks
x_wp = Float64.(wp.grid.x.ticks); y_wp = Float64.(wp.grid.y.ticks); z_wp = Float64.(wp.grid.z.ticks)
x_ep = Float64.(ep.grid.x.ticks); y_ep = Float64.(ep.grid.y.ticks); z_ep = Float64.(ep.grid.z.ticks)

iy_wp = argmin(abs.(y_wp .- 2.5e-3)); ix_wp = argmin(abs.(x_wp))
iy_ep = argmin(abs.(y_ep .- 2.5e-3)); ix_ep = argmin(abs.(x_ep))
println("  WP grid: $(length(x_wp)) × $(length(y_wp)) × $(length(z_wp)) — drift axis at (x[$(ix_wp)]=$(round(x_wp[ix_wp]*1000; digits=3)) mm, y[$(iy_wp)]=$(round(y_wp[iy_wp]*1000; digits=3)) mm)")
println("  EP grid: $(length(x_ep)) × $(length(y_ep)) × $(length(z_ep)) — drift axis at (x[$(ix_ep)]=$(round(x_ep[ix_ep]*1000; digits=3)) mm, y[$(iy_ep)]=$(round(y_ep[iy_ep]*1000; digits=3)) mm)")

# 1D drift-axis W_A(z) from WP grid
W_1d = [wp.data[ix_wp, iy_wp, iz] for iz in 1:length(z_wp)]
# 1D drift-axis V(z) from EP grid
V_1d = [ep.data[ix_ep, iy_ep, iz] for iz in 1:length(z_ep)]

# 2D W_A(x, z) at y=2.5 mm (WP grid)
W_2d = [wp.data[ix, iy_wp, iz] for ix in 1:length(x_wp), iz in 1:length(z_wp)]

outjson = joinpath(OUTDIR, "wp_anode_axis.json")
open(outjson, "w") do io
    JSON3.write(io, Dict(
        "anode_id" => ANODE_ID,
        "z_grid_mm_wp" => z_wp .* 1000,
        "z_grid_mm_ep" => z_ep .* 1000,
        "x_grid_mm_wp" => x_wp .* 1000,
        "y_axis_mm_wp" => y_wp[iy_wp] * 1000,
        "y_axis_mm_ep" => y_ep[iy_ep] * 1000,
        "W_A_1d" => W_1d,
        "V_1d_V" => V_1d,
        "W_A_2d" => W_2d,
    ))
end
println("\nSaved → $outjson")
println("W_A(z) values along drift axis: min=$(round(minimum(W_1d); digits=4))  max=$(round(maximum(W_1d); digits=4))")

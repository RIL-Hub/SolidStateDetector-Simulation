"""
Extract 2D weighting-potential (WP) slices at y = 2.5 mm for three contacts:
  anode    (id=3)   — collecting electrode; shows small-pixel effect
  steering (id=6)   — biased at −80 V; focuses charge toward anode strips
  cathode  (id=8)   — primary cathode row at y>0

Each slice is the xz plane at the nearest grid point to y=2.5 mm.
Grid is in metres; exported x_mm and z_mm arrays are in mm.
2D W arrays are written column-major — reshape in Python with:
  np.array(d["W_2d"]).reshape((nx, nz), order="F")

Output: output/wp_three_contacts.json
  Structure: {"contacts": {"anode": {id, x_mm, z_mm, y_slice_mm, W_2d}, ...}}

Run:
  julia --project=. -t8 scripts/export_wp_three_contacts.jl
"""

using SolidStateDetectors
using Unitful
using JSON3

const REPO     = abspath(joinpath(@__DIR__, ".."))
const GEOMETRY = joinpath(REPO, "geometries", "czt_cross_strip.yaml")
const OUTDIR   = joinpath(REPO, "output")
const REFINE   = [0.2, 0.1, 0.05]

# Contact IDs
const CONTACTS = Dict("anode" => 3, "steering" => 6, "cathode" => 8)

mkpath(OUTDIR)

println("Loading geometry ...")
sim = Simulation{Float32}(GEOMETRY)
println("  $(length(sim.detector.contacts)) contacts")

println("Solving electric potential ...")
t0 = @elapsed calculate_electric_potential!(sim;
    refinement_limits = REFINE, convergence_limit = 1e-6, depletion_handling = true)
println("  done ($(round(t0; digits=1))s)")

output = Dict{String,Any}("contacts" => Dict{String,Any}())

for (label, cid) in CONTACTS
    println("Solving W_$(label) (contact id $(cid)) ...")
    t1 = @elapsed calculate_weighting_potential!(sim, cid;
        refinement_limits = REFINE, convergence_limit = 1e-6)
    println("  done ($(round(t1; digits=1))s)")
    wp = sim.weighting_potentials[cid]
    x = Float64.(wp.grid.x.ticks); y = Float64.(wp.grid.y.ticks); z = Float64.(wp.grid.z.ticks)
    iy = argmin(abs.(y .- 2.5e-3))
    println("  slice at y[$(iy)] = $(round(y[iy]*1000; digits=3)) mm")
    W = [wp.data[ix, iy, iz] for ix in 1:length(x), iz in 1:length(z)]
    output["contacts"][label] = Dict(
        "id" => cid,
        "x_mm" => x .* 1000,
        "z_mm" => z .* 1000,
        "y_slice_mm" => y[iy] * 1000,
        "W_2d" => W,
    )
end

outjson = joinpath(OUTDIR, "wp_three_contacts.json")
open(outjson, "w") do io
    JSON3.write(io, output)
end
println("\nSaved → $outjson")

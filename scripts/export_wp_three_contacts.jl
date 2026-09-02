"""Extract 2D weighting-potential slices at y = 2.5 mm (primary
cathode row) for three contacts: target anode (id=3), steering
electrode (id=6), and primary cathode (id=8).  Exports to
output/wp_three_contacts.json for Python plotting.
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

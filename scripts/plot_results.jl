using JLD2
using Plots
using SolidStateDetectors
using Unitful

# Headless backend for CI
if get(ENV, "CI", "false") == "true"
    ENV["GKSwstype"] = "100"
end

# Load simulation results
input_file = joinpath(@__DIR__, "..", "output", "simulation_results.jld2")
println("Loading simulation results from $input_file...")
sim = load(input_file, "sim")

output_dir = joinpath(@__DIR__, "..", "output")
mkpath(output_dir)

# Plot electric potential - xz cross-section at y=0
println("Plotting electric potential (xz at y=0)...")
p1 = plot(sim.electric_potential, y = 0.0u"mm",
    title = "Electric Potential (y=0)",
    xlabel = "x [mm]", ylabel = "z [mm]",
    colorbar_title = "V")
savefig(p1, joinpath(output_dir, "electric_potential_xz.png"))

# Plot electric potential - xy cross-section at z=0
println("Plotting electric potential (xy at z=0)...")
p2 = plot(sim.electric_potential, z = 0.0u"mm",
    title = "Electric Potential (z=0)",
    xlabel = "x [mm]", ylabel = "y [mm]",
    colorbar_title = "V")
savefig(p2, joinpath(output_dir, "electric_potential_xy.png"))

# Plot weighting potentials for each contact
for (i, wp) in enumerate(sim.weighting_potentials)
    isnothing(wp) && continue
    println("Plotting weighting potential for contact $i...")
    p = plot(wp, y = 0.0u"mm",
        title = "Weighting Potential - Contact $i (y=0)",
        xlabel = "x [mm]", ylabel = "z [mm]")
    savefig(p, joinpath(output_dir, "weighting_potential_contact_$(i).png"))
end

println("All plots saved to $output_dir")

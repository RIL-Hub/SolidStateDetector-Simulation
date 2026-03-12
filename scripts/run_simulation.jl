using SolidStateDetectors
using Unitful
using JLD2

# Geometry file: use CLI arg or default
geometry_file = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "geometries", "czt_cross_strip.yaml")
geometry_file = abspath(geometry_file)

println("=" ^ 60)
println("CZT Cross-Strip Detector Simulation")
println("=" ^ 60)
println("Geometry file: $geometry_file")
println("Julia threads:  $(Threads.nthreads())")
println()

# Load detector configuration
println("Loading detector configuration...")
t_load = @elapsed sim = Simulation{Float32}(geometry_file)
println("  Done in $(round(t_load, digits=2))s")
println("  Number of contacts: $(length(sim.detector.contacts))")

# Calculate electric potential
println("\nCalculating electric potential...")
t_pot = @elapsed calculate_electric_potential!(sim,
    refinement_limits = [0.2, 0.1, 0.05],
    convergence_limit = 1e-6,
    depletion_handling = true,
)
println("  Done in $(round(t_pot, digits=2))s")

# Calculate electric field
println("\nCalculating electric field...")
t_field = @elapsed calculate_electric_field!(sim)
println("  Done in $(round(t_field, digits=2))s")

# Calculate weighting potentials for all contacts
println("\nCalculating weighting potentials...")
t_wp = @elapsed for contact in sim.detector.contacts
    println("  Contact $(contact.id)...")
    calculate_weighting_potential!(sim, contact.id)
end
println("  Done in $(round(t_wp, digits=2))s")

# Save results
output_dir = joinpath(@__DIR__, "..", "output")
mkpath(output_dir)
output_file = joinpath(output_dir, "simulation_results.jld2")

println("\nSaving results to $output_file...")
t_save = @elapsed jldsave(output_file; sim)
println("  Done in $(round(t_save, digits=2))s")

# Summary
println("\n" * "=" ^ 60)
println("Simulation Summary")
println("=" ^ 60)
println("  Load:               $(round(t_load, digits=2))s")
println("  Electric potential:  $(round(t_pot, digits=2))s")
println("  Electric field:      $(round(t_field, digits=2))s")
println("  Weighting potentials: $(round(t_wp, digits=2))s")
println("  Save:               $(round(t_save, digits=2))s")
println("  Total:              $(round(t_load + t_pot + t_field + t_wp + t_save, digits=2))s")
println("=" ^ 60)

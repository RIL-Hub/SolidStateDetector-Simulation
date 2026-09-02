"""
Solve electric potential and field for the simplified CZT geometry and export
an xz cross-section slice (at y=0 mm) to JSON for Python plotting.

The SSD grid is in metres; all exported arrays are converted to mm.
2D arrays are written column-major (Julia default) — reshape in Python with:
  np.array(data["pot_V"]).reshape((nx, nz), order="F")

Output: output/efield_xsection.json
  Keys: x_mm (nx,), z_mm (nz,), pot_V (nx*nz,), Ex_Vm (nx*nz,), Ez_Vm (nx*nz,)

Produces 3 figures (via plot_efield_xsection.py):
  efield_xsection_lines.png     — |E| heatmap + field streamlines
  efield_xsection_contour.png   — |E| heatmap + equipotential contours
  efield_xsection_potential.png — potential heatmap + streamlines

Run:
  julia --project=. -t8 scripts/plot_efield_xsection.jl
  python scripts/plot_efield_xsection.py
"""

using SolidStateDetectors
using Unitful
using JSON3

const REPO     = abspath(joinpath(@__DIR__, ".."))
const GEOMETRY = joinpath(REPO, "geometries", "czt_cross_strip.yaml")
const REFINE   = [0.2, 0.1, 0.05]
const OUTDIR   = joinpath(REPO, "output")
mkpath(OUTDIR)

println("Loading geometry...")
sim = Simulation{Float32}(GEOMETRY)
println("Solving electric potential...")
t0 = @elapsed calculate_electric_potential!(sim;
    refinement_limits=REFINE, convergence_limit=1e-6, depletion_handling=true)
println("  done ($(round(t0; digits=1))s)")
calculate_electric_field!(sim)

ep = sim.electric_potential
ef = sim.electric_field

x_m = Float64.(ep.grid.x.ticks)   # metres
y_m = Float64.(ep.grid.y.ticks)
z_m = Float64.(ep.grid.z.ticks)

iy = argmin(abs.(y_m))
println("y-slice at y = $(round(y_m[iy]*1000; digits=2)) mm (index $iy of $(length(y_m)))")

nx = length(x_m)
nz = length(z_m)

# Extract xz slices — potential and field components
pot = Matrix{Float64}(undef, nx, nz)
Ex  = Matrix{Float64}(undef, nx, nz)
Ez  = Matrix{Float64}(undef, nx, nz)

for ix in 1:nx, iz in 1:nz
    pot[ix, iz] = ep.data[ix, iy, iz]
    v = ef.data[ix, iy, iz]
    Ex[ix, iz] = v[1]
    Ez[ix, iz] = v[3]
end

out = Dict(
    "x_mm"  => x_m .* 1000,
    "z_mm"  => z_m .* 1000,
    "pot_V" => pot,
    "Ex_Vm" => Ex,
    "Ez_Vm" => Ez,
)

outjson = joinpath(OUTDIR, "efield_xsection.json")
open(outjson, "w") do io
    JSON3.write(io, out)
end
println("Saved → $outjson")
println("Grid: $(nx) x-points × $(nz) z-points")

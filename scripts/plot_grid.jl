"""
Visualize the adaptive grid produced by SolidStateDetectors.jl field solve.

Two panels:
  Left  — full xz cross-section of the detector showing overall grid density
  Right — zoomed view near the anode strips showing fine refinement

Run:  julia --project=. -t8 scripts/plot_grid.jl
"""

using SolidStateDetectors
using Unitful
using Plots; gr()
using Serialization

const REPO     = abspath(joinpath(@__DIR__, ".."))
const GEOMETRY = joinpath(REPO, "geometries", "czt_cross_strip.yaml")
const CACHE    = get(ENV, "SSD_CACHE", joinpath(REPO, "output", "sweep", "sim_cache.jls"))
const REFINE   = [0.2, 0.1, 0.05, 0.02]
const OUTDIR   = joinpath(REPO, "output")

# ── Load or solve ─────────────────────────────────────────────────────────────
sim = if isfile(CACHE)
    print("Loading cache … "); flush(stdout)
    t = @elapsed s = deserialize(CACHE)
    println("$(round(t;digits=1))s")
    s
else
    print("Solving electric potential … "); flush(stdout)
    s = Simulation{Float32}(GEOMETRY)
    t = @elapsed calculate_electric_potential!(s;
        refinement_limits = REFINE, convergence_limit = 1e-6, depletion_handling = true)
    println("$(round(t;digits=1))s")
    calculate_electric_field!(s)
    s
end

# ── Extract grid ticks (convert m → mm) ──────────────────────────────────────
ep  = sim.electric_potential
x_mm = Float64.(ep.grid.x.ticks) .* 1000
z_mm = Float64.(ep.grid.z.ticks) .* 1000

println("Grid: $(length(x_mm)) x-ticks × $(length(z_mm)) z-ticks")

# ── Helper: draw grid lines on a plot ────────────────────────────────────────
function add_gridlines!(p, xs, zs; lc=:grey, lw=0.4, la=0.5,
                        xlims=nothing, zlims=nothing)
    xlo, xhi = xlims === nothing ? (xs[1],   xs[end])  : xlims
    zlo, zhi = zlims === nothing ? (zs[1],   zs[end])  : zlims
    for x in xs
        xlo ≤ x ≤ xhi || continue
        plot!(p, [x, x], [zlo, zhi]; lc=lc, lw=lw, la=la, label=false)
    end
    for z in zs
        zlo ≤ z ≤ zhi || continue
        plot!(p, [xlo, xhi], [z, z]; lc=lc, lw=lw, la=la, label=false)
    end
end

# ── Plot 1: full detector xz view ────────────────────────────────────────────
# full detector overview (inset context)
p0 = plot(; xlabel="x (mm)", ylabel="z (mm)",
            title="Full detector",
            aspect_ratio=:equal, legend=false,
            xlims=(-22, 22), ylims=(-4, 4),
            size=(320, 220), dpi=150,
            background_color=:white, grid=false)
add_gridlines!(p0, x_mm, z_mm; lc=:steelblue, lw=0.3, la=0.5)
plot!(p0, [-20,20,20,-20,-20],[2.5,2.5,-2.5,-2.5,2.5]; lc=:black, lw=1.5, label=false)
plot!(p0, [-20,20],[-2.5,-2.5]; lc=:royalblue, lw=2, label=false)
for x in [-2,-1,0,1,2]
    plot!(p0,[x-0.05,x+0.05],[2.5,2.5]; lc=:orange, lw=2, label=false)
end

# medium zoom — anode region, all 5 strips visible
p1 = plot(; xlabel="x (mm)", ylabel="z (mm)",
            title="Adaptive grid — anode region",
            aspect_ratio=:equal, legend=false,
            xlims=(-3.5, 3.5), ylims=(1.6, 3.2),
            size=(520, 420), dpi=150,
            background_color=:white, grid=false)

add_gridlines!(p1, x_mm, z_mm; lc=:steelblue, lw=0.6, la=0.7,
               xlims=(-3.5, 3.5), zlims=(1.6, 3.2))

# crystal top surface
plot!(p1, [-3.5, 3.5], [2.5, 2.5]; lc=:black, lw=1.5, ls=:dash, label=false)

# anode strips
for x in [-2, -1, 0, 1, 2]
    plot!(p1, [x-0.05, x+0.05], [2.5, 2.5]; lc=:orange, lw=5, label=false)
    annotate!(p1, x, 2.67, text("a$(x+3)", 8, :orange, :center))
end

# steering strips
for x in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    -3.5 ≤ x ≤ 3.5 || continue
    plot!(p1, [x-0.2, x+0.2], [2.5, 2.5]; lc=:purple, lw=3, label=false)
end
annotate!(p1, -2.5, 2.78, text("steering", 7, :purple, :left))
annotate!(p1, -2.5, 1.75, text("crystal bulk →", 8, :grey, :left))

# ── Plot 2: zoomed near anodes ───────────────────────────────────────────────
xz_lo, xz_hi = -1.6,  1.6    # two anode pitches either side of centre
zz_lo, zz_hi =  2.1,  2.65   # tight window approaching anode face at z = +2.5

p2 = plot(; xlabel="x (mm)", ylabel="z (mm)",
            title="Adaptive grid — near anodes (zoomed)",
            aspect_ratio=:equal, legend=false,
            xlims=(xz_lo, xz_hi), ylims=(zz_lo, zz_hi),
            size=(500, 350), dpi=150,
            background_color=:white, grid=false)

add_gridlines!(p2, x_mm, z_mm; lc=:steelblue, lw=0.6, la=0.7,
               xlims=(xz_lo, xz_hi), zlims=(zz_lo, zz_hi))

# crystal top surface
plot!(p2, [xz_lo, xz_hi], [2.5, 2.5]; lc=:black, lw=1.5, label=false)

# anode strips within zoom window
for x in [-2, -1, 0, 1, 2]
    xz_lo ≤ x ≤ xz_hi || continue
    plot!(p2, [x-0.05, x+0.05], [2.5, 2.5]; lc=:orange, lw=4, label=false)
    annotate!(p2, x, 2.57, text("a$(x+3)", 7, :orange, :center))
end

# steering strips (between anodes)
for x in [-1.5, -0.5, 0.5, 1.5]
    xz_lo ≤ x ≤ xz_hi || continue
    plot!(p2, [x-0.2, x+0.2], [2.5, 2.5]; lc=:purple, lw=3, label=false)
end
annotate!(p2, -1.5, 2.65, text("steering", 7, :purple, :left))

# ── Combine and save ──────────────────────────────────────────────────────────
combined = plot(p0, p1, p2;
    layout=(1, 3), size=(1500, 520), dpi=150,
    plot_title="SolidStateDetectors.jl — Adaptive Grid Refinement",
    plot_titlefontsize=11)

mkpath(OUTDIR)
outfile = joinpath(OUTDIR, "grid_refinement.png")
savefig(combined, outfile)
println("Saved → $outfile")

# Also print grid spacing stats
dx = diff(x_mm); dz = diff(z_mm)
println("\nGrid spacing summary:")
println("  x: min=$(round(minimum(dx)*1000;digits=0)) µm  max=$(round(maximum(dx)*1000;digits=0)) µm  n=$(length(x_mm))")
println("  z: min=$(round(minimum(dz)*1000;digits=0)) µm  max=$(round(maximum(dz)*1000;digits=0)) µm  n=$(length(z_mm))")

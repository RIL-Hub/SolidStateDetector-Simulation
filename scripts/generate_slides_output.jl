"""
Generate all output images needed for the tutorial slides.
Run:  julia --project=. -t8 scripts/generate_slides_output.jl
"""

using SolidStateDetectors
using Unitful
using Plots; gr()

const REPO     = abspath(joinpath(@__DIR__, ".."))
const GEOMETRY = joinpath(REPO, "geometries", "czt_cross_strip.yaml")
const OUTDIR   = joinpath(REPO, "output")
const REFINE   = [0.2, 0.1, 0.05]

mkpath(OUTDIR)

println("Loading geometry...")
sim = Simulation{Float32}(GEOMETRY)
println("  $(length(sim.detector.contacts)) contacts")

# ── Electric potential ────────────────────────────────────────────────────────
println("Solving electric potential...")
t0 = @elapsed calculate_electric_potential!(sim;
    refinement_limits = REFINE,
    convergence_limit = 1e-6,
    depletion_handling = true,
)
println("  done ($(round(t0; digits=1))s)")
calculate_electric_field!(sim)

p_ep = plot(sim.electric_potential, y = 0.0u"mm",
    title = "Electric potential — xz slice at y = 0 mm",
    xlims = (-21, 21), ylims = (-3, 3),
    size = (700, 350), dpi = 150)
savefig(p_ep, joinpath(OUTDIR, "electric_potential_xz.png"))
println("Saved electric_potential_xz.png")

# ── Weighting potentials ──────────────────────────────────────────────────────
println("Solving weighting potentials...")
for contact in sim.detector.contacts
    t1 = @elapsed calculate_weighting_potential!(sim, contact.id;
        refinement_limits = REFINE,
        convergence_limit = 1e-6,
    )
    println("  contact $(contact.id): $(round(t1; digits=1))s")
end

# Center anode (contact 3) — shows small-pixel effect (for slide 7)
p_wp_anode = plot(sim.weighting_potentials[3], y = 0.0u"mm",
    title = "Weighting potential — anode 3 (center strip, small-pixel effect)",
    xlims = (-21, 21), ylims = (-3, 3),
    clims = (0, 1), size = (700, 350), dpi = 150)
savefig(p_wp_anode, joinpath(OUTDIR, "weighting_potential_contact.png"))
println("Saved weighting_potential_contact.png")

# Combined anode vs cathode contrast (for slides 7 and 11)
p_wp_cathode = plot(sim.weighting_potentials[8], y = 2.5u"mm",
    title = "Weighting potential — cathode 8 (y > 0, depth encoding)",
    xlims = (-21, 21), ylims = (-3, 3),
    clims = (0, 1), size = (700, 350), dpi = 150)
p_wps = plot(p_wp_anode, p_wp_cathode;
    layout = (1, 2), size = (1400, 350), dpi = 150,
    plot_title = "Weighting potentials — anode small-pixel (left) vs cathode depth-encoding (right)")
savefig(p_wps, joinpath(OUTDIR, "weighting_potentials.png"))
println("Saved weighting_potentials.png")

# ── Single event simulation ───────────────────────────────────────────────────
println("Simulating event at (0, 2.5, 0) mm...")
pos = CartesianPoint{Float32}(0f0, Float32(2.5 / 1000), 0f0)
evt = Event([pos], [662f0 * u"keV"], 50;
    number_of_shells = 2,
    radius = [0.1u"mm"],
)
t2 = @elapsed simulate!(evt, sim; Δt = 0.1u"ns", max_nsteps = 50_000)
println("  done ($(round(t2; digits=1))s)")

# Descriptive label and color for each contact type
contact_info = Dict(
    1 => ("anode 1",          :royalblue),
    2 => ("anode 2",          :royalblue),
    3 => ("anode 3 (center)", :royalblue),
    4 => ("anode 4",          :royalblue),
    5 => ("anode 5",          :royalblue),
    6 => ("steering (−80V)",  :purple),
    7 => ("cathode 7 (y<0)",  :crimson),
    8 => ("cathode 8 (y>0)",  :crimson),
)

p_wf = plot(
    title  = "Waveforms — 662 keV at (0, 2.5, 0) mm",
    xlabel = "Time (ns)", ylabel = "Induced charge (a.u.)",
    legend = :outerright, size = (1000, 500), dpi = 150)
for contact in sim.detector.contacts
    wf = evt.waveforms[contact.id]
    ismissing(wf) && continue
    t_ns = ustrip.(u"ns", collect(wf.time))
    q    = ustrip.(collect(wf.signal))
    label, color = get(contact_info, contact.id,
                       ("contact $(contact.id)", :grey))
    plot!(p_wf, t_ns, q; label, color)
end
savefig(p_wf, joinpath(OUTDIR, "single_event.png"))
println("Saved single_event.png")

# ── Drift paths (3D) ──────────────────────────────────────────────────────────
try
    p_3d = plot(sim.detector; size = (800, 600), dpi = 150, legend = :outertopright)
    for (i, dp) in enumerate(evt.drift_paths)
        e_lbl = i == 1 ? "electrons (→ anodes)"  : ""
        h_lbl = i == 1 ? "holes (→ cathodes)"    : ""
        if !isempty(dp.e_path)
            xs = [p.x * 1000 for p in dp.e_path]
            ys = [p.y * 1000 for p in dp.e_path]
            zs = [p.z * 1000 for p in dp.e_path]
            plot!(p_3d, xs, ys, zs; label = e_lbl, color = :royalblue, lw = 0.8, alpha = 0.7)
        end
        if !isempty(dp.h_path)
            xs = [p.x * 1000 for p in dp.h_path]
            ys = [p.y * 1000 for p in dp.h_path]
            zs = [p.z * 1000 for p in dp.h_path]
            plot!(p_3d, xs, ys, zs; label = h_lbl, color = :crimson, lw = 0.8, alpha = 0.7)
        end
    end
    savefig(p_3d, joinpath(OUTDIR, "drift_paths.png"))
    println("Saved drift_paths.png")
catch e
    println("drift_paths skipped: $(typeof(e)): $e")
end

println("\nAll done. Output in $OUTDIR")

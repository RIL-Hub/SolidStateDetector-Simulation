using SolidStateDetectors
using Unitful
using Plots

# ── Geometry ──────────────────────────────────────────────────────────────────
GEOMETRY = joinpath(@__DIR__, "..", "geometries", "thin_film.yaml")

# ── Field solve knobs ─────────────────────────────────────────────────────────
REFINEMENT_LIMITS = [0.2, 0.1, 0.05]   # mm — coarse→fine adaptive grid stages
CONVERGENCE_LIMIT = 1e-6
DEPLETION_HANDLING = true

# ── Charge cloud knobs ────────────────────────────────────────────────────────
ENERGY_KEV        = 5.9                # deposited energy (keV) — e.g. Fe-55 source
CLOUD_RADIUS_MM   = 0.01              # thin film: much smaller cloud than bulk
N_CARRIERS        = 50
N_SHELLS          = 2

# ── Drift knobs ───────────────────────────────────────────────────────────────
DT_NS             = 0.1               # time step (ns)
MAX_STEPS         = 50_000            # max drift steps

# ── Interaction point (SSD frame, mm) ─────────────────────────────────────────
X_MM, Y_MM, Z_MM  = 0.0, 0.0, 0.0   # centre of the thin film

# ── Run ───────────────────────────────────────────────────────────────────────
println("Loading geometry...")
sim = Simulation{Float32}(GEOMETRY)
println("  $(length(sim.detector.contacts)) contacts")

println("Solving electric potential...")
calculate_electric_potential!(sim;
    refinement_limits = REFINEMENT_LIMITS,
    convergence_limit = CONVERGENCE_LIMIT,
    depletion_handling = DEPLETION_HANDLING,
)
calculate_electric_field!(sim)

println("Solving weighting potentials...")
for contact in sim.detector.contacts
    calculate_weighting_potential!(sim, contact.id;
        refinement_limits = REFINEMENT_LIMITS,
        convergence_limit = CONVERGENCE_LIMIT,
    )
end

println("Simulating event at ($X_MM, $Y_MM, $Z_MM) mm...")
pos = CartesianPoint{Float32}(Float32(X_MM/1000), Float32(Y_MM/1000), Float32(Z_MM/1000))
evt = Event([pos], [Float32(ENERGY_KEV) * u"keV"], N_CARRIERS;
    number_of_shells = N_SHELLS,
    radius           = [CLOUD_RADIUS_MM * u"mm"],
)
simulate!(evt, sim; Δt = DT_NS * u"ns", max_nsteps = MAX_STEPS)

# ── Plot waveforms ────────────────────────────────────────────────────────────
p = plot(title="Thin film — single event at ($X_MM, $Y_MM, $Z_MM) mm",
         xlabel="Time (ns)", ylabel="Induced charge (a.u.)", legend=:outerright)

for contact in sim.detector.contacts
    wf = evt.waveforms[contact.id]
    ismissing(wf) && continue
    t = ustrip.(u"ns", collect(wf.time))
    q = ustrip.(collect(wf.signal))
    plot!(p, t, q, label="contact $(contact.id)")
end

outdir = joinpath(@__DIR__, "..", "output")
mkpath(outdir)
outfile = joinpath(outdir, "single_event_thin_film.png")
savefig(p, outfile)
println("Saved → $outfile")

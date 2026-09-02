using SolidStateDetectors
using Unitful
using Plots

# ── Geometry ──────────────────────────────────────────────────────────────────
GEOMETRY = joinpath(@__DIR__, "..", "geometries", "czt_cross_strip.yaml")

# ── Field solve knobs ─────────────────────────────────────────────────────────
# how fine to make the grid — more stages = finer near contacts, slower
REFINEMENT_LIMITS = [0.2, 0.1, 0.05]

# stop refining when the solution changes less than this between iterations
CONVERGENCE_LIMIT = 1e-6

# if bias < depletion voltage → part of crystal has no field, find that boundary
# if bias >= depletion voltage → no effect, safe to leave true
DEPLETION_HANDLING = true

# ── Charge cloud knobs ────────────────────────────────────────────────────────
# total energy deposited by the incoming radiation
ENERGY_KEV        = 662.0

# physical size of the charge cloud at the moment of creation
# too large → cloud shells span electrode gap → staircase artifact in waveform
CLOUD_RADIUS_MM   = 0.1

# how many points to sample the cloud with — more = smoother, slower
N_CARRIERS        = 50

# number of concentric shells the cloud is divided into (radii at 1r, 2r, ...)
N_SHELLS          = 2

# ── Drift knobs ───────────────────────────────────────────────────────────────
# time resolution of the output waveform
DT_NS             = 0.1

# maximum drift time = DT_NS × MAX_STEPS (here: 0.1 ns × 50000 = 5 µs)
MAX_STEPS         = 50_000

# ── Interaction point ─────────────────────────────────────────────────────────
# where the radiation deposits its energy, in mm (SSD detector frame)
X_MM, Y_MM, Z_MM  = 0.0, 2.5, 0.0

# ── Run ───────────────────────────────────────────────────────────────────────

# load geometry → build detector object with all contacts
println("Loading geometry...")
sim = Simulation{Float32}(GEOMETRY)
println("  $(length(sim.detector.contacts)) contacts")

# solve Poisson equation on adaptive grid → electric potential everywhere
println("Solving electric potential...")
calculate_electric_potential!(sim;
    refinement_limits = REFINEMENT_LIMITS,
    convergence_limit = CONVERGENCE_LIMIT,
    depletion_handling = DEPLETION_HANDLING,
)

# gradient of potential → electric field (tells carriers which way to drift)
calculate_electric_field!(sim)

# for each electrode: solve Laplace equation with that contact at 1V, rest at 0V
# result = weighting potential → used to compute induced charge via Shockley-Ramo
println("Solving weighting potentials...")
for contact in sim.detector.contacts
    calculate_weighting_potential!(sim, contact.id;
        refinement_limits = REFINEMENT_LIMITS,
        convergence_limit = CONVERGENCE_LIMIT,
    )
end

# place charge cloud at interaction point, drift carriers through E-field,
# integrate induced charge on each electrode over time → waveforms
println("Simulating event at ($X_MM, $Y_MM, $Z_MM) mm...")
pos = CartesianPoint{Float32}(Float32(X_MM/1000), Float32(Y_MM/1000), Float32(Z_MM/1000))
evt = Event([pos], [Float32(ENERGY_KEV) * u"keV"], N_CARRIERS;
    number_of_shells = N_SHELLS,
    radius           = [CLOUD_RADIUS_MM * u"mm"],
)
simulate!(evt, sim; Δt = DT_NS * u"ns", max_nsteps = MAX_STEPS)

# ── Plot waveforms ────────────────────────────────────────────────────────────

# for each contact: extract time axis and induced charge signal, plot
p = plot(title="Waveforms — single event at ($X_MM, $Y_MM, $Z_MM) mm",
         xlabel="Time (ns)", ylabel="Induced charge (a.u.)", legend=:outerright)

for contact in sim.detector.contacts
    wf = evt.waveforms[contact.id]
    ismissing(wf) && continue   # skip contacts with no weighting potential solved
    t  = ustrip.(u"ns", collect(wf.time))
    q  = ustrip.(collect(wf.signal))
    plot!(p, t, q, label="contact $(contact.id)")
end

outdir = joinpath(@__DIR__, "..", "output")
mkpath(outdir)
outfile = joinpath(outdir, "single_event.png")
savefig(p, outfile)
println("Saved → $outfile")

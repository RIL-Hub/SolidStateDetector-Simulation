"""
Sweep 9 interaction depths (0.5–4.5 mm from anode in 0.5 mm steps) and
decompose the induced anode charge into electron and hole contributions —
both with and without charge-carrier trapping — using Ramo's theorem.

Physics:
  ΔQ_e(t) = +ΔΦ_w(r_e(t))           electrons moving toward anode
  ΔQ_h(t) = -ΔΦ_w(r_h(t))           holes moving toward cathode
  q_trap(T) = Σ_j exp(-j·dt/τ)·ΔΦ_w  incremental survival weighting

Small-pixel effect: anode WP ≈ 0 in the bulk, so holes contribute little
except for events very close to the anode.

Geometry: czt_cross_strip.yaml (8-contact simplified geometry)
  τ_e = 5 µs, τ_h = 3 µs  (set in TAU_E_US / TAU_H_US constants below;
  must match the values in the YAML charge_trapping_model block)

Output: output/eh_polarization.json
  Per-depth arrays: time_ns, q_electron_free, q_electron_trap,
                    q_hole_free, q_hole_trap, q_total_free, q_total_trap

Run:
  julia --project=. -t8 scripts/plot_z_sweep_polarization.jl
  python scripts/plot_z_sweep_polarization.py
"""

using SolidStateDetectors
using Unitful
using Unitful: ustrip
using JSON3

const REPO     = abspath(joinpath(@__DIR__, ".."))
const GEOMETRY = joinpath(REPO, "geometries", "czt_cross_strip.yaml")
const OUTJSON  = joinpath(REPO, "output", "eh_polarization.json")
const REFINE   = [0.2, 0.1, 0.05]

const ANODE_ID   = 3      # center anode (0 V)
const CATHODE_ID = 8      # cathode y>0 (-600 V)
const STEER_ID   = 6      # steering (-80 V)

const DT_NS      = 0.1
const MAX_STEPS  = 100_000   # 10 us
const N_CARRIERS = 50
const ENERGY_KEV = 662.0f0

# Solve fields
println("Loading geometry...")
sim = Simulation{Float32}(GEOMETRY)

println("Electric potential...")
calculate_electric_potential!(sim; refinement_limits=REFINE,
    convergence_limit=1e-6, depletion_handling=true)
calculate_electric_field!(sim)

println("Weighting potentials...")
for id in [ANODE_ID, CATHODE_ID, STEER_ID]
    print("  contact $id ... "); flush(stdout)
    calculate_weighting_potential!(sim, id; refinement_limits=REFINE, convergence_limit=1e-6)
    println("done")
end

# Trapping lifetimes (must match geometry YAML)
const TAU_E_US = 5.0   # μs
const TAU_H_US = 3.0   # μs

# Helper: e/h contributions via Ramo's theorem, with and without trapping.
# Returns (t_ns, q_e_free, q_e_trap, q_h_free, q_h_trap)
# "free"  = no trapping weight (ideal carrier)
# "trap"  = incremental contributions weighted by survival exp(-t/τ)
function eh_contributions(drift_paths, wp, dt_ns, max_steps)
    n_grid = max_steps + 1
    q_e_free = zeros(Float64, n_grid)
    q_e_trap = zeros(Float64, n_grid)
    q_h_free = zeros(Float64, n_grid)
    q_h_trap = zeros(Float64, n_grid)

    itp = SolidStateDetectors.interpolated_scalarfield(wp)
    eval_wp(r) = Float64(itp(r.x, r.y, r.z))

    for dp in drift_paths
        # ── electrons ────────────────────────────────────────────────────────
        if !isempty(dp.e_path)
            Phi0 = eval_wp(dp.e_path[1])
            n_e  = length(dp.e_path)
            cumulative_trap = 0.0
            Phi_prev = Phi0
            for i in 1:n_e
                Phi_i    = eval_wp(dp.e_path[i])
                dPhi     = Phi_i - Phi_prev
                survival = exp(-(i-1) * dt_ns * 1e-3 / TAU_E_US)
                cumulative_trap += survival * dPhi
                q_e_free[i] += Phi_i - Phi0
                q_e_trap[i] += cumulative_trap
                Phi_prev = Phi_i
            end
            final_free = Phi_prev - Phi0
            final_trap = cumulative_trap  # use local var, not q_e_trap[n_e]
            if n_e < n_grid
                q_e_free[n_e+1:end] .+= final_free
                q_e_trap[n_e+1:end] .+= final_trap
            end
        end

        # ── holes ─────────────────────────────────────────────────────────────
        if !isempty(dp.h_path)
            Phi0 = eval_wp(dp.h_path[1])
            n_h  = length(dp.h_path)
            cumulative_trap = 0.0
            Phi_prev = Phi0
            for i in 1:n_h
                Phi_i    = eval_wp(dp.h_path[i])
                dPhi     = Phi_i - Phi_prev
                survival = exp(-(i-1) * dt_ns * 1e-3 / TAU_H_US)
                cumulative_trap += survival * (-dPhi)
                q_h_free[i] += -(Phi_i - Phi0)
                q_h_trap[i] += cumulative_trap
                Phi_prev = Phi_i
            end
            final_free = -(Phi_prev - Phi0)
            final_trap = cumulative_trap  # use local var, not q_h_trap[n_h]
            if n_h < n_grid
                q_h_free[n_h+1:end] .+= final_free
                q_h_trap[n_h+1:end] .+= final_trap
            end
        end
    end

    n = length(drift_paths)
    t_ns = collect(0:n_grid-1) .* dt_ns
    return t_ns, q_e_free./n, q_e_trap./n, q_h_free./n, q_h_trap./n
end

# Sweep three depths: 0.5, 2.0, 3.0 mm from anode
# SSD z = anode_z - depth = 2.5 - depth
DEPTHS_MM = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]   # distance from anode face
X_MM      = 0.0
Y_MM      = 2.5

wp_a = sim.weighting_potentials[ANODE_ID]

depth_records = Dict{String,Any}[]

for depth in DEPTHS_MM
    z_mm = 2.5 - depth
    println("\nSimulating 662 keV at z=$(z_mm) mm  [$(depth) mm from anode]...")
    pos = CartesianPoint{Float32}(Float32(X_MM/1000), Float32(Y_MM/1000), Float32(z_mm/1000))
    evt = Event([pos], [ENERGY_KEV * u"keV"], N_CARRIERS; number_of_shells=2)
    t_sim = @elapsed simulate!(evt, sim; Δt=DT_NS * u"ns", max_nsteps=MAX_STEPS)
    println("  done ($(round(t_sim; digits=1))s), $(length(evt.drift_paths)) drift paths")

    println("  Computing e/h contributions (with trapping)...")
    t_eh, q_e_free, q_e_trap, q_h_free, q_h_trap = eh_contributions(evt.drift_paths, wp_a, DT_NS, MAX_STEPS)

    last_nz = findlast(x -> abs(x) > 1e-10, q_e_free .+ q_h_free)
    last_nz = isnothing(last_nz) ? length(t_eh) : min(last_nz + 1000, length(t_eh))
    t_eh     = t_eh[1:last_nz]
    q_e_free = q_e_free[1:last_nz]
    q_e_trap = q_e_trap[1:last_nz]
    q_h_free = q_h_free[1:last_nz]
    q_h_trap = q_h_trap[1:last_nz]

    wf_a  = evt.waveforms[ANODE_ID]
    t_tot = ismissing(wf_a) ? Float64[] : Float64.(ustrip.(u"ns", collect(wf_a.time)))
    q_tot = ismissing(wf_a) ? Float64[] : Float64.(ustrip.(collect(wf_a.signal)))

    frac_e = q_e_free[end] / max(q_e_free[end] + q_h_free[end], 1e-15)
    println("  e fraction = $(round(frac_e*100; digits=1))%  h fraction = $(round((1-frac_e)*100; digits=1))%")
    h_lost = (q_h_free[end] - q_h_trap[end]) / max(q_h_free[end], 1e-15)
    println("  hole trapping loss = $(round(h_lost*100; digits=1))%  (τ_h=$(TAU_H_US) μs)")

    push!(depth_records, Dict{String,Any}(
        "depth_from_anode_mm" => depth,
        "z_mm"                => z_mm,
        "tau_h_us"            => TAU_H_US,
        "tau_e_us"            => TAU_E_US,
        "eh" => Dict(
            "time_ns"    => t_eh,
            "q_electron_free" => q_e_free,
            "q_electron_trap" => q_e_trap,
            "q_hole_free"     => q_h_free,
            "q_hole_trap"     => q_h_trap,
            "q_total_free"    => q_e_free .+ q_h_free,
            "q_total_trap"    => q_e_trap .+ q_h_trap,
        ),
        "ssd_total" => Dict(
            "time_ns" => t_tot,
            "signal"  => q_tot,
        ),
    ))
end

out = Dict(
    "energy_keV" => Float64(ENERGY_KEV),
    "n_carriers" => N_CARRIERS,
    "dt_ns"      => DT_NS,
    "depths"     => depth_records,
)

mkpath(dirname(OUTJSON))
open(OUTJSON, "w") do io; JSON3.write(io, out); end
println("\nSaved -> $OUTJSON")

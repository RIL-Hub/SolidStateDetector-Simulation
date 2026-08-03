#!/usr/bin/env julia
"""
Same 15 events as export_drift_paths.jl but with thermal diffusion AND
self-repulsion enabled during drift. Used to compare against the
ballistic (no-diffusion, no-self-repulsion) baseline.

Writes: output/lateral/drift_paths_export_diffuse.json
"""

const REPO = abspath(joinpath(@__DIR__, ".."))
const GEOMETRY = get(ENV, "GEOMETRY",
    joinpath(REPO, "geometries", "czt_cross_strip_full.yaml"))
const OUT_JSON = get(ENV, "OUTPUT",
    joinpath(REPO, "output", "lateral", "drift_paths_export_diffuse.json"))

const REFINE = [0.2, 0.1, 0.05]
const X_LIST = [-0.4, -0.2, 0.0, +0.2, +0.4]
const Z_LIST = [2.0, 0.0, -2.0]
const Y_MM = 2.5
const ENERGY_KEV = 662.0
const N_CARRIERS = 50
const CLOUD_RADIUS_MM = 0.2
const NSHELLS = 2
const DT_NS = 0.1
const MAX_NSTEPS = 50000

print("Loading SolidStateDetectors … "); flush(stdout)
using SolidStateDetectors
using Unitful; using Unitful: ustrip
println("ok")

# Override CdZnTe material properties to include diffusion coefficients
# (default package definition lacks De/Dh so diffusion=true is ignored).
# Einstein relation D = μ kT/e at 300 K:
#   μ_e ≈ 1000 cm²/(V·s) → De ≈ 26 cm²/s
#   μ_h ≈  100 cm²/(V·s) → Dh ≈ 2.6 cm²/s
if haskey(SolidStateDetectors.material_properties, :CdZnTe)
    old = SolidStateDetectors.material_properties[:CdZnTe]
    new = merge(NamedTuple(pairs(old)),
                  (; De = 26u"cm^2/s", Dh = 2.6u"cm^2/s"))
    SolidStateDetectors.material_properties[:CdZnTe] = new
    println("CdZnTe: added De=26, Dh=2.6 cm²/s for diffusion")
else
    println("WARN: CdZnTe material not found in material_properties")
end

println("Geometry: $GEOMETRY")
print("Parsing … "); flush(stdout)
sim = Simulation{Float32}(GEOMETRY)
println("$(length(sim.detector.contacts)) contacts")

print("Electric potential … "); flush(stdout)
t = @elapsed calculate_electric_potential!(sim; refinement_limits=REFINE,
    convergence_limit=1e-6, depletion_handling=true)
println("$(round(t;digits=1))s")
calculate_electric_field!(sim)

to_json(v::AbstractVector{<:Number}) = "[" * join((isnan(x)||isinf(x) ? "null" : string(Float64(x)) for x in v), ",") * "]"
to_json(v::AbstractVector{<:AbstractString}) = "[" * join((to_json(s) for s in v), ",") * "]"
to_json(v::Number) = (isnan(v)||isinf(v)) ? "null" : string(Float64(v))
to_json(v::String) = "\"$(replace(v, "\"" => "\\\""))\""
function to_json(d::AbstractDict)
    io = IOBuffer(); print(io, "{")
    for (i, (k, v)) in enumerate(d)
        i > 1 && print(io, ",")
        print(io, "\"$k\":")
        v isa AbstractDict ? print(io, to_json(v)) : v isa AbstractVector ? print(io, to_json(v)) :
        v isa Number ? print(io, to_json(v)) : v isa String ? print(io, to_json(v)) : print(io, "\"$(v)\"")
    end
    print(io, "}"); String(take!(io))
end
to_json(v::AbstractVector{<:AbstractDict}) = "[" * join(to_json.(v), ",") * "]"

function extract_paths(evt)
    dps = evt.drift_paths
    carriers = Dict{String,Any}[]
    for i in 1:length(dps)
        dp = dps[i]
        e_xs = Float64[Float64(ustrip(u"mm", p.x * u"m")) for p in dp.e_path]
        e_ys = Float64[Float64(ustrip(u"mm", p.y * u"m")) for p in dp.e_path]
        e_zs = Float64[Float64(ustrip(u"mm", p.z * u"m")) for p in dp.e_path]
        h_xs = Float64[Float64(ustrip(u"mm", p.x * u"m")) for p in dp.h_path]
        h_ys = Float64[Float64(ustrip(u"mm", p.y * u"m")) for p in dp.h_path]
        h_zs = Float64[Float64(ustrip(u"mm", p.z * u"m")) for p in dp.h_path]
        t_e_ns = Float64[Float64(ustrip(u"ns", t * u"s")) for t in dp.timestamps_e]
        t_h_ns = Float64[Float64(ustrip(u"ns", t * u"s")) for t in dp.timestamps_h]
        push!(carriers, Dict{String,Any}(
            "e_x_mm"=>e_xs, "e_y_mm"=>e_ys, "e_z_mm"=>e_zs,
            "h_x_mm"=>h_xs, "h_y_mm"=>h_ys, "h_z_mm"=>h_zs,
            "e_time_ns"=>t_e_ns, "h_time_ns"=>t_h_ns))
    end
    return carriers
end

records = Dict{String,Any}[]
println("Simulating $(length(X_LIST) * length(Z_LIST)) events "
        * "(diffusion=true, self_repulsion=true):")
for z_mm in Z_LIST
    depth = 2.5 - z_mm
    for x_mm in X_LIST
        print("  x=$(x_mm), z=$(z_mm) mm … "); flush(stdout)
        pos = CartesianPoint{Float32}(Float32(x_mm/1000),
                                        Float32(Y_MM/1000),
                                        Float32(z_mm/1000))
        evt = Event([pos], [Float32(ENERGY_KEV) * u"keV"], N_CARRIERS;
            number_of_shells=NSHELLS, radius=[CLOUD_RADIUS_MM * u"mm"])
        tt = @elapsed simulate!(evt, sim; Δt=DT_NS * u"ns",
                                  max_nsteps=MAX_NSTEPS,
                                  diffusion=true, self_repulsion=true)
        push!(records, Dict{String,Any}(
            "x_mm"=>x_mm, "y_mm"=>Y_MM, "z_mm"=>z_mm,
            "depth_from_anode_mm"=>depth, "energy_keV"=>ENERGY_KEV,
            "n_carriers"=>length(evt.drift_paths),
            "carriers"=>extract_paths(evt)))
        println("$(round(tt;digits=1))s   "
                * "($(length(evt.drift_paths)) drift paths)")
    end
end

output = Dict{String,Any}(
    "simulator"=>"SolidStateDetectors.jl",
    "geometry_file"=>GEOMETRY,
    "diffusion"=>true, "self_repulsion"=>true,
    "y_mm"=>Y_MM, "energy_keV"=>ENERGY_KEV,
    "cloud_radius_mm"=>CLOUD_RADIUS_MM,
    "n_carriers"=>N_CARRIERS, "n_shells"=>NSHELLS,
    "dt_ns"=>DT_NS, "events"=>records)
mkpath(dirname(OUT_JSON))
write(OUT_JSON, to_json(output) * "\n")
println("Wrote $OUT_JSON")

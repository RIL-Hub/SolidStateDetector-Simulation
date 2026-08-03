#!/usr/bin/env julia
"""Export the steering-electrode weighting potential W_steer(x, z) at
y = 2.5 mm from the FULL 39-anode geometry cache (sim_cache.jls).

Steering contact ID in the full geometry is 40 (not 6, which was
the id in the smaller czt_cross_strip.yaml).

Writes: output/wp_full_steering.json
"""

const REPO     = abspath(joinpath(@__DIR__, ".."))
const CACHE    = get(ENV, "SSD_CACHE",
    joinpath(REPO, "output", "sweep", "sim_cache.jls"))
const OUT_JSON = joinpath(REPO, "output", "wp_full_steering.json")
const STEER_ID = 40
const Y_SLICE_MM = 2.5

print("Loading SolidStateDetectors … "); flush(stdout)
using SolidStateDetectors
using Unitful; using Unitful: ustrip
using Serialization
println("ok")

isfile(CACHE) || error("Cache not found: $CACHE")
print("Loading cache ($(round(filesize(CACHE)/1e6;digits=0)) MB) … ")
flush(stdout)
t = @elapsed (sim = deserialize(CACHE)); println("$(round(t;digits=1))s")

wp = sim.weighting_potentials[STEER_ID]
x = Float64.(wp.grid.x.ticks)
y = Float64.(wp.grid.y.ticks)
z = Float64.(wp.grid.z.ticks)

iy = argmin(abs.(y .- Y_SLICE_MM * 1e-3))
println("Grid: nx=$(length(x)), ny=$(length(y)), nz=$(length(z))")
println("y slice at index $iy → y = $(round(y[iy]*1000; digits=3)) mm")

W = [wp.data[ix, iy, iz] for ix in 1:length(x), iz in 1:length(z)]

# JSON helpers
to_json(v::AbstractVector{<:Number}) = "[" * join((isnan(x)||isinf(x) ? "null" : string(Float64(x)) for x in v), ",") * "]"
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

output = Dict{String,Any}(
    "geometry" => "czt_cross_strip_full.yaml (39 anodes)",
    "contact_id" => STEER_ID,
    "contact_label" => "steering",
    "x_mm" => x .* 1000,
    "z_mm" => z .* 1000,
    "y_slice_mm" => y[iy] * 1000,
    "W_2d" => vec(W),     # column-major Julia → row-major-safe with order="F"
)
write(OUT_JSON, to_json(output) * "\n")
println("Wrote $OUT_JSON")

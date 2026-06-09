#!/usr/bin/env julia
"""
Generate the FULL CZT cross-strip detector geometry:
  - 39 anode strips   (ids 1..39):  1 mm pitch, 100 µm wide, x = -19..+19, z=+2.5, 0 V
  - steering grid     (id 40):       400 µm strips at half-integer x = -19.5..+19.5, z=+2.5, -80 V
  - 8 cathode strips  (ids 41..48):  5 mm pitch, 4.9 mm wide, y centers ±2.5,±7.5,±12.5,±17.5, z=-2.5, -600 V

Same conventions/bulk as geometries/czt_cross_strip.yaml. Contact ids are kept
contiguous (1..48) so that index in evt.waveforms == contact id.

Writes:
  geometries/czt_cross_strip_full.yaml
  output/contact_map.json   (id -> name/type/position, for the sim + plot scripts)
"""

const REPO = abspath(joinpath(@__DIR__, ".."))
const OUT_YAML = joinpath(REPO, "geometries", "czt_cross_strip_full.yaml")
const OUT_MAP  = joinpath(REPO, "output", "contact_map.json")

# ── Detector parameters ──
const N_ANODE        = 39
const ANODE_PITCH    = 1.0    # mm
const ANODE_HX       = 0.05   # mm half-width (100 µm wide)
const ANODE_HY       = 20.0   # mm (full y span)
const ANODE_Z        = 2.5    # mm (top face)
const ANODE_V        = 0

const STEER_HX       = 0.2    # mm half-width (400 µm wide)
const STEER_V        = -80

const N_CATHODE      = 8
const CATHODE_PITCH  = 5.0    # mm
const CATHODE_HY     = 2.45   # mm half-width (4.9 mm wide, 100 µm gaps)
const CATHODE_HX     = 20.0   # mm (full x span)
const CATHODE_Z      = -2.5   # mm (bottom face)
const CATHODE_V      = -600

# Anode x positions: symmetric about 0 at 1 mm pitch -> -19..+19
anode_x(i) = (i - (N_ANODE + 1) / 2) * ANODE_PITCH           # i = 1..39
# Steering x positions: half-integers spanning the array edges -> -19.5..+19.5
steer_xs() = collect(-(N_ANODE) / 2 : 1.0 : (N_ANODE) / 2)   # 40 strips
# Cathode y centers: symmetric about 0 at 5 mm pitch -> ±2.5,±7.5,±12.5,±17.5
cathode_y(j) = (j - (N_CATHODE + 1) / 2) * CATHODE_PITCH      # j = 1..8

fnum(x) = (x == round(x)) ? string(Int(round(x))) : string(x)

function build_yaml()
    io = IOBuffer()
    println(io, "name: CZT Strip Detector (full 39 anodes / steering / 8 cathodes)")
    println(io)
    println(io, "units:")
    println(io, "  length: mm")
    println(io, "  angle: deg")
    println(io, "  potential: V")
    println(io, "  temperature: K")
    println(io)
    println(io, "grid:")
    println(io, "  coordinates: cartesian")
    println(io, "  axes:")
    for (ax, lo, hi) in (("x", -22, 22), ("y", -22, 22), ("z", -4, 4))
        println(io, "    $ax:")
        println(io, "      from: $lo")
        println(io, "      to: $hi")
        println(io, "      boundaries:")
        println(io, "        left: reflecting")
        println(io, "        right: reflecting")
    end
    println(io)
    println(io, "medium: vacuum")
    println(io)
    println(io, "detectors:")
    println(io, "  - semiconductor:")
    println(io, "      material: CdZnTe")
    println(io, "      temperature: 293")
    println(io, "      geometry:")
    println(io, "        box:")
    println(io, "          widths: [40, 40, 5]")
    println(io, "      charge_drift_model:")
    println(io, "        model: IsotropicChargeDriftModel")
    println(io, "        mobilities:")
    println(io, "          e: 1000cm^2/(V*s)")
    println(io, "          h: 50cm^2/(V*s)")
    println(io, "      impurity_density:")
    println(io, "        name: constant")
    println(io, "        value: 0")
    println(io, "      charge_trapping_model:")
    println(io, "        model: ConstantLifetime")
    println(io, "        parameters:")
    println(io, "          τe: 10μs")
    println(io, "          τh: 1μs")
    println(io)
    println(io, "    contacts:")

    contact_map = String[]  # JSON lines

    # ── Anodes (ids 1..39) ──
    println(io, "      # ── Anode strips (z=+2.5 mm): 39 strips, 100 µm wide, 1 mm pitch, 0 V ──")
    for i in 1:N_ANODE
        x = anode_x(i)
        println(io, "      - id: $i")
        println(io, "        potential: $ANODE_V")
        println(io, "        geometry:")
        println(io, "          box:")
        println(io, "            hX: $ANODE_HX")
        println(io, "            hY: $ANODE_HY")
        println(io, "            hZ: 0")
        println(io, "            origin: [$(fnum(x)), 0, $(fnum(ANODE_Z))]")
        push!(contact_map, "\"$i\":{\"name\":\"anode_$i\",\"type\":\"anode\",\"x\":$x,\"y\":0,\"z\":$ANODE_Z}")
    end

    # ── Steering (id 40) ──
    println(io, "      # ── Steering electrode (z=+2.5 mm): 400 µm strips at half-integer x, -80 V ──")
    steer_id = N_ANODE + 1
    println(io, "      - id: $steer_id")
    println(io, "        potential: $STEER_V")
    println(io, "        geometry:")
    println(io, "          union:")
    for x in steer_xs()
        println(io, "            - box:")
        println(io, "                hX: $STEER_HX")
        println(io, "                hY: $ANODE_HY")
        println(io, "                hZ: 0")
        println(io, "                origin: [$(fnum(x)), 0, $(fnum(ANODE_Z))]")
    end
    push!(contact_map, "\"$steer_id\":{\"name\":\"steering\",\"type\":\"steering\",\"x\":0,\"y\":0,\"z\":$ANODE_Z}")

    # ── Cathodes (ids 41..48) ──
    println(io, "      # ── Cathode strips (z=-2.5 mm): 8 strips, 4.9 mm wide, 5 mm pitch, -600 V ──")
    for j in 1:N_CATHODE
        id = N_ANODE + 1 + j
        y = cathode_y(j)
        println(io, "      - id: $id")
        println(io, "        potential: $CATHODE_V")
        println(io, "        geometry:")
        println(io, "          box:")
        println(io, "            hX: $CATHODE_HX")
        println(io, "            hY: $CATHODE_HY")
        println(io, "            hZ: 0")
        println(io, "            origin: [0, $(fnum(y)), $(fnum(CATHODE_Z))]")
        push!(contact_map, "\"$id\":{\"name\":\"cathode_$j\",\"type\":\"cathode\",\"x\":0,\"y\":$y,\"z\":$CATHODE_Z}")
    end

    return String(take!(io)), contact_map
end

yaml_str, contact_map = build_yaml()
mkpath(dirname(OUT_YAML))
write(OUT_YAML, yaml_str)
mkpath(dirname(OUT_MAP))
write(OUT_MAP, "{" * join(contact_map, ",") * "}\n")

n_contacts = N_ANODE + 1 + N_CATHODE
println("Wrote $OUT_YAML")
println("Wrote $OUT_MAP")
println("Contacts: $n_contacts  (anodes 1..$N_ANODE, steering $(N_ANODE+1), cathodes $(N_ANODE+2)..$n_contacts)")
println("Anode x range:   $(anode_x(1)) .. $(anode_x(N_ANODE)) mm")
println("Steering x range: $(minimum(steer_xs())) .. $(maximum(steer_xs())) mm ($(length(steer_xs())) strips)")
println("Cathode y centers: $(round.([cathode_y(j) for j in 1:N_CATHODE]; digits=1))")

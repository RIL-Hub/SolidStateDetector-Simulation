#!/usr/bin/env bash
# Shard the (z × E × rep) SSD sweep into N parallel Julia processes.
#
# Each shard covers a disjoint z-range and writes its own JLD2 output
# to output/zE_sweep/z<idx>_z<lo>_z<hi>.jld2.  Each process independently
# loads the 559 MB WP cache — plan for ~600 MB × N_SHARDS of RAM.
#
# Usage:
#   scripts/shard_launcher.sh [N_SHARDS] [Z_LO] [Z_HI]
#
# Defaults: 51 shards, z ∈ [0.0, 5.0] mm (0.1 mm step per shard).
#
# Recommended per-shard threads:
#   Local Mac (8 physical cores): N_SHARDS=1..4  with -t8
#   Hummingbird node (usually 32+ cores): N_SHARDS = ncores / 4
#
# Environment:
#   SSD_CACHE   Path to sim_cache.jls (default: output/sweep/sim_cache.jls)
#   JULIA_THREADS  Passed as -t to each julia process (default 4)
#   OUT_DIR     Where to write JLD2 shards (default: output/zE_sweep)
#   SEED_BASE   Base RNG seed (default 42). Per-shard seed = SEED_BASE + shard_idx*10000.
#   E_LO, E_HI, E_STEP, N_REPS   Passed through to zE_sweep.jl.

set -euo pipefail

N_SHARDS="${1:-51}"
Z_LO="${2:-0.0}"
Z_HI="${3:-5.0}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
: "${SSD_CACHE:=$REPO/output/sweep/sim_cache.jls}"
: "${JULIA_THREADS:=4}"
: "${OUT_DIR:=$REPO/output/zE_sweep}"
: "${SEED_BASE:=42}"
: "${E_LO:=10.0}"
: "${E_HI:=2000.0}"
: "${E_STEP:=10.0}"
: "${N_REPS:=10}"

mkdir -p "$OUT_DIR"

if [ ! -f "$SSD_CACHE" ]; then
    echo "ERROR: no cache at $SSD_CACHE"
    echo "Build first with: julia --project=. scripts/build_wp_cache.jl"
    exit 1
fi

# Split [Z_LO, Z_HI] into N_SHARDS equal chunks (in mm).
Z_STEP_SHARD=$(python3 -c "print(($Z_HI - $Z_LO) / $N_SHARDS)")
Z_STEP_INNER="0.1"   # step used inside each shard (must match plan)

echo "[launcher] Sweep: z ∈ [$Z_LO, $Z_HI] mm, $N_SHARDS shards × ${Z_STEP_SHARD} mm"
echo "[launcher] E    : [$E_LO, $E_HI] keV at $E_STEP keV step"
echo "[launcher] Reps : $N_REPS per (z, E) point"
echo "[launcher] Threads per shard: $JULIA_THREADS"
echo "[launcher] Cache: $SSD_CACHE"
echo "[launcher] Out  : $OUT_DIR"
echo

for i in $(seq 0 $((N_SHARDS - 1))); do
    Z_SHARD_LO=$(python3 -c "print($Z_LO + $i * $Z_STEP_SHARD)")
    Z_SHARD_HI=$(python3 -c "print($Z_LO + ($i + 1) * $Z_STEP_SHARD - $Z_STEP_INNER)")
    # Clip Z_SHARD_HI on the last shard so it hits Z_HI exactly
    if [ "$i" -eq "$((N_SHARDS - 1))" ]; then
        Z_SHARD_HI="$Z_HI"
    fi
    SHARD_SEED=$((SEED_BASE + i * 10000))
    OUT_FILE="$OUT_DIR/shard_${i}_z${Z_SHARD_LO}_z${Z_SHARD_HI}.jld2"
    LOG_FILE="$OUT_DIR/shard_${i}.log"
    echo "[launcher] shard $i: z ∈ [$Z_SHARD_LO, $Z_SHARD_HI] → $OUT_FILE (seed $SHARD_SEED)"
    SSD_CACHE="$SSD_CACHE" \
    julia --project="$REPO" -t "$JULIA_THREADS" \
        "$REPO/scripts/zE_sweep.jl" \
        --z-lo "$Z_SHARD_LO" --z-hi "$Z_SHARD_HI" --z-step "$Z_STEP_INNER" \
        --e-lo "$E_LO" --e-hi "$E_HI" --e-step "$E_STEP" \
        --n-reps "$N_REPS" \
        --seed "$SHARD_SEED" \
        --out "$OUT_FILE" \
        > "$LOG_FILE" 2>&1 &
done

echo
echo "[launcher] Launched $N_SHARDS shards. Waiting for all to complete…"
wait
echo "[launcher] All shards done. Outputs in $OUT_DIR/"

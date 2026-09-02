#!/usr/bin/env bash
# Assemble a MINIMAL agent-handoff bundle for the (z × E × rep) SSD
# sweep on Hummingbird HPC.
#
# Assumes:
# - Hummingbird has internet access (pkg.julialang.org, PyPI)
# - A separate agent will receive this bundle, discover cluster
#   specifics (Julia module, partitions, quotas), generate a
#   customized SLURM script from the template, and run the sweep.
#
# Bundle contents (tiny, ~30 KB compressed):
#   AGENT_HANDOFF.md         Main instructions for the receiving agent
#   Project.toml             Julia deps
#   Manifest.toml            Exact Julia versions (pinned)
#   requirements.txt         Python deps
#   geometries/              CZT geometry YAML
#   scripts/                 Julia driver + benchmark + cache builder + SLURM template
#   python/                  Feature extractor + preview plotter
#   benchmark_reference.json Local benchmark results (for ETA comparison)
#
# Usage:
#   scripts/package_for_hummingbird.sh [OUT_TARBALL]

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
: "${CZT_ROOT:=$HOME/Documents/Claude/Waveform Processing/czt_doi}"

OUT="${1:-$HOME/zE_sweep_handoff.tar.gz}"

STAGE="$(mktemp -d)/zE_sweep_handoff"
mkdir -p "$STAGE"/{scripts,geometries,python}

echo "[pack] Staging in $STAGE"

# ── Julia scripts (driver, benchmark, cache builder, launcher) ──
cp "$REPO/scripts/build_wp_cache.jl"       "$STAGE/scripts/"
cp "$REPO/scripts/benchmark_zE_sweep.jl"   "$STAGE/scripts/"
cp "$REPO/scripts/zE_sweep.jl"             "$STAGE/scripts/"
cp "$REPO/scripts/shard_launcher.sh"       "$STAGE/scripts/"

# ── Geometry ──
cp "$REPO/geometries/czt_cross_strip_full.yaml" "$STAGE/geometries/"

# ── Julia deps ──
cp "$REPO/Project.toml"  "$STAGE/"
if [ -f "$REPO/Manifest.toml" ]; then
    cp "$REPO/Manifest.toml" "$STAGE/"
fi

# ── Python (feature extractor + plotter + requirements) ──
if [ -d "$CZT_ROOT/scripts" ]; then
    cp "$CZT_ROOT/scripts/extract_zE_features.py" "$STAGE/python/"
    cp "$CZT_ROOT/scripts/plot_zE_sweep.py"       "$STAGE/python/"
fi
cat > "$STAGE/requirements.txt" << 'EOF'
h5py>=3.0
numpy>=1.20
pandas>=1.3
pyarrow>=6.0
matplotlib>=3.5
EOF

# ── Benchmark reference (from local run, for cluster ETA comparison) ──
if [ -f "$REPO/output/benchmark_zE_sweep.json" ]; then
    cp "$REPO/output/benchmark_zE_sweep.json" \
        "$STAGE/benchmark_reference.json"
fi

# ── SLURM template ──
cat > "$STAGE/scripts/slurm_template.sbatch" << 'EOF'
#!/usr/bin/env bash
# SLURM template for the (z × E × rep) SSD sweep.
# The receiving agent should edit the fields marked [ADAPT] to match
# Hummingbird's specifics (partition, account, module names, mem, time).

#SBATCH --job-name=zE_sweep
#SBATCH --array=0-50%10                # 51 shards, up to 10 concurrent [ADAPT concurrency]
#SBATCH --cpus-per-task=4              # 4 threads/shard [ADAPT to match node]
#SBATCH --mem=4G                       # 600 MB cache + event buffer [ADAPT]
#SBATCH --time=03:30:00                # 2.5h expected + margin [ADAPT to worst-case shard]
#SBATCH --output=logs/shard_%a.out
#SBATCH --error=logs/shard_%a.err
# [ADAPT]:
#   --partition=<partition-name>
#   --account=<charge-account>
#   --mail-user=<email>
#   --mail-type=END,FAIL

set -euo pipefail

# ── Environment ── [ADAPT to Hummingbird's module system]
module load julia/1.10                # or: export PATH=/opt/julia/bin:$PATH
module load python/3.11               # or your preferred python
source $HOME/zE_sweep_venv/bin/activate

BUNDLE="$(cd "$(dirname "$0")/.." && pwd)"
export JULIA_PROJECT="$BUNDLE"

# WP cache location — must be pre-built via build_wp_cache.jl.
# Recommend staging on parallel filesystem (accessible from all nodes).
SSD_CACHE="${SSD_CACHE:-$BUNDLE/output/sweep/sim_cache.jls}"
if [ ! -f "$SSD_CACHE" ]; then
    echo "[sbatch] No cache at $SSD_CACHE"
    echo "[sbatch] Build first with: julia --project=. -t 8 scripts/build_wp_cache.jl"
    exit 1
fi

# ── Shard params ──
N_SHARDS=51
Z_LO_BASE=0.0
Z_HI_BASE=5.0
Z_STEP=0.1

IDX=$SLURM_ARRAY_TASK_ID
Z_SHARD_LO=$(python3 -c "print(round($Z_LO_BASE + $IDX * ($Z_HI_BASE - $Z_LO_BASE) / $N_SHARDS, 6))")
if [ "$IDX" -eq "$((N_SHARDS - 1))" ]; then
    Z_SHARD_HI="$Z_HI_BASE"
else
    Z_SHARD_HI=$(python3 -c "print(round($Z_LO_BASE + ($IDX + 1) * ($Z_HI_BASE - $Z_LO_BASE) / $N_SHARDS - $Z_STEP, 6))")
fi
SEED=$((42 + IDX * 10000))
OUT="$BUNDLE/output/zE_sweep/shard_${IDX}_z${Z_SHARD_LO}_z${Z_SHARD_HI}.jld2"
mkdir -p "$(dirname "$OUT")" logs

echo "[sbatch] Shard $IDX: z ∈ [$Z_SHARD_LO, $Z_SHARD_HI], seed $SEED → $OUT"

SSD_CACHE="$SSD_CACHE" \
julia --project="$BUNDLE" -t "$SLURM_CPUS_PER_TASK" \
    "$BUNDLE/scripts/zE_sweep.jl" \
    --z-lo "$Z_SHARD_LO" --z-hi "$Z_SHARD_HI" --z-step "$Z_STEP" \
    --e-lo 10.0 --e-hi 2000.0 --e-step 10.0 \
    --n-reps 10 \
    --seed "$SEED" \
    --out "$OUT"

echo "[sbatch] Shard $IDX done"
EOF

# ── Aggregation SLURM script (runs after all shards complete) ──
cat > "$STAGE/scripts/slurm_aggregate.sbatch" << 'EOF'
#!/usr/bin/env bash
# Runs feature extraction + preview plots after the sweep completes.
# Submit with: sbatch --dependency=afterok:<zE_sweep_jobid> slurm_aggregate.sbatch

#SBATCH --job-name=zE_aggregate
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G                       # holds 16 GB of waveforms during extraction [ADAPT if bigger]
#SBATCH --time=01:00:00
#SBATCH --output=logs/aggregate.out
#SBATCH --error=logs/aggregate.err

set -euo pipefail
module load python/3.11
source $HOME/zE_sweep_venv/bin/activate

BUNDLE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$BUNDLE/output/zE_sweep_features"

python3 "$BUNDLE/python/extract_zE_features.py" \
    --glob "$BUNDLE/output/zE_sweep/*.jld2" \
    --out  "$BUNDLE/output/zE_sweep_features"

python3 "$BUNDLE/python/plot_zE_sweep.py" \
    --features "$BUNDLE/output/zE_sweep_features" \
    --out      "$BUNDLE/output/zE_sweep_features/summary_figures.png"

echo "[aggregate] Done."
EOF

# ── AGENT HANDOFF DOC (the primary deliverable) ──
cat > "$STAGE/AGENT_HANDOFF.md" << 'EOF'
# Agent handoff: (z × E × rep) SSD sweep on Hummingbird

You are the receiving agent. This bundle is self-contained; your job
is to (1) discover Hummingbird's specifics, (2) adapt the SLURM
template, (3) install deps, (4) run the sweep, (5) aggregate outputs.

## What the sweep is

Ideal SolidStateDetectors.jl simulation of a cross-strip CZT (39
anodes, 1 mm pitch, 100 µm strips, 5 mm thick) at gamma energies
10 keV → 2 MeV in 10 keV steps, depths 0 → 5 mm from anode in 0.1 mm
steps, 10 stochastic repetitions per (z, E) point.

**Total: 102,000 events. Records preamp waveforms on 4 channels
(A19, A20, A21, steering). Fixed x = 0 (center of primary anode).**

Purpose: characterize primary-anode response (peak plateau + timing
at fixed 3 mV threshold) across the full physical parameter space —
input to a downstream depth-of-interaction and energy-resolution
analysis.

## Cluster discovery — DO THIS FIRST

Answer these before submitting anything:

1. **Julia version available?**
   ```
   module avail julia
   julia --version                # need >= 1.10
   ```
   If no Julia module, install via juliaup or download binary.

2. **Python version?**
   ```
   module avail python
   python3 --version              # need >= 3.9
   ```

3. **Partition + resource limits?**
   ```
   sinfo                          # available partitions
   sacctmgr show assoc user=$USER # your quotas
   scontrol show partition <name> # per-partition limits
   ```

4. **Storage layout?**
   ```
   df -h                          # find /scratch, /project, quotas
   quota -s                       # user quotas
   ```
   The sweep produces ~16 GB of JLD2 output. Put it on parallel/scratch
   filesystem, not home.

5. **Concurrency limit?**
   Look at your fair-share allocation. The SLURM template caps at
   `--array=0-50%10` (10 concurrent shards). Increase or decrease
   based on quota. Higher concurrency = shorter total wall time.

Fill in the [ADAPT] fields in `scripts/slurm_template.sbatch`
using the answers above.

## Setup steps (in order)

### 1. Install Julia deps
```
cd zE_sweep_handoff
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```
- Instantiates from bundled `Project.toml` + `Manifest.toml`
- Downloads platform-specific artifacts from `pkg.julialang.org`
  (needs outbound HTTPS)
- Expect ~2 min on first run

### 2. Install Python deps (in a venv)
```
python3 -m venv $HOME/zE_sweep_venv
source $HOME/zE_sweep_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Build the WP cache (one-time, ~4 min)
```
julia --project=. -t 8 scripts/build_wp_cache.jl
```
Produces `output/sweep/sim_cache.jls` (~560 MB). The cache is Julia +
SSD version-locked — if either changes, rebuild.

Cache goes on shared filesystem so all SLURM shards can read it.

### 4. Verify per-event timing (~5 min, 50 events)
```
julia --project=. -t 8 scripts/benchmark_zE_sweep.jl
```
Compare `output/benchmark_zE_sweep.json` to `benchmark_reference.json`
(shipped in this bundle). Reference numbers (from local Mac M-series,
8 threads):
- Per-event mean: **4.36 s**, std 1.82 s
- Range: 1.17 s (near-cathode / high E) → 8.30 s (near-anode)
- Per-z: 5.7 s @ z=0.1 mm → 1.25 s @ z=4.9 mm (holes dominate near-anode)

Hummingbird's Linux CPUs should be within ±30% of these numbers. If
much slower, check the sbatch memory allocation isn't paging the cache.

### 5. Adapt the SLURM template
Edit `scripts/slurm_template.sbatch`. All [ADAPT] markers must be
filled in. In particular:
- `--partition=<>` — pick the correct compute partition
- `--time=03:30:00` — set to 2.5 × slowest expected shard runtime
- `--array=0-50%<N>` — N depends on your concurrency quota
- `--cpus-per-task=<>` — match node core count if you can (defaults 4)
- `module load julia/<version>` — match what discovery step found
- `source .../activate` — venv path from step 2

### 6. Submit
```
sbatch scripts/slurm_template.sbatch
```
Then submit the aggregation script with dependency:
```
sbatch --dependency=afterok:<jobid> scripts/slurm_aggregate.sbatch
```

Monitor with `squeue -u $USER`, tail `logs/shard_*.out` for progress.

## Expected runtime (full sweep)

At benchmark timing on Hummingbird:
- Sequential (single-thread, no shards): ~124 hours
- 51 shards, 4 threads each: **~2.5 hours wall time**
- 10 concurrent shards, 4 threads: ~12.5 hours (partition-limited)
- 20+ concurrent shards: ~6 hours or better

Aggregation step: ~10 min (extract + plot).

## Storage projections

| Item | Size |
|---|---|
| `output/sweep/sim_cache.jls` | 560 MB (one-time) |
| `output/zE_sweep/*.jld2` (51 shards) | ~16 GB total |
| `output/zE_sweep_features/features.parquet` | ~50 MB |
| `output/zE_sweep_features/summary.parquet` | ~1 MB |
| `output/zE_sweep_features/summary_figures.png` | ~2 MB |

Peak simultaneous I/O: 51 shards × ~10 MB writes over 2.5 h → ~30 MB/s
aggregate. Fine on any modern parallel FS.

## Validation checklist (after sweep completes)

- [ ] All 51 shards present, no missing files
- [ ] Each shard is ~320 MB
- [ ] `features.parquet` has exactly 102,000 rows
- [ ] A20 timing yield > 99% for E ≥ 300 keV (`summary.parquet`)
- [ ] `summary_figures.png` shows: monotonic A20 drift-time-vs-depth
      at 662 keV, plateau ~305 mV at 662 keV/mid-depth
- [ ] Per-rep spread (std) is nonzero (proves jitter worked)

## What to send back

- `output/zE_sweep_features/features.parquet` (~50 MB)
- `output/zE_sweep_features/summary.parquet` (~1 MB)
- `output/zE_sweep_features/summary.json` (headline stats)
- `output/zE_sweep_features/summary_figures.png`
- Per-shard timing summary from `logs/shard_*.out`

The 16 GB of raw JLD2 waveforms should stay on Hummingbird (too big
to move); mount or scp only if further waveform-level analysis is
needed downstream.

## Physics parameters (documented for the record)

| Parameter | Value | Rationale |
|---|---|---|
| Geometry | `czt_cross_strip_full.yaml` | 39 anodes, 1 mm pitch, 100 µm strip width, 5 mm thick |
| N_carriers | 50 | Standard SSD event resolution |
| n_shells | 2 | Cloud shell tessellation |
| Cloud radius | 100 µm | Matches physical 662 keV cloud σ ≈ 50–100 µm |
| Δt (sim) | 0.1 ns | SSD internal timestep |
| max_nsteps | 50_000 | 5 µs — covers full carrier drift at CZT µ·E |
| Position jitter | ±25 µm (uniform) | Per-rep stochasticity proxy (SSD's native diffusion is dropped by cached sim objects) |
| Preamp | b0=1400, a1=0.9999992857 | IIR, τ ≈ 140 µs |
| C_f | 75 fF | 662 keV full collection ≈ 305 mV |
| Threshold | 3 mV | 1% of 662 keV reference plateau |

## Question for the user BEFORE launching the full sweep

Two SSD physics features are OFF by default and were left off for the
initial local dry-run. **Ask the user whether to enable either for the
full Hummingbird sweep** — they change fidelity and cost.

### Thermal diffusion (SSD `diffusion=true`)

- What it does: adds thermal random walk to each carrier during drift.
  Replaces the current position-jitter workaround with physical
  stochasticity.
- Cost: none per event (only marginal RNG overhead).
- Requires: `SolidStateDetectors.material_properties[:CdZnTe]` must
  include `De ≈ 25u"cm^2/s"` and `Dh ≈ 3u"cm^2/s"` (Einstein relation:
  D = µ·kT/e at 300 K). CZT has no De/Dh set upstream, so this needs
  a one-time override.
- **CRITICAL**: the WP cache must be rebuilt AFTER injecting De/Dh.
  Cached `Simulation` objects freeze `material_properties` at cache
  time; a runtime override does NOT reach a deserialized sim. This is
  a ~4 min one-time cost. Also update `zE_sweep.jl` and
  `benchmark_zE_sweep.jl` to pass `diffusion=true` to `simulate!`.
- Marked as "experimental" in the SSD 0.11 docs
  (`docs/src/man/charge_drift.md`).
- Recommend: **yes**, enable for publication-grade sweep. The position-
  jitter workaround is a proxy; real diffusion is defensible.

### Self-repulsion (SSD `self_repulsion=true`)

- What it does: same-species Coulomb repulsion between carriers within
  one cloud (electrons repel electrons, holes repel holes). NOT
  electron-hole attraction (SSD does not model that).
- Cost: **2–5× slower per event.** Full sweep goes from ~2.5 h to
  ~6–12 h on 51 shards.
- Requires: NBodyChargeCloud (already used by our sim). No cache
  rebuild.
- Physical relevance: significant for high-density interactions
  (high-Z, low-energy X-rays); modest (~few %) for 662 keV in CZT with
  100 µm cloud radius.
- Also marked "experimental" in SSD 0.11 docs.
- Recommend: **probably not** for the full sweep. Enable selectively
  at a handful of (z, E) points as a sensitivity check if needed.

If the user says "enable diffusion":
1. Rebuild the cache with material overrides — add the following
   BEFORE `Simulation{Float32}(GEOMETRY)` in `scripts/build_wp_cache.jl`:
   ```julia
   let base = SolidStateDetectors.material_properties[:CdZnTe]
       SolidStateDetectors.material_properties[:CdZnTe] = merge(base, (
           De = 25.0u"cm^2/s", Dh = 3.0u"cm^2/s",
       ))
   end
   ```
2. Rerun `julia --project=. -t 8 scripts/build_wp_cache.jl`.
3. In `zE_sweep.jl` and `benchmark_zE_sweep.jl`: add `diffusion=true`
   to every `simulate!` call. Remove the position-jitter block (search
   for `JITTER_MM` and delete surrounding code).
4. Re-run the benchmark to confirm timing is unchanged.
5. Then submit the full sweep.

If the user says "enable self-repulsion": add `self_repulsion=true`
to the `simulate!` calls in `zE_sweep.jl` and `benchmark_zE_sweep.jl`.
No cache rebuild. Expect ~3× wall time — adjust SLURM `--time=`
accordingly.

## Where to look if things break

- Julia script errors → `logs/shard_N.err`
- Missing cache → step 3 wasn't run, OR cache is on non-shared FS
- Stale Manifest error → `julia --project=. -e 'using Pkg; Pkg.update()'`
- OOM on shards → increase `--mem=` (each shard uses ~700 MB peak)
- Slower than reference by > 2× → check node isn't oversubscribed;
  check `--cpus-per-task` isn't higher than physically allocated
- `Pkg.instantiate` hangs → cluster firewall blocking pkg.julialang.org
- Python `ImportError` → wrong venv activated
EOF

# ── Build tarball ──
BASENAME="$(basename "$STAGE")"
STAGE_PARENT="$(dirname "$STAGE")"
echo "[pack] Building tarball → $OUT"
tar -czf "$OUT" -C "$STAGE_PARENT" "$BASENAME"
SZ_KB=$(du -k "$OUT" | cut -f1)
echo "[pack] Wrote $OUT ($SZ_KB KB)"
rm -rf "$STAGE_PARENT"

echo
echo "[pack] Bundle contents:"
tar -tzf "$OUT"

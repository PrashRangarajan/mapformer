#!/usr/bin/env bash
# The MISSING 8th cell of the ALLOCENTRIC MiniGrid factorial: PoPE-Hier
# (PoPE + index + hierarchy), n=3.
#
# WHY IT MATTERS. MINIGRID_ALLOCENTRIC_2X2X2.md has 7 of 8 cells: both launchers
# (run_minigrid_allo.sh, run_minigrid_2x2x2.sh) carry a 7-variant VARS list, and
# the RAW factorial's 8th cell was filled by a separate top-up (run_pope_hier.sh)
# that was never run in the allocentric condition. The gap is not random:
# PoPE-Hier is the BEST arm in the raw factorial (0.964/0.955), so omitting it
# removes the strongest INDEX arm and biases the path-int - index effect upward.
# With an estimated PoPE-Hier the T=1024 effect drops +0.026 -> ~+0.021 (flip
# survives) and the ordering claim "all four path-int arms outrank all index arms"
# (margin only +0.003) does NOT survive.
#
# BATCH DISCIPLINE (rule 3: do not compare a fresh arm to stored ones). Same
# argument as run_pope_hier.sh, verified for this batch: MiniGrid trains from a
# FIXED on-disk 25K allocentric buffer (byte-identical data, not merely
# distribution-matched), and minigrid_env.py / train_variant.py / model*.py are
# UNCHANGED since the allocentric run (commit 7d295a2, 2026-08-24; no uncommitted
# edits). PoPE-Flat is retrained alongside as a REPRODUCIBILITY CONTROL: if it
# reproduces its stored number (T=512 0.8279, T=1024 0.8069) the new cell can be
# read against the existing 7-arm grid; if it drifts more than the +0.003 margin,
# the full 8-cell rerun is the only honest fix and this run says so.
#
# Single GPU (cuda:1) -- the hierarchy grid sweep owns cuda:0.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/minigrid_2x2x2_allo"; mkdir -p "$R"
LOG="$REPO/minigrid_allo_8thcell.log"; echo "8th-cell start $(date)" > "$LOG"

# 1. Buffers already exist from the original allocentric run; touch them to be safe
#    (sequential, loads from cache).
echo "$(date +%H:%M) ensuring allocentric buffers (cached)" >> "$LOG"
for SEED in 0 1 2; do
  python3 -u -c "from mapformer.minigrid_env import MiniGridWorld_Cached as C; \
e=C(env_name='MiniGrid-DoorKey-16x16-v0',tokenization='obj_color',seed=$SEED,\
buffer_size=25000,allocentric=True); e.generate_batch(2,128)" >> "$LOG" 2>&1
done
echo "$(date +%H:%M) buffers ready; training 6 runs on cuda:1" >> "$LOG"

# 2. PoPE-Hier (the missing cell) + PoPE-Flat-repro (the control), 3 seeds, 2/GPU.
for SEED in 0 1 2; do
  for V in PoPE-Hier PoPE-Flat-repro; do
    VAR="${V%-repro}"
    D="$R/${V}_s${SEED}"
    [ -f "$D/${VAR}.pt" ] && { echo "skip $V s$SEED" >> "$LOG"; continue; }
    echo "$(date +%H:%M) $V s$SEED -> cuda:1" >> "$LOG"
    python3 -u -m mapformer.train_variant --variant "$VAR" --seed "$SEED" \
      --n-landmarks 0 --epochs 50 --n-batches 156 --n-layers 3 \
      --env minigrid_doorkey16 --minigrid-tokenization obj_color \
      --minigrid-allocentric --minigrid-cached-buffer 25000 \
      --device cuda:1 --output-dir "$D" > "$R/train_${V}_s${SEED}.log" 2>&1 &
  done
  wait
  echo "$(date +%H:%M) seed $SEED done" >> "$LOG"
done

# 3. Report: control reproduction first (licenses or voids the comparison), then
#    the corrected position effect with the 8th cell included.
python3 -u - "$R" >> "$REPO/MINIGRID_ALLO_8THCELL.md" 2>>"$LOG" <<'PYEOF'
import glob, json, os, sys
import numpy as np
R = sys.argv[1]
STORED = {"512": 0.8279, "1024": 0.8069}          # PoPE-Flat, from the 7-arm table
def arm(name, L):
    vals = []
    for d in sorted(glob.glob(os.path.join(R, f"{name}_s*"))):
        for f in glob.glob(os.path.join(d, "*.json")):
            try:
                r = json.load(open(f))
                v = r.get(str(L))
                if isinstance(v, dict):
                    v = v.get("acc", v.get("nb_acc"))
                if v is not None: vals.append(float(v))
            except Exception: pass
    return vals
out = ["# MiniGrid allocentric — the missing 8th cell (PoPE-Hier) + repro control", ""]
ok = True
for L in (512, 1024):
    rep = arm("PoPE-Flat-repro", L)
    drift = (np.mean(rep) - STORED[str(L)]) if rep else float("nan")
    verdict = "REPRODUCES" if abs(drift) <= 0.003 else "DRIFTS > margin"
    if not (abs(drift) <= 0.003): ok = False
    out.append(f"- **T={L} control:** PoPE-Flat-repro {np.mean(rep):.4f} (n={len(rep)}) "
               f"vs stored {STORED[str(L)]:.4f} -> drift {drift:+.4f} — **{verdict}**"
               if rep else f"- T={L} control: MISSING")
out.append("")
# corrected effect with the 8th cell
alloc = json.load(open(os.path.join(os.path.dirname(R), "..", "MINIGRID_ALLOCENTRIC_2X2X2.json"))) \
    if os.path.exists(os.path.join(os.path.dirname(R), "..", "MINIGRID_ALLOCENTRIC_2X2X2.json")) else None
out += ["| T | PoPE-Hier (new) | index mean (n=4) | path-int mean | corrected effect |", "|---|---|---|---|---|"]
for L in (512, 1024):
    ph = arm("PoPE-Hier", L)
    if not ph or alloc is None:
        out.append(f"| {L} | — | — | — | (incomplete) |"); continue
    A = alloc["acc"]
    PI = ["Vanilla","Hourglass_k2","MapPoPE-Flat","MapPoPE-Hier"]
    IX = ["RoPE","PlainHourglass","PoPE-Flat"]
    p = np.mean([A[n][str(L)]["mean"] for n in PI])
    i = np.mean([A[n][str(L)]["mean"] for n in IX] + [np.mean(ph)])
    out.append(f"| {L} | {np.mean(ph):.4f} (n={len(ph)}) | {i:.4f} | {p:.4f} | **{p-i:+.4f}** |")
out += ["", ("> Control reproduces within the +0.003 margin -> the 8th cell is readable "
             "against the stored 7-arm grid." if ok else
             "> **Control DRIFTS beyond the margin -> this cross-batch comparison is NOT "
             "valid; the full 8-cell rerun in one batch is required.**")]
print("\n".join(out))
PYEOF
touch "$REPO/.mg_8thcell_done"
echo "$(date) DONE" >> "$LOG"

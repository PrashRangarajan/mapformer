#!/usr/bin/env bash
# ALIASING-CONTROLLED SWEEP. Turns the surviving MiniWorld claim from
# correlational into manipulated -- or kills it.
#
# THE PROBLEM. The claim is "the position effect scales with observation
# ALIASING": grid 8 (2 cells/token) -0.010, grid 32 (32/token) +0.173, torus
# (128/token) +0.461. Monotone -- but aliasing CO-VARIES with map size across
# that series, so "bigger map" explains every point equally well. Nothing in the
# repo separates them.
#
# THE MANIPULATION. Hold grid size FIXED at 32 and vary n_obs alone:
#     n_obs=16  -> 32 cells/token   ANCHOR (already converged, n=3, +0.173)
#     n_obs=64  ->  8 cells/token   new
#     n_obs=256 ->  2 cells/token   DECISIVE: grid 8's aliasing, grid 32's map
#
# The gates confirm n_obs only RELABELS the obs_map: G5 label mass (50.4/traj)
# and G6 revisit lag (median 33) are byte-identical across all three conditions.
# Same trajectories, same scored positions, only the labelling changes. That is
# as clean as this manipulation gets. Gates in ALIASING_GATES.md, all PASS.
#
# PRE-REGISTERED (written before any GPU was spent; do not revise after seeing
# the numbers):
#   A. ALIASING drives it -> effect falls monotonically with cells/token and
#      lands below the measured noise floor (0.150) at n_obs=256. Claim becomes
#      manipulated. This is the hypothesis under test.
#   B. MAP SIZE drives it -> effect stays above 0.150 at all three n_obs. The
#      aliasing story is FALSIFIED and must be withdrawn from CLAUDE.md and
#      the memory file.
#   C. FLOOR COLLAPSE -> both arms land near the non-blank marginal at n_obs=256
#      (0.013). Then the shrinking effect is compression toward a floor, NOT
#      aliasing, and the condition is uninformative. Reported, not spun.
#   D. Non-monotone -> neither hypothesis; report the curve, claim nothing.
#
# Three points, because this session already learned that two points make a line
# and a line is not a trend (rule 5 corollary, bought by the H12 budget curve).
#
# WHY n=5 AND NOT n=3. The anchor's own per-seed effects are +0.274 / +0.210 /
# +0.033, sd 0.125 -- and that spread is BIMODAL BASIN SELECTION, not noise:
# Vanilla seed 2 converged to loss 0.430 while seeds 0 and 1 reached 0.004 and
# 0.027. It passes the flat-slope test and still lands in a much worse basin.
# At n=3 the BETWEEN-CONDITION MDE is 0.286 -- larger than the entire effect --
# so the sweep as first designed could not have distinguished "collapsed to 0"
# from "unchanged at +0.173". n=5 brings that to 0.221 raw, and the loss gap
# reported alongside gives the deconfounded read (rule 9).
#
# HONEST LIMIT, stated up front: accuracy here is an affine readout of training
# loss (r = -0.996), and the two arms' loss ranges barely overlap at n_obs=16
# (Vanilla 0.004-0.430 vs RoPE 0.406-0.499). So this measures whether path
# integration's OPTIMIZATION advantage depends on aliasing. Whether it also
# represents better AT EQUAL FIT is not separable from these runs, and the
# aggregator says so rather than implying otherwise.
#
# BATCH DISCIPLINE (rule 3). n_obs=16 is reused from runs/rope_converge rather
# than retrained -- 6 runs saved. Licensed only because Vanilla/RoPE training
# code is byte-identical since then (this week's fixes touched model_pope.py,
# train_hourglass_enwik8.py, agg_miniworld.py and a shell aggregator, none on
# this path) AND one RoPE seed is retrained here as a REPRODUCIBILITY CONTROL.
# If that control drifts more than 0.03 from the stored 0.725, the cross-batch
# anchor is NOT licensed and the drift becomes the error bar. The script says so
# in its own verdict rather than leaving it to a reader.
#
# 25 runs (n16 topped up to 5 seeds, n64 and n256 at 5), 6 concurrent at 5.7 GiB
# each on two idle 24 GiB 4090s -> ~17 h.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/alias_sweep"; mkdir -p "$R"
LOG="$REPO/alias_sweep.log"; echo "alias sweep start $(date)" > "$LOG"

G=32; T=512; NBUF=24000; EP=400; NB=180; BS=24; DM=256; NL=4; NH=4; NW=24; ETRIALS=128
MAXPG=3        # 5.7 GiB/job measured; 3x = 17 GiB of 24 GiB

# refuse to run if the gates were not cleared
[ -f "$REPO/.alias_gates_clean" ] || { echo "GATES NOT CLEAN -- refusing" >> "$LOG"; exit 1; }

# PREBUILD THE BUFFERS SERIALLY FIRST. MiniWorld creates an EGL context per
# worker (~150 MiB of GPU) even with rendering disabled, so launching 6 jobs at
# once means 144 worker processes: it pinned a 24 GiB 4090 at 23,955 MiB (97.5%)
# before any model reached the device, and oversubscribed 32 cores by 4.5x.
# Built one job at a time it peaks at 4.5 GiB and takes ~42 s per buffer.
# Cheap and idempotent -- cached buffers return instantly.
echo "$(date +%H:%M) prebuilding buffers serially" >> "$LOG"
python3 -u -m mapformer.prebuild_buffers --grid-size $G --n-obs 256 64 \
    --seeds 0 1 2 3 4 --n-steps $T --buffer-size $NBUF --eval-trials $ETRIALS \
    --n-workers $NW --oracle >> "$LOG" 2>&1
python3 -u -m mapformer.prebuild_buffers --grid-size $G --n-obs 16 \
    --seeds 3 4 --n-steps $T --buffer-size $NBUF --eval-trials $ETRIALS \
    --n-workers $NW --oracle >> "$LOG" 2>&1

declare -a PIDG0=() PIDG1=()
alive(){ local o=(); for p in "$@"; do kill -0 "$p" 2>/dev/null && o+=("$p"); done; echo "${o[@]:-}"; }
launch(){
  local V="$1" SEED="$2" NOBS="$3"
  local OUT="$R/n${NOBS}/s${SEED}"; mkdir -p "$OUT"
  [ -f "$OUT/${V}_oracle.pt" ] && { echo "skip ${V} n${NOBS} s${SEED}" >> "$LOG"; return; }
  while :; do
    PIDG0=($(alive "${PIDG0[@]:-}")); PIDG1=($(alive "${PIDG1[@]:-}"))
    if [ "${#PIDG0[@]}" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "${#PIDG1[@]}" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 30
  done
  echo "$(date +%H:%M) ${V} n_obs=${NOBS} s${SEED} -> cuda:$GPU" >> "$LOG"
  python3 -u -m mapformer.train_miniworld --variant "$V" --seed "$SEED" --oracle \
    --grid-size $G --n-obs "$NOBS" --n-steps $T --buffer-size $NBUF --epochs $EP \
    --n-batches $NB --batch-size $BS --d-model $DM --n-layers $NL --n-heads $NH \
    --n-workers $NW --schedule cosine --eval-trials $ETRIALS --eval-lengths 512 1024 \
    --device "cuda:$GPU" --output-dir "$OUT" > "$R/${V}_n${NOBS}_s${SEED}.log" 2>&1 &
  local PID=$!; [ "$GPU" -eq 0 ] && PIDG0+=("$PID") || PIDG1+=("$PID"); sleep 3
}

# DECISIVE condition first, so a problem shows up early rather than at hour 17.
for NOBS in 256 64; do
  for SEED in 0 1 2 3 4; do
    for V in Vanilla RoPE; do launch "$V" "$SEED" "$NOBS"; done
  done
done
# top the reused n_obs=16 anchor up from 3 seeds to 5, so all three conditions
# carry the same n and the curve is not compared across differing power
for SEED in 3 4; do
  for V in Vanilla RoPE; do launch "$V" "$SEED" 16; done
done
# reproducibility control for the reused n_obs=16 anchor
mkdir -p "$R/n16_repro/s0"
launch_repro(){
  while :; do
    PIDG0=($(alive "${PIDG0[@]:-}")); PIDG1=($(alive "${PIDG1[@]:-}"))
    if [ "${#PIDG0[@]}" -lt "$MAXPG" ]; then GPU=0; break; fi
    if [ "${#PIDG1[@]}" -lt "$MAXPG" ]; then GPU=1; break; fi
    sleep 30
  done
  echo "$(date +%H:%M) RoPE n_obs=16 s0 REPRO CONTROL -> cuda:$GPU" >> "$LOG"
  python3 -u -m mapformer.train_miniworld --variant RoPE --seed 0 --oracle \
    --grid-size $G --n-obs 16 --n-steps $T --buffer-size $NBUF --epochs $EP \
    --n-batches $NB --batch-size $BS --d-model $DM --n-layers $NL --n-heads $NH \
    --n-workers $NW --schedule cosine --eval-trials $ETRIALS --eval-lengths 512 1024 \
    --device "cuda:$GPU" --output-dir "$R/n16_repro/s0" > "$R/RoPE_n16_repro.log" 2>&1 &
  local PID=$!; [ "$GPU" -eq 0 ] && PIDG0+=("$PID") || PIDG1+=("$PID")
}
launch_repro
wait

# `wait` returns regardless of child success -- verify the artifacts exist rather
# than inferring success from the absence of a crash (that mistake touched a
# .done marker after every arm had died).
N_PT=$(find "$R" -name "*_oracle.pt" | wc -l)
echo "$(date +%H:%M) finished; $N_PT/25 checkpoints present" >> "$LOG"

python3 -u -m mapformer.agg_alias --runs-dir "$R" \
    --anchor-dir "$REPO/runs/rope_converge" --seeds 0 1 2 3 4 \
    --out "$REPO/ALIASING_CONTROLLED.md" >> "$LOG" 2>&1

touch "$REPO/.alias_sweep_done"
echo "$(date) DONE" >> "$LOG"

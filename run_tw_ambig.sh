#!/usr/bin/env bash
# Direction words as actions and as observed content. Pre-registration: TW_AMBIG_PREREG.md.
#   batch:  nohup setsid bash run_tw_ambig.sh > /dev/null 2>&1 &
#   pilot:  TAG=tw_ambig_pilot SEEDS="140" nohup setsid bash run_tw_ambig.sh > /dev/null 2>&1 &   (no verdicts below n=8)
# Knobs (env): TAG (run dir name), SEEDS, EPOCHS (1800: Amendment 1 -- the default lives here and is md5-covered),
# MAXPG (3 jobs per GPU), RUN_TIMEOUT (per run, 8h), ANA_TIMEOUT (readouts + analysis, 6h). Every run trains all 8 arms
# (the analysis reads all of them and asserts one epoch count).
set -uo pipefail
REPO=/home/prashr/mapformer
TAG="${TAG:-tw_ambig}"
SEEDS="${SEEDS:-40 41 42 43 44 45 46 47}"
EPOCHS="${EPOCHS:-1800}"
RUN_TIMEOUT="${RUN_TIMEOUT:-8h}"; ANA_TIMEOUT="${ANA_TIMEOUT:-6h}"
ARMS_LIST="HSR CF2 RoleTag2 RoPE2 MapWM RoleTag DirOnlyRole RoPE1"   # seed outer, arm inner; 2-layer arms first
LOG="$REPO/$TAG.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-3}"; DRV_SPACING="${SPACING:-20}"; DRV_MINFREE="${MINFREE:-4500}"   # after sourcing (it sets defaults)
drv_lock "$REPO/.run_$TAG.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/$TAG"; mkdir -p "$R/p0"
MODULES="__init__.py analyze_tw_ambig.py data_parallel.py environment.py environment_textworld.py
  environment_textworld_ctx3.py environment_tw_ambig.py evaluate.py lie_groups.py model.py model_baseline_rope.py
  model_codes.py model_context_step.py model_inekf_level15.py model_looped.py model_rank.py model_tw_ambig.py
  prefix_scan.py stats_core.py train.py train_tw_ambig.py tw_ambig_readouts.py lib_driver.sh run_tw_ambig.sh"
_drv_log "start $TAG seeds [$SEEDS] epochs $EPOCHS arms [$ARMS_LIST] maxpg $DRV_MAXPG"
{ echo "== $(date '+%F %T') $TAG"; git -C "$REPO" rev-parse HEAD; git -C "$REPO" status --porcelain -- $MODULES; } >> "$R/git_head.txt"
# shellcheck disable=SC2086
drv_md5_guard "$R" $MODULES || exit 1

# exact-token duplicate guard: is any python3 running with this output dir as an argv TOKEN (not a substring:
# MapWM_s4 must not match MapWM_s40)?
running() {
  ps -u "$USER" -o comm=,args= | awk -v o="$1" '$1=="python3" { for (i = 2; i <= NF; i++) if ($i == o) f = 1 } END { exit !f }'
}

N_EXP=0
for S in $SEEDS; do for A in $ARMS_LIST; do
  N_EXP=$((N_EXP + 1))
  OUT="$R/p0/${A}_s${S}"
  [ -f "$OUT/eval.json" ] && { _drv_log "skip $OUT (eval.json present)"; continue; }
  if running "$OUT"; then _drv_log "skip $OUT (already running)"; continue; fi
  # shellcheck disable=SC2086
  drv_md5_guard "$R" $MODULES || drv_fail "code changed during the batch"
  G=$(drv_wait_slot)
  if running "$OUT"; then _drv_log "skip $OUT (started while waiting)"; continue; fi
  mkdir -p "$OUT"
  _drv_log "$A s$S -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" timeout --kill-after=120 "$RUN_TIMEOUT" python3 -u -m mapformer.train_tw_ambig --arm "$A" --seed "$S" \
    --epochs "$EPOCHS" --device "cuda:$G" --output-dir "$OUT"
done; done
drv_wait_dir "$R/p0/"
N=$(ls "$R"/p0/*/eval.json 2>/dev/null | wc -l); [ "$N" -eq "$N_EXP" ] || drv_fail "only $N/$N_EXP eval.json"
# shellcheck disable=SC2086
drv_md5_guard "$R" $MODULES || drv_fail "code changed before the readouts / analysis"
{ echo "== $(date '+%F %T') analysis"; git -C "$REPO" rev-parse HEAD; } >> "$R/git_head.txt"
SEEDS_CSV=$(echo $SEEDS | tr ' ' ',')
if [ "$TAG" = tw_ambig ]; then
  OUTJ="$REPO/TW_AMBIG.json"; OUTA="$REPO/TW_AMBIG_ANALYSIS.txt"
else
  OUTJ="$R/readouts.json"; OUTA="$R/analysis.txt"
fi
timeout --kill-after=120 "$ANA_TIMEOUT" python3 -u -m mapformer.analyze_tw_ambig --readouts --runs-dir "$R/p0" --seeds "$SEEDS_CSV" --out "$OUTJ" > "$OUTA" 2>&1 \
  || drv_fail "analyze (non-zero: VOID, a failed assertion or the timeout; see $OUTA)"
drv_done "$REPO/.${TAG}_done" "$OUTJ" "$OUTA"

#!/usr/bin/env bash
# State-change clauses in the text world. Pre-registration: TW_STATECHANGE_PREREG.md. Recipe = the text world's
# (T = 1024 words, batch 16, 900 epochs x 98 batches, lr 1e-3, warmup + cosine, 1 layer, 2 heads, d 128, r = 4 shared,
# --data-workers 3); p_take = p_drop = 0.4. Arms MapWM NormStep DirOnly RoPE x seeds 50-57 (n = 8) = 32 runs, one batch;
# seed outer, arm inner. Readouts and analysis on CPU after every checkpoint exists. Launch:
#   cd /home/prashr/mapformer && setsid nohup bash run_tw_statechange.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/tw_statechange.log"
source "$REPO/lib_driver.sh"
# lib_driver.sh (sourced above) sets DRV_* defaults, so these are assigned unconditionally AFTER it (GAIN_PHASE Amendment 2)
DRV_MAXPG="${TWSC_MAXPG:-4}"            # jobs per GPU; the picker counts every mapformer.train_ job (shared budget)
DRV_SPACING="${TWSC_SPACING:-15}"
DRV_MINFREE="${TWSC_MINFREE:-5500}"     # MiB free before each launch (T = 1024 jobs measured ~4.7 GB in GAIN_PHASE's pilot)
drv_lock "$REPO/.run_tw_statechange.lock" || exit 1
# Amendment 1: bounded waits. lib_driver's drv_wait_slot / drv_wait_dir loop forever, and drv_wait_dir counts ANY python3
# with the run dir in argv (an orphaned or unrelated process would block it); these wait on this batch's TRAINERS only.
SLOT_TIMEOUT="${TWSC_SLOT_TIMEOUT:-172800}"   # 48 h for a free slot, then fail
DIR_TIMEOUT="${TWSC_DIR_TIMEOUT:-86400}"      # 24 h after the last launch for every trainer to exit, then fail
twsc_wait_slot() {
  local g="" t0; t0=$(date +%s)
  while [ -z "$g" ]; do
    g=$(drv_pick)
    if [ -z "$g" ]; then
      [ $(( $(date +%s) - t0 )) -gt "$SLOT_TIMEOUT" ] && return 1
      sleep "$DRV_POLL"
    fi
  done
  echo "$g"
}
twsc_ntrainers() {   # python3 trainers of THIS batch (module name AND run dir in argv)
  ps -u "$USER" -o comm=,args= | awk -v r="$1" '$1=="python3" && /mapformer[.]train_tw_statechange/ && index($0, r)' | wc -l
}
twsc_wait_dir() {
  local t0; t0=$(date +%s)
  while [ "$(twsc_ntrainers "$1")" -gt 0 ]; do
    [ $(( $(date +%s) - t0 )) -gt "$DIR_TIMEOUT" ] && return 1
    sleep "${DRV_WAIT_POLL:-60}"
  done
}
cd "$REPO/.."
R="$REPO/runs/tw_statechange"; mkdir -p "$R/p0"
ARMS="MapWM NormStep DirOnly RoPE"
SEEDS="50 51 52 53 54 55 56 57"
# every mapformer module the trainer, the readouts and the analysis import (sys.modules after importing them)
GUARD=(__init__.py analyze_tw_statechange.py data_parallel.py environment.py environment_addition.py
       environment_textworld.py environment_tw_statechange.py evaluate.py hourglass_plain.py lie_groups.py model.py
       model_ablations.py model_baseline_nope.py model_baseline_rope.py model_baselines_extra.py model_bounded_mem.py
       model_cho_coupled.py model_cho_positions.py model_code_decay.py model_codes.py model_counter_installed.py
       model_coupled_ape.py model_coupled_rope.py model_em_dof.py model_em_fixed.py model_em_magonly.py
       model_em_noleak.py model_em_pairconst.py model_em_pairorigin.py model_em_phase.py model_em_unfreeze.py
       model_em_warm.py model_fixed_omega.py model_forget.py model_gated.py model_grid.py model_grid_l15_pc.py
       model_hier_attn.py model_hourglass.py model_inekf_cascade.py model_inekf_gsf.py model_inekf_gsf_modeomega.py
       model_inekf_gsf_nodrop.py model_inekf_level15.py model_inekf_level15_beta.py model_inekf_level15_em.py
       model_inekf_level15_em_perscale.py model_inekf_level15_extrahead.py model_inekf_level15_hopfield.py
       model_inekf_level15_hopfield_nomainap.py model_inekf_level15_nodrop.py model_inekf_level15_perscale.py
       model_inekf_level15_sr.py model_inekf_level2.py model_inekf_parallel.py model_level15_dog.py model_level15_pc.py
       model_level15_pc_v2.py model_level15_pc_v3.py model_level15_pc_v4.py model_looped.py model_mapformer_nc.py
       model_monotone.py model_pope.py model_pope_ablate.py model_pope_decay.py model_pope_index_hier.py
       model_pope_t3.py model_predictive_coding.py model_rank.py model_rank_perhead.py model_recursive.py
       model_rope_canonical.py model_route_attn.py model_selective.py model_sign.py model_spacetime_hier.py
       model_srope_components.py model_tem.py model_tem_faithful.py model_tem_ffn.py model_tem_recency.py
       model_tem_scaling.py model_tem_t.py model_textstep.py model_vanilla_extrahead.py model_vanilla_nodrop.py
       prefix_scan.py stats_core.py train.py train_textworld.py train_tw_statechange.py train_variant.py
       tw_statechange_readouts.py)
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version at launch: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && git diff --quiet && echo clean || echo DIRTY)"
for S in $SEEDS; do for ARM in $ARMS; do
  OUT="$R/p0/${ARM}_s${S}"
  [ -f "$OUT/eval.json" ] && { echo "skip $OUT" >> "$LOG"; continue; }
  # rule 21: never launch a duplicate of a run already training; exact --output-dir token match (s5 must not match s55).
  # Amendment 1: also a RELATIVE --output-dir of the same run (runs/tw_statechange/p0/X or any path ending /runs/...).
  REL="runs/tw_statechange/p0/${ARM}_s${S}"
  if [ -n "$(ps -u "$USER" -o comm=,args= | awk -v o="$OUT" -v rel="$REL" '$1=="python3" { for (i = 2; i < NF; i++) if ($i == "--output-dir") { t = $(i + 1); sub(/\/+$/, "", t); if (t == o || t == rel || (length(t) > length(rel) && substr(t, length(t) - length(rel)) == "/" rel)) { print; break } } }')" ]; then
    echo "skip $OUT (already running)" >> "$LOG"; continue
  fi
  mkdir -p "$OUT"; G=$(twsc_wait_slot) || drv_fail "no free GPU slot within ${SLOT_TIMEOUT} s"
  echo "$(date +%H:%M:%S) $ARM s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=2 drv_launch "$R/p0/${ARM}_s${S}.log" python3 -u -m mapformer.train_tw_statechange --arm "$ARM" \
    --seed "$S" --epochs 900 --n-steps 1024 --batch-size 16 --p-take 0.4 --p-drop 0.4 --device "cuda:$G" --output-dir "$OUT"
done; done
twsc_wait_dir "$R/" || drv_fail "trainers still running ${DIR_TIMEOUT} s after the last launch"
REQ=(); for S in $SEEDS; do for ARM in $ARMS; do REQ+=("$R/p0/${ARM}_s${S}/${ARM}.pt" "$R/p0/${ARM}_s${S}/eval.json"); done; done
[ "${#REQ[@]}" -eq 64 ] || drv_fail "expected 64 artifacts, listed ${#REQ[@]}"
drv_require "${REQ[@]}" || drv_fail "missing checkpoints or eval.json"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the readouts"
_drv_log "readouts (CPU) starting"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 python3 -u -m mapformer.analyze_tw_statechange --readouts \
  > "$REPO/TW_STATECHANGE_READOUTS.txt" 2>&1 || drv_fail readouts
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the analysis"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_tw_statechange > "$REPO/TW_STATECHANGE_ANALYSIS.txt" 2>&1 \
  || drv_fail analyze
drv_done "$REPO/.tw_statechange_done" "$REPO/TW_STATECHANGE.json" "$REPO/TW_STATECHANGE_READOUTS.txt" \
  "$REPO/TW_STATECHANGE_ANALYSIS.txt" "$REPO/TW_STATECHANGE_VERDICTS.json"

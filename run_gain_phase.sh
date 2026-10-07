#!/usr/bin/env bash
# The gain-phase map on the new-object task: 2x2 STEP {raw, NormStep} x SCORE {rotary, gain}. Pre-registration:
# GAIN_PHASE_PREREG.md. Recipe = run_leak.sh's exactly (T=1024, batch 16, 900 epochs x 98 batches, lr 1e-3, cosine,
# 1 layer, 2 heads, d 128, rank 4, data-workers 3, pool 1000). Arms MapWM NormStep GainRaw GainPhase x seeds 8-15
# (n=8) = 32 runs, one batch; seed outer, arm inner. Launch:
#   cd /home/prashr/mapformer && setsid nohup bash run_gain_phase.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/gain_phase.log"
source "$REPO/lib_driver.sh"
# Amendment 2 (bug): lib_driver.sh, sourced above, already sets DRV_SPACING=45 and DRV_MINFREE=4500, so "${X:-default}"
# here was a no-op; these are assigned unconditionally (overridable through GP_* variables).
DRV_MAXPG="${GP_MAXPG:-4}"              # 4/GPU as LEAK; the picker counts every mapformer.train_ job (Amendment 3: GP_*)
DRV_SPACING="${GP_SPACING:-15}"
DRV_MINFREE="${GP_MINFREE:-5500}"       # the pilot measured ~4.7 GB per job; 5.5 GB free before each launch
drv_lock "$REPO/.run_gain_phase.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/gain_phase"; mkdir -p "$R/p0"
ARMS="MapWM NormStep GainRaw GainPhase"
SEEDS="8 9 10 11 12 13 14 15"
# every mapformer module the trainer / evaluator / analysis import (sys.modules after importing them, plus the lazily
# imported data_parallel), and the secondary script
GUARD=(__init__.py analyze_gain_phase.py data_parallel.py environment.py environment_addition.py environment_nd.py
       environment_newobj.py eval_gain_phase.py evaluate.py gain_phase_eval.py hourglass_plain.py leak_eval.py
       lie_groups.py model.py model_ablations.py model_baseline_nope.py model_baseline_rope.py model_baselines_extra.py
       model_bounded_mem.py model_cho_coupled.py model_cho_positions.py model_code_decay.py model_codes.py
       model_counter_installed.py model_coupled_ape.py model_coupled_rope.py model_em_dof.py model_em_fixed.py
       model_em_magonly.py model_em_noleak.py model_em_pairconst.py model_em_pairorigin.py model_em_phase.py
       model_em_pope.py model_em_unfreeze.py model_em_warm.py model_fixed_omega.py model_forget.py model_gain_phase.py
       model_gated.py model_grid.py model_grid_l15_pc.py model_hier_attn.py model_hourglass.py model_inekf_cascade.py
       model_inekf_gsf.py model_inekf_gsf_modeomega.py model_inekf_gsf_nodrop.py model_inekf_level15.py
       model_inekf_level15_beta.py model_inekf_level15_em.py model_inekf_level15_em_perscale.py
       model_inekf_level15_extrahead.py model_inekf_level15_hopfield.py model_inekf_level15_hopfield_nomainap.py
       model_inekf_level15_nodrop.py model_inekf_level15_perscale.py model_inekf_level15_sr.py model_inekf_level2.py
       model_inekf_parallel.py model_level15_dog.py model_level15_pc.py model_level15_pc_v2.py model_level15_pc_v3.py
       model_level15_pc_v4.py model_looped.py model_mapformer_nc.py model_monotone.py model_pope.py
       model_pope_ablate.py model_pope_decay.py model_pope_index_hier.py model_pope_t3.py model_predictive_coding.py
       model_rank.py model_rank_perhead.py model_recursive.py model_rope_canonical.py model_route_attn.py
       model_selective.py model_sign.py model_spacetime_hier.py model_srope_components.py model_tem.py
       model_tem_faithful.py model_tem_ffn.py model_tem_recency.py model_tem_scaling.py model_tem_t.py
       model_vanilla_extrahead.py model_vanilla_nodrop.py prefix_scan.py rescore_hook.py stats_core.py train.py
       train_gain_phase.py train_newobj.py train_variant.py docs/audits/2026-10-06/gain_phase_secondary.py)
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version at launch: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && git diff --quiet && echo clean || echo DIRTY)"
for S in $SEEDS; do for ARM in $ARMS; do
  OUT="$R/p0/${ARM}_s${S}"
  [ -f "$OUT/${ARM}.pt" ] && { echo "skip $OUT" >> "$LOG"; continue; }
  # Amendment 1 (rule 21): never launch a duplicate of a run that is already training (and never truncate its log)
  # Amendment 2: exact-token match (a prefix such as ..._s1 must not match ..._s15)
  if [ -n "$(ps -u "$USER" -o comm=,args= | awk -v o="$OUT" '$1=="python3" { for (i = 2; i < NF; i++) if ($i == "--output-dir" && $(i + 1) == o) { print; break } }')" ]; then
    echo "skip $OUT (already running)" >> "$LOG"; continue
  fi
  mkdir -p "$OUT"; G=$(drv_wait_slot)
  echo "$(date +%H:%M:%S) $ARM s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=3 drv_launch "$R/p0/${ARM}_s${S}.log" python3 -u -m mapformer.train_gain_phase --variant "$ARM" \
    --seed "$S" --epochs 900 --n-steps 1024 --batch-size 16 --device "cuda:$G" --output-dir "$OUT"
done; done
drv_wait_dir "$R/"
REQ=(); for S in $SEEDS; do for ARM in $ARMS; do REQ+=("$R/p0/${ARM}_s${S}/${ARM}.pt"); done; done
[ "${#REQ[@]}" -eq 32 ] || drv_fail "expected 32 checkpoints, listed ${#REQ[@]}"
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before eval"
# Amendment 1: eval on the GPU with the most free memory, and only when it has > 6 GB free. Amendment 2: logged, guarded
# against empty nvidia-smi output, and bounded (12 h, then drv_fail).
_drv_log "eval: waiting for a GPU with > 6000 MiB free"
W0=$(date +%s)
while :; do
  F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -nr | head -1 | tr -dc '0-9')
  [ -n "$F" ] && [ "$F" -gt 6000 ] && break
  [ $(( $(date +%s) - W0 )) -gt 43200 ] && drv_fail "eval: no GPU with > 6000 MiB free (or nvidia-smi empty) for 12 h"
  sleep 60
done
_drv_log "eval: starting (max free ${F} MiB)"
python3 -u -m mapformer.eval_gain_phase auto >> "$LOG" 2>&1 || drv_fail eval
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before analysis"
python3 -u -m mapformer.analyze_gain_phase > "$REPO/GAIN_PHASE_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.gain_phase_done" "$REPO/GAIN_PHASE_EVAL.json" "$REPO/GAIN_PHASE_ANALYSIS.txt" "$REPO/GAIN_PHASE_VERDICTS.json"
# declared secondaries, after the registered artifacts and the marker (a failure here does not touch the registered result)
python3 -u "$REPO/docs/audits/2026-10-06/gain_phase_secondary.py" > "$REPO/docs/audits/2026-10-06/gain_phase_secondary_out.txt" 2>&1 \
  && _drv_log "secondaries done" || _drv_log "secondaries FAILED (registered result unaffected)"

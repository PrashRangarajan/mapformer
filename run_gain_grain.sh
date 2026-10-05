#!/usr/bin/env bash
# Gain granularity between MapEM and MapPoPE on the paper torus. Pre-registration: GAIN_GRAIN_PREREG.md.
# Recipe = run_mappope_pair.sh's exactly (T=128, 300 epochs x 98 batches, B128, lr 1e-3, cosine, 1 layer, 2 heads,
# d 128, no landmarks, data-workers 0, explicit attention path). Six arms x seeds 26-45 (n=20) = 120 runs, one batch.
# Seed outer, arm inner. Launch:
#   cd /home/prashr/mapformer && setsid nohup bash run_gain_grain.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/gain_grain.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-4}"; DRV_SPACING="${DRV_SPACING:-8}"     # 4/GPU, as MAPPOPE_PAIR (pilot-measured timing in the prereg)
drv_lock "$REPO/.run_gain_grain.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/gain_grain"; mkdir -p "$R/p0"
ARMS="Vanilla MapPoPE-Pair GainScalar GainMod4 VanillaEM VanillaEM_NonNeg"
SEEDS="$(seq -s ' ' 26 45)"
# every mapformer module the trainer / evaluator / analysis import (sys.modules at registration), the wrappers, the
# remap secondary and its two helpers
GUARD=(__init__.py analyze_gain_grain.py ckpt_guard.py data_parallel.py environment.py environment_addition.py
       eval_noise_refine.py evaluate.py hourglass_plain.py lie_groups.py model.py model_ablations.py
       model_baseline_nope.py model_baseline_rope.py model_baselines_extra.py model_bounded_mem.py
       model_cho_coupled.py model_cho_positions.py model_code_decay.py model_counter_installed.py
       model_coupled_ape.py model_coupled_rope.py model_em_dof.py model_em_fixed.py model_em_magonly.py
       model_em_noleak.py model_em_pairconst.py model_em_pairorigin.py model_em_phase.py model_em_pope.py
       model_em_unfreeze.py model_em_warm.py model_fixed_omega.py model_forget.py model_gated.py model_grid.py
       model_grid_l15_pc.py model_hier_attn.py model_hourglass.py model_inekf_cascade.py model_inekf_gsf.py
       model_inekf_gsf_modeomega.py model_inekf_gsf_nodrop.py model_inekf_level15.py model_inekf_level15_beta.py
       model_inekf_level15_em.py model_inekf_level15_em_perscale.py model_inekf_level15_extrahead.py
       model_inekf_level15_hopfield.py model_inekf_level15_hopfield_nomainap.py model_inekf_level15_nodrop.py
       model_inekf_level15_perscale.py model_inekf_level15_sr.py model_inekf_level2.py model_inekf_parallel.py
       model_level15_dog.py model_level15_pc.py model_level15_pc_v2.py model_level15_pc_v3.py model_level15_pc_v4.py
       model_looped.py model_mapformer_nc.py model_monotone.py model_pope.py model_pope_ablate.py
       model_pope_decay.py model_pope_index_hier.py model_pope_pair.py model_pope_t3.py model_predictive_coding.py
       model_rank.py model_rank_perhead.py model_recursive.py model_rope_canonical.py model_route_attn.py
       model_selective.py model_sign.py model_spacetime_hier.py model_srope_components.py model_tem.py
       model_tem_faithful.py model_tem_ffn.py model_tem_recency.py model_tem_scaling.py model_tem_t.py
       model_vanilla_extrahead.py model_vanilla_nodrop.py prefix_scan.py stats_core.py train.py train_gain_grain.py
       train_variant.py eval_gain_grain.py docs/audits/2026-10-05/gain_grain_remap.py
       docs/audits/2026-10-05/remap_probe.py docs/audits/2026-09-27/probe_whatwhere.py)
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
for S in $SEEDS; do for V in $ARMS; do
  OUT="$R/p0/${V}_s${S}"
  [ -f "$OUT/${V}.pt" ] && { echo "skip $OUT" >> "$LOG"; continue; }
  G=$(drv_wait_slot); mkdir -p "$OUT"
  echo "$(date +%H:%M:%S) $V s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=2 drv_launch "$R/${V}_s${S}.log" python3 -u -m mapformer.train_gain_grain --variant "$V" --seed "$S" \
    --epochs 300 --n-batches 98 --batch-size 128 --n-steps 128 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
    --schedule cosine --lr 1e-3 --device "cuda:$G" --output-dir "$OUT"
done; done
drv_wait_dir "$R/"
REQ=(); for S in $SEEDS; do for V in $ARMS; do REQ+=("$R/p0/${V}_s${S}/${V}.pt"); done; done
[ "${#REQ[@]}" -eq 120 ] || drv_fail "expected 120 checkpoints, listed ${#REQ[@]}"
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before eval"
python3 -u -m mapformer.eval_gain_grain mapformer.eval_noise_refine --runs-dir "$R" --variants $ARMS --noises 0.0 \
  --seeds $SEEDS --lengths 128 512 1024 --n-trials 100 --device cuda:0 --out "$REPO/GAIN_GRAIN_EVAL.md" \
  --title "Gain granularity between MapEM and MapPoPE, paper torus, rank 2, trained at T=128" >> "$LOG" 2>&1 || drv_fail eval
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before analysis"
python3 -u -m mapformer.analyze_gain_grain > "$REPO/GAIN_GRAIN_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.gain_grain_done" "$REPO/GAIN_GRAIN_EVAL.json" "$REPO/GAIN_GRAIN_ANALYSIS.txt"
# declared secondary, after the registered artifacts and the marker (its failure does not touch the registered result)
python3 -u "$REPO/docs/audits/2026-10-05/gain_grain_remap.py" > "$REPO/docs/audits/2026-10-05/gain_grain_remap_out.txt" 2>&1 \
  && _drv_log "remap secondary done" || _drv_log "remap secondary FAILED (registered result unaffected)"

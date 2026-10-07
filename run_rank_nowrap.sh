#!/usr/bin/env bash
# Is per-head rank 2's failure on the 2D torus about wrap-around (the exactly periodic code)? Pre-registration:
# RANK_NOWRAP_PREREG.md. Recipe = run_rank_nd.sh's 2D cells exactly (T=1024, 900 epochs x 98 batches, B16, lr 1e-3,
# cosine, 1 layer, 2 heads, d 128, no landmarks, data-workers 3, explicit attention path, --save-full-state); arms
# through train_rank_nowrap (omega initialised at grid 32 on every torus). Cells (Amendment 1): per-head rank 2 / 3 on
# the 32-torus and on the 256-torus, plus rank 2 on the 32-torus with the map redrawn every trajectory (memorisation
# control); seeds 60-69 (n=10); 50 runs; seed outer, cell inner. Launch:
#   cd /home/prashr/mapformer && setsid nohup bash run_rank_nowrap.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/rank_nowrap.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-4}"; DRV_SPACING="${DRV_SPACING:-15}"   # 4/GPU as RANK_ND; slots are shared with any other batch
drv_lock "$REPO/.run_rank_nowrap.lock" || exit 1
export PYTHONUNBUFFERED=1
cd "$REPO/.."
R="$REPO/runs/rank_nowrap"; mkdir -p "$R"
SEEDS="$(seq -s ' ' 60 69)"
CELLS=("32 Vanilla_r2ph_om32" "32 Vanilla_r3ph_om32" "32 Vanilla_r2ph_om32_redraw" "256 Vanilla_r2ph_om32" "256 Vanilla_r3ph_om32")
# every mapformer module the trainer / evaluators / analysis import (sys.modules after importing train_rank_nowrap,
# analyze_rank_nowrap, eval_nd, rescore_hook), plus the runpy'd eval wrapper
GUARD=(__init__.py analyze_rank_nowrap.py data_parallel.py environment.py environment_addition.py environment_nd.py
       environment_nd_redraw.py eval_nd.py eval_rank_nowrap.py evaluate.py hourglass_plain.py lie_groups.py model.py model_ablations.py
       model_baseline_nope.py model_baseline_rope.py model_baselines_extra.py model_bounded_mem.py model_cho_coupled.py
       model_cho_positions.py model_code_decay.py model_counter_installed.py model_coupled_ape.py model_coupled_rope.py
       model_em_dof.py model_em_fixed.py model_em_magonly.py model_em_noleak.py model_em_pairconst.py
       model_em_pairorigin.py model_em_phase.py model_em_unfreeze.py model_em_warm.py model_fixed_omega.py
       model_forget.py model_gated.py model_grid.py model_grid_l15_pc.py model_hier_attn.py model_hourglass.py
       model_inekf_cascade.py model_inekf_gsf.py model_inekf_gsf_modeomega.py model_inekf_gsf_nodrop.py
       model_inekf_level15.py model_inekf_level15_beta.py model_inekf_level15_em.py model_inekf_level15_em_perscale.py
       model_inekf_level15_extrahead.py model_inekf_level15_hopfield.py model_inekf_level15_hopfield_nomainap.py
       model_inekf_level15_nodrop.py model_inekf_level15_perscale.py model_inekf_level15_sr.py model_inekf_level2.py
       model_inekf_parallel.py model_level15_dog.py model_level15_pc.py model_level15_pc_v2.py model_level15_pc_v3.py
       model_level15_pc_v4.py model_looped.py model_mapformer_nc.py model_monotone.py model_pope.py
       model_pope_ablate.py model_pope_decay.py model_pope_index_hier.py model_pope_t3.py model_predictive_coding.py
       model_rank.py model_rank_perhead.py model_recursive.py model_rope_canonical.py model_route_attn.py
       model_selective.py model_sign.py model_spacetime_hier.py model_srope_components.py model_tem.py
       model_tem_faithful.py model_tem_ffn.py model_tem_recency.py model_tem_scaling.py model_tem_t.py
       model_vanilla_extrahead.py model_vanilla_nodrop.py prefix_scan.py rescore_hook.py stats_core.py train.py
       train_rank_nowrap.py train_variant.py)
running() { ps -u "$USER" -o comm=,args= | awk -v d="--output-dir $1 " '$1=="python3" && index($0, d)' | wc -l; }
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version at launch: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && git diff --quiet HEAD -- && echo clean || echo DIRTY)"
for S in $SEEDS; do for c in "${CELLS[@]}"; do set -- $c
  OUT="$R/N$1/D2/${2}_s${S}"
  [ -f "$OUT/$2.pt" ] && { echo "skip $OUT" >> "$LOG"; continue; }
  # duplicate-launch guard (Amendment 1, N8; rule 23: python3 by comm, never pgrep -f)
  if [ "$(running "$OUT")" -gt 0 ]; then echo "$(date +%H:%M:%S) skip $OUT: a trainer for it is already running" >> "$LOG"; continue; fi
  G=$(drv_wait_slot); mkdir -p "$OUT"
  [ "$(running "$OUT")" -gt 0 ] && { echo "$(date +%H:%M:%S) skip $OUT: started while waiting for a slot" >> "$LOG"; continue; }
  echo "$(date +%H:%M:%S) N$1 $2 s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=2 drv_launch "$R/N$1_${2}_s${S}.log" python3 -u -m mapformer.train_rank_nowrap --variant "$2" \
    --env nd --n-dims 2 --grid-size "$1" --seed "$S" --epochs 900 --lr 1e-3 --n-batches 98 --batch-size 16 \
    --n-steps 1024 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
    --data-workers 3 --device "cuda:$G" --output-dir "$OUT" --save-full-state
done; done
drv_wait_dir "$R/"
REQ=(); for S in $SEEDS; do for c in "${CELLS[@]}"; do set -- $c; REQ+=("$R/N$1/D2/${2}_s${S}/$2.pt"); done; done
[ "${#REQ[@]}" -eq 50 ] || drv_fail "expected 50 checkpoints, listed ${#REQ[@]}"
drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before eval"
for N in 32 256; do
  ARMS="Vanilla_r2ph_om32,Vanilla_r3ph_om32"; [ "$N" = 32 ] && ARMS="$ARMS,Vanilla_r2ph_om32_redraw"
  python3 -u -m mapformer.eval_rank_nowrap --runs-dir "$R/N$N" --configs "2:$N:$ARMS" \
    --seeds $SEEDS --lengths 1024 2048 --n-trials 100 --device cuda:0 --out "$R/N$N/EVAL_D2.md" >> "$LOG" 2>&1 \
    || drv_fail "eval N$N"
  python3 -u -m mapformer.rescore_hook --scale auto -- mapformer.eval_rank_nowrap --runs-dir "$R/N$N" \
    --configs "2:$N:$ARMS" --seeds $SEEDS --lengths 1024 --n-trials 100 --device cuda:0 \
    --out "$REPO/RANK_NOWRAP_RESCORE_N$N.md" >> "$LOG" 2>&1 || drv_fail "rescore N$N"
done
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before analysis"
python3 -u -m mapformer.analyze_rank_nowrap > "$REPO/RANK_NOWRAP_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.rank_nowrap_done" "$R/N32/EVAL_D2.json" "$R/N256/EVAL_D2.json" "$REPO/RANK_NOWRAP_RESCORE_N32.json" \
  "$REPO/RANK_NOWRAP_RESCORE_N256.json" "$REPO/RANK_NOWRAP_ANALYSIS.txt" "$REPO/RANK_NOWRAP.json"

#!/usr/bin/env bash
# Landmarks vs path integration in words. Pre-registration: TW_LANDMARK_PREREG.md. Recipe = run_textworld.sh's (T=1024
# words, batch 16, 900 epochs x 98 batches, lr 1e-3, warmup + cosine, d 128, 2 heads, r=4 shared, data-workers 3).
# Cells: P2 = MapWM 2 layers at name rate 0 / 0.5 / 1; R2 = RoPE 2 layers at 0 / 0.5 / 1; P1 = MapWM 1 layer at 0 / 1.
# Seeds 50-55 (fresh), seed outer, cell inner: 48 runs, one batch. Then (md5 re-checked) the readouts on CPU in 4
# shards, plain and dropout re-scored (rescore_hook), then (md5 re-checked) the analysis, then the done marker.
# Launch (after the GPU pilot and the audit amendment):
#   cd /home/prashr/mapformer && setsid nohup bash run_tw_landmark.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="${TL_LOG:-$REPO/tw_landmark.log}"
source "$REPO/lib_driver.sh"
# knobs AFTER sourcing lib_driver.sh (it assigns its own defaults); overridable through TL_* variables
DRV_MAXPG="${TL_MAXPG:-4}"              # 8 slots on the two 4090s; the picker counts every mapformer.train_ job
# Amendment 1 (N7, rule 24): DRV_MINFREE = 1.25 x the pilot's measured peak per 2-layer job, and DRV_SPACING >= the
# pilot's measured time from launch to that peak, so a job not yet holding its memory is not double-booked. Defaults
# until the pilot: 60 s spacing, 7000 MiB.
DRV_SPACING="${TL_SPACING:-60}"
DRV_MINFREE="${TL_MINFREE:-7000}"
RUN_TIMEOUT="${TL_RUN_TIMEOUT:-8h}"     # Amendment 1 (N6): per-run wall limit (expected ~1.5 h per 2-layer run)
EVAL_TIMEOUT="${TL_EVAL_TIMEOUT:-4h}"
drv_lock "$REPO/.run_tw_landmark.lock" || exit 1
cd "$REPO/.."
R="${TL_RUNS:-$REPO/runs/tw_landmark}"; mkdir -p "$R/p0"   # TL_RUNS / TL_OUT: end-to-end test only
OUTD="${TL_OUT:-$REPO}"; NW="${TL_NWALKS:-200}"
SEEDS="50 51 52 53 54 55"
CELLS="MapWM:2:0.0 MapWM:2:0.5 MapWM:2:1.0 RoPE:2:0.0 RoPE:2:0.5 RoPE:2:1.0 MapWM:1:0.0 MapWM:1:1.0"
# every mapformer module the trainer / readouts / analysis / rescore hook import (sys.modules after importing them,
# plus the lazily imported data_parallel)
GUARD=(__init__.py analyze_tw_landmark.py data_parallel.py environment.py environment_addition.py
       environment_textworld.py environment_tw_landmark.py evaluate.py hourglass_plain.py lie_groups.py model.py
       model_ablations.py model_baseline_nope.py model_baseline_rope.py model_baselines_extra.py model_bounded_mem.py
       model_cho_coupled.py model_cho_positions.py model_code_decay.py model_counter_installed.py model_coupled_ape.py
       model_coupled_rope.py model_em_dof.py model_em_fixed.py model_em_magonly.py model_em_noleak.py
       model_em_pairconst.py model_em_pairorigin.py model_em_phase.py model_em_unfreeze.py model_em_warm.py
       model_fixed_omega.py model_forget.py model_gated.py model_grid.py model_grid_l15_pc.py model_hier_attn.py
       model_hourglass.py model_inekf_cascade.py model_inekf_gsf.py model_inekf_gsf_modeomega.py
       model_inekf_gsf_nodrop.py model_inekf_level15.py model_inekf_level15_beta.py model_inekf_level15_em.py
       model_inekf_level15_em_perscale.py model_inekf_level15_extrahead.py model_inekf_level15_hopfield.py
       model_inekf_level15_hopfield_nomainap.py model_inekf_level15_nodrop.py model_inekf_level15_perscale.py
       model_inekf_level15_sr.py model_inekf_level2.py model_inekf_parallel.py model_level15_dog.py model_level15_pc.py
       model_level15_pc_v2.py model_level15_pc_v3.py model_level15_pc_v4.py model_looped.py model_mapformer_nc.py
       model_monotone.py model_pope.py model_pope_ablate.py model_pope_decay.py model_pope_index_hier.py
       model_pope_t3.py model_predictive_coding.py model_rank.py model_rank_perhead.py model_recursive.py
       model_rope_canonical.py model_route_attn.py model_selective.py model_sign.py model_spacetime_hier.py
       model_srope_components.py model_tem.py model_tem_faithful.py model_tem_ffn.py model_tem_recency.py
       model_tem_scaling.py model_tem_t.py model_vanilla_extrahead.py model_vanilla_nodrop.py prefix_scan.py
       rescore_hook.py stats_core.py train.py train_tw_landmark.py train_variant.py tw_landmark_eval.py)
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version at launch: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && [ -z "$(git status --porcelain -- "${GUARD[@]}")" ] && echo clean || echo "DIRTY (a guarded file is modified or untracked)")"
RUNS=()
for S in $SEEDS; do for C in $CELLS; do
  IFS=: read -r ARM L RATE <<< "$C"
  OUT="$R/p0/${ARM}_L${L}_r${RATE}_s${S}"; RUNS+=("$OUT")
  [ -f "$OUT/train.json" ] && [ -f "$OUT/${ARM}.pt" ] && { echo "skip $OUT" >> "$LOG"; continue; }
  # never launch a duplicate of a run that is already training (exact-token match on the --output-dir value, so a
  # prefix such as ..._s5 can never match ..._s55; rule 21)
  if [ -n "$(ps -u "$USER" -o comm=,args= | awk -v o="$OUT" '$1=="python3" { for (i = 2; i < NF; i++) if ($i == "--output-dir" && $(i + 1) == o) { print; break } }')" ]; then
    echo "skip $OUT (already running)" >> "$LOG"; continue
  fi
  mkdir -p "$OUT"; G=$(drv_wait_slot)
  echo "$(date +%H:%M:%S) $ARM L$L r$RATE s$S -> cuda:$G" >> "$LOG"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" timeout "$RUN_TIMEOUT" python3 -u -m mapformer.train_tw_landmark --arm "$ARM" --n-layers "$L" \
    --name-rate "$RATE" --seed "$S" --epochs 900 --device "cuda:$G" --output-dir "$OUT"
done; done
drv_wait_dir "$R/"
[ "${#RUNS[@]}" -eq 48 ] || drv_fail "expected 48 runs, listed ${#RUNS[@]}"
REQ=(); for D in "${RUNS[@]}"; do REQ+=("$D/train.json"); done
drv_require "${REQ[@]}" || drv_fail "missing runs"
# readouts on CPU (deterministic; no GPU needed), 4 shards of 12 runs, plain and re-scored
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before eval"
_drv_log "eval: starting (4 CPU shards x 2 passes)"
# stale shard outputs are not trusted blindly: tw_landmark_eval reuses a stored run only if its checkpoint md5 and
# walk count match (Amendment 1, N5)
for PASS in plain rescored; do
  PIDS=()
  for K in 0 1 2 3; do
    SH=("${RUNS[@]:$((K * 12)):12}")
    if [ "$PASS" = plain ]; then
      CUDA_VISIBLE_DEVICES="" timeout "$EVAL_TIMEOUT" python3 -u -m mapformer.tw_landmark_eval --runs "${SH[@]}" --threads 6 --n-walks "$NW" \
        --out "$R/eval_${PASS}_${K}.json" > "$R/eval_${PASS}_${K}.log" 2>&1 & PIDS+=($!)
    else
      CUDA_VISIBLE_DEVICES="" timeout "$EVAL_TIMEOUT" python3 -u -m mapformer.rescore_hook --scale auto -- mapformer.tw_landmark_eval \
        --runs "${SH[@]}" --threads 6 --n-walks "$NW" --no-mc --out "$R/eval_${PASS}_${K}.json" > "$R/eval_${PASS}_${K}.log" 2>&1 & PIDS+=($!)
    fi
  done
  # Amendment 1 (N6): a failed shard stops its siblings before drv_fail (no orphaned eval processes)
  for P in "${PIDS[@]}"; do
    wait "$P" || { kill "${PIDS[@]}" 2>/dev/null; drv_fail "eval $PASS shard failed (pid $P); sibling shards stopped"; }
  done
done
python3 - "$R" "$OUTD" <<'EOF' || drv_fail "merge"
import json, sys
R, REPO = sys.argv[1], sys.argv[2]
for p, out in (("plain", "TW_LANDMARK.json"), ("rescored", "TW_LANDMARK_RESCORED.json")):
    m = {}
    for k in range(4):
        m.update(json.load(open(f"{R}/eval_{p}_{k}.json")))
    assert len(m) == 48, (p, len(m))
    json.dump(m, open(f"{REPO}/{out}", "w"), indent=1)
EOF
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before analysis"
python3 -u -m mapformer.analyze_tw_landmark --json "$OUTD/TW_LANDMARK.json" --rescored "$OUTD/TW_LANDMARK_RESCORED.json" \
  --verdicts-out "$OUTD/TW_LANDMARK_VERDICTS.json" > "$OUTD/TW_LANDMARK_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$OUTD/.tw_landmark_done" "$OUTD/TW_LANDMARK.json" "$OUTD/TW_LANDMARK_RESCORED.json" \
  "$OUTD/TW_LANDMARK_ANALYSIS.txt" "$OUTD/TW_LANDMARK_VERDICTS.json"

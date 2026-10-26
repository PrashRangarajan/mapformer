#!/usr/bin/env bash
# RNN_TORUS_PREREG.md: LSTM / GRU baselines (parameter-matched) vs path MapWM 1L and index RoPE 1L on the torus, T = 1024,
# 900 epochs, FACTORIAL_N8's recipe; seeds 3300-3307 (fresh). Anchors through mapformer.train_variant with --fast-attn
# (FACTORIAL_N8's flags exactly); RNN arms through mapformer.train_rnn (same CLI and recipe; no --fast-attn; cuDNN TF32 off).
# DRV_GPUS and DRV_MAXPG are REQUIRED in the environment (no defaults). Planned launch (kalman, both GPUs, 2 jobs each),
# AFTER the EXPTORUS batch (Amendment 1 A1: refuses unless .exptorus_done exists, or RNN_SKIP_EXPTORUS=1, logged; refuses
# while any other mapformer.train_* process of this user runs, or while another user holds a GPU process):
#   cd /home/prashr/mapformer && DRV_GPUS="0 1" DRV_MAXPG=2 setsid nohup bash run_rnn_torus.sh > /dev/null 2>&1 &
# Dry run (logs launches only; own run dir and log; the clean-tree / committed / sequencing checks become logged warnings,
# and slot choice does not wait on real GPU occupancy, so it completes while another batch trains):
#   cd /home/prashr/mapformer && DRV_DRYRUN=1 DRV_GPUS="0 1" DRV_MAXPG=2 bash run_rnn_torus.sh
# RNN_TORUS_PREREG.md is md5-guarded: it must be FINAL before launch.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
DRY="${DRV_DRYRUN:-0}"
if [ "$DRY" = 1 ]; then LOG="$REPO/rnn_torus_dryrun.log"; R="$REPO/runs/rnn_torus_dryrun"
else LOG="$REPO/rnn_torus.log"; R="$REPO/runs/rnn_torus"; fi
REQ_GPUS="${DRV_GPUS:-}"; REQ_MAXPG="${DRV_MAXPG:-}"          # captured BEFORE lib_driver.sh assigns its defaults
source "$REPO/lib_driver.sh"
[ -n "$REQ_GPUS" ] && [ -n "$REQ_MAXPG" ] || { _drv_log "REFUSED: set DRV_GPUS and DRV_MAXPG explicitly (e.g. DRV_GPUS=\"0 1\" DRV_MAXPG=2)"; exit 1; }
[[ "$REQ_GPUS" =~ ^[0-9]+( [0-9]+)*$ && "$REQ_MAXPG" =~ ^[1-8]$ ]] || { _drv_log "REFUSED: bad DRV_GPUS '$REQ_GPUS' / DRV_MAXPG '$REQ_MAXPG'"; exit 1; }
DRV_GPUS="$REQ_GPUS"; export DRV_GPUS
DRV_MAXPG="$REQ_MAXPG"; DRV_MINFREE=4000; DRV_SPACING=30     # assigned after sourcing (rule 27)
export CUDA_DEVICE_ORDER=PCI_BUS_ID                           # cuda:N == nvidia-smi index N (drv_freemem reads nvidia-smi)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
drv_lock "$REPO/.run_rnn_torus.lock" || exit 1
echo "start $(date)" >> "$LOG"
# Amendment 1 (A1, audit M1): sequencing. RNN_TORUS runs after EXPTORUS, never beside another batch.
seq_refuse(){ if [ "$DRY" = 1 ]; then _drv_log "DRYRUN WARNING: $* (a real launch would refuse)"; else _drv_log "REFUSED: $*"; exit 1; fi; }
if [ ! -f "$REPO/.exptorus_done" ]; then
  if [ "${RNN_SKIP_EXPTORUS:-0}" = 1 ]; then _drv_log "OVERRIDE: RNN_SKIP_EXPTORUS=1 -- starting without .exptorus_done"
  else seq_refuse ".exptorus_done is absent (EXPTORUS runs first; override RNN_SKIP_EXPTORUS=1)"; fi
fi
OTHER=$(ps -u "$USER" -o pid=,comm=,args= | awk -v r="$R/" '$2=="python3" && /mapformer[.]train_/ && !index($0, r)' | head -1)
[ -n "$OTHER" ] && seq_refuse "another mapformer.train_ process is running: $OTHER"
for P in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  U=$(ps -o user= -p "$P" 2>/dev/null | tr -d ' ')
  [ -n "$U" ] && [ "$U" != "$USER" ] && { seq_refuse "GPU process $P belongs to user $U"; break; }
done
if (cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .); then
  [ "$DRY" = 1 ] && _drv_log "DRYRUN WARNING: tracked files modified (a real launch would refuse)" ||
    { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
fi
cd "$REPO/.."
SEEDS=(3300 3301 3302 3303 3304 3305 3306 3307)
# cell = root:variant:layers:lr:kind (kind fa = train_variant --fast-attn, rnn = train_rnn). Longest first within a seed.
# root of an RNN cell = analyze_rnn_torus.lr_root(lr) = f"lr{lr:.0e}"; test_rnn_torus.py checks this table against
# analyze_rnn_torus.all_cells().
# RNN rates: picked by analyze_rnn_torus.pilot_pick on the lr pilot (RNN_TORUS_PREREG.md, Pilot); the 4x LSTM at both
# 1e-3 and 3e-3 (Amendment 1 A3).
CELLS=("lr1e-03:LSTM_L1_h313:1:1e-3:rnn" "lr3e-03:LSTM_L1_h313:1:3e-3:rnn"
       "lr3e-03:LSTM_L1_h157:1:3e-3:rnn" "lr1e-03:GRU_L1_h181:1:1e-3:rnn"
       "lr3e-03:LSTM_L2_h111:2:3e-3:rnn" "anchor:Vanilla_r4:1:1e-3:fa" "anchor:RoPE:1:1e-3:fa"
       "lr3e-03:GRU_L2_h128:2:3e-3:rnn")
ROOTS=($(printf '%s\n' "${CELLS[@]}" | cut -d: -f1 | sort -u))
for c in "${ROOTS[@]}"; do mkdir -p "$R/$c/p0"; done
mapfile -t GUARD < <(cd "$REPO/.." && CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_variant, mapformer.train_rnn, mapformer.model_rnn, mapformer.data_parallel
import mapformer.eval_rnn_torus, mapformer.eval_noise_refine, mapformer.eval_rank_strata
import mapformer.analyze_rnn_torus, mapformer.analyze_factorial_n8
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
[ "${#GUARD[@]}" -ge 20 ] || { _drv_log "ABORT: module list for the md5 guard has ${#GUARD[@]} entries"; exit 1; }
GUARD+=(ckpt_guard.py lib_driver.sh run_rnn_torus.sh RNN_TORUS_PREREG.md)
if ! (cd "$REPO" && git ls-files --error-unmatch "${GUARD[@]}" > /dev/null 2>&1); then
  [ "$DRY" = 1 ] && _drv_log "DRYRUN WARNING: a guarded file is not committed (a real launch would refuse)" ||
    { _drv_log "REFUSED: a guarded file is not committed"; exit 1; }
fi
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) $(cd "$REPO" && git diff --quiet && echo clean || echo DIRTY); GPUs $DRV_GPUS; MAXPG $DRV_MAXPG; md5 guard over ${#GUARD[@]} files"
slot(){ if [ "$DRY" = 1 ]; then local g; g=$(drv_pick 99 0); echo "${g:-0}"; else drv_wait_slot; fi; }
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
# FACTORIAL_N8's exact training flags; only seed / device / output dir (and, per cell, variant / layers / lr / numerics) differ
COMMON=(--epochs 900 --n-batches 98 --batch-size 16 --n-steps 1024 --n-heads 2 --d-model 128 --n-landmarks 0
        --schedule cosine --data-workers 3)
for S in "${SEEDS[@]}"; do for C in "${CELLS[@]}"; do      # seed outer, cell inner: a partial batch stays paired by seed
  IFS=: read -r ROOT V NL LR KIND <<< "$C"; OUT="$R/$ROOT/p0/${V}_s$S"
  [ -f "$OUT/$V.pt" ] && { _drv_log "skip $ROOT $V s$S (done)"; continue; }
  running "$OUT" && { _drv_log "skip $ROOT $V s$S (running)"; continue; }
  drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed during the batch"      # per launch (rule 22)
  G=$(slot); mkdir -p "$OUT"; _drv_log "$ROOT $V L$NL lr $LR s$S -> cuda:$G"
  if [ "$KIND" = fa ]; then
    OMP_NUM_THREADS=3 drv_launch "$R/$ROOT/${V}_s${S}.log" python3 -u -m mapformer.train_variant --variant "$V" --seed "$S" \
      --lr "$LR" "${COMMON[@]}" --fast-attn --n-layers "$NL" --device "cuda:$G" --output-dir "$OUT"
  else
    OMP_NUM_THREADS=3 drv_launch "$R/$ROOT/${V}_s${S}.log" python3 -u -m mapformer.train_rnn --variant "$V" --seed "$S" \
      --lr "$LR" "${COMMON[@]}" --n-layers "$NL" --device "cuda:$G" --output-dir "$OUT"
  fi
done; done
EV(){ if [ "$DRY" = 1 ]; then _drv_log "DRYRUN eval: $*"; else "$@" >> "$LOG" 2>&1; fi; }
if [ "$DRY" != 1 ]; then
  sleep 60
  DRV_WAIT_POLL=120 drv_wait_dir "$R/"
  REQ=(); for S in "${SEEDS[@]}"; do for C in "${CELLS[@]}"; do IFS=: read -r ROOT V NL LR KIND <<< "$C"; REQ+=("$R/$ROOT/p0/${V}_s$S/$V.pt"); done; done
  [ "${#REQ[@]}" -eq $(( ${#SEEDS[@]} * ${#CELLS[@]} )) ] || drv_fail "checkpoint list size"
  drv_require "${REQ[@]}" || drv_fail "missing checkpoints"
  drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before eval"
fi
G=$(slot)
for ROOT in "${ROOTS[@]}"; do
  VS=$(printf '%s\n' "${CELLS[@]}" | awk -F: -v r="$ROOT" '$1==r {print $2}' | tr '\n' ' ')
  EV python3 -u -m mapformer.eval_rnn_torus noise_refine --runs-dir "$R/$ROOT" --variants $VS --noises 0.0 --seeds "${SEEDS[@]}" \
    --lengths 512 1024 2048 --n-trials 100 --device "cuda:$G" --title "RNN torus $ROOT" \
    --out "$REPO/RNN_TORUS_${ROOT}.md" || drv_fail "eval $ROOT"
  EV python3 -u -m mapformer.eval_rnn_torus strata --runs-dir "$R/$ROOT" --variants $VS --seeds "${SEEDS[@]}" --lengths 1024 \
    --n-trials 100 --device "cuda:$G" --check-json "$REPO/RNN_TORUS_${ROOT}.json" \
    --out "$REPO/RNN_TORUS_${ROOT}_STRATA.json" || drv_fail "strata $ROOT"
  for S in "${SEEDS[@]}"; do        # each run on its OWN training map (env seed = run seed), secondary S5
    EV python3 -u -m mapformer.eval_rnn_torus noise_refine --runs-dir "$R/$ROOT" --variants $VS --noises 0.0 --seeds "$S" \
      --env-seed "$S" --lengths 1024 --n-trials 100 --device "cuda:$G" --title "RNN torus $ROOT own map s$S" \
      --out "$REPO/RNN_TORUS_${ROOT}_OWN_s${S}.md" || _drv_log "own-map eval $ROOT s$S failed (secondary)"
  done
done
EV python3 -u -m mapformer.analyze_rnn_torus --context-destruction --device "cuda:$G" || _drv_log "context destruction failed (secondary S8)"
[ "$DRY" = 1 ] && { _drv_log "DRYRUN analysis: CUDA_VISIBLE_DEVICES= python3 -u -m mapformer.analyze_rnn_torus > RNN_TORUS_ANALYSIS.txt"; _drv_log "DRYRUN complete"; exit 0; }
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the analysis"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_rnn_torus > "$REPO/RNN_TORUS_ANALYSIS.txt" 2>&1 || drv_fail analyze
ART=("$REPO/RNN_TORUS_ANALYSIS.txt" "$REPO/RNN_TORUS_VERDICTS.json")
for ROOT in "${ROOTS[@]}"; do ART+=("$REPO/RNN_TORUS_${ROOT}.json" "$REPO/RNN_TORUS_${ROOT}_STRATA.json"); done
drv_done "$REPO/.rnn_torus_done" "${ART[@]}"

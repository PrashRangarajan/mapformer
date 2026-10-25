#!/usr/bin/env bash
# G3_SSM_PREREG.md batch (G-3, the sign test in a one-layer diagonal complex SSM), as amended (Amendment 1).
# NOT LAUNCHED by the preparing session. The configuration is a launch-time choice, G3_CONFIG (the presets live in
# analyze_ssm_sign.CONFIGS, read from there so driver and analysis cannot disagree):
#   minimal_h2 (default)  A at m = 32 (40 runs) + H (32) + H2 (32)                        104 runs, 18,400 run-epochs
#   core                  A at m = 32 and 8 (80) + H (32)                                 112 runs, 27,200 run-epochs
#   full                  A at m = 32 and 8 (80) + A2 (40) + B (32) + H (32) + H2 (32)    216 runs, 52,000 run-epochs
#   a2_after_minimal_h2   (Amendment 2) block A2 only (40 runs, 12,000 run-epochs) into the finished minimal_h2 root:
#                         A / H / H2 must be complete there and are never re-run; their guarded code must equal the
#                         recorded code_md5.txt except analyze_ssm_sign.py, run_ssm_sign.sh, G3_SSM_PREREG.md; A2's own
#                         config.txt and code_md5.txt live in p0/A2; outputs G3_SSM_A2_{ANALYSIS.txt,VERDICTS.json}
# Blocks: A  seeds 2100-2107 x m x arms signed abs softplus exp index, steps = moves, lr 3e-3, 300 epochs
#         A2 the same five arms at lr 1e-3, m = 32                                   B  every token steps, m = 32, 3e-3
#         H  per seed, construct_ssm_sign.py from A's signed m = 32 run (CPU), then 100-epoch continuations of the abs /
#            softplus / exp constructions and the signed control at 3e-3;           H2 the SAME constructions at 1e-3
# The first launch writes runs/ssm_sign/p0/config.txt; a relaunch with another configuration is refused.
# Seed-major order. One shared 4090 by default (DRV_GPUS=1), at most 4 of our runs per GPU (DRV_MAXPG=4; measured beside
# the other user's jobs: 4 concurrent runs 5.0-6.1 s/epoch each vs 2.1 s alone). Slots are shared with this user's other
# batches (lib_driver counts every mapformer.train_* process on a GPU).
#   cd /home/prashr/mapformer && G3_CONFIG=minimal_h2 setsid nohup bash run_ssm_sign.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/ssm_sign.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${G3_MAXPG:-4}"; DRV_MINFREE=2500; DRV_SPACING=5      # assigned after sourcing (rule 27)
DRV_GPUS_GIVEN="${DRV_GPUS+set}"                                 # Amendment 3: A2-only mode needs it explicit
export DRV_GPUS="${DRV_GPUS:-1}"
CONFIG="${G3_CONFIG:-minimal_h2}"
drv_lock "$REPO/.run_ssm_sign.lock" || exit 1
cd "$REPO/.."
read -r SIZES_STR BLOCK_A2 BLOCK_B BLOCK_H2 BASE < <(CUDA_VISIBLE_DEVICES="" python3 -c "
from mapformer.analyze_ssm_sign import CONFIGS
c = CONFIGS['$CONFIG']
print(' '.join(map(str, c['sizes'])).replace(' ', ','), int(c['a2']), int(c['b']), int(c['h2']), c.get('base', '-'))" 2>> "$LOG") ||
  { _drv_log "ABORT: unknown G3_CONFIG '$CONFIG'"; exit 1; }
[ -n "${BASE:-}" ] || { _drv_log "ABORT: unknown G3_CONFIG '$CONFIG'"; exit 1; }
IFS=, read -r -a SIZES <<< "$SIZES_STR"
R="$REPO/runs/ssm_sign/p0"
# Amendment 2: A2ONLY = block A2 added to a finished base root. CFGD holds this launch's config.txt and code_md5.txt.
A2ONLY=0; CFGD="$R"; OUTV="$REPO/G3_SSM_VERDICTS.json"; OUTA="$REPO/G3_SSM_ANALYSIS.txt"; DONE="$REPO/.ssm_sign_done"
if [ "$BASE" != "-" ]; then
  A2ONLY=1; CFGD="$R/A2"; LOG="$REPO/ssm_sign_a2.log"
  OUTV="$REPO/G3_SSM_A2_VERDICTS.json"; OUTA="$REPO/G3_SSM_A2_ANALYSIS.txt"; DONE="$REPO/.ssm_sign_a2_done"
  [ "$BLOCK_A2" = 1 ] && [ "$BLOCK_B" = 0 ] || { _drv_log "ABORT: an A2-only configuration must have a2 and not b"; exit 1; }
  [ -f "$R/config.txt" ] && [ "$(cat "$R/config.txt")" = "$BASE" ] ||
    { _drv_log "REFUSED: $R is not a finished $BASE root (config.txt: $(cat "$R/config.txt" 2>/dev/null))"; exit 1; }
  [ -f "$R/code_md5.txt" ] || { _drv_log "REFUSED: $R/code_md5.txt (block A / H / H2 provenance) missing"; exit 1; }
  # Amendment 3: the default DRV_GPUS=1 is GPU index 1; an A2-only launch must name its GPU(s)
  [ "$DRV_GPUS_GIVEN" = set ] || { _drv_log "REFUSED: A2-only mode needs an explicit DRV_GPUS (e.g. DRV_GPUS=0)"; exit 1; }
else
  mkdir -p "$R/A" "$R/H/built"
fi
if [ -f "$CFGD/config.txt" ] && [ "$(cat "$CFGD/config.txt")" != "$CONFIG" ]; then
  _drv_log "REFUSED: $CFGD was launched as $(cat "$CFGD/config.txt"), not $CONFIG"; exit 1
fi
SEEDS=(2100 2101 2102 2103 2104 2105 2106 2107)
ARMS_A=(signed abs softplus exp index)
ARMS_B=(signed abs softplus exp)
CLASSES_H=(abs softplus exp signed)
FILES=(model_ssm_sign.py train_ssm_sign.py construct_ssm_sign.py analyze_ssm_sign.py gate_ssm_sign.py test_ssm_sign.py
       run_ssm_sign.sh G3_SSM_PREREG.md)
echo "start $(date)" >> "$LOG"

clean_tree() {   # at start: tracked files unmodified AND every batch file committed (rule 7)
  (cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .) &&
    { _drv_log "REFUSED: tracked files modified (commit first)"; return 1; }
  (cd "$REPO" && git ls-files --error-unmatch "${FILES[@]}" > /dev/null 2>&1) ||
    { _drv_log "REFUSED: a G-3 file is not committed"; return 1; }
}
batch_tree() {   # mid-batch (Amendment 1.7): only the batch's own files and the guarded modules, so another session's
                 # edits elsewhere in the repo cannot fail a running batch (the md5 guard covers the code as well)
  (cd "$REPO" && git diff --quiet HEAD -- "${FILES[@]}" "${GUARD[@]}") ||
    { _drv_log "REFUSED: a batch file or guarded module differs from HEAD"; return 1; }
}
clean_tree || exit 1
# md5 guard over EVERY mapformer module the batch imports (as run_signroute.sh); data_parallel is imported lazily by train()
mapfile -t GUARD < <(cd "$REPO/.." && CUDA_VISIBLE_DEVICES="" python3 -c "
import sys
import mapformer.train_ssm_sign, mapformer.construct_ssm_sign, mapformer.data_parallel, mapformer.analyze_ssm_sign
for m in sorted(k for k, v in sys.modules.items() if k.startswith('mapformer.') and getattr(v, '__file__', None)):
    print(m.split('.', 1)[1].replace('.', '/') + '.py')
" 2>> "$LOG")
[ "${#GUARD[@]}" -ge 15 ] || { _drv_log "ABORT: module list for the md5 guard has ${#GUARD[@]} entries"; exit 1; }
GUARD+=(lib_driver.sh run_ssm_sign.sh G3_SSM_PREREG.md)
if [ "$A2ONLY" = 1 ]; then
  # A2 trains with the code that trained A / H / H2: the same guarded file set, and every file's md5 equal to the
  # recorded one except the three this amendment edits (analysis preset, driver, pre-registration)
  CUDA_VISIBLE_DEVICES="" python3 - "$REPO" "$R/code_md5.txt" "${GUARD[@]}" >> "$LOG" 2>&1 <<'PYEOF' ||
import hashlib, sys
repo, rec, files = sys.argv[1], sys.argv[2], sys.argv[3:]
old = {ln.split()[1]: ln.split()[0] for ln in open(rec) if ln.strip()}
exempt = {"analyze_ssm_sign.py", "run_ssm_sign.sh", "G3_SSM_PREREG.md"}
bad = sorted(set(old) ^ set(files))
bad += [f for f in files if f in old and f not in exempt
        and hashlib.md5(open(f"{repo}/{f}", "rb").read()).hexdigest() != old[f]]
print(f"A2 provenance: {len(files)} guarded files, {len(exempt)} exempt, mismatches {bad}")
sys.exit(1 if bad else 0)
PYEOF
    { _drv_log "REFUSED: guarded code differs from the code that trained block A / H / H2 ($R/code_md5.txt)"; exit 1; }
  BASEREQ=()
  for S in "${SEEDS[@]}"; do
    for A in "${ARMS_A[@]}"; do BASEREQ+=("$R/A/${A}_moves_m32_s$S/eval.json" "$R/A/${A}_moves_m32_s$S/${A}.pt"); done
    for C in "${CLASSES_H[@]}"; do
      BASEREQ+=("$R/H/${C}_built_moves_m32_s$S/eval.json" "$R/H/built/s$S/${C}_built.pt"
                "$R/H/${C}_built_moves_m32_s$S/${C}.pt"                         # Amendment 3 (MINOR 2)
                "$R/H2/${C}_built_moves_m32_lr1e-3_s$S/eval.json" "$R/H2/${C}_built_moves_m32_lr1e-3_s$S/${C}.pt")
    done
  done
  drv_require "${BASEREQ[@]}" || { _drv_log "REFUSED: the $BASE root is incomplete (A / H / H2 are not re-run)"; exit 1; }
fi
drv_md5_guard "$CFGD" "${GUARD[@]}" || exit 1
[ -f "$CFGD/config.txt" ] || echo "$CONFIG" > "$CFGD/config.txt"
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) clean; GPUs $DRV_GPUS max/GPU $DRV_MAXPG; config $CONFIG (sizes ${SIZES[*]}, A2 $BLOCK_A2, B $BLOCK_B, H2 $BLOCK_H2, A2-only $A2ONLY); md5 guard over ${#GUARD[@]} files in $CFGD; torch $(python3 -c 'import torch; print(torch.__version__)')"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
launch() {   # $1 out dir, $2 lr, rest = trainer args
  local OUT="$1" LR="$2"; shift 2
  [ -f "$OUT/eval.json" ] && { _drv_log "skip $OUT (done)"; return 0; }
  running "$OUT" && { _drv_log "skip $OUT (running)"; return 0; }
  local G; G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$OUT -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m mapformer.train_ssm_sign "$@" --lr "$LR" --device "cuda:$G" \
    --output-dir "$OUT"
}

# ---- stage 1 (seed-major; A before B within the batch so stage 2's sources finish first)
[ "$A2ONLY" = 1 ] || for S in "${SEEDS[@]}"; do for M in "${SIZES[@]}"; do for A in "${ARMS_A[@]}"; do
  launch "$R/A/${A}_moves_m${M}_s$S" 3e-3 --arm "$A" --steps moves --size "$M" --seed "$S"
done; done; done
if [ "$BLOCK_A2" = 1 ]; then
  for S in "${SEEDS[@]}"; do for A in "${ARMS_A[@]}"; do
    launch "$R/A2/${A}_moves_m32_lr1e-3_s$S" 1e-3 --arm "$A" --steps moves --size 32 --seed "$S"
  done; done
fi
if [ "$BLOCK_B" = 1 ]; then
  for S in "${SEEDS[@]}"; do for A in "${ARMS_B[@]}"; do
    launch "$R/B/${A}_all_m32_s$S" 3e-3 --arm "$A" --steps all --size 32 --seed "$S"
  done; done
fi

# ---- stage 2: constructions need every signed m = 32 source
batch_tree || drv_fail "tree changed during stage 1"
if [ "$A2ONLY" = 0 ]; then
sleep 30
DRV_WAIT_POLL=60 drv_wait_dir "$R/A/signed_moves_m32_"
SRC=(); for S in "${SEEDS[@]}"; do SRC+=("$R/A/signed_moves_m32_s$S/signed.pt" "$R/A/signed_moves_m32_s$S/eval.json"); done
drv_require "${SRC[@]}" || drv_fail "missing signed m = 32 sources"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the constructions"
for S in "${SEEDS[@]}"; do
  B="$R/H/built/s$S"
  if [ ! -f "$B/report.json" ]; then
    mkdir -p "$B"
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python3 -u -m mapformer.construct_ssm_sign \
      --source "$R/A/signed_moves_m32_s$S/signed.pt" --save "$B" --json "$B/report.json" > "$B/construct.log" 2>&1 ||
      drv_fail "construction s$S"
  fi
  for C in "${CLASSES_H[@]}"; do drv_require "$B/${C}_built.pt" || drv_fail "missing $B/${C}_built.pt"; done
  for C in "${CLASSES_H[@]}"; do
    launch "$R/H/${C}_built_moves_m32_s$S" 3e-3 --arm "$C" --steps moves --size 32 --seed "$S" --epochs 100 \
      --init-built "$B/${C}_built.pt"
    [ "$BLOCK_H2" = 1 ] && launch "$R/H2/${C}_built_moves_m32_lr1e-3_s$S" 1e-3 --arm "$C" --steps moves --size 32 \
      --seed "$S" --epochs 100 --init-built "$B/${C}_built.pt"
  done
done
fi   # A2ONLY: no constructions, no continuations (block H / H2 are the base root's)

# ---- wait, verify every artifact, analyse
batch_tree || drv_fail "tree changed during stage 2"
sleep 60
DRV_WAIT_POLL=60 drv_wait_dir "$R/"
REQ=()
for S in "${SEEDS[@]}"; do
  for M in "${SIZES[@]}"; do for A in "${ARMS_A[@]}"; do
    REQ+=("$R/A/${A}_moves_m${M}_s$S/eval.json" "$R/A/${A}_moves_m${M}_s$S/${A}.pt"); done; done
  if [ "$BLOCK_A2" = 1 ]; then for A in "${ARMS_A[@]}"; do
    REQ+=("$R/A2/${A}_moves_m32_lr1e-3_s$S/eval.json" "$R/A2/${A}_moves_m32_lr1e-3_s$S/${A}.pt"); done; fi
  if [ "$BLOCK_B" = 1 ]; then for A in "${ARMS_B[@]}"; do
    REQ+=("$R/B/${A}_all_m32_s$S/eval.json" "$R/B/${A}_all_m32_s$S/${A}.pt"); done; fi
  for C in "${CLASSES_H[@]}"; do
    REQ+=("$R/H/${C}_built_moves_m32_s$S/eval.json" "$R/H/${C}_built_moves_m32_s$S/${C}.pt" "$R/H/built/s$S/${C}_built.pt")
    [ "$BLOCK_H2" = 1 ] && REQ+=("$R/H2/${C}_built_moves_m32_lr1e-3_s$S/eval.json" "$R/H2/${C}_built_moves_m32_lr1e-3_s$S/${C}.pt")
  done
done
drv_require "${REQ[@]}" || drv_fail "missing runs"
NWANT=$(( ${#SEEDS[@]} * (2 * ${#SIZES[@]} * ${#ARMS_A[@]} + 3 * ${#CLASSES_H[@]}) + BLOCK_B * ${#SEEDS[@]} * 2 * ${#ARMS_B[@]} + BLOCK_A2 * ${#SEEDS[@]} * 2 * ${#ARMS_A[@]} + BLOCK_H2 * ${#SEEDS[@]} * 2 * ${#CLASSES_H[@]} ))
[ "${#REQ[@]}" -eq "$NWANT" ] || drv_fail "expected $NWANT artifacts, listed ${#REQ[@]}"
drv_md5_guard "$CFGD" "${GUARD[@]}" || drv_fail "code changed before the analysis"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_ssm_sign --config "$CONFIG" --root "$R" \
  --out "$OUTV" > "$OUTA" 2>&1 ||
  drv_fail analyze
drv_done "$DONE" "$OUTA" "$OUTV"

#!/usr/bin/env bash
# Dyck position effect at MATCHED nesting depth. Pre-registration: DYCK_MDEPTH_PREREG.md.
# Gates first (validate_dyck_mdepth, exits non-zero on failure); then, in one batch:
#   repro  2 runs  ladder defaults (L32 D4), must reproduce runs/dyck_ladder weights bitwise
#   T12x3  48 runs 4L, trained at L32 D12, 3x budget: RoPE PoPE RoPE_b32 PoPE_b32 MapWM MapPoPE  (PRIMARY)
#   T12   128 runs 1-4L, trained at L32 D12, paper budget: RoPE PoPE MapWM MapPoPE   (depth shape)
#   Tmix   32 runs 4L, trained at L32 with D uniform in {4..12} per batch             (in-support control)
# The long 3x runs go first so the primary lands first and short runs fill the slots.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/dyck_mdepth.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG="${MAXPG:-2}"; DRV_SPACING="${DRV_SPACING:-4}"; DRV_POLL="${DRV_POLL:-10}"
# count only Dyck trainers, so this batch and a concurrent train_variant batch each get their own slots
DRV_MODULE_RE="mapformer[.]train_dyck"
DRV_MINFREE="${DRV_MINFREE:-1500}"          # a 4-layer Dyck run needs well under 1 GB
drv_lock "$REPO/.run_dyck_mdepth.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/dyck_mdepth"; mkdir -p "$R"/{repro,T12x3,T12,Tmix}
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" train_dyck.py environment_dyck.py model.py model_pope.py model_baseline_rope.py \
  eval_dyck_literature.py probe_dyck_stack.py validate_dyck.py dyck_mdepth_common.py validate_dyck_mdepth.py \
  analyze_dyck_mdepth.py || exit 1

python3 -u -m mapformer.validate_dyck_mdepth --out "$REPO/DYCK_MDEPTH_GATES.md" >> "$LOG" 2>&1 \
  || drv_fail "gates (DYCK_MDEPTH_GATES.md)"

nm_of(){ local a=$1 L=$2 sfx=$3 n; n="${a%_b32}-${L}L"; case "$a" in Map*) n="${n}_r2";; esac
  case "$a" in *_b32) n="${n}_b32";; esac; echo "${n}${sfx}"; }
launch(){  # $1 sub-dir, $2 arm, $3 layers, $4 seed, $5 name suffix, rest = extra train_dyck flags
  local sub=$1 a=$2 L=$3 s=$4 sfx=$5; shift 5
  local nm; nm=$(nm_of "$a" "$L" "$sfx")
  local out="$R/$sub/${nm}_s${s}"
  [ -f "$out/${nm}.json" ] && { echo "skip $sub $nm s$s" >> "$LOG"; return; }
  local rb=(); case "$a" in *_b32) rb=(--rope-base 32);; esac
  local G; G=$(drv_wait_slot)
  OMP_NUM_THREADS=2 drv_launch "$R/$sub/${nm}_s${s}.log" python3 -u -m mapformer.train_dyck \
    --arch "${a%_b32}" --n-layers "$L" --n-heads 2 --rank 2 --seed "$s" ${rb[@]+"${rb[@]}"} "$@" \
    --device "cuda:$G" --output-dir "$out"
}
REQ=()
for a in RoPE MapWM; do launch repro "$a" 4 0 ""; REQ+=("$R/repro/$(nm_of $a 4 "")_s0/$(nm_of $a 4 "").json"); done
for s in 0 1 2 3 4 5 6 7; do
  for a in RoPE PoPE RoPE_b32 PoPE_b32 MapWM MapPoPE; do
    launch T12x3 "$a" 4 "$s" _tL32D12 --train-D 12 --n-sequences 1680000
    REQ+=("$R/T12x3/$(nm_of $a 4 _tL32D12)_s$s/$(nm_of $a 4 _tL32D12).json")
  done
done
for s in 0 1 2 3 4 5 6 7; do
  for L in 1 2 3 4; do for a in RoPE PoPE MapWM MapPoPE; do
    launch T12 "$a" "$L" "$s" _tL32D12 --train-D 12
    REQ+=("$R/T12/$(nm_of $a $L _tL32D12)_s$s/$(nm_of $a $L _tL32D12).json")
  done; done
  for a in RoPE PoPE MapWM MapPoPE; do
    launch Tmix "$a" 4 "$s" _tL32Dset4-12 --train-D-set 4,5,6,7,8,9,10,11,12
    REQ+=("$R/Tmix/$(nm_of $a 4 _tL32Dset4-12)_s$s/$(nm_of $a 4 _tL32Dset4-12).json")
  done
done
drv_wait_dir "$R/"
drv_require "${REQ[@]}" || drv_fail "missing runs (${#REQ[@]} required)"
python3 -u -m mapformer.analyze_dyck_mdepth --gates "$REPO/DYCK_MDEPTH_GATES.json" \
  --out-stem "$REPO/DYCK_MDEPTH_RESULTS" >> "$LOG" 2>&1 || drv_fail analyze
drv_done "$REPO/.dyck_mdepth_done" "$REPO/DYCK_MDEPTH_GATES.md" "$REPO/DYCK_MDEPTH_RESULTS.md" "$REPO/DYCK_MDEPTH_RESULTS.json"

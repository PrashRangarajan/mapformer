#!/usr/bin/env bash
# ACCOUNTS_PREREG.md GPU batch: T1 (weight decay, ring continuations from the WRAP_EXISTENCE sources), T2 (frozen move
# latents), T4 (CoPE on a non-wrapping line). T3 (CPU toy) runs separately (runs/accounts/t3/run_t3.sh). GPUs from DRV_GPUS.
#   cd /home/prashr/mapformer && DRV_GPUS="0 1" setsid nohup bash run_accounts.sh > /dev/null 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/accounts.log"
source "$REPO/lib_driver.sh"
DRV_MAXPG=6; DRV_MINFREE=3000; DRV_SPACING=15    # assigned after sourcing (rule 27)
export DRV_GPUS="${DRV_GPUS:-0 1}"
drv_lock "$REPO/.run_accounts.lock" || exit 1
cd "$REPO/.."
R="$REPO/runs/accounts"; mkdir -p "$R/t1" "$R/t2" "$R/t4"
GUARD=(train_wrap_exist.py train_wrap_mech.py train_ring_variant.py model_wrap_exist.py model_mono_baselines.py model_sign.py
       model_baseline_rope.py train_cancel.py environment_cancel.py environment_nd.py model.py train_variant.py lie_groups.py
       train.py data_parallel.py environment.py analyze_accounts.py analyze_wrap_mech.py stats_core.py lib_driver.sh
       run_accounts.sh rescore_hook.py)
echo "start $(date)" >> "$LOG"
drv_md5_guard "$R" "${GUARD[@]}" || exit 1
(cd "$REPO" && git status --porcelain --untracked-files=no | grep -q .) && { _drv_log "REFUSED: tracked files modified (commit first)"; exit 1; }
_drv_log "code version: $(cd "$REPO" && git rev-parse HEAD) clean; GPUs $DRV_GPUS"
running(){ ps -u "$USER" -o comm=,args= | awk -v o="--output-dir $1" '$1=="python3" && index($0" ", o" ")' | grep -q .; }
go(){  # out-dir, then the module and its args
  local OUT="$1" G; shift
  [ -f "$OUT/eval.json" ] && { _drv_log "skip $OUT (done)"; return 0; }
  running "$OUT" && { _drv_log "skip $OUT (running)"; return 0; }
  G=$(drv_wait_slot); mkdir -p "$OUT"; _drv_log "$(basename "$OUT") -> cuda:$G"
  OMP_NUM_THREADS=2 drv_launch "$OUT.log" python3 -u -m "$@" --device "cuda:$G" --output-dir "$OUT"
}
for S in 600 601 602 603 604 605 606 607; do
  SRC="$REPO/runs/wrap_existence/p0/SO_s$S/SO.pt"; [ -f "$SRC" ] || drv_fail "missing source $SRC"
  for A in AO_bal SO_bal; do
    go "$R/t1/${A}_wd_s$S" mapformer.train_wrap_exist --arm "$A" --seed "$S" --init "$SRC" --weight-decay 0.05
    go "$R/t1/${A}_wd0_s$S" mapformer.train_wrap_exist --arm "$A" --seed "$S" --init "$SRC" --weight-decay 0
  done
done
for S in 1000 1001 1002 1003 1004 1005 1006 1007; do for A in SO_same SO_absmag SO_same_fz SO_absmag_fz; do
  go "$R/t2/${A}_s$S" mapformer.train_wrap_mech --arm "$A" --seed "$S"
done; done
for S in 1100 1101 1102 1103 1104 1105 1106 1107; do
  for A in CoPEF Signed_r4 RoPE; do go "$R/t4/${A}_line_s$S" mapformer.train_ring_variant --arm "$A" --p-plus 0.5 --size 256 --seed "$S"; done
  go "$R/t4/CoPEF_ring_s$S" mapformer.train_ring_variant --arm CoPEF --p-plus 0.5 --size 32 --seed "$S"
done
sleep 60
DRV_WAIT_POLL=60 drv_wait_dir "$R/"
REQ=()
for S in 600 601 602 603 604 605 606 607; do for A in AO_bal SO_bal; do for W in wd wd0; do
  REQ+=("$R/t1/${A}_${W}_s$S/eval.json" "$R/t1/${A}_${W}_s$S/$A.pt"); done; done; done
for S in 1000 1001 1002 1003 1004 1005 1006 1007; do for A in SO_same SO_absmag SO_same_fz SO_absmag_fz; do
  REQ+=("$R/t2/${A}_s$S/eval.json" "$R/t2/${A}_s$S/$A.pt"); done; done
for S in 1100 1101 1102 1103 1104 1105 1106 1107; do
  for A in CoPEF Signed_r4 RoPE; do REQ+=("$R/t4/${A}_line_s$S/eval.json" "$R/t4/${A}_line_s$S/$A.pt"); done
  REQ+=("$R/t4/CoPEF_ring_s$S/eval.json" "$R/t4/CoPEF_ring_s$S/CoPEF.pt")
done
drv_require "${REQ[@]}" || drv_fail "missing runs"
NF=0; for f in "${REQ[@]}"; do [ -f "$f" ] && NF=$((NF + 1)); done
[ "$NF" -eq 192 ] || drv_fail "expected 192 artifacts, found $NF"
drv_md5_guard "$R" "${GUARD[@]}" || drv_fail "code changed before the analysis"
CUDA_VISIBLE_DEVICES="" python3 -u -m mapformer.analyze_accounts > "$REPO/ACCOUNTS_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.accounts_done" "$REPO/ACCOUNTS_ANALYSIS.txt" "$REPO/ACCOUNTS_VERDICTS.json"

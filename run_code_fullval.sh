#!/usr/bin/env bash
# GUARD (2026-10-29 disk cleanup): the checkpoints of runs this driver writes or reads were
# deleted; it would retrain those runs or overwrite tracked results. Losses/configs survive in
# runs/<dir>/CKPT_META.jsonl. See docs/cleanup/DISK_TIERS_ANALYSIS.md section 4, D2_MANIFEST.md.
if [ "${MAPFORMER_ALLOW_DELETED_RERUN:-0}" != "1" ]; then
  echo "REFUSED: $(basename "$0"): checkpoints of its runs were deleted on 2026-10-29 (docs/cleanup/DISK_TIERS_ANALYSIS.md, section 4); relaunching would retrain them or overwrite tracked results. Set MAPFORMER_ALLOW_DELETED_RERUN=1 to run anyway (rule 30: reproduce archived batches from their pre-registration commit)." >&2
  exit 3
fi
# Full-val rescoring of the code checkpoints. Pre-registration: CODE_FULLVAL_PREREG.md.
set -uo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="$REPO/code_fullval.log"
source "$REPO/lib_driver.sh"
drv_lock "$REPO/.run_code_fullval.lock" || exit 1
cd "$REPO/.."
G="${GPU:-1}"
echo "start $(date)" >> "$LOG"
for spec in "code2048 2048" "code_decay 512" "code 512"; do
  set -- $spec
  python3 -u -m mapformer.eval_code_long --runs-dir "$REPO/runs/$1" --pattern "*.final" --seq-len "$2" \
    --device "cuda:$G" --out "$REPO/runs/$1/FULLVAL_$2.json" >> "$LOG" 2>&1 || drv_fail "eval $1"
done
python3 -u -m mapformer.analyze_code_fullval > "$REPO/CODE_FULLVAL_ANALYSIS.txt" 2>&1 || drv_fail analyze
drv_done "$REPO/.code_fullval_done" "$REPO/runs/code2048/FULLVAL_2048.json" "$REPO/runs/code_decay/FULLVAL_512.json" \
  "$REPO/runs/code/FULLVAL_512.json" "$REPO/CODE_FULLVAL_ANALYSIS.txt"

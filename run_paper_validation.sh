#!/usr/bin/env bash
# KEYSTONE: does our reimplementation reproduce the PAPER's own result?
# Paper task = aliased-observation revisit on the torus (environment.py).
# Paper config = 1 layer, 2 heads, d=128, batch 128, T=128, 200K sequences
# (16 epochs x 98 batches x 128). CLAUDE.md claims WM 0.955 / EM 0.999 but the
# figures_v6/ evidence is gitignored and absent from this clone.
#   Vanilla         = MapFormer-WM  (reference: claimed 0.955)
#   VanillaEM       = MapFormer-EM, our original (Hadamard on signed logits)
#   VanillaEM_Fixed = MapFormer-EM, paper eq.13 (Hadamard on probabilities)
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/paper_validation.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
VS=(Vanilla VanillaEM VanillaEM_Fixed)
JOBS=(); for s in 0 1 2; do for v in "${VS[@]}"; do JOBS+=("$v:$s"); done; done
run(){ local g=$1 j=$2; IFS=: read -r v s <<<"$j"
  local o="$REPO/runs/paper_task/${v}_s${s}"
  [ -f "$o/${v}.pt" ] && { echo "$(date +%H:%M) skip $j" >> "$LOG"; return; }
  echo "$(date +%H:%M) [gpu$g] $v seed=$s" >> "$LOG"
  python3 -u -m mapformer.train_variant --variant "$v" --seed "$s" \
    --epochs 16 --n-batches 98 --batch-size 128 --n-steps 128 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
    --device "cuda:$g" --output-dir "$o" >> "$LOG" 2>&1; }
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 0 ] && run 0 "${JOBS[$i]}"; done; echo "$(date +%H:%M) GPU0 DONE">>"$LOG" ) & P0=$!
( for i in "${!JOBS[@]}"; do [ $((i%2)) -eq 1 ] && run 1 "${JOBS[$i]}"; done; echo "$(date +%H:%M) GPU1 DONE">>"$LOG" ) & P1=$!
wait $P0 $P1
echo "$(date +%H:%M) ALL TRAINED" >> "$LOG"
{ echo "# Paper-task validation: does our reimplementation reproduce MapFormer?"
  echo ""
  echo "Paper task (aliased-obs revisit, torus), paper config: 1 layer, 2 heads,"
  echo "d=128, batch 128, T=128, 200K sequences. CLAIMED in CLAUDE.md: WM 0.955 / EM 0.999."
  echo ""
  echo "| variant | seed | final train loss | held-out revisit acc |"
  echo "|---|---|---|---|"
  for v in "${VS[@]}"; do for s in 0 1 2; do
    f="$REPO/runs/paper_task/${v}_s${s}/${v}.pt"
    [ -f "$f" ] && python3 -c "
import torch,sys
c=torch.load('$f',map_location='cpu',weights_only=False)
L=c.get('losses') or c.get('train_losses') or []
acc=c.get('test_acc', c.get('final_acc','n/a'))
print(f\"| $v | $s | {L[-1]:.4f} | {acc if isinstance(acc,str) else f'{acc:.4f}'} |\")" 2>/dev/null
  done; done; } > "$REPO/PAPER_VALIDATION.md"
cd "$REPO"; git add run_paper_validation.sh PAPER_VALIDATION.md 2>/dev/null
git diff --cached --quiet || { git commit -q -m "Paper-task validation: run WM/EM/EM_Fixed on the paper's own task

Keystone check -- CLAUDE.md claims WM 0.955 / EM 0.999 but figures_v6/ is
gitignored and absent from this clone, and no results file records the number.
Runs all three at the paper's config (1 layer, 2 heads, d=128, 200K sequences).
Auto-committed; interpretation pending review."; git push origin main >> "$LOG" 2>&1; }
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.paper_val_done"

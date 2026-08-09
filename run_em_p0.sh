#!/usr/bin/env bash
# EM corrected to paper eq.3: SINGLE origin p_0, A_P = P.P^T.
# Tests on BOTH the paper's own task (does it reproduce?) and compositional.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/em_p0.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
V=VanillaEM_P0
pt(){ local g=$1 s=$2 o="$REPO/runs/paper_task/${V}_s$2"
  [ -f "$o/${V}.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] paper s$s" >> "$LOG"
  python3 -u -m mapformer.train_variant --variant "$V" --seed "$s" \
    --epochs 16 --n-batches 98 --batch-size 128 --n-steps 128 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 \
    --device "cuda:$g" --output-dir "$o" >> "$LOG" 2>&1; }
cp(){ local g=$1 s=$2 o="$REPO/runs/comp_multiseed/seed$2"
  [ -f "$o/${V}.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] comp s$s" >> "$LOG"
  python3 -u -m mapformer.train_compositional --variant "$V" --target motif --n-steps 256 \
    --epochs 50 --n-batches 156 --n-layers 3 --seed "$s" --device "cuda:$g" --output-dir "$o" >> "$LOG" 2>&1; }
( pt 0 0; pt 0 2; cp 0 0; cp 0 2; echo "$(date +%H:%M) GPU0 DONE" >> "$LOG" ) & P0=$!
( pt 1 1; cp 1 1;                 echo "$(date +%H:%M) GPU1 DONE" >> "$LOG" ) & P1=$!
wait $P0 $P1
{ echo "# EM eq.3 correction (single origin p_0): does it reproduce the paper?"; echo ""
  echo "| variant | seed | paper-task final loss |"; echo "|---|---|---|"
  for v in Vanilla VanillaEM VanillaEM_P0; do for s in 0 1 2; do
    f="$REPO/runs/paper_task/${v}_s${s}/${v}.pt"
    [ -f "$f" ] && python3 -c "
import torch;c=torch.load('$f',map_location='cpu',weights_only=False);L=c.get('losses') or c.get('train_losses') or []
print(f'| $v | $s | {L[-1]:.4f} |')" 2>/dev/null; done; done; } > "$REPO/EM_P0_PAPER.md"
python3 -u -m mapformer.agg_comp_multiseed --runs-dir "$REPO/runs/comp_multiseed" --seeds 0 1 2 \
  --variants VanillaEM VanillaEM_P0 Vanilla Hourglass_k2 PlainFlat \
  --lengths 256 512 1024 2048 --n-traj 200 --batch 16 --device cuda:0 \
  --out "$REPO/EM_P0_COMP.md" >> "$LOG" 2>&1
cd "$REPO"; git add model_em_fixed.py train_variant.py run_em_p0.sh EM_P0_PAPER.md EM_P0_COMP.md EM_P0_COMP.json 2>/dev/null
git diff --cached --quiet || { git commit -q -m "EM eq.3 correction: single origin p_0 (A_P = P.P^T)

RETRACTS the earlier softmax 'fix'. Read the paper directly: eq.3 is
softmax(A_X o A_P)V on RAW scaled scores -- our original layer was already
correct. The real deviation is that the paper uses a SINGLE learned origin p_0
on both sides (A_P = P.P^T, an autocorrelation kernel), while we used separate
q0_pos/k0_pos. Measured on a trained VanillaEM, zero-displacement A_P was
NEGATIVE on 50% of positions and the row max on only 16% of rows (q0,k0 had
learned to be nearly anti-aligned, cos=-0.73) -- the model scored 'same place'
as a mismatch. With a single p_0: 0% negative, row max on 100%.
Auto-committed; interpretation pending review."; git push origin main >> "$LOG" 2>&1; }
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.em_p0_done"

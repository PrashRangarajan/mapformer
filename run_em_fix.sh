#!/usr/bin/env bash
# A/B the EM fix: Hadamard on PROBABILITIES (paper eq.13) vs on raw signed
# logits (our original). Compositional is the decisive test -- the original
# scored 0.097 there, worst of all variants, which the sign pathology predicts.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO/.."
LOG="$REPO/em_fix.log"; : > "$LOG"; echo "start $(date)" >> "$LOG"
V=VanillaEM_Fixed
cp(){ local g=$1 s=$2 o="$REPO/runs/comp_multiseed/seed$2"
  [ -f "$o/${V}.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] comp s$s" >> "$LOG"
  python3 -u -m mapformer.train_compositional --variant "$V" --target motif --n-steps 256 \
    --epochs 50 --n-batches 156 --n-layers 3 --seed "$s" --device cuda:$g --output-dir "$o" >> "$LOG" 2>&1; }
hg(){ local g=$1 s=$2 o="$REPO/runs/hiergoal_multiseed/seed$2"
  [ -f "$o/${V}_hiergoal.pt" ] && return
  echo "$(date +%H:%M) [gpu$g] hg s$s" >> "$LOG"
  python3 -u -m mapformer.train_hier_goal --variant "$V" --seed "$s" --epochs 25 --n-batches 64 \
    --T-explore 64 --T-navigate 64 --eval-explore 64 128 192 256 --n-layers 3 \
    --device cuda:$g --output-dir "$o" >> "$LOG" 2>&1; }
( cp 0 0; cp 0 2; hg 0 0; hg 0 2; echo "$(date +%H:%M) GPU0 DONE" >> "$LOG" ) & P0=$!
( cp 1 1; hg 1 1;                 echo "$(date +%H:%M) GPU1 DONE" >> "$LOG" ) & P1=$!
wait $P0 $P1
echo "$(date +%H:%M) aggregating" >> "$LOG"
python3 -u -m mapformer.agg_comp_multiseed --runs-dir "$REPO/runs/comp_multiseed" --seeds 0 1 2 \
  --variants VanillaEM VanillaEM_Fixed Vanilla Hourglass_k2 PlainFlat \
  --lengths 256 512 1024 2048 --n-traj 200 --batch 16 --device cuda:0 \
  --out "$REPO/EM_FIX_COMP.md" >> "$LOG" 2>&1
python3 -u -m mapformer.agg_hiergoal --runs-dir "$REPO/runs/hiergoal_multiseed" --seeds 0 1 2 \
  --variants VanillaEM_Fixed Vanilla Hourglass_k2 PlainFlat \
  --lengths 64 128 192 256 --out "$REPO/EM_FIX_HIERGOAL.md" >> "$LOG" 2>&1
cd "$REPO"; git add model_em_fixed.py train_variant.py run_em_fix.sh EM_FIX_COMP.md EM_FIX_HIERGOAL.md \
  EM_FIX_COMP.json EM_FIX_HIERGOAL.json 2>/dev/null
git diff --cached --quiet || { git commit -q -m "Fix MapFormer-EM: Hadamard on probabilities (paper eq.13), not signed logits

Our EMTransformerLayer computed softmax(A_X_logits o A_P_logits). The paper's
eq.13 is (Att(Q,K) o Att(Q_P,K_P))V, where Att() includes the softmax -- so both
are attention matrices and A_P is a mask in [0,1]. Multiplying signed logits
makes the gate an XNOR: measured on a trained VanillaEM, 35.5% of causal pairs
had A_X<0 AND A_P<0, and 69.9% of positive scores came from double-mismatches,
while content-match/position-mismatch (cross-instance retrieval) was driven
negative. Adds MapFormerEM_Fixed; param-identical, causal, A_P now 0% negative.
Auto-committed; interpretation pending review."; git push origin main >> "$LOG" 2>&1; }
echo "$(date +%H:%M) DONE" >> "$LOG"; touch "$REPO/.em_fix_done"

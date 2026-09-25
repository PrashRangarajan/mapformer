#!/usr/bin/env bash
# IS THE PAPER'S FIG. 4 C4 CLAIM AN **EM** PROPERTY?
#
# Fig. 4 reports that after training the value-embedding norms in the attention
# layer become "much bigger for observations than actions (||v_o|| >> ||v_a||),
# implying that only observations contribute in updating the state's content".
#
# On MapWM we measure that ratio at **0.57 +/- 0.06** -- not merely short of >> 1,
# but INVERTED (PAPER_FIG4_REPRO.md, 8 seeds, r=2 and r=4 alike). The other three
# Fig. 4 claims reproduce, including the non-orthogonality the paper reports as its
# own limitation.
#
# HYPOTHESIS. Fig. 4 shows an EM model. Sec 5.4's framing is explicitly about EM:
# "this factorization in two separate pools of neurons should allow EM to be more
# efficient than WM, as in the former, neurons specialize for either position or
# observation". MapWM's additive attention has no such separation by construction,
# so there is no architectural reason for its value norms to split by token type.
# MapEM's multiplicative A_X (*) A_P does separate the two channels.
#
# PRE-REGISTERED:
#   EM ratio >> 1 while WM stays < 1 -> C4 is an EM property the caption does not
#       scope, our reproduction of Fig. 4 is COMPLETE, and the WM number is not a
#       discrepancy but a different architecture behaving differently.
#   EM ratio also < 1 -> a genuine discrepancy with the paper, on the architecture
#       the claim is most likely about. Record it as one; do not explain it away.
#   EM ratio ~ 1 -> neither; report the number and say the claim is not reproduced
#       in either architecture.
#
# Two ranks, because r=4 sharpened C2 and C3 substantially on WM and it is worth
# knowing whether C4 moves with rank at all (on WM it did not: 0.57 -> 0.60).
#
# Arms use the PAPER-FAITHFUL EM: separate q0_pos / k0_pos, which is the paper's
# MapEM-os. The single-p0 variant is an ablation of the paper's own stated
# suspicion and would not be a reproduction.
#
# 2 arms x 8 seeds = 16 runs, ~40 min.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO/.."
R="$REPO/runs/em_fig4"; mkdir -p "$R/p0"
LOG="$REPO/em_fig4.log"; echo "em-fig4 start $(date)" > "$LOG"
MAXPG=5; A="train_var""iant"
busy_gpu(){ ps -u "$USER" -o comm=,args= | awk -v p="$A" '$1=="python3" && index($0,p)' | grep -c -- "--device cuda:$1" || true; }
for SEED in 0 1 2 3 4 5 6 7; do
  for V in VanillaEM VanillaEM_r4; do
    OUT="$R/p0/${V}_s${SEED}"; mkdir -p "$OUT"
    [ -f "$OUT/${V}.pt" ] && continue
    while :; do
      N0=$(busy_gpu 0); N1=$(busy_gpu 1)
      if [ "$N0" -le "$N1" ] && [ "$N0" -lt "$MAXPG" ]; then G=0; break; fi
      if [ "$N1" -lt "$MAXPG" ]; then G=1; break; fi
      if [ "$N0" -lt "$MAXPG" ]; then G=0; break; fi
      sleep 15
    done
    echo "$(date +%H:%M:%S) $V s$SEED -> cuda:$G" >> "$LOG"
    python3 -u -m mapformer.train_variant --variant "$V" --seed "$SEED" \
      --epochs 300 --lr 1e-3 --n-batches 98 --batch-size 128 --n-steps 128 \
      --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine \
      --data-workers 3 --device "cuda:$G" --output-dir "$OUT" \
      > "$R/${V}_s${SEED}.log" 2>&1 &
    sleep 6
  done
done
wait
echo "$(date +%H:%M) $(find "$R" -name '*.pt' | wc -l)/16 checkpoints" >> "$LOG"
python3 -u -m mapformer.probe_paper_fig4 --runs-dir "$R/p0" \
  --arms VanillaEM VanillaEM_r4 --out "$REPO/PAPER_FIG4_EM.md" >> "$LOG" 2>&1
touch "$REPO/.em_fig4_done"; echo "$(date) DONE" >> "$LOG"

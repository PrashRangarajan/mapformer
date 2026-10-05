#!/usr/bin/env bash
# Re-score of the torus batches (eval_noise_refine, registered streams) under rescore_hook: scale none (must reproduce the
# registered JSON) and scale auto (attention x 1/(1-p)). Matched length only (paper2x2 128, the rest 1024).
set -uo pipefail
REPO=/home/prashr/mapformer; O=$REPO/runs_rescore; mkdir -p $O; cd /home/prashr
run() {  # name runs-dir length variants...
  local n=$1 d=$2 L=$3; shift 3
  for sc in none auto; do
    python3 -m mapformer.rescore_hook --scale $sc -- mapformer.eval_noise_refine --runs-dir $d --variants "$@" --noises 0.0 \
      --seeds 0 1 2 3 4 5 6 7 --lengths $L --n-trials 100 --device cuda:0 --out $O/${n}_${sc}.md > $O/${n}_${sc}.log 2>&1 \
      || echo "FAILED $n $sc"
  done
}
run PAPER2X2 $REPO/runs/paper2x2 128 $(python3 -c "print(' '.join(__import__('json').load(open('$REPO/_PAPER2X2_RAW.json')).keys()))" | tr ' ' '\n' | cut -d'|' -f2 | sort -u | tr '\n' ' ')
run RANK_MI $REPO/runs/rank_mi 1024 Vanilla Vanilla_r2ph Vanilla_r4mi
run RANK_SEP $REPO/runs/rank_sep 1024 Vanilla_r4mibd Vanilla_r4ph
run RANK3 $REPO/runs/rank3 1024 Vanilla_r3ph
run LOOP_RANK $REPO/runs/loop_rank 1024 Looped Vanilla_L4
run LOOP_RANK_E1800_P1 $REPO/runs/loop_rank_e1800 1024 Vanilla Vanilla_r4mi
run SIGN_MATCHED $REPO/runs/sign_matched 1024 Signed_r4 Abs_r4 Pos_r4 RoPE
echo done

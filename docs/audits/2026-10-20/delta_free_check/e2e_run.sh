#!/usr/bin/env bash
S=/tmp/claude-1002/-home-prashr-mapformer/224d7c23-b570-4513-8742-cf8a41acc4a3/scratchpad/delta_free_check/e2e
cd /home/prashr
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
ARMS="PoPE-Flat PoPE-FreeDelta MapPoPE-Flat MapPoPE-FreeDelta MapPoPE-FrozenDelta"
for s in 1670 1671 1672 1673 1674 1675 1676 1677; do for v in $ARMS; do
  python3 -m mapformer.train_variant --variant $v --seed $s --epochs 2 --n-batches 2 --batch-size 8 --n-steps 32 \
    --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine --lr 1e-3 --device cpu \
    --output-dir $S/p0/${v}_s$s > $S/${v}_s$s.log 2>&1 &
done; wait; done
python3 -m mapformer.eval_noise_refine --runs-dir $S --variants $ARMS --noises 0.0 --seeds 1670 1671 1672 1673 1674 1675 1676 1677 \
  --lengths 128 512 1024 --n-trials 3 --device cpu --out $S/RAW.md --title e2e > $S/eval.log 2>&1
python3 -m mapformer.analyze_delta_free --root $S --raw $S/RAW.json --seeds 1670-1677 --floors none --probe-trials 3 \
  --json-out $S/V.json > $S/ANALYSIS.txt 2>&1
echo $? > $S/.done

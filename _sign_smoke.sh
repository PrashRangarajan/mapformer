#!/usr/bin/env bash
# GATE: every arm trains, and the constraint survives training.
set -u; cd /home/prashr
R=/home/prashr/mapformer/runs/_sign_smoke; rm -rf "$R"; mkdir -p "$R"
for V in Vanilla_r4 Signed_r4 Abs_r4 Pos_r4 CARoPE_r4 RoPE; do
  python3 -u -m mapformer.train_variant --variant "$V" --seed 0 \
    --epochs 8 --n-batches 40 --batch-size 128 --n-steps 128 --n-layers 1 \
    --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine --lr 1e-3 \
    --device cuda:0 --output-dir "$R/$V" > "$R/$V.log" 2>&1
  echo "$V  final_loss=$(grep -oE 'loss[= ]+[0-9.]+' "$R/$V.log" | tail -1)"
done
touch "$R/.done"

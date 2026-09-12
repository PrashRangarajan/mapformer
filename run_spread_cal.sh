#!/usr/bin/env bash
# Calibration pilot for the off-ceiling SPREAD re-run: at what budget does the m=4 arm land
# near 0.8 instead of 1.000? SPREAD_RESULTS.md: m4 at 300 ep is 1.000 on 8/8 (ceiling), which
# is why the exposure-matched contrast could not fire.
set -u
REPO=/home/prashr/mapformer; cd /home/prashr
OUT=$REPO/runs/spread_cal; mkdir -p "$OUT/logs"
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_recency/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); [ "$a" -le "$b" ] && echo 0 || echo 1; }
for ep in 20 40 80; do for s in 0 1; do
  while [ "$(busy)" -ge 6 ]; do sleep 10; done
  g=$(pick); echo "launch m4 e$ep s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency --variant VanillaEM_P0_r4 --seed "$s" \
    --k-max 64 --k-set "1,4,16,64" --T 1024 --eval-T 1024 --epochs "$ep" --n-batches 48 \
    --batch-size 16 --schedule cosine --lr 1e-3 --n-layers 1 --d-model 128 --n-heads 2 \
    --device "cuda:$g" --output-dir "$OUT/m4_e${ep}_s${s}" > "$OUT/logs/m4_e${ep}_s${s}.log" 2>&1 &
  sleep 5
done; done
while [ "$(busy)" -gt 0 ]; do sleep 15; done
python3 - <<'PY'
import json, glob, numpy as np, re
for ep in (20, 40, 80):
    a = []
    for f in sorted(glob.glob(f"/home/prashr/mapformer/runs/spread_cal/m4_e{ep}_s*/VanillaEM_P0_r4_recency.json")):
        r = json.load(open(f))["1024"]
        pk = {int(k): v for k, v in r["per_k"].items() if v is not None}
        a.append(np.mean([pk[k] for k in (4, 16, 64) if k in pk]))
    print(f"m4 at {ep:3d} epochs: primary {[round(x,3) for x in a]}  mean {np.mean(a):.3f}")
PY
touch "$OUT/.done"; echo "calibration finished $(date)"

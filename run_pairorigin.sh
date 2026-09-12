#!/usr/bin/env bash
# PAIRORIGIN_PREREG.md: does PER-PAIR kernel freedom recover EM's varying-k deficit?
# 3 arms x 8 seeds, ONE batch: EMPair_r4 vs VanillaEM_P0_r4 vs Vanilla_r4 (WM).
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
OUT=$REPO/runs/pairorigin
mkdir -p "$OUT/logs"
MAXPG=4
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /mapformer\.train_recency/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
go(){ local v=$1 s=$2
  [ -f "$OUT/${v}_s${s}/${v}_recency.json" ] && { echo "skip $v s$s"; return; }
  while [ "$(busy)" -ge $((2*MAXPG)) ]; do sleep 20; done
  g=$(pick); echo "launch $v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency \
    --variant "$v" --seed "$s" --k-max 64 --T 1024 --eval-T 1024 2048 \
    --epochs 300 --n-batches 48 --batch-size 16 \
    --schedule cosine --lr 1e-3 --n-layers 1 --d-model 128 --n-heads 2 \
    --device "cuda:$g" --output-dir "$OUT/${v}_s${s}" \
    > "$OUT/logs/${v}_s${s}.log" 2>&1 &
  sleep 8; }
# seed outer, arm inner: a full one-seed table lands first
for s in $(seq 0 7); do
  go EMPair_r4 "$s"
  go VanillaEM_P0_r4 "$s"
  go Vanilla_r4 "$s"
done
while [ "$(busy)" -gt 0 ]; do sleep 30; done
python3 - <<'PY' > "$OUT/MANIPULATION.txt" 2>&1
# The checks PAIRORIGIN_PREREG.md requires before its verdicts may be read.
import torch
from mapformer.ckpt_guard import load_checkpoint, REPO
from mapformer.train_variant import VARIANT_MAP

def build(name, seed):
    torch.manual_seed(seed)
    return VARIANT_MAP[name](vocab_size=89, d_model=128, n_heads=2, n_layers=1, grid_size=64).eval()

a, b = build("VanillaEM_P0_r4", 0), build("EMPair_r4", 0)
torch.manual_seed(1); x = torch.randint(0, 89, (4, 256))
with torch.no_grad():
    d = (a(x) - b(x)).abs().max().item()
print(f"CHECK 1 same function at init: max|logit diff| {d:.3e} -> {'PASS' if d < 1e-6 else 'FAIL'}")

moved = []
for s in range(8):
    ck = load_checkpoint(REPO / f"runs/pairorigin/EMPair_r4_s{s}/EMPair_r4_recency.pt")
    init = build("EMPair_r4", s).state_dict()
    for k in ("q_origin_out.weight", "k_origin_out.weight"):
        moved.append(float((ck.state[k] - init[k]).abs().max()))
print(f"CHECK 2 origin pathway moved: min over 16 tensors {min(moved):.3e}, "
      f"max {max(moved):.3e} -> {'PASS' if min(moved) > 1e-6 else 'FAIL -- parameter-count control only'}")
PY
missing=0
for v in EMPair_r4 VanillaEM_P0_r4 Vanilla_r4; do for s in $(seq 0 7); do
  [ -f "$OUT/${v}_s${s}/${v}_recency.json" ] || { echo "MISSING $v s$s"; missing=$((missing+1)); }
done; done
echo "missing=$missing"; cat "$OUT/MANIPULATION.txt"
[ "$missing" -eq 0 ] && touch "$OUT/.done"
echo "batch finished $(date)"

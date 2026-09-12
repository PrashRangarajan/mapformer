#!/usr/bin/env bash
# PAIRCONST_PREREG.md: is PAIRORIGIN's +0.280 per-pair freedom, or 2,048 parameters?
# Gated behind SPREAD2 so the batches run in sequence. Counts only its own runs.
set -u
REPO=/home/prashr/mapformer
cd /home/prashr
OUT=$REPO/runs/pairconst
mkdir -p "$OUT/logs"
MAXPG=4
mine(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_recency/ && /runs\/pairconst/' | wc -l; }
others(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_/ && !/runs\/pairconst/' | wc -l; }
on_gpu(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" '$1=="python3" && /runs\/pairconst/ && index($0,g)' | wc -l; }
pick(){ local a b; a=$(on_gpu 0); b=$(on_gpu 1); if [ "$a" -le "$b" ]; then echo 0; else echo 1; fi; }
go(){ local v=$1 s=$2 dir=$3
  [ -f "$dir/${v}_s${s}/${v}_recency.json" ] && { echo "skip $v s$s"; return; }
  while [ "$(mine)" -ge $((2*MAXPG)) ]; do sleep 20; done
  g=$(pick); echo "launch $v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency \
    --variant "$v" --seed "$s" --k-max 64 --T 1024 --eval-T 1024 2048 \
    --epochs 300 --n-batches 48 --batch-size 16 \
    --schedule cosine --lr 1e-3 --n-layers 1 --d-model 128 --n-heads 2 \
    --device "cuda:$g" --output-dir "$dir/${v}_s${s}" \
    > "$dir/logs/${v}_s${s}.log" 2>&1 &
  sleep 8; }

echo "$(date +%H:%M) waiting for SPREAD2"
until [ -f "$REPO/runs/spread2/.done" ] || [ "$(others)" -eq 0 ]; do sleep 60; done
echo "$(date +%H:%M) GPUs free; starting"

go EMPair_r4 0 "$OUT"                      # determinism re-check, first so a failure is early
for s in $(seq 0 7); do go EMPairConst_r4 "$s" "$OUT"; done
while [ "$(mine)" -gt 0 ]; do sleep 30; done

python3 - <<'PY' > "$OUT/MANIPULATION.txt" 2>&1
import torch
from mapformer.ckpt_guard import REPO, compare_checkpoints, load_checkpoint
from mapformer.train_variant import VARIANT_MAP
print(compare_checkpoints("runs/pairconst/EMPair_r4_s0/EMPair_r4_recency.pt",
                          "runs/pairorigin/EMPair_r4_s0/EMPair_r4_recency.pt").report())
# origins must be CONSTANT across tokens in EMPairConst and vary in EMPair
def spread(variant, path, s):
    ck = load_checkpoint(REPO / path)
    torch.manual_seed(s)
    m = VARIANT_MAP[variant](vocab_size=89, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    m.load_state_dict(ck.state); m.eval()
    x = torch.randint(0, 89, (2, 128))
    with torch.no_grad():
        q, _ = m._origins(m.token_emb(x))
    return float(q.std(dim=2).mean())
c = spread("EMPairConst_r4", "runs/pairconst/EMPairConst_r4_s0/EMPairConst_r4_recency.pt", 0)
p = spread("EMPair_r4", "runs/pairorigin/EMPair_r4_s0/EMPair_r4_recency.pt", 0)
print(f"origin spread across tokens: EMPairConst {c:.3e} (must be 0), EMPair {p:.3e} (must be > 0)"
      f" -> {'PASS' if c < 1e-9 < p else 'FAIL'}")
PY
missing=0
for s in $(seq 0 7); do [ -f "$OUT/EMPairConst_r4_s${s}/EMPairConst_r4_recency.json" ] || { echo "MISSING s$s"; missing=$((missing+1)); }; done
echo "missing=$missing"; cat "$OUT/MANIPULATION.txt"
[ "$missing" -eq 0 ] && touch "$OUT/.done"
echo "batch finished $(date)"

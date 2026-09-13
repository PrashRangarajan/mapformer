#!/usr/bin/env bash
# MONOTONE_PREREG.md: the sign ablation on MapFormer-EM (recency) and on Selective RoPE's
# generator (torus + recency). 96 training runs, one batch. Recipes copied from
# run_pairorigin.sh (recency) and run_sign.sh (torus).
set -u
REPO=/home/prashr/mapformer
cd /home/prashr                      # python3 -m mapformer.X must run from the parent
OUT=$REPO/runs/monotone
mkdir -p "$OUT/recency" "$OUT/torus/p0" "$OUT/torus_repro/p0" "$OUT/logs"
CAP=8                                # load units per GPU: recency run = 1, torus run = 2

cnt(){ ps -u "$USER" -o comm=,args= | awk -v g="cuda:$1" -v t="$2" '$1=="python3" && $0 ~ ("mapformer\\." t) && index($0,g)' | wc -l; }
load(){ echo $(( $(cnt "$1" train_recency) + 2*$(cnt "$1" train_variant) )); }
busy(){ ps -u "$USER" -o comm=,args= | awk '$1=="python3" && /mapformer\.train_(recency|variant)/' | wc -l; }
slot(){ local need=$1 a b
  while :; do a=$(load 0); b=$(load 1)
    if [ "$a" -le "$b" ] && [ $((a+need)) -le $CAP ]; then echo 0; return; fi
    if [ $((b+need)) -le $CAP ]; then echo 1; return; fi
    if [ $((a+need)) -le $CAP ]; then echo 0; return; fi
    sleep 20; done; }

echo "$(date +%H:%M) M-C2 construction check"
python3 - <<'PY' || { echo "construction check FAILED -- not launching"; exit 1; }
import torch
from mapformer.train_variant import VARIANT_MAP as V
from mapformer.ckpt_guard import assert_same_function_at_init
def b(n, s):
    torch.manual_seed(s); return V[n](vocab_size=89, d_model=128, n_heads=2, n_layers=1, grid_size=64)
x = torch.randint(0, 89, (4, 300))
for s in (0, 11):
    assert_same_function_at_init(b("VanillaEM_P0_r4", s), b("EM_P0_Signed_r4", s), x)
    for p, c in (("VanillaEM_P0_r4", "EM_P0_Abs_r4"), ("SRoPEGen", "SRoPEGen_Abs")):
        A, B = b(p, s).state_dict(), b(c, s).state_dict()
        assert A.keys() == B.keys() and all(torch.equal(A[k], B[k]) for k in A), (p, c, s)
print("M-C2 PASS: signed twin bitwise equal; constrained arms share every parameter at init")
PY

rec(){ local v=$1 s=$2
  [ -f "$OUT/recency/${v}_s${s}/${v}_recency.json" ] && { echo "skip rec $v s$s"; return; }
  local g; g=$(slot 1); echo "$(date +%H:%M) rec $v s$s -> cuda:$g"
  setsid nohup python3 -u -m mapformer.train_recency \
    --variant "$v" --seed "$s" --k-max 64 --T 1024 --eval-T 1024 2048 \
    --epochs 300 --n-batches 48 --batch-size 16 \
    --schedule cosine --lr 1e-3 --n-layers 1 --d-model 128 --n-heads 2 \
    --device "cuda:$g" --output-dir "$OUT/recency/${v}_s${s}" \
    > "$OUT/logs/rec_${v}_s${s}.log" 2>&1 &
  sleep 8; }
tor(){ local v=$1 s=$2 dir=$3
  [ -f "$dir/p0/${v}_s${s}/${v}.pt" ] && { echo "skip tor $v s$s"; return; }
  local g; g=$(slot 2); echo "$(date +%H:%M) tor $v s$s ($dir) -> cuda:$g"
  mkdir -p "$dir/p0/${v}_s${s}"
  setsid nohup python3 -u -m mapformer.train_variant --variant "$v" --seed "$s" \
    --epochs 300 --n-batches 98 --batch-size 128 --n-steps 128 --n-layers 1 \
    --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine --lr 1e-3 \
    --device "cuda:$g" --output-dir "$dir/p0/${v}_s${s}" \
    > "$dir/${v}_s${s}.log" 2>&1 &
  sleep 8; }

tor Signed_r4 0 "$OUT/torus_repro"
tor Abs_r4 0 "$OUT/torus_repro"
# seed outer, arm inner: a complete one-seed table lands first
for s in $(seq 0 11); do
  tor SRoPEGen "$s" "$OUT/torus"; tor SRoPEGen_Abs "$s" "$OUT/torus"
  for v in VanillaEM_P0_r4 EM_P0_Abs_r4 Signed_r4 Abs_r4 SRoPEGen SRoPEGen_Abs; do rec "$v" "$s"; done
done
while [ "$(busy)" -gt 0 ]; do sleep 30; done
echo "$(date +%H:%M) training finished"

missing=0
for v in VanillaEM_P0_r4 EM_P0_Abs_r4 Signed_r4 Abs_r4 SRoPEGen SRoPEGen_Abs; do for s in $(seq 0 11); do
  [ -f "$OUT/recency/${v}_s${s}/${v}_recency.json" ] || { echo "MISSING rec $v s$s"; missing=$((missing+1)); }; done; done
for v in SRoPEGen SRoPEGen_Abs; do for s in $(seq 0 11); do
  [ -f "$OUT/torus/p0/${v}_s${s}/${v}.pt" ] || { echo "MISSING tor $v s$s"; missing=$((missing+1)); }; done; done
for v in Signed_r4 Abs_r4; do
  [ -f "$OUT/torus_repro/p0/${v}_s0/${v}.pt" ] || { echo "MISSING repro $v"; missing=$((missing+1)); }; done
echo "missing=$missing"
[ "$missing" -gt 0 ] && { echo "not evaluating an incomplete batch"; exit 1; }

python3 -u -m mapformer.eval_noise_refine --runs-dir "$OUT/torus" \
  --variants SRoPEGen SRoPEGen_Abs --noises 0.0 --seeds $(seq 0 11) --lengths 128 512 1024 \
  --n-trials 100 --device cuda:0 --out "$REPO/_MONOTONE_TORUS.md" \
  --title "MONOTONE torus, raw evaluation" > "$OUT/logs/eval_torus.log" 2>&1
echo "torus eval exit $?"
[ -s "$REPO/_MONOTONE_TORUS.json" ] || { echo "no torus json -- stopping"; exit 1; }

python3 -u -m mapformer.analyze_monotone > "$OUT/logs/analyze.log" 2>&1
echo "analysis exit $?"
# the marker certifies the ARTIFACT, not the absence of a crash
if [ -s "$REPO/MONOTONE_RAW.json" ]; then touch "$OUT/.done"; else echo "no MONOTONE_RAW.json -- NOT touching .done"; exit 1; fi
echo "batch finished $(date)"

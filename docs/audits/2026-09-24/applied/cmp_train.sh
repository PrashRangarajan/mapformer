#!/usr/bin/env bash
# usage: cmp_train.sh <outdir> [config names...]
# Trains each config with the pristine HEAD worktree (OLD) and the repo (NEW) on the SAME GPU,
# one after the other, then compares per-epoch losses, every parameter and (if saved) the
# optimizer state bitwise.
set -u
SP="${MF_SCRATCH:?set MF_SCRATCH to a dir holding base/mapformer, a git worktree of the pre-change commit}"
O="$1"; shift; mkdir -p "$O"
declare -A CFG
CFG[rank_dw3]="--variant Vanilla_r4 --n-steps 1024 --batch-size 16 --data-workers 3 --schedule cosine --lr 1e-3 --save-full-state"
CFG[rank_r2_dw3]="--variant Vanilla --n-steps 1024 --batch-size 16 --data-workers 3 --schedule cosine --lr 1e-3"
CFG[rank_serial]="--variant Vanilla --n-steps 1024 --batch-size 16 --data-workers 0 --schedule cosine --lr 1e-3"
CFG[torus_dw3]="--variant Vanilla --n-steps 128 --batch-size 128 --data-workers 3"
CFG[torus_serial]="--variant Vanilla --n-steps 128 --batch-size 128 --data-workers 0"
CFG[perhead_dw3]="--variant Vanilla_r2ph --n-steps 1024 --batch-size 16 --data-workers 3 --schedule cosine --lr 1e-3"
CFG[em_serial]="--variant VanillaEM --n-steps 128 --batch-size 128 --data-workers 0"
CFG[dhead48_serial]="--variant Vanilla --d-model 96 --n-steps 128 --batch-size 32 --data-workers 0"
CFG[dog_noise_serial]="--variant Level15_DoG --n-steps 128 --batch-size 64 --data-workers 0 --p-action-noise 0.1"
CFG[looped_dw3]="--variant Looped --n-steps 256 --batch-size 32 --data-workers 3 --schedule cosine --lr 1e-3"
NAMES="${*:-rank_dw3 rank_r2_dw3 rank_serial torus_dw3 torus_serial perhead_dw3 em_serial dhead48_serial dog_noise_serial looped_dw3}"
GPU="${GPU:-0}"
for nm in $NAMES; do
  for side in old new; do
    [ $side = old ] && CP=$SP/base || CP=/home/prashr
    ( cd $CP && OMP_NUM_THREADS=4 python3 -u -m mapformer.train_variant ${CFG[$nm]} --seed 0 --epochs 3 --n-batches 20 \
        --n-layers 1 --n-heads 2 --n-landmarks 0 --device cuda:$GPU --output-dir $O/$nm/$side ) > $O/$nm.$side.log 2>&1 \
      || echo "TRAIN FAILED $nm $side" >> $O/summary.txt
  done
  # the continuation path: --init-from the rank_dw3 checkpoint, fresh stream via offset 1
  if [ $nm = rank_dw3 ]; then
    for side in old new; do
      [ $side = old ] && CP=$SP/base || CP=/home/prashr
      ( cd $CP && OMP_NUM_THREADS=4 python3 -u -m mapformer.train_variant ${CFG[$nm]} --seed 0 --epochs 2 --n-batches 20 \
          --n-layers 1 --n-heads 2 --n-landmarks 0 --device cuda:$GPU --output-dir $O/cont/$side \
          --init-from $O/rank_dw3/old/Vanilla_r4.pt --data-seed-offset 1 ) > $O/cont.$side.log 2>&1 \
        || echo "TRAIN FAILED cont $side" >> $O/summary.txt
    done
  fi
done
python3 - "$O" $NAMES <<'PY' >> $O/summary.txt 2>&1
import sys, glob, os, torch
O = sys.argv[1]; names = sys.argv[2:] + (["cont"] if "rank_dw3" in sys.argv[2:] else [])
def eq(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    return a == b
for nm in names:
    po = glob.glob(f"{O}/{nm}/old/*.pt"); pn = glob.glob(f"{O}/{nm}/new/*.pt")
    if not po or not pn: print(f"{nm:18s} MISSING checkpoint"); continue
    a = torch.load(po[0], map_location="cpu", weights_only=False); b = torch.load(pn[0], map_location="cpu", weights_only=False)
    L = a["losses"] == b["losses"]; P = eq(a["model_state_dict"], b["model_state_dict"])
    Op = eq(a.get("optimizer_state_dict"), b.get("optimizer_state_dict")) if "optimizer_state_dict" in a else None
    lp = eq(a.get("losses_prior"), b.get("losses_prior"))
    ok = L and P and (Op is not False) and lp
    print(f"{nm:18s} {'BIT-IDENTICAL' if ok else 'DIFFERENT'}  losses {'==' if L else '!='}  params {'==' if P else '!='}"
          f"  optim {'-' if Op is None else ('==' if Op else '!=')}  losses_prior {'==' if lp else '!='}"
          f"  final loss {a['losses'][-1]!r} / {b['losses'][-1]!r}  n_params {sum(v.numel() for v in a['model_state_dict'].values())}")
PY
echo DONE >> $O/summary.txt

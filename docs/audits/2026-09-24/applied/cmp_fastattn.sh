#!/usr/bin/env bash
# --fast-attn / --deterministic through the real CLI, rank config, one GPU, sequential.
set -u
O="$1"; GPU="${GPU:-0}"; mkdir -p "$O"; cd /home/prashr
RC="--variant Vanilla --seed 0 --n-steps 1024 --batch-size 16 --data-workers 3 --schedule cosine --lr 1e-3 --n-layers 1 --n-heads 2 --n-landmarks 0 --device cuda:$GPU"
t(){ local nm=$1; shift; local t0=$(date +%s.%N)
     OMP_NUM_THREADS=4 python3 -u -m mapformer.train_variant $RC "$@" --output-dir $O/$nm > $O/$nm.log 2>&1 || echo "FAILED $nm" >> $O/summary.txt
     echo "$nm wall $(python3 -c "print(round($(date +%s.%N)-$t0,1))") s; epoch-5 time: $(grep -o 'Epoch   5/  5.*' $O/$nm.log | grep -o '[0-9.]*s$')" >> $O/summary.txt; }
# timing: 5 epochs x 98 batches (the 5th epoch's time is printed by train())
t time_explicit --epochs 5 --n-batches 98
t time_fast --epochs 5 --n-batches 98 --fast-attn
t time_fast_det --epochs 5 --n-batches 98 --fast-attn --deterministic
t time_det --epochs 5 --n-batches 98 --deterministic
# run-to-run reproducibility: 2 epochs x 20 batches, twice each
for k in a b; do t rep_fast_$k --epochs 2 --n-batches 20 --fast-attn; t rep_fastdet_$k --epochs 2 --n-batches 20 --fast-attn --deterministic; done
# refusal: a variant with no plain WMTransformerLayer
OMP_NUM_THREADS=4 python3 -m mapformer.train_variant --variant VanillaEM --seed 0 --n-steps 16 --batch-size 4 --epochs 1 --n-batches 2 --device cuda:$GPU --output-dir $O/em --fast-attn > $O/em.log 2>&1; echo "VanillaEM --fast-attn exit $?: $(tail -n 1 $O/em.log)" >> $O/summary.txt
python3 - "$O" <<'PY' >> "$O/summary.txt"
import sys, torch
O = sys.argv[1]
L = lambda n: torch.load(f"{O}/{n}/Vanilla.pt", map_location="cpu", weights_only=False)
def eq(a, b): return a["losses"] == b["losses"] and all(torch.equal(x, y) for x, y in zip(a["model_state_dict"].values(), b["model_state_dict"].values()))
for n1, n2, what in [("rep_fast_a", "rep_fast_b", "--fast-attn run-to-run"),
                     ("rep_fastdet_a", "rep_fastdet_b", "--fast-attn --deterministic run-to-run"),
                     ("time_explicit", "time_det", "explicit vs explicit --deterministic (5 x 98)"),
                     ("time_explicit", "time_fast", "explicit vs --fast-attn (5 x 98; expected to differ)")]:
    a, b = L(n1), L(n2)
    print(f"{what:52s} {'BITWISE IDENTICAL' if eq(a, b) else 'DIFFERENT'}   final losses {a['losses'][-1]:.6f} / {b['losses'][-1]:.6f}")
c = L("time_fast_det")["config"]; print("config keys:", {k: c[k] for k in ("fast_attn", "deterministic")}, "args has them:", c["args"]["fast_attn"], c["args"]["deterministic"])
PY
echo DONE >> $O/summary.txt

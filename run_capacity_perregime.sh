#!/bin/bash
# Per-regime capacity control + length sweep.
#
# The capacity control (CAPACITY_CONTROL.md) was lm200-only. This makes the
# architecture-vs-capacity question per-regime, and tests the bounded-error
# claim by sweeping length instead of probing a single point (T=512).
#
#   (1) Train Vanilla_ExtraHead on clean + noise (lm200 already done).
#   (2) Length sweep: Vanilla / Vanilla_ExtraHead / Level15 at
#       T in {512, 1024, 2048} on clean / noise / lm200.
#   (3) Vanilla_ExtraHead on NumberLine OOD arithmetic chains.
#
# GPU 1 (free as of launch; another user's job finished). Strictly
# sequential -- one process at a time, so a single small MapFormer fits
# even if the other user returns. Runs parallel to the TEM pipeline on GPU 0.
set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs
ED=$REPO/paper_figures/capacity_perregime
mkdir -p "$LOGS" "$ED"
GPU=1

train_grid() {
    local variant=$1 cfg=$2 seed=$3 lm=$4 noise=$5
    local dir="$REPO/runs/${variant}_${cfg}/seed${seed}"
    if [ -f "$dir/${variant}.pt" ]; then
        echo "[$(date)] skip ${variant}_${cfg} s$seed (exists)"; return
    fi
    mkdir -p "$dir"
    echo "[$(date)] train ${variant}_${cfg} s$seed (lm=$lm noise=$noise)"
    CUDA_VISIBLE_DEVICES=$GPU python3 -u -m mapformer.train_variant \
        --variant "$variant" --seed $seed \
        --n-landmarks $lm --grid-size 64 --p-action-noise $noise \
        --epochs 50 --n-batches 156 --device cuda \
        --output-dir mapformer/runs/${variant}_${cfg}/seed${seed} \
        > "$LOGS/${variant}_${cfg}_s${seed}_pr.log" 2>&1
}

train_numberline() {
    local variant=$1 seed=$2
    local dir="$REPO/runs/${variant}_numberline/seed${seed}"
    if [ -f "$dir/${variant}_numberline.pt" ]; then
        echo "[$(date)] skip ${variant}_numberline s$seed (exists)"; return
    fi
    mkdir -p "$dir"
    echo "[$(date)] train ${variant}_numberline s$seed"
    CUDA_VISIBLE_DEVICES=$GPU python3 -u -m mapformer.train_numberline \
        --variant "$variant" --seed $seed \
        --size 64 --n-obs-types 16 --n-landmarks 0 \
        --epochs 50 --n-batches 156 --device cuda \
        --output-dir mapformer/runs/${variant}_numberline/seed${seed} \
        > "$LOGS/${variant}_numberline_s${seed}_pr.log" 2>&1
}

# --- (1) train Vanilla_ExtraHead on clean + noise ---
for seed in 0 1 2; do
    train_grid Vanilla_ExtraHead clean $seed 0 0.0
    train_grid Vanilla_ExtraHead noise $seed 0 0.10
done
# --- (3-train) Vanilla_ExtraHead on NumberLine ---
for seed in 0 1 2; do
    train_numberline Vanilla_ExtraHead $seed
done
echo "[$(date)] training done"

# --- (2) length sweep eval ---
eval_one() {
    local variant=$1 cfg=$2 seed=$3 pan=$4 T2=$5
    local ckpt="$REPO/runs/${variant}_${cfg}/seed${seed}/${variant}.pt"
    [ -f "$ckpt" ] || { echo "  miss ${variant}_${cfg} s$seed"; return; }
    CUDA_VISIBLE_DEVICES=$GPU python3 -m mapformer.eval_single_env \
        --variant "$variant" --checkpoint "$ckpt" \
        --p-action-noise "$pan" --eval-T2 "$T2" \
        > "$ED/${variant}_${cfg}_s${seed}_T${T2}.json" \
        2>"$LOGS/cpr_eval_${variant}_${cfg}_s${seed}_T${T2}.err"
}
echo "[$(date)] length-sweep eval"
for T2 in 512 1024 2048; do
    for s in 0 1 2; do
        for v in Vanilla Vanilla_ExtraHead Level15; do
            eval_one "$v" clean "$s" 0.0  "$T2"
            eval_one "$v" noise "$s" 0.10 "$T2"
            eval_one "$v" lm200 "$s" 0.0  "$T2"
        done
    done
done

# --- aggregate ---
cd "$REPO"
python3 -u <<'PYEOF' > "$REPO/CAPACITY_PERREGIME.md" 2>"$LOGS/cpr_agg.err"
import json, torch, numpy as np
from pathlib import Path

VARIANTS = ["Vanilla", "Vanilla_ExtraHead", "Level15"]
PARAMS = {"Vanilla": "256K", "Vanilla_ExtraHead": "322K", "Level15": "305K"}

def fmt(arr):
    if not arr: return "—"
    return f"{np.mean(arr):.3f} ± {np.std(arr):.3f}"

def grab(variant, cfg, T2, key):
    out = []
    for s in [0, 1, 2]:
        p = Path(f"paper_figures/capacity_perregime/{variant}_{cfg}_s{s}_T{T2}.json")
        if not p.exists(): continue
        j = json.loads(p.read_text())
        v = j.get(key)            # acc_T512 / nll_T512 hold the requested-T2 value
        if v is not None: out.append(v)
    return out

print("# Per-regime capacity control + length sweep\n")
print("Is the Level15-over-Vanilla win architecture or parameters? The first")
print("capacity control (CAPACITY_CONTROL.md) tested only lm200 at one length.")
print("This answers it per-regime and sweeps length to test the bounded-error")
print("claim (a drifting accumulator degrades without bound; a wrapped filter")
print("plateaus — visible only across lengths, not at a single T).\n")
print("`Vanilla_ExtraHead` = Vanilla + a generic extra attention head (322K")
print("params > Level15's 305K). n=3 seeds, eval_single_env.\n")

print("## Capacity by regime (T=512 OOD)\n")
print("| Regime | Vanilla | Vanilla_ExtraHead | Level15 | verdict |")
print("|---|---|---|---|---|")
for cfg in ["clean", "noise", "lm200"]:
    accs = {v: grab(v, cfg, 512, "acc_T512") for v in VARIANTS}
    cells = [fmt(accs[v]) for v in VARIANTS]
    va = np.mean(accs["Vanilla"]) if accs["Vanilla"] else None
    xa = np.mean(accs["Vanilla_ExtraHead"]) if accs["Vanilla_ExtraHead"] else None
    la = np.mean(accs["Level15"]) if accs["Level15"] else None
    verdict = "—"
    if None not in (va, xa, la):
        if xa >= la - 0.02:   verdict = "CAPACITY"
        elif xa <= va + 0.02: verdict = "ARCHITECTURE"
        else:                 verdict = "PARTIAL"
    print(f"| **{cfg}** | {cells[0]} | {cells[1]} | {cells[2]} | {verdict} |")
print()
print("Verdict rule: CAPACITY if ExtraHead reaches Level15 (within 2pp);")
print("ARCHITECTURE if ExtraHead stays at Vanilla; PARTIAL otherwise.\n")

print("## Length sweep — accuracy\n")
for cfg in ["clean", "noise", "lm200"]:
    print(f"### {cfg}\n")
    print("| Variant | T=512 | T=1024 | T=2048 |")
    print("|---|---|---|---|")
    for v in VARIANTS:
        cells = [fmt(grab(v, cfg, T2, "acc_T512")) for T2 in [512, 1024, 2048]]
        print(f"| **{v}** | " + " | ".join(cells) + " |")
    print()

print("## Length sweep — NLL (calibration)\n")
for cfg in ["clean", "noise", "lm200"]:
    print(f"### {cfg}\n")
    print("| Variant | T=512 | T=1024 | T=2048 |")
    print("|---|---|---|---|")
    for v in VARIANTS:
        cells = [fmt(grab(v, cfg, T2, "nll_T512")) for T2 in [512, 1024, 2048]]
        print(f"| **{v}** | " + " | ".join(cells) + " |")
    print()

print("## NumberLine — does capacity close the arithmetic-chain gap?\n")
print("OOD chain = 512 ops (4x trained). If Vanilla_ExtraHead tracks Level15")
print("here, even arithmetic extrapolation is capacity; if it stays at Vanilla,")
print("the self-correcting accumulator is a genuine architectural effect.\n")
print("| Variant | in-dist T=128 | OOD chain T=512 | T=512 NLL |")
print("|---|---|---|---|")
for v in ["Vanilla", "Vanilla_ExtraHead", "Level15"]:
    t1, t4, n4 = [], [], []
    for s in [0, 1, 2]:
        p = Path(f"runs/{v}_numberline/seed{s}/{v}_numberline.pt")
        if not p.exists(): continue
        c = torch.load(p, map_location="cpu", weights_only=False)
        if c.get("test_acc") is not None: t1.append(c["test_acc"])
        if c.get("test_acc_T2") is not None: t4.append(c["test_acc_T2"])
        if c.get("test_nll_T2") is not None: n4.append(c["test_nll_T2"])
    print(f"| **{v}** | {fmt(t1)} | {fmt(t4)} | {fmt(n4)} |")
print()
print("*Auto-generated by run_capacity_perregime.sh*")
PYEOF

git pull --rebase 2>&1 | tail -2
git add run_capacity_perregime.sh CAPACITY_PERREGIME.md \
    paper_figures/capacity_perregime/ 2>/dev/null
for cfg in clean noise; do
    for s in 0 1 2; do
        git add runs/Vanilla_ExtraHead_${cfg}/seed${s}/*.pt 2>/dev/null || true
    done
done
for s in 0 1 2; do
    git add runs/Vanilla_ExtraHead_numberline/seed${s}/*.pt 2>/dev/null || true
done
git commit -m "Per-regime capacity control + length sweep

Extends the lm200-only capacity control to clean and noise, and sweeps
T in {512,1024,2048} to test the bounded-error claim across lengths
rather than at a single point. Adds Vanilla_ExtraHead on NumberLine to
check whether capacity closes the arithmetic-chain extrapolation gap.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -2
echo "[$(date)] per-regime capacity control done."

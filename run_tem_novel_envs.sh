#!/bin/bash
# TEMFaithful on novel-env tests: cross-topology, cross-scale, multi-env held-out.
# Sanity check that our TEMFaithful reimplementation matches the published
# TEM paper's behavior on held-out environments, plus completes the n=3
# comparison table.
set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs
mkdir -p "$LOGS"

is_gpu_free() {
    local mem util
    while IFS=', ' read -r mem util; do
        mem=${mem//[^0-9]/}; util=${util//[^0-9]/}
        [ -z "$mem" ] && mem=0; [ -z "$util" ] && util=0
        if [ "$mem" -ge 5000 ] || [ "$util" -ge 50 ]; then return 1; fi
    done < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    return 0
}
wait_gpus() {
    echo "[$(date)] Polling for free GPUs..."
    while ! is_gpu_free; do sleep 60; done
    echo "[$(date)] GPUs free."
}

# -------- training helpers ----------
topology_train() {
    local seed=$1 gpu=$2
    mkdir -p "$REPO/runs/TEMFaithful_topology/seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_topology \
        --variant TEMFaithful --seed $seed \
        --topologies torus open walls \
        --size 64 --n-envs-per-topology 20 --n-landmarks 200 \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/TEMFaithful_topology/seed${seed} \
        > "$LOGS/TEMFaithful_topology_s${seed}.log" 2>&1
}
multisize_train() {
    local seed=$1 gpu=$2
    mkdir -p "$REPO/runs/TEMFaithful_multisize/seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_multisize \
        --variant TEMFaithful --seed $seed \
        --sizes 32 64 128 --n-envs-per-size 20 --n-landmarks 200 \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/TEMFaithful_multisize/seed${seed} \
        > "$LOGS/TEMFaithful_multisize_s${seed}.log" 2>&1
}
multienv_train() {
    local seed=$1 gpu=$2
    mkdir -p "$REPO/runs/TEMFaithful_multienv/seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_multienv \
        --variant TEMFaithful --seed $seed \
        --n-landmarks 200 --n-train-envs 50 --n-test-envs 50 \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/TEMFaithful_multienv/seed${seed} \
        > "$LOGS/TEMFaithful_multienv_s${seed}.log" 2>&1
}

# Step 1: multi-env held-out (paper sanity check — TEM's classic novel-env test)
wait_gpus
echo "[$(date)] === Step 1: TEMFaithful multi-env held-out (seeds 0, 1, 2) ==="
multienv_train 0 0 & P1=$!
multienv_train 1 1 & P2=$!
wait $P1 $P2
multienv_train 2 0 & P1=$!
wait $P1
echo "[$(date)] Step 1 done."

# Step 2: cross-topology
wait_gpus
echo "[$(date)] === Step 2: TEMFaithful cross-topology (seeds 0, 1, 2) ==="
topology_train 0 0 & P1=$!
topology_train 1 1 & P2=$!
wait $P1 $P2
topology_train 2 0 & P1=$!
wait $P1
echo "[$(date)] Step 2 done."

# Step 3: cross-scale
wait_gpus
echo "[$(date)] === Step 3: TEMFaithful cross-scale (seeds 0, 1, 2) ==="
multisize_train 0 0 & P1=$!
multisize_train 1 1 & P2=$!
wait $P1 $P2
multisize_train 2 0 & P1=$!
wait $P1
echo "[$(date)] Step 3 done."

# Step 4: regenerate aggregated tables with TEMFaithful rows added.
cd "$REPO"
python3 -u <<'PYEOF' > "$REPO/TEM_NOVEL_ENV_RESULTS.md" 2>"$LOGS/tem_novel_eval.err"
import torch, numpy as np
from pathlib import Path
print("# TEMFaithful on novel-environment tests (n=3)\n")
print("Sanity check: does our TEMFaithful reimplementation generalize to held-out")
print("environments the way Whittington 2020 reports? Plus apples-to-apples row in")
print("the multi-seed comparison tables alongside RoPE/Vanilla/Level15/GSF.\n")

def fmt(arr):
    if not arr: return "—"
    return f"{np.mean(arr):.3f} ± {np.std(arr):.3f} (n={len(arr)})"

# ---------- Multi-env held-out (TEM's home regime) ----------
print("## Multi-env held-out (TEM's classic novel-env test)\n")
print("50 train envs, 50 held-out test envs, lm200, size 64.\n")
print("| Variant | Train mean ± std | Held T=128 mean ± std | Held T=512 mean ± std |")
print("|---|---|---|---|")
for v in ["RoPE", "Vanilla", "Level15", "Level15GSF_NoDrop", "TEMFaithful"]:
    tr, t128, t512 = [], [], []
    for s in [0, 1, 2]:
        ckpt = Path(f"mapformer/runs/{v}_multienv/seed{s}/{v}_multienv.pt")
        if not ckpt.exists(): continue
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        if c.get("train_acc") is not None: tr.append(c["train_acc"])
        if c.get("test_acc") is not None: t128.append(c["test_acc"])
        if c.get("test_acc_T2") is not None: t512.append(c["test_acc_T2"])
    print(f"| **{v}** | {fmt(tr)} | {fmt(t128)} | {fmt(t512)} |")
print()

# ---------- Cross-topology ----------
print("## Cross-topology (TEM OOD-d analog)\n")
print("Train on torus + open + walls mix; eval per-topology on held-out envs.\n")
print("| Variant | torus T=512 | open T=512 | walls T=512 |")
print("|---|---|---|---|")
for v in ["RoPE", "Vanilla", "Level15GSF_NoDrop", "Level15GSF_NoDrop_K16", "TEMFaithful"]:
    per_topo = {"torus": [], "open": [], "walls": []}
    for s in [0, 1, 2]:
        ckpt = Path(f"mapformer/runs/{v}_topology/seed{s}/{v}_topology.pt")
        if not ckpt.exists(): continue
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        er = c.get("eval_results", {})
        for topo in ["torus", "open", "walls"]:
            r = er.get(topo, {})
            if r.get("held_T512_acc") is not None:
                per_topo[topo].append(r["held_T512_acc"])
    cells = [fmt(per_topo[t]) for t in ["torus", "open", "walls"]]
    print(f"| **{v}** | " + " | ".join(cells) + " |")
print()

# ---------- Cross-scale ----------
print("## Cross-scale (TEM OOD-s analog)\n")
print("Train on sizes 32 + 64 + 128 mix; eval per-size on held-out envs.\n")
print("| Variant | size 32 T=512 | size 64 T=512 | size 128 T=512 |")
print("|---|---|---|---|")
for v in ["RoPE", "Vanilla", "Level15GSF_NoDrop", "Level15GSF_NoDrop_K16", "TEMFaithful"]:
    per_size = {32: [], 64: [], 128: []}
    for s in [0, 1, 2]:
        ckpt = Path(f"mapformer/runs/{v}_multisize/seed{s}/{v}_multisize.pt")
        if not ckpt.exists(): continue
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        er = c.get("eval_results", {})
        for sz in [32, 64, 128]:
            r = er.get(sz, {})
            if r.get("held_T512_acc") is not None:
                per_size[sz].append(r["held_T512_acc"])
    cells = [fmt(per_size[sz]) for sz in [32, 64, 128]]
    print(f"| **{v}** | " + " | ".join(cells) + " |")
print()

print("\n*Auto-generated by run_tem_novel_envs.sh*\n")
print("**Caveat on cross-class (torus + DoorKey):** TEMFaithful's forward uses")
print("`tokens < n_actions` to distinguish actions from observations. The cross-class")
print("unified vocab has env-prefix tokens (0, 1) and 11 action tokens spread over")
print("ids 2–12, so this binary split breaks. A TEMFaithful row in MULTICLASS_*")
print("would require a custom is-action mask. Skipped here.")
PYEOF

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add TEM_NOVEL_ENV_RESULTS.md run_tem_novel_envs.sh
for setting in topology multisize multienv; do
    for seed in 0 1 2; do
        git add runs/TEMFaithful_${setting}/seed${seed}/TEMFaithful_${setting}.pt 2>/dev/null || true
    done
done
git commit -m "TEMFaithful on novel-env tests: n=3 sanity check + comparison rows

Adds TEMFaithful to: multi-env held-out (TEM's classic test), cross-topology,
cross-scale. Sanity-checks that our TEMFaithful reimplementation generalizes
the way Whittington 2020 reports, and completes the n=3 comparison table
alongside RoPE / Vanilla / Level15 / Level15GSF_NoDrop.

Cross-class (torus+DoorKey) skipped: TEMFaithful's tokens<n_actions split
doesn't accommodate the unified-vocab env-prefix tokens.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] [tem-novel-envs] All done."

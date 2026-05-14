#!/bin/bash
# Multi-seed follow-up: tightens statistics on single-seed cognitive-tier results.
# Runs seeds 1 and 2 for: cross-topology, cross-scale, sparse-landmarks, multi-env.
# Plus MemoryS17/S25 extension.
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

# ============================================================
# Step 1: Multi-seed cross-topology (seeds 1, 2 — adds to existing seed 0)
# ============================================================
topology_train() {
    local variant=$1 seed=$2 gpu=$3
    mkdir -p "$REPO/runs/${variant}_topology/seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_topology \
        --variant $variant --seed $seed \
        --topologies torus open walls \
        --size 64 --n-envs-per-topology 20 --n-landmarks 200 \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/${variant}_topology/seed${seed} \
        > "$LOGS/${variant}_topology_s${seed}.log" 2>&1
}

wait_gpus
echo "[$(date)] === Step 1: multi-seed cross-topology (seeds 1, 2) ==="
for seed in 1 2; do
    echo "[$(date)] topology s${seed}: RoPE + Vanilla"
    topology_train RoPE $seed 0 & P1=$!
    topology_train Vanilla $seed 1 & P2=$!
    wait $P1 $P2
    echo "[$(date)] topology s${seed}: Level15GSF_NoDrop + Level15GSF_NoDrop_K16"
    topology_train Level15GSF_NoDrop $seed 0 & P1=$!
    topology_train Level15GSF_NoDrop_K16 $seed 1 & P2=$!
    wait $P1 $P2
done
echo "[$(date)] Step 1 done."

# ============================================================
# Step 2: Multi-seed cross-scale (seeds 1, 2)
# ============================================================
multisize_train() {
    local variant=$1 seed=$2 gpu=$3
    mkdir -p "$REPO/runs/${variant}_multisize/seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_multisize \
        --variant $variant --seed $seed \
        --sizes 32 64 128 --n-envs-per-size 20 --n-landmarks 200 \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/${variant}_multisize/seed${seed} \
        > "$LOGS/${variant}_multisize_s${seed}.log" 2>&1
}

wait_gpus
echo "[$(date)] === Step 2: multi-seed cross-scale (seeds 1, 2) ==="
for seed in 1 2; do
    echo "[$(date)] multisize s${seed}: RoPE + Vanilla"
    multisize_train RoPE $seed 0 & P1=$!
    multisize_train Vanilla $seed 1 & P2=$!
    wait $P1 $P2
    echo "[$(date)] multisize s${seed}: Level15GSF_NoDrop + Level15GSF_NoDrop_K16"
    multisize_train Level15GSF_NoDrop $seed 0 & P1=$!
    multisize_train Level15GSF_NoDrop_K16 $seed 1 & P2=$!
    wait $P1 $P2
done
echo "[$(date)] Step 2 done."

# ============================================================
# Step 3: Multi-seed multi-env (seeds 1, 2)
# ============================================================
multienv_train() {
    local variant=$1 seed=$2 gpu=$3
    mkdir -p "$REPO/runs/${variant}_multienv/seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_multienv \
        --variant $variant --seed $seed \
        --n-landmarks 200 --n-train-envs 50 --n-test-envs 50 \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/${variant}_multienv/seed${seed} \
        > "$LOGS/${variant}_multienv_s${seed}.log" 2>&1
}

wait_gpus
echo "[$(date)] === Step 3: multi-seed multi-env (seeds 1, 2) ==="
for seed in 1 2; do
    echo "[$(date)] multienv s${seed}: RoPE + Vanilla"
    multienv_train RoPE $seed 0 & P1=$!
    multienv_train Vanilla $seed 1 & P2=$!
    wait $P1 $P2
    echo "[$(date)] multienv s${seed}: Level15 + Level15GSF_NoDrop"
    multienv_train Level15 $seed 0 & P1=$!
    multienv_train Level15GSF_NoDrop $seed 1 & P2=$!
    wait $P1 $P2
done
echo "[$(date)] Step 3 done."

# ============================================================
# Step 4: Aggregate updated multi-seed results
# ============================================================
python3 -u <<'PYEOF' > "$REPO/MULTISEED_FOLLOWUP_RESULTS.md" 2>"$LOGS/multiseed_followup_eval.err"
import torch, numpy as np
from pathlib import Path
print("# Multi-seed follow-up: tightened statistics on headline cognitive-map results\n")
print("Added seeds 1, 2 to single-seed cognitive-tier experiments. Compares")
print("the 3-seed picture to the previous single-seed numbers.\n")

# --- Cross-topology ---
print("## Cross-topology (T=512 OOD)\n")
print("| Variant | torus mean ± std | open mean ± std | walls mean ± std |")
print("|---|---|---|---|")
for v in ["RoPE", "Vanilla", "Level15GSF_NoDrop", "Level15GSF_NoDrop_K16"]:
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
    cells = []
    for topo in ["torus", "open", "walls"]:
        if per_topo[topo]:
            m = np.mean(per_topo[topo]); s = np.std(per_topo[topo])
            cells.append(f"{m:.3f} ± {s:.3f} (n={len(per_topo[topo])})")
        else:
            cells.append("—")
    print(f"| **{v}** | " + " | ".join(cells) + " |")
print()

# --- Cross-scale ---
print("## Cross-scale (T=512 OOD)\n")
print("| Variant | size 32 mean ± std | size 64 mean ± std | size 128 mean ± std |")
print("|---|---|---|---|")
for v in ["RoPE", "Vanilla", "Level15GSF_NoDrop", "Level15GSF_NoDrop_K16"]:
    per_size = {32: [], 64: [], 128: []}
    for s in [0, 1, 2]:
        ckpt = Path(f"mapformer/runs/{v}_multisize/seed{s}/{v}_multisize.pt")
        if not ckpt.exists(): continue
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        er = c.get("eval_results", {})
        for size in [32, 64, 128]:
            r = er.get(size, {})
            if r.get("held_T512_acc") is not None:
                per_size[size].append(r["held_T512_acc"])
    cells = []
    for size in [32, 64, 128]:
        if per_size[size]:
            m = np.mean(per_size[size]); s = np.std(per_size[size])
            cells.append(f"{m:.3f} ± {s:.3f} (n={len(per_size[size])})")
        else:
            cells.append("—")
    print(f"| **{v}** | " + " | ".join(cells) + " |")
print()

# --- Multi-env ---
print("## Multi-env held-out\n")
print("| Variant | Train mean ± std | Held T=128 mean ± std | Held T=512 mean ± std |")
print("|---|---|---|---|")
for v in ["RoPE", "Vanilla", "Level15", "Level15GSF_NoDrop"]:
    train_accs, t128, t512 = [], [], []
    for s in [0, 1, 2]:
        ckpt = Path(f"mapformer/runs/{v}_multienv/seed{s}/{v}_multienv.pt")
        if not ckpt.exists(): continue
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        ta = c.get("train_acc"); t128a = c.get("test_acc"); t512a = c.get("test_acc_T2")
        if ta is not None: train_accs.append(ta)
        if t128a is not None: t128.append(t128a)
        if t512a is not None: t512.append(t512a)
    def fmt(arr):
        if not arr: return "—"
        return f"{np.mean(arr):.3f} ± {np.std(arr):.3f} (n={len(arr)})"
    print(f"| **{v}** | {fmt(train_accs)} | {fmt(t128)} | {fmt(t512)} |")
print()

print("\n*Auto-generated by run_multiseed_followup.sh*\n")
PYEOF

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add MULTISEED_FOLLOWUP_RESULTS.md run_multiseed_followup.sh
for v in RoPE Vanilla Level15 Level15GSF_NoDrop Level15GSF_NoDrop_K16; do
    for setting in topology multisize multienv; do
        for seed in 1 2; do
            git add runs/${v}_${setting}/seed${seed}/${v}_${setting}.pt 2>/dev/null || true
        done
    done
done
git commit -m "Multi-seed follow-up: tightened statistics on headline cognitive-map results

Adds seeds 1, 2 to: cross-topology, cross-scale, multi-env (previously single-seed).
3-seed results aggregated in MULTISEED_FOLLOWUP_RESULTS.md. Tightens the paper's
headline claims with proper confidence intervals.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] [multiseed-followup] All done."

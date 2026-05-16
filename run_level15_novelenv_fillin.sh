#!/bin/bash
# Fill-in: Level15 (single, no GSF) on cross-topology + cross-scale.
# These are the two remaining gaps for the "TEM setting of novel envs on
# MapFormer" comparison table.
#
# Seed-outer-loop per feedback_seed_ordering.md: each seed lands both
# settings before the next seed starts.
set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs
mkdir -p "$LOGS"

topology_train() {
    local seed=$1 gpu=$2
    mkdir -p "$REPO/runs/Level15_topology/seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_topology \
        --variant Level15 --seed $seed \
        --topologies torus open walls \
        --size 64 --n-envs-per-topology 20 --n-landmarks 200 \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/Level15_topology/seed${seed} \
        > "$LOGS/Level15_topology_s${seed}.log" 2>&1
}
multisize_train() {
    local seed=$1 gpu=$2
    mkdir -p "$REPO/runs/Level15_multisize/seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python3 -u -m mapformer.train_multisize \
        --variant Level15 --seed $seed \
        --sizes 32 64 128 --n-envs-per-size 20 --n-landmarks 200 \
        --epochs 50 --n-batches 156 \
        --device cuda \
        --output-dir mapformer/runs/Level15_multisize/seed${seed} \
        > "$LOGS/Level15_multisize_s${seed}.log" 2>&1
}

for seed in 0 1 2; do
    echo "[$(date)] [fillin] seed ${seed}: topology + multisize in parallel"
    topology_train $seed 0 & P1=$!
    multisize_train $seed 1 & P2=$!
    wait $P1 $P2
done
echo "[$(date)] [fillin] all training done"

# ---- Re-aggregate TEM_NOVEL_ENV_RESULTS.md with new Level15 rows ----
python3 -u <<'PYEOF' > "$REPO/TEM_NOVEL_ENV_RESULTS.md" 2>"$LOGS/tem_novel_reagg.err"
import torch, numpy as np
from pathlib import Path
print("# TEMFaithful on novel-environment tests (n=3)\n")
print("Apples-to-apples comparison across RoPE / Vanilla / Level15 /")
print("Level15GSF_NoDrop[_K16] / TEMFaithful on TEM's three classic novel-env")
print("regimes, plus cross-class (beyond TEM).\n")

def fmt(arr):
    if not arr: return "—"
    return f"{np.mean(arr):.3f} ± {np.std(arr):.3f} (n={len(arr)})"

# --- Multi-env held-out (TEM's home regime) ---
print("## Multi-env held-out (TEM's classic novel-env test)\n")
print("50 train envs, 50 held-out test envs, lm200, size 64.\n")
print("| Variant | Train | Held T=128 | Held T=512 OOD |")
print("|---|---|---|---|")
for v in ["RoPE", "Vanilla", "Level15", "Level15GSF_NoDrop", "TEMFaithful"]:
    tr, t128, t512 = [], [], []
    for s in [0, 1, 2]:
        p = Path(f"runs/{v}_multienv/seed{s}/{v}_multienv.pt")
        if not p.exists(): continue
        c = torch.load(p, map_location="cpu", weights_only=False)
        if c.get("train_acc") is not None: tr.append(c["train_acc"])
        if c.get("test_acc") is not None: t128.append(c["test_acc"])
        if c.get("test_acc_T2") is not None: t512.append(c["test_acc_T2"])
    print(f"| **{v}** | {fmt(tr)} | {fmt(t128)} | {fmt(t512)} |")
print()

# --- Cross-topology ---
print("## Cross-topology (TEM OOD-d analog)\n")
print("Train on torus + open + walls mix; eval per-topology on held-out envs.\n")
print("| Variant | torus T=512 | open T=512 | walls T=512 |")
print("|---|---|---|---|")
for v in ["RoPE", "Vanilla", "Level15", "Level15GSF_NoDrop", "Level15GSF_NoDrop_K16", "TEMFaithful"]:
    per_topo = {"torus": [], "open": [], "walls": []}
    for s in [0, 1, 2]:
        p = Path(f"runs/{v}_topology/seed{s}/{v}_topology.pt")
        if not p.exists(): continue
        c = torch.load(p, map_location="cpu", weights_only=False)
        er = c.get("eval_results", {})
        for topo in ["torus", "open", "walls"]:
            r = er.get(topo, {})
            if r.get("held_T512_acc") is not None:
                per_topo[topo].append(r["held_T512_acc"])
    cells = [fmt(per_topo[t]) for t in ["torus", "open", "walls"]]
    print(f"| **{v}** | " + " | ".join(cells) + " |")
print()

# --- Cross-scale ---
print("## Cross-scale (TEM OOD-s analog)\n")
print("Train on sizes 32 + 64 + 128 mix; eval per-size on held-out envs.\n")
print("| Variant | size 32 T=512 | size 64 T=512 | size 128 T=512 |")
print("|---|---|---|---|")
for v in ["RoPE", "Vanilla", "Level15", "Level15GSF_NoDrop", "Level15GSF_NoDrop_K16", "TEMFaithful"]:
    per_size = {32: [], 64: [], 128: []}
    for s in [0, 1, 2]:
        p = Path(f"runs/{v}_multisize/seed{s}/{v}_multisize.pt")
        if not p.exists(): continue
        c = torch.load(p, map_location="cpu", weights_only=False)
        er = c.get("eval_results", {})
        for sz in [32, 64, 128]:
            r = er.get(sz, {})
            if r.get("held_T512_acc") is not None:
                per_size[sz].append(r["held_T512_acc"])
    cells = [fmt(per_size[sz]) for sz in [32, 64, 128]]
    print(f"| **{v}** | " + " | ".join(cells) + " |")
print()

# --- Cross-class ---
print("## Cross-class (beyond TEM): torus + MiniGrid-DoorKey\n")
print("Different action vocab (4 vs 7) + different obs spaces. Disjoint vocab\n")
print("with env-prefix tokens. TEMFaithful skipped (`tokens<n_actions` doesn't\n")
print("fit the unified vocab).\n")
print("| Variant | Torus T=128 | Torus T=512 | DoorKey T=128 | DoorKey T=512 |")
print("|---|---|---|---|---|")
for v in ["RoPE", "Vanilla", "Level15", "Level15GSF_NoDrop_K16"]:
    t128, t512, d128, d512 = [], [], [], []
    for s in [0, 1, 2]:
        p = Path(f"runs/{v}_multiclass/seed{s}/{v}_multiclass.pt")
        if not p.exists(): continue
        c = torch.load(p, map_location="cpu", weights_only=False)
        er = c.get("eval_results", {})
        tr_, dr_ = er.get("torus", {}), er.get("doorkey", {})
        if tr_.get("held_T128_acc") is not None: t128.append(tr_["held_T128_acc"])
        if tr_.get("held_T512_acc") is not None: t512.append(tr_["held_T512_acc"])
        if dr_.get("held_T128_acc") is not None: d128.append(dr_["held_T128_acc"])
        if dr_.get("held_T512_acc") is not None: d512.append(dr_["held_T512_acc"])
    print(f"| **{v}** | {fmt(t128)} | {fmt(t512)} | {fmt(d128)} | {fmt(d512)} |")
print(f"| **TEMFaithful** | n/a | n/a | n/a | n/a |")
print()

print("\n*Auto-generated by run_level15_novelenv_fillin.sh*\n")
print("**Caveat on cross-class for TEMFaithful:** Its forward uses\n")
print("`tokens<n_actions` to split actions from observations; the cross-class\n")
print("unified vocab has env-prefix tokens at IDs 0/1 and 11 action tokens spread\n")
print("over IDs 2-12, so this split is not directly applicable. Filling it would\n")
print("require a custom is-action mask in TEMFaithful.\n")
PYEOF

cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add TEM_NOVEL_ENV_RESULTS.md run_level15_novelenv_fillin.sh
for seed in 0 1 2; do
    for setting in topology multisize; do
        git add runs/Level15_${setting}/seed${seed}/Level15_${setting}.pt 2>/dev/null || true
    done
done
git commit -m "Level15 on cross-topology + cross-scale: complete the novel-env table

Adds the two missing rows (Level15 single, not GSF) to TEM_NOVEL_ENV_RESULTS.md.
Cross-class for TEMFaithful remains n/a due to vocab incompatibility (would need
a custom is-action mask).
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] [fillin] done."

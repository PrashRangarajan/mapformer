#!/bin/bash
# Watchdog for the orphaned MiniGrid-DoorKey trainings.
#
# The original run_minigrid_doorkey.sh launched 4 trainings via $(...)
# command substitution, which made the python processes children of an
# already-exited subshell. The script's `wait` returned immediately,
# so the eval ran on non-existent checkpoints and wrote empty results.
#
# This watchdog polls until all 4 checkpoints land, then re-runs the
# eval block and commits + pushes the populated MINIGRID_DOORKEY_RESULTS.md.
#
# Idempotent: safe to run while trainings are still ongoing.

set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs

CKPTS=(
    "mapformer/runs/minigrid_doorkey_clean/Vanilla/seed0/Vanilla.pt"
    "mapformer/runs/minigrid_doorkey_clean/Level15/seed0/Level15.pt"
    "mapformer/runs/minigrid_doorkey_noise/Vanilla/seed0/Vanilla.pt"
    "mapformer/runs/minigrid_doorkey_noise/Level15/seed0/Level15.pt"
)

echo "[$(date)] Watching for 4 MiniGrid-DoorKey checkpoints..."
while true; do
    missing=0
    for c in "${CKPTS[@]}"; do
        if [ ! -f "$c" ]; then missing=$((missing+1)); fi
    done
    if [ $missing -eq 0 ]; then
        echo "[$(date)] All 4 checkpoints present."
        break
    fi
    # Print a heartbeat every poll showing progress (epoch number from logs)
    cv=$(grep -E 'Epoch +[0-9]+/50' $LOGS/mg_doorkey_clean_Vanilla_s0.log 2>/dev/null | tail -1 | awk '{print $2}')
    cl=$(grep -E 'Epoch +[0-9]+/50' $LOGS/mg_doorkey_clean_Level15_s0.log 2>/dev/null | tail -1 | awk '{print $2}')
    nv=$(grep -E 'Epoch +[0-9]+/50' $LOGS/mg_doorkey_noise_Vanilla_s0.log 2>/dev/null | tail -1 | awk '{print $2}')
    nl=$(grep -E 'Epoch +[0-9]+/50' $LOGS/mg_doorkey_noise_Level15_s0.log 2>/dev/null | tail -1 | awk '{print $2}')
    echo "[$(date)] missing=$missing — clean V=${cv:-0/50} L=${cl:-0/50}  noise V=${nv:-0/50} L=${nl:-0/50}"
    sleep 120
done

echo "[$(date)] Running eval on all 4 checkpoints..."
python3 -u <<'PYEOF' > "$REPO/MINIGRID_DOORKEY_RESULTS.md" 2>"$LOGS/mg_eval_watchdog.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.minigrid_env import MiniGridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF

VARIANT_CLS = {"Vanilla": MapFormerWM, "Level15": MapFormerWM_Level15InEKF}

def build(variant, ckpt_path):
    c = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    cfg = c.get("config", {})
    cls = VARIANT_CLS[variant]
    m = cls(vocab_size=cfg["vocab_size"], d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 2), n_layers=cfg.get("n_layers", 1),
            grid_size=cfg.get("grid_size", 8))
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()

def eval_revisit(model, env, T, n_trials, seed, p_action_noise=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    c = tot = 0; nll_sum = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, rm = env.generate_trajectory(T, p_action_noise=p_action_noise)
            tt = tokens.unsqueeze(0).cuda()
            try:
                logits = model(tt[:, :-1])
            except Exception:
                return None, None
            lp = F.log_softmax(logits, dim=-1)
            preds = lp.argmax(-1)[0]; tgts = tt[0, 1:]; mask = rm[1:].cuda()
            if mask.sum() == 0: continue
            c += (preds[mask] == tgts[mask]).sum().item()
            tot += mask.sum().item()
            idx = torch.arange(lp.shape[1], device="cuda")[mask]
            nll_sum += -lp[0, idx, tgts[mask]].sum().item()
    return (c / tot if tot else None, nll_sum / tot if tot else None)


print("# MiniGrid-DoorKey-8x8: Vanilla vs Level 1.5 InEKF\n")
print("Single seed (s0). DoorKey naturally has landmarks (key, door, goal — ")
print("unique cells with unique tokens). Tokenization: `obj_color` (66 obs ")
print("types). Random forward-biased policy (65% forward, 30% turns, 5% other).\n")
print("**Primary (no action noise):** real-task setting; tests Level 1.5's ")
print("landmark-utilisation claim without artificial perturbation.\n")
print("**Ablation (10% action noise):** constructed-noise regime; confirms ")
print("noise-robustness claim transfers from torus to MiniGrid.\n")

variants = ["Vanilla", "Level15"]

def make_env(seed):
    return MiniGridWorld(env_name="MiniGrid-DoorKey-8x8-v0",
                          tokenization="obj_color", seed=seed)

for noise_label, noise_dir, eval_noise in [
    ("Primary (no noise)",   "minigrid_doorkey_clean", 0.0),
    ("Ablation (10% noise)", "minigrid_doorkey_noise", 0.10),
]:
    print(f"## {noise_label}\n")
    print("| Variant | T=128 in-dist | T=512 in-dist | T=128 OOD | T=512 OOD |")
    print("|---|---|---|---|---|")
    for v in variants:
        ckpt = Path(f"mapformer/runs/{noise_dir}/{v}/seed0/{v}.pt")
        if not ckpt.exists():
            print(f"| {v} | (no ckpt) | — | — | — |")
            continue
        m = build(v, ckpt)
        env_id = make_env(seed=0)
        a128_id, n128_id = eval_revisit(m, env_id, 128, 100, seed=1000,
                                          p_action_noise=eval_noise)
        a512_id, n512_id = eval_revisit(m, env_id, 512, 50, seed=1000,
                                          p_action_noise=eval_noise)
        env_ood = make_env(seed=1000)
        a128_ood, n128_ood = eval_revisit(m, env_ood, 128, 100, seed=2000,
                                            p_action_noise=eval_noise)
        a512_ood, n512_ood = eval_revisit(m, env_ood, 512, 50, seed=2000,
                                            p_action_noise=eval_noise)
        print(f"| **{v}** | "
              f"{a128_id:.3f} (NLL {n128_id:.3f}) | "
              f"{a512_id:.3f} (NLL {n512_id:.3f}) | "
              f"{a128_ood:.3f} (NLL {n128_ood:.3f}) | "
              f"{a512_ood:.3f} (NLL {n512_ood:.3f}) |")
        del m; torch.cuda.empty_cache()
    print()

print("\n**Predicted (if torus result transfers to MiniGrid):**")
print("- Level 1.5 should beat Vanilla on T=512 OOD in BOTH primary and ")
print("  ablation regimes — DoorKey landmarks make Level 1.5's Kalman ")
print("  correction useful even without injected noise.")
print("- The gap should be larger in the ablation regime (action noise ")
print("  amplifies Level 1.5's bounded-error advantage).")
print("\n*Auto-generated by watch_minigrid_doorkey.sh (re-eval after watchdog).*\n")
PYEOF

echo "[$(date)] Eval done. Committing + pushing..."
cd "$REPO"
git pull --rebase 2>&1 | tail -3
git add MINIGRID_DOORKEY_RESULTS.md run_minigrid_doorkey.sh watch_minigrid_doorkey.sh
git add runs/minigrid_doorkey_clean/Vanilla/seed0/*.pt \
        runs/minigrid_doorkey_clean/Level15/seed0/*.pt \
        runs/minigrid_doorkey_noise/Vanilla/seed0/*.pt \
        runs/minigrid_doorkey_noise/Level15/seed0/*.pt 2>/dev/null || true
git commit -m "MiniGrid-DoorKey: re-eval after watchdog (orphaned-training fix)

The original run_minigrid_doorkey.sh used \$(launch ...) command
substitution to capture PIDs, which forked python in a subshell. When
the subshell exited, the python process was reparented to init and
\`wait \$PID\` returned immediately. The eval block ran 3 seconds after
launch with no checkpoints, populating MINIGRID_DOORKEY_RESULTS.md
with empty rows.

This commit:
  - Fixes run_minigrid_doorkey.sh to launch python directly (no
    command substitution), so wait actually waits.
  - Adds watch_minigrid_doorkey.sh: a watchdog that polls until all 4
    checkpoints land, then runs the same eval and commits the result.
  - Replaces the empty MINIGRID_DOORKEY_RESULTS.md with the real
    numbers from the now-completed orphaned trainings.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] Done."

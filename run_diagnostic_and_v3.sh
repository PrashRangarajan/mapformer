#!/bin/bash
# Stage 1: length diagnostic on existing Level15/Level15PC/Level15PC_NoBypass
# Stage 2: train Level15PC_v3 on clean s0 + lm200 s0 in parallel
# Stage 3: extended results comparison
# Stage 4: commit + push

set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs
mkdir -p "$LOGS"

echo "[$(date)] Stage 1: length diagnostic..."
CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.length_diagnostic \
    --runs-dir mapformer/runs --config lm200 --seed 0 --n-trials 5 \
    --output-md "$REPO/LENGTH_DIAGNOSTIC.md" \
    --output-figs "$REPO/paper_figures" 2>"$LOGS/length_diag.err"
echo "[$(date)] Stage 1 done."

echo "[$(date)] Stage 2: training Level15PC_v3 on clean s0 + lm200 s0..."
mkdir -p "$REPO/runs/Level15PC_v3_clean/seed0" "$REPO/runs/Level15PC_v3_lm200/seed0"

CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.train_variant \
    --variant Level15PC_v3 --seed 0 \
    --n-landmarks 0 --p-action-noise 0.0 \
    --epochs 50 --n-batches 156 --aux-coef 0.1 \
    --device cuda \
    --output-dir mapformer/runs/Level15PC_v3_clean/seed0 \
    > "$LOGS/Level15PC_v3_clean_s0.log" 2>&1 &
P_CLEAN=$!

CUDA_VISIBLE_DEVICES=1 python3 -u -m mapformer.train_variant \
    --variant Level15PC_v3 --seed 0 \
    --n-landmarks 200 --p-action-noise 0.10 \
    --epochs 50 --n-batches 156 --aux-coef 0.1 \
    --device cuda \
    --output-dir mapformer/runs/Level15PC_v3_lm200/seed0 \
    > "$LOGS/Level15PC_v3_lm200_s0.log" 2>&1 &
P_LM=$!

wait $P_CLEAN $P_LM
echo "[$(date)] Training done."
echo "  clean final: $(grep 'Epoch  50/50' $LOGS/Level15PC_v3_clean_s0.log | tail -1)"
echo "  lm200 final: $(grep 'Epoch  50/50' $LOGS/Level15PC_v3_lm200_s0.log | tail -1)"

echo "[$(date)] Stage 3: revisit accuracy + R distribution comparison..."
cd /home/prashr
python3 -u <<'PYEOF' > "$REPO/V3_RESULTS.md" 2>"$LOGS/v3_eval.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_level15_pc import MapFormerWM_Level15PC
from mapformer.model_level15_pc_v2 import MapFormerWM_Level15PC_NoBypass
from mapformer.model_level15_pc_v3 import MapFormerWM_Level15PC_v3

VARIANT_CLS = {
    "Level15": MapFormerWM_Level15InEKF,
    "Level15PC": MapFormerWM_Level15PC,
    "Level15PC_NoBypass": MapFormerWM_Level15PC_NoBypass,
    "Level15PC_v3": MapFormerWM_Level15PC_v3,
}

def build(variant, ckpt_path):
    c = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    cfg = c.get("config", {})
    cls = VARIANT_CLS[variant]
    m = cls(vocab_size=cfg["vocab_size"], d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 2), n_layers=cfg.get("n_layers", 1),
            grid_size=cfg.get("grid_size", 64))
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval()

def eval_revisit(model, env, T, n_trials, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    c = tot = 0; nll_sum = 0.0
    with torch.no_grad():
        for _ in range(n_trials):
            tokens, _, rm = env.generate_trajectory(T)
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


print("# Level15PC_v3 — Fix 5 + Fix 6 + tighter R clamp [-1, 5]\n")
print("Single seed (s0). Compares 4 variants on clean + lm200, T=128 / T=512, in-dist + OOD.\n")

variants = ["Level15", "Level15PC", "Level15PC_NoBypass", "Level15PC_v3"]
for cfg_tag, n_lm in [("clean", 0), ("lm200", 200)]:
    print(f"## Config: {cfg_tag}\n")
    print("| Variant | T=128 in-dist | T=512 in-dist | T=128 OOD | T=512 OOD |")
    print("|---|---|---|---|---|")
    for v in variants:
        ckpt = Path(f"mapformer/runs/{v}_{cfg_tag}/seed0/{v}.pt")
        if not ckpt.exists():
            print(f"| {v} | (no ckpt) | — | — | — |")
            continue
        m = build(v, ckpt)
        env_id = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=n_lm, seed=0)
        a128_id, n128_id = eval_revisit(m, env_id, 128, 200, seed=1000)
        a512_id, n512_id = eval_revisit(m, env_id, 512, 100, seed=1000)
        env_ood = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=n_lm, seed=1000)
        a128_ood, n128_ood = eval_revisit(m, env_ood, 128, 200, seed=2000)
        a512_ood, n512_ood = eval_revisit(m, env_ood, 512, 100, seed=2000)
        print(f"| **{v}** | "
              f"{a128_id:.3f} (NLL {n128_id:.3f}) | "
              f"{a512_id:.3f} (NLL {n512_id:.3f}) | "
              f"{a128_ood:.3f} (NLL {n128_ood:.3f}) | "
              f"{a512_ood:.3f} (NLL {n512_ood:.3f}) |")
        del m; torch.cuda.empty_cache()
    print()

# R distribution
print("## R_t distribution comparison (lm200, fresh env seed=12345)\n")
print("| Variant | action mean log_R | blank | aliased | landmark | spread |")
print("|---|---|---|---|---|---|")
for v in variants:
    ckpt = Path(f"mapformer/runs/{v}_lm200/seed0/{v}.pt")
    if not ckpt.exists():
        continue
    m = build(v, ckpt)
    if not hasattr(m, "inekf"):
        del m; continue
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=200, seed=12345)
    np.random.seed(12345); torch.manual_seed(12345)
    log_R_lm, log_R_obs, log_R_blank, log_R_act = [], [], [], []
    with torch.no_grad():
        for _ in range(30):
            tokens, _, _ = env.generate_trajectory(512)
            tt = tokens.unsqueeze(0).cuda()
            try:
                _ = m(tt[:, :-1])
            except Exception:
                continue
            R = m.last_R[0]
            log_R_per_tok = torch.log(R).mean(dim=(1, 2)).cpu().numpy()
            toks = tt[0, :len(log_R_per_tok)].cpu().numpy()
            for i, tok in enumerate(toks):
                tok = int(tok)
                if i % 2 == 0:
                    log_R_act.append(log_R_per_tok[i])
                else:
                    obs_id = tok - env.obs_offset
                    if obs_id < 0: continue
                    if obs_id == env.blank_token:
                        log_R_blank.append(log_R_per_tok[i])
                    elif obs_id < env.n_obs_types:
                        log_R_obs.append(log_R_per_tok[i])
                    else:
                        log_R_lm.append(log_R_per_tok[i])
    means = {
        "action": float(np.mean(log_R_act)) if log_R_act else float("nan"),
        "blank": float(np.mean(log_R_blank)) if log_R_blank else float("nan"),
        "aliased": float(np.mean(log_R_obs)) if log_R_obs else float("nan"),
        "landmark": float(np.mean(log_R_lm)) if log_R_lm else float("nan"),
    }
    spread = max(means.values()) - min(means.values())
    print(f"| **{v}** | "
          f"{means['action']:+.3f} | {means['blank']:+.3f} | "
          f"{means['aliased']:+.3f} | {means['landmark']:+.3f} | "
          f"**{spread:.3f}** |")
    del m; torch.cuda.empty_cache()

print("\n**Predicted (if v3's tighter clamp closes the saturation route):**")
print("- v3 should match Level15 on lm200 OOD T=512 (~0.82) and clean T=512 (~0.99)")
print("- v3's R distribution should resemble Level15's (positive log_R values bounded by -1)")
print("\n*Auto-generated by run_diagnostic_and_v3.sh*\n")
PYEOF
echo "[$(date)] Stage 3 done."

cd "$REPO"
git add LENGTH_DIAGNOSTIC.md V3_RESULTS.md \
        length_diagnostic.py model_level15_pc_v3.py train_variant.py \
        run_diagnostic_and_v3.sh
git add paper_figures/fig_length_diag*.png 2>/dev/null || true
git add runs/Level15PC_v3_clean/seed0/*.pt \
        runs/Level15PC_v3_lm200/seed0/*.pt 2>/dev/null || true
git commit -m "Length-generalization diagnostic + Level15PC_v3 (Fix 5+6+tighter R clamp)

LENGTH_DIAGNOSTIC.md: per-position drift statistics for Level15,
Level15PC, NoBypass at T=128/T=512. Looks for where each variant's θ̂
goes OOD on long trajectories. Plots in paper_figures/fig_length_diag*.

model_level15_pc_v3.py: adds InEKFLevel15TightClamp (clamps log_R
to [-1, 5] instead of [-5, 5]), capping K at ~0.73. Inherits NoBypass's
stop-gradient + landmark-mask fixes. Tests whether bounded K closes
the indirect R-saturation route via shared action_to_lie parameters
that NoBypass alone didn't close.

V3_RESULTS.md: revisit accuracy + R distribution comparison across
all four variants (Level15, Level15PC, NoBypass, v3) on clean + lm200,
in-dist + OOD at T=128/T=512.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] All done."

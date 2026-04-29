#!/bin/bash
# run_v4_test.sh — train Level15PC_v4 (full PC isolation) on clean + lm200,
# then build the final 5-variant comparison table.

set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs
mkdir -p "$LOGS"

echo "[$(date)] Stage 1: train Level15PC_v4 on clean s0 + lm200 s0..."
mkdir -p "$REPO/runs/Level15PC_v4_clean/seed0" "$REPO/runs/Level15PC_v4_lm200/seed0"

CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.train_variant \
    --variant Level15PC_v4 --seed 0 \
    --n-landmarks 0 --p-action-noise 0.0 \
    --epochs 50 --n-batches 156 --aux-coef 0.1 \
    --device cuda \
    --output-dir mapformer/runs/Level15PC_v4_clean/seed0 \
    > "$LOGS/Level15PC_v4_clean_s0.log" 2>&1 &
P_CLEAN=$!

CUDA_VISIBLE_DEVICES=1 python3 -u -m mapformer.train_variant \
    --variant Level15PC_v4 --seed 0 \
    --n-landmarks 200 --p-action-noise 0.10 \
    --epochs 50 --n-batches 156 --aux-coef 0.1 \
    --device cuda \
    --output-dir mapformer/runs/Level15PC_v4_lm200/seed0 \
    > "$LOGS/Level15PC_v4_lm200_s0.log" 2>&1 &
P_LM=$!

wait $P_CLEAN $P_LM
echo "[$(date)] Training done."
echo "  clean: $(grep 'Epoch  50/50' $LOGS/Level15PC_v4_clean_s0.log | tail -1)"
echo "  lm200: $(grep 'Epoch  50/50' $LOGS/Level15PC_v4_lm200_s0.log | tail -1)"

echo "[$(date)] Stage 2: 5-variant comparison + θ̂ scale check..."
cd /home/prashr
python3 -u <<'PYEOF' > "$REPO/V4_RESULTS.md" 2>"$LOGS/v4_eval.err"
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_level15_pc import MapFormerWM_Level15PC
from mapformer.model_level15_pc_v2 import MapFormerWM_Level15PC_NoBypass
from mapformer.model_level15_pc_v3 import MapFormerWM_Level15PC_v3
from mapformer.model_level15_pc_v4 import MapFormerWM_Level15PC_v4

VARIANT_CLS = {
    "Level15": MapFormerWM_Level15InEKF,
    "Level15PC": MapFormerWM_Level15PC,
    "Level15PC_NoBypass": MapFormerWM_Level15PC_NoBypass,
    "Level15PC_v3": MapFormerWM_Level15PC_v3,
    "Level15PC_v4": MapFormerWM_Level15PC_v4,
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


print("# Level15PC variant series — final comparison\n")
print("| Variant | Fixes |")
print("|---|---|")
print("| Level15 | (baseline; no PC) |")
print("| Level15PC | original Level15 + PC aux loss |")
print("| Level15PC_NoBypass | Fix 5 (stop-grad d_t) + Fix 6 (mask landmarks) |")
print("| Level15PC_v3 | NoBypass + Fix 7 (R clamp [-1, 5]) |")
print("| Level15PC_v4 | NoBypass + Fix 8 (full detach: theta_hat AND embed for PC) |")
print()

variants = ["Level15", "Level15PC", "Level15PC_NoBypass", "Level15PC_v3", "Level15PC_v4"]
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

# theta_hat scale check on lm200 (the smoking gun from LENGTH_DIAGNOSTIC.md)
print("## θ̂ scale comparison (lm200, T=512, fresh env)\n")
print("If PC's indirect route is fully closed, |θ̂| should match Level15's. "
      "If not, |θ̂| will balloon as in NoBypass (~3800).\n")
print("| Variant | mean |θ̂| | std |θ̂| | mean |d_t| (correction) |")
print("|---|---|---|---|")
for v in variants:
    ckpt = Path(f"mapformer/runs/{v}_lm200/seed0/{v}.pt")
    if not ckpt.exists(): continue
    m = build(v, ckpt)
    if not hasattr(m, "last_theta_path"):
        # vanilla MapFormer doesn't expose these. skip.
        del m; continue
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=200, seed=12345)
    np.random.seed(12345); torch.manual_seed(12345)
    th_list, d_list = [], []
    with torch.no_grad():
        for _ in range(5):
            tokens, _, _ = env.generate_trajectory(512)
            tt = tokens.unsqueeze(0).cuda()
            try:
                _ = m(tt[:, :-1])
            except Exception:
                continue
            th_list.append(m.last_theta_hat[0].cpu().numpy())
            d_list.append((m.last_theta_hat[0] - m.last_theta_path[0]).cpu().numpy())
    th = np.stack(th_list); d = np.stack(d_list)
    print(f"| **{v}** | "
          f"{np.abs(th).mean():.2f} | "
          f"{np.abs(th).std():.2f} | "
          f"{np.abs(d).mean():.3f} |")
    del m; torch.cuda.empty_cache()

print("\n**Predictions if Fix 8 closes the indirect route:**")
print("- v4 should match Level15 *exactly* on clean and lm200 (PC is purely a passive observer)")
print("- v4's |θ̂| should match Level15's (~80-100), not NoBypass's (~3800)")
print()
print("If v4 matches Level15: confirms ANY PC influence on the model degrades")
print("performance in this regime. Combinable PC + Kalman is mechanistically")
print("falsified — they require full architectural separation.")
print("If v4 still differs: something we haven't accounted for. Debug further.")
print("\n*Auto-generated by run_v4_test.sh*\n")
PYEOF

echo "[$(date)] Stage 2 done."

cd "$REPO"
git add V4_RESULTS.md model_level15_pc_v4.py train_variant.py run_v4_test.sh
git add runs/Level15PC_v4_clean/seed0/*.pt \
        runs/Level15PC_v4_lm200/seed0/*.pt 2>/dev/null || true
git commit -m "Level15PC_v4: full PC isolation (Fix 5 + 6 + 8)

Adds MapFormerWM_Level15PC_v4 with all three detachment fixes:
  Fix 5: stop-gradient on d_t (InEKF correction) inside PC aux loss
  Fix 6: mask aux loss at landmark token positions
  Fix 8: also detach theta_hat AND token embedding inside PC aux loss

This makes PC's aux loss a pure forward_model trainer — cannot affect
action_to_lie, omega, InEKF parameters, or token embeddings. Verified
mechanically by gradient norm check (forward_model: nonzero, all others:
None).

V4_RESULTS.md compares Level15, Level15PC, NoBypass, v3, v4 on:
  - revisit accuracy at T=128/512 in-dist + OOD on clean and lm200
  - |theta_hat| magnitude (the smoking-gun diagnostic from
    LENGTH_DIAGNOSTIC.md showing NoBypass blew theta_path up to ~3800)

If v4 matches Level15: combinable PC + Kalman is mechanistically
falsified — they require full architectural separation. The forward
model is a passive representation probe, not a gradient contributor.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3
echo "[$(date)] Done."

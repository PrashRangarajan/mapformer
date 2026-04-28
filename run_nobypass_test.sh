#!/bin/bash
# run_nobypass_test.sh — train + evaluate Level15PC_NoBypass on lm200.
#
# Hypothesis test: if R-saturation (driving K → 1, bypassing path
# integration) is the mechanism behind Level15PC's lm200 regression,
# then Level15PC_NoBypass (Fix 5: stop-gradient on InEKF correction
# inside PC aux loss + Fix 6: mask aux loss at landmark tokens) should:
#
#   - Match or exceed Level15-alone's lm200 OOD accuracy (~0.821)
#   - Show R_t distribution similar to Level15 (large spread across
#     token types), NOT saturated like Level15PC
#   - Inherit PC's clone-separation benefit on aliased tokens
#
# Strategy:
#   1. Wait for run_interference_tests.sh pipeline to finish
#   2. Train Level15PC_NoBypass on lm200 single seed (s0)
#   3. Train on clean s0 too (sanity: should match Level15's clean perf)
#   4. Run R_t distribution test (extended to include the new variant)
#   5. Run revisit accuracy at T=128 / T=512, in-dist + OOD
#   6. Run clone-separation test (in-dist + OOD)
#   7. Commit + push
#
# Launch:
#   nohup bash /home/prashr/mapformer/run_nobypass_test.sh \
#       > /home/prashr/mapformer/logs/nobypass_test.log 2>&1 &

set -u
cd /home/prashr
REPO=/home/prashr/mapformer
LOGS=$REPO/logs
mkdir -p "$LOGS"

is_running() {
    ps aux | grep -E "$1" | grep -v grep | grep -v run_nobypass_test >/dev/null
}

echo "[$(date)] Waiting for current interference pipeline to finish..."
while is_running "run_interference_tests|aux_coef_sweep|train_variant.*aux"; do
    sleep 60
done
echo "[$(date)] Interference pipeline done. Starting Level15PC_NoBypass tests."

# ---- Train clean s0 (GPU 0) + lm200 s0 (GPU 1) in parallel ----
echo "[$(date)] Training Level15PC_NoBypass on clean s0 + lm200 s0..."
mkdir -p "$REPO/runs/Level15PC_NoBypass_clean/seed0"
mkdir -p "$REPO/runs/Level15PC_NoBypass_lm200/seed0"

CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.train_variant \
    --variant Level15PC_NoBypass --seed 0 \
    --n-landmarks 0 --p-action-noise 0.0 \
    --epochs 50 --n-batches 156 --aux-coef 0.1 \
    --device cuda \
    --output-dir mapformer/runs/Level15PC_NoBypass_clean/seed0 \
    > "$LOGS/Level15PC_NoBypass_clean_s0.log" 2>&1 &
P_CLEAN=$!

CUDA_VISIBLE_DEVICES=1 python3 -u -m mapformer.train_variant \
    --variant Level15PC_NoBypass --seed 0 \
    --n-landmarks 200 --p-action-noise 0.10 \
    --epochs 50 --n-batches 156 --aux-coef 0.1 \
    --device cuda \
    --output-dir mapformer/runs/Level15PC_NoBypass_lm200/seed0 \
    > "$LOGS/Level15PC_NoBypass_lm200_s0.log" 2>&1 &
P_LM=$!

wait $P_CLEAN $P_LM
echo "[$(date)] Both trainings done."
echo "  clean final: $(grep 'Epoch  50/50' $LOGS/Level15PC_NoBypass_clean_s0.log | tail -1)"
echo "  lm200 final: $(grep 'Epoch  50/50' $LOGS/Level15PC_NoBypass_lm200_s0.log | tail -1)"

# ---- R_t distribution comparison: extend to include NoBypass ----
echo "[$(date)] Stage 2: R_t distribution comparison (Level15 vs Level15PC vs NoBypass)..."
cd /home/prashr
# Modify r_t_distribution_test on the fly to include the new variant
python3 -u -c "
import sys
sys.path.insert(0, '/home/prashr')
from mapformer import r_t_distribution_test as t1
t1.VARIANT_CLS['Level15PC_NoBypass'] = __import__(
    'mapformer.model_level15_pc_v2', fromlist=['MapFormerWM_Level15PC_NoBypass']
).MapFormerWM_Level15PC_NoBypass

# Temporarily override the variants list inside main()
import argparse
sys.argv = [
    'r_t_distribution_test',
    '--runs-dir', 'mapformer/runs',
    '--config', 'lm200',
    '--seed', '0',
    '--T', '512', '--n-trials', '30',
    '--output', 'mapformer/R_T_DISTRIBUTION_3WAY.md',
]

# Patch main to include the new variant
orig_main = t1.main
def patched_main():
    # Hijack at the variants line
    import argparse
    from pathlib import Path
    import numpy as np, torch
    from mapformer.environment import GridWorld
    args = argparse.Namespace(
        runs_dir='mapformer/runs', config='lm200', seed=0,
        T=512, n_trials=30, test_seed=12345,
        output='mapformer/R_T_DISTRIBUTION_3WAY.md', device='cuda',
    )
    runs = Path(args.runs_dir)
    n_lm = 200
    results = {}
    for variant in ['Level15', 'Level15PC', 'Level15PC_NoBypass']:
        ckpt = runs / f'{variant}_lm200' / f'seed{args.seed}' / f'{variant}.pt'
        if not ckpt.exists():
            print(f'[skip] {variant}: no ckpt at {ckpt}', file=sys.stderr); continue
        model, cfg = t1.build_model(variant, ckpt, args.device)
        env = GridWorld(size=cfg.get('grid_size', 64),
                        n_obs_types=cfg.get('n_obs_types', 16),
                        p_empty=cfg.get('p_empty', 0.5),
                        n_landmarks=cfg.get('n_landmarks', n_lm),
                        seed=args.test_seed)
        np.random.seed(args.test_seed); torch.manual_seed(args.test_seed)
        d = t1.collect_R_distribution(model, env, args.T, args.n_trials, args.device)
        results[variant] = d
        print(f'  {variant}: spread = '
              f'{max(d[k].mean() for k in d) - min(d[k].mean() for k in d):.3f}',
              file=sys.stderr)
    md = ['# R_t distribution: Level15 vs Level15PC vs Level15PC_NoBypass (lm200, seed 0)\n']
    md.append('Lower log_R = sharper Kalman gain. Predicted ordering: '
              'landmark < aliased < blank < action.\n')
    md.append('| Variant | action | blank | aliased | landmark | spread |')
    md.append('|---|---|---|---|---|---|')
    for v in results:
        d = results[v]
        means = {k: d[k].mean() if len(d[k]) else float('nan') for k in d}
        spread = max(means.values()) - min(means.values())
        md.append(f'| **{v}** | {means[\"action\"]:+.3f} | '
                  f'{means[\"blank\"]:+.3f} | {means[\"aliased\"]:+.3f} | '
                  f'{means[\"landmark\"]:+.3f} | **{spread:.3f}** |')
    md.append('\n*Auto-generated*\n')
    Path(args.output).write_text('\n'.join(md))
    print('wrote', args.output, file=sys.stderr)

patched_main()
" 2>"$LOGS/r_t_3way.err"
echo "[$(date)] R_t distribution done."

# ---- Revisit accuracy comparison: T=128 + T=512, in-dist + OOD ----
echo "[$(date)] Stage 3: Revisit accuracy on Level15PC_NoBypass..."
cd /home/prashr
python3 -u <<'PYEOF' > "$REPO/NOBYPASS_RESULTS.md" 2>"$LOGS/nobypass_eval.err"
import torch, torch.nn.functional as F, numpy as np, statistics as st
from pathlib import Path
from mapformer.environment import GridWorld
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_level15_pc import MapFormerWM_Level15PC
from mapformer.model_level15_pc_v2 import MapFormerWM_Level15PC_NoBypass

VARIANT_CLS = {
    "Level15": MapFormerWM_Level15InEKF,
    "Level15PC": MapFormerWM_Level15PC,
    "Level15PC_NoBypass": MapFormerWM_Level15PC_NoBypass,
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

print("# Level15PC_NoBypass — revisit accuracy comparison\n")
print("Single seed (s0). For each variant, evaluate at T=128 (training length) "
      "and T=512 (4x OOD), on both in-distribution env and a fresh obs_map "
      "(seed+1000).\n")

variants = ["Level15", "Level15PC", "Level15PC_NoBypass"]
for cfg_tag, n_lm, p_noise in [("clean", 0, 0.0), ("lm200", 200, 0.10)]:
    print(f"## Config: {cfg_tag}\n")
    print("| Variant | T=128 in-dist | T=512 in-dist | T=128 OOD | T=512 OOD |")
    print("|---|---|---|---|---|")
    for v in variants:
        ckpt = Path(f"mapformer/runs/{v}_{cfg_tag}/seed0/{v}.pt")
        if not ckpt.exists():
            print(f"| {v} | — | — | — | — | (no ckpt) |")
            continue
        m = build(v, ckpt)
        # in-dist
        env_id = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                           n_landmarks=n_lm, seed=0)
        a128_id, n128_id = eval_revisit(m, env_id, 128, 200, seed=1000)
        a512_id, n512_id = eval_revisit(m, env_id, 512, 100, seed=1000)
        # OOD
        env_ood = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                            n_landmarks=n_lm, seed=1000)
        a128_ood, n128_ood = eval_revisit(m, env_ood, 128, 200, seed=2000)
        a512_ood, n512_ood = eval_revisit(m, env_ood, 512, 100, seed=2000)
        print(f"| **{v}** | "
              f"{a128_id:.3f} (NLL {n128_id:.3f}) | "
              f"{a512_id:.3f} (NLL {n512_id:.3f}) | "
              f"{a128_ood:.3f} (NLL {n128_ood:.3f}) | "
              f"{a512_ood:.3f} (NLL {n512_ood:.3f}) |")
        del m; torch.cuda.empty_cache()
    print()

print("\n**Predicted (if R-saturation diagnosis is correct):**")
print("- Level15PC_NoBypass should match Level15 on lm200 OOD T=512 (~0.82)")
print("- Level15PC_NoBypass should match Level15 on clean (already at ceiling)")
print("- Level15PC (current) should still show the lm200 regression to ~0.59")
print("\n*Auto-generated by run_nobypass_test.sh*\n")
PYEOF
echo "[$(date)] Revisit accuracy eval done."

# ---- Clone-separation test (NoBypass should inherit PC's win on aliased) ----
echo "[$(date)] Stage 4: Clone-separation test on NoBypass..."
cd /home/prashr
CUDA_VISIBLE_DEVICES=0 python3 -u -m mapformer.clone_transfer_test \
    --runs-dir mapformer/runs --config clean --seed 0 \
    --variants Vanilla Level15 PC Level15PC Level15PC_NoBypass \
    --T 128 --n-trials 200 \
    --output "$REPO/CLONE_TRANSFER_NOBYPASS.md" 2>"$LOGS/clone_nobypass.err" || true
echo "[$(date)] Clone-separation done."

# ---- Commit + push ----
cd "$REPO"
git add R_T_DISTRIBUTION_3WAY.md NOBYPASS_RESULTS.md CLONE_TRANSFER_NOBYPASS.md \
        model_level15_pc_v2.py train_variant.py \
        long_sequence_eval.py per_visit_eval.py zero_shot_eval.py \
        calibration_analysis.py hippocampal_hidden_eval.py \
        run_nobypass_test.sh
git add runs/Level15PC_NoBypass_clean/seed0/*.pt \
        runs/Level15PC_NoBypass_lm200/seed0/*.pt 2>/dev/null || true
git commit -m "Level15PC_NoBypass: stop-gradient + landmark mask fixes for R-saturation

Adds MapFormerWM_Level15PC_NoBypass with two architectural fixes targeting
the R-saturation autoencoder bypass diagnosed in Test 1 (R_T_DISTRIBUTION.md):

  Fix 5 (stop-gradient on InEKF correction inside PC aux loss): PC can no
    longer drive R_t -> 0 to bypass path integration via the measurement.
  Fix 6 (mask aux loss at landmark tokens): removes the noise gradient at
    one-shot tokens that motivated the saturation in the first place.

Sanity check verified: PC aux loss has no gradient path to R-head, z-head,
or log_Pi (confirmed by zero gradient on backward), but does flow through
action_to_lie (legitimate path-integration improvement route).

Trained on clean + lm200 single seed (s0). Compared against Level15 and
Level15PC on:
  - R_t distribution by token type (R_T_DISTRIBUTION_3WAY.md)
  - Revisit accuracy in-dist + OOD at T=128/512 (NOBYPASS_RESULTS.md)
  - Clone-separation in-dist vs OOD env (CLONE_TRANSFER_NOBYPASS.md)

If diagnosis is right: NoBypass recovers Level15's lm200 OOD performance.
" 2>&1 | tail -3
git push origin main 2>&1 | tail -3

echo "[$(date)] run_nobypass_test.sh complete."

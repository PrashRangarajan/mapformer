#!/usr/bin/env python3
"""Test 1: R_t distribution comparison — Level15 vs Level15PC on lm200.

Tests whether PC's aux loss "blocks" the per-token R_t head from learning
the landmark/aliased/blank/action gating that Level 1.5 alone learns.

Compute mean log_R_t broken out by token type:
  - action tokens (even positions, vocab 0..3)
  - blank obs (vocab 20)
  - aliased obs (vocab 4..19, the 16 K-many aliased types)
  - landmark obs (vocab 21+, unique single-use IDs)

Hypothesis (interference story):
  Level15:    R_landmark < R_aliased ≤ R_blank ≤ R_action  (landmarks lowest;
              actions highest because they carry no measurement)
  Level15PC:  distribution should be FLATTER — the PC aux loss noise prevents
              R from differentiating sharply.

If the distributions look similar, the interference story is wrong and the
combined-model failure has a different mechanism.

Output:
  R_T_DISTRIBUTION.md — table comparing mean log_R per token type per variant.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_level15_pc import MapFormerWM_Level15PC


VARIANT_CLS = {
    "Level15":   MapFormerWM_Level15InEKF,
    "Level15PC": MapFormerWM_Level15PC,
}


def build_model(variant, ckpt_path, device="cuda"):
    c = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = c.get("config", {})
    cls = VARIANT_CLS[variant]
    m = cls(
        vocab_size=cfg.get("vocab_size"),
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 2),
        n_layers=cfg.get("n_layers", 1),
        grid_size=cfg.get("grid_size", 64),
    )
    m.load_state_dict(c["model_state_dict"])
    return m.to(device).eval(), cfg


@torch.no_grad()
def collect_R_distribution(model, env, T=512, n_trials=30, device="cuda"):
    """For Level 1.5 variants: collect log-R_t broken out by token type.

    Returns dict: {"action": [...], "blank": [...], "aliased": [...], "landmark": [...]}
    """
    if not hasattr(model, "inekf"):
        return None

    out = {"action": [], "blank": [], "aliased": [], "landmark": []}

    for _ in range(n_trials):
        tokens, _, _ = env.generate_trajectory(T)
        tt = tokens.unsqueeze(0).to(device)
        try:
            _ = model(tt[:, :-1])
        except Exception:
            continue
        if not hasattr(model, "last_R") or model.last_R is None:
            return None
        # last_R: (B, L, H, NB) → mean over heads × blocks → (L,)
        R = model.last_R[0]
        log_R_per_tok = torch.log(R).mean(dim=(1, 2)).cpu().numpy()
        toks = tt[0, :len(log_R_per_tok)].cpu().numpy()
        for i, tok in enumerate(toks):
            tok = int(tok)
            if i % 2 == 0:
                # action token (even position)
                out["action"].append(log_R_per_tok[i])
            else:
                # obs token (odd position)
                obs_id = tok - env.obs_offset
                if obs_id < 0:
                    continue
                if obs_id == env.blank_token:
                    out["blank"].append(log_R_per_tok[i])
                elif obs_id < env.n_obs_types:
                    out["aliased"].append(log_R_per_tok[i])
                else:
                    out["landmark"].append(log_R_per_tok[i])
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--config", default="lm200")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--test-seed", type=int, default=12345)
    p.add_argument("--output", default="R_T_DISTRIBUTION.md")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    runs = Path(args.runs_dir)
    n_lm = 200 if args.config == "lm200" else 0

    results = {}
    for variant in ["Level15", "Level15PC"]:
        ckpt = runs / f"{variant}_{args.config}" / f"seed{args.seed}" / f"{variant}.pt"
        if not ckpt.exists():
            print(f"[skip] {variant}: no ckpt at {ckpt}", file=sys.stderr)
            continue
        try:
            model, cfg = build_model(variant, ckpt, args.device)
        except Exception as e:
            print(f"[skip] {variant}: {e}", file=sys.stderr); continue
        env = GridWorld(
            size=cfg.get("grid_size", 64),
            n_obs_types=cfg.get("n_obs_types", 16),
            p_empty=cfg.get("p_empty", 0.5),
            n_landmarks=cfg.get("n_landmarks", n_lm),
            seed=args.test_seed,
        )
        np.random.seed(args.test_seed); torch.manual_seed(args.test_seed)
        d = collect_R_distribution(model, env, args.T, args.n_trials, args.device)
        if d is None:
            print(f"[skip] {variant}: model has no last_R", file=sys.stderr); continue
        results[variant] = d
        print(f"  {variant} done: "
              f"action n={len(d['action'])}, blank n={len(d['blank'])}, "
              f"aliased n={len(d['aliased'])}, landmark n={len(d['landmark'])}",
              file=sys.stderr)

    # Build markdown
    md = ["# Test 1: R_t distribution by token type (Level15 vs Level15PC on lm200)\n"]
    md.append("**Hypothesis:** PC's aux loss flattens R_t's per-token-type "
              "differentiation, blocking the gating that lets Level 1.5 use "
              "landmarks effectively.\n")
    md.append("Lower log_R = sharper Kalman gain = stronger correction. "
              "Predicted ordering for Level15 (works on lm200): "
              "`landmark < aliased < blank < action`.\n")

    md.append("| Variant | action mean | blank mean | aliased mean | landmark mean | spread (max-min) |")
    md.append("|---|---|---|---|---|---|")
    for v in results:
        d = results[v]
        means = {k: d[k].mean() if len(d[k]) else float("nan") for k in d}
        spread = max(means.values()) - min(means.values())
        md.append(
            f"| **{v}** | "
            f"{means['action']:+.3f} | "
            f"{means['blank']:+.3f} | "
            f"{means['aliased']:+.3f} | "
            f"{means['landmark']:+.3f} | "
            f"**{spread:.3f}** |"
        )

    md.append("\n**Interpretation:**\n")
    md.append("- If `Level15` shows large spread (≥1.0) and the predicted ordering, "
              "and `Level15PC` shows small spread (≤0.5) or jumbled ordering: "
              "**interference confirmed**. PC actively prevents R-gating.")
    md.append("- If both show similar spread/ordering: interference NOT the mechanism. "
              "Failure is something else (gradient quality, optimisation, or capacity).")
    md.append("\n## Per-token-type std (within-class variance)\n")
    md.append("| Variant | action std | blank std | aliased std | landmark std |")
    md.append("|---|---|---|---|---|")
    for v in results:
        d = results[v]
        stds = {k: d[k].std() if len(d[k]) else float("nan") for k in d}
        md.append(
            f"| {v} | "
            f"{stds['action']:.3f} | "
            f"{stds['blank']:.3f} | "
            f"{stds['aliased']:.3f} | "
            f"{stds['landmark']:.3f} |"
        )

    md.append("\n*Auto-generated by `r_t_distribution_test.py`.*\n")
    Path(args.output).write_text("\n".join(md))
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

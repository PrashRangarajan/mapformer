"""Linear-probe of frozen prediction-trained models for goal-directed action info.

For each prediction-trained checkpoint:
  1. Freeze the model.
  2. Generate goal-directed episodes (same as train_goal.py).
  3. Forward pass; capture the pre-`out_proj` hidden state at action-target
     positions.
  4. Train a single Linear(d_model -> 4) head to predict the BFS-optimal
     action. Backbone is frozen — gradient flows only through the head.

Compare across variants: how much goal-directed information is *already* in
each model's representation, before any goal-directed training?

This is a cleaner test of cognitive-map content than full fine-tuning
because the backbone wasn't trained for the task — anything the head can
extract was emergent from prediction training alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment_goal import GoalDirectedGridWorld
from mapformer.model import MapFormerWM, MapFormerEM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_level15_em import MapFormerEM_Level15InEKF
from mapformer.model_inekf_level15_nodrop import MapFormerWM_Level15NoDrop
from mapformer.model_tem_faithful import TEMFaithful


VARIANT_CLS = {
    "Vanilla":         MapFormerWM,
    "VanillaEM":       MapFormerEM,
    "Level15":         MapFormerWM_Level15InEKF,
    "Level15EM":       MapFormerEM_Level15InEKF,
    "Level15NoDrop":   MapFormerWM_Level15NoDrop,
    "TEMFaithful":     TEMFaithful,
}


class _HiddenCapture:
    """Hook that captures the input to `out_proj` (the pre-output hidden)."""
    def __init__(self):
        self.hidden = None
    def __call__(self, module, inp, out):
        # inp is a tuple; out_proj is Linear, takes one arg
        self.hidden = inp[0].detach()


def get_hidden(model, tokens):
    cap = _HiddenCapture()
    handle = model.out_proj.register_forward_hook(cap)
    with torch.no_grad():
        _ = model(tokens)
    handle.remove()
    return cap.hidden  # (B, L, d_model)


def build(variant, ckpt):
    c = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = c["config"]; cls = VARIANT_CLS[variant]
    m = cls(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            grid_size=cfg["grid_size"])
    m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval(), cfg


def collect_features(model, env, n_episodes, T_explore, T_navigate, device="cuda"):
    """Run frozen forward pass, return (features, targets) for all action-target positions."""
    rng = np.random.RandomState(0)
    feats, tgts = [], []
    with torch.no_grad():
        for _ in range(n_episodes):
            t, _, am, _ = env.generate_goal_episode(T_explore=T_explore,
                                                     T_navigate=T_navigate, rng=rng)
            tt = t.unsqueeze(0).to(device)
            am1 = am.unsqueeze(0).to(device)
            inp = tt[:, :-1]; mask = am1[:, :-1]; targets = tt[:, 1:]
            h = get_hidden(model, inp)  # (1, L-1, d)
            feats.append(h[mask].cpu())
            tgts.append(targets[mask].cpu())
    return torch.cat(feats), torch.cat(tgts)


def train_linear_probe(feats, targets, n_classes=4, epochs=100, lr=1e-2, device="cuda"):
    """Train a linear head: (d_model) -> (4 actions). targets are action token ids in [0,4)."""
    d = feats.shape[1]
    head = nn.Linear(d, n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    feats_d = feats.to(device); tgt_d = targets.to(device).long()
    # action tokens are 0-3 directly; no offset
    for ep in range(epochs):
        logits = head(feats_d)
        loss = F.cross_entropy(logits, tgt_d)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (head(feats_d).argmax(-1) == tgt_d).float().mean().item()
    return head, acc


def evaluate_probe(head, model, env, n_episodes, T_explore, T_navigate, device="cuda", seed=2000):
    rng = np.random.RandomState(seed)
    correct = 0; total = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            t, _, am, _ = env.generate_goal_episode(T_explore=T_explore,
                                                     T_navigate=T_navigate, rng=rng)
            tt = t.unsqueeze(0).to(device)
            am1 = am.unsqueeze(0).to(device)
            inp = tt[:, :-1]; mask = am1[:, :-1]; targets = tt[:, 1:]
            h = get_hidden(model, inp)
            preds = head(h[mask]).argmax(-1)
            correct += (preds == targets[mask].long()).sum().item()
            total += mask.sum().item()
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-md", default="PROBE_GOAL_RESULTS.md")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=200)
    args = ap.parse_args()

    env_train = GoalDirectedGridWorld(size=64, n_obs_types=16, p_empty=0.5,
                                      n_landmarks=200, seed=0)
    env_test  = GoalDirectedGridWorld(size=64, n_obs_types=16, p_empty=0.5,
                                      n_landmarks=200, seed=1000)

    out_lines = []
    out_lines.append("# Linear probe of frozen prediction-trained models for goal-directed info\n")
    out_lines.append("Setup: take each prediction-trained lm200 checkpoint (seed 0). Freeze it.")
    out_lines.append("Capture pre-out_proj hidden state at navigate-phase action-target positions")
    out_lines.append("on goal-directed episodes (random-walk explore → BFS to goal landmark).")
    out_lines.append("Train a single Linear(d_model→4) head to predict the BFS-optimal action.")
    out_lines.append("Evaluate on held-out env (different obs_map). Chance = 0.25.\n")
    out_lines.append("This isolates *what the cognitive map encodes about goal-directed action*")
    out_lines.append("WITHOUT any goal-directed training of the backbone. Compare across variants")
    out_lines.append("to test whether correction's gain in BC fine-tuning reflects richer underlying")
    out_lines.append("representation, or merely better learnability.\n")
    out_lines.append("| Variant | Train-probe acc | Held-out probe acc |")
    out_lines.append("|---|---|---|")

    for variant in ["Vanilla", "Level15", "Level15EM", "Level15NoDrop"]:
        ckpt = Path(f"mapformer/runs/{variant}_lm200/seed0/{variant}.pt")
        if not ckpt.exists():
            out_lines.append(f"| {variant} | — | — |"); continue
        try:
            model, cfg = build(variant, ckpt)
        except Exception as e:
            out_lines.append(f"| {variant} | err: {e!s:.40} | — |"); continue
        # Collect train features
        feats_tr, tgts_tr = collect_features(model, env_train, args.n_train,
                                             T_explore=64, T_navigate=64)
        head, tr_acc = train_linear_probe(feats_tr, tgts_tr.long())
        te_acc = evaluate_probe(head, model, env_test, args.n_eval,
                                T_explore=64, T_navigate=64)
        out_lines.append(f"| **{variant}** | {tr_acc:.3f} | {te_acc:.3f} |")
        del model; torch.cuda.empty_cache()

    out_lines.append("\n## Interpretation\n")
    out_lines.append("- Held-out probe acc near 0.25: the cognitive map carries no goal-directed")
    out_lines.append("  info; the BC win was pure learnability (the body adapts).")
    out_lines.append("- Held-out probe acc well above 0.25, ordered like BC results: the map")
    out_lines.append("  ITSELF encodes goal-relevant info; correction enriches the representation.")
    out_lines.append("- TEMFaithful for reference (post-fix lm200 leader on prediction).\n")
    out_lines.append("*Auto-generated by probe_goal_linear.py*\n")

    with open(args.output_md, "w") as f:
        f.write("\n".join(out_lines))
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()

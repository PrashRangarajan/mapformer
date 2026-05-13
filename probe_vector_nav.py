"""Vector navigation probe — the cleanest Tolman cognitive-map test we can do.

Given a frozen prediction-trained model that has only seen random walks
(no goal-directed training, no goal tokens), probe whether its representation
supports navigation to ARBITRARY landmarks.

Setup:
  1. Pick a prediction-trained lm200 checkpoint. Freeze it.
  2. Generate random-walk trajectories on a held-out env.
  3. At each time t, the agent is at position (x_t, y_t). Some landmarks
     have been visited; some haven't.
  4. For a randomly chosen target landmark (which may or may not have been
     visited), compute the BFS-optimal next action from (x_t, y_t) to it.
  5. Train a linear probe: (hidden_t, target_landmark_embedding) -> action.

If the probe accuracy is high, the model's representation linearly encodes
both "where am I" AND "where is landmark X" — the prerequisite for vector
navigation.

For the strongest test: include cases where the target landmark was NEVER
visited during the random walk. If accuracy stays high, the cognitive map
has integrated landmark positions even without revisits — closest to the
Tolman cognitive-map claim.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM
from mapformer.model_inekf_level15 import MapFormerWM_Level15InEKF
from mapformer.model_inekf_gsf_nodrop import MapFormerWM_Level15GSF_NoDrop
from mapformer.model_baseline_rope import MapFormerWM_RoPE
from mapformer.environment_goal import bfs_torus

VARIANT_CLS = {
    "RoPE":               MapFormerWM_RoPE,
    "Vanilla":            MapFormerWM,
    "Level15":            MapFormerWM_Level15InEKF,
    "Level15GSF_NoDrop":  MapFormerWM_Level15GSF_NoDrop,
}


class _HiddenCapture:
    def __init__(self): self.h = None
    def __call__(self, mod, inp, out): self.h = inp[0].detach()


def get_hidden(model, tokens):
    cap = _HiddenCapture()
    handle = model.out_proj.register_forward_hook(cap)
    with torch.no_grad():
        _ = model(tokens)
    handle.remove()
    h = cap.h
    if h is None: return None
    if h.dim() == 4: h = h[:, 0]  # GSF: take mode 0
    return h


def build(variant, ckpt):
    c = torch.load(ckpt, map_location="cuda", weights_only=False)
    cfg = c["config"]; cls = VARIANT_CLS[variant]
    kw = dict(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
              n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
              grid_size=cfg["grid_size"])
    if variant == "Level15GSF_NoDrop": kw["n_modes"] = 8
    m = cls(**kw); m.load_state_dict(c["model_state_dict"])
    return m.cuda().eval(), cfg


def collect_probe_data(model, env, n_trajectories=200, T=128, device="cuda",
                       n_landmarks_per_traj=20, seed=0):
    """For each trajectory:
      - Generate random walk on env
      - Capture hidden state at obs positions
      - For each time t and a randomly chosen target landmark, compute BFS
        next-action from current position to that landmark
      - Record (hidden_t, landmark_idx, current_pos, bfs_action, was_visited)
    """
    rng = np.random.RandomState(seed)
    feats = []        # (N, d_model)
    landmark_idxs = []  # (N,) which landmark is target
    targets = []      # (N,) BFS action
    was_visited = []  # (N,) bool whether target landmark was visited so far

    all_lm = env.landmark_cells  # list of (x, y, lm_idx)

    for traj in range(n_trajectories):
        tokens, _, _ = env.generate_trajectory(T)
        positions = list(env.visited_locations)
        tt = tokens.unsqueeze(0).to(device)
        h = get_hidden(model, tt)  # (1, 2T-1 or 2T, d_model)

        # Determine which landmarks were visited at each point in time
        visited_so_far = set()
        landmark_xy_set = {(l[0], l[1]): l[2] for l in all_lm}

        for t in range(min(T, len(positions))):
            cur_x, cur_y = positions[t]
            obs_pos_in_seq = 2 * t + 1
            if obs_pos_in_seq >= h.shape[1]: continue

            if (cur_x, cur_y) in landmark_xy_set:
                visited_so_far.add(landmark_xy_set[(cur_x, cur_y)])

            # Sample n_landmarks_per_traj target landmarks at random
            if t < 10: continue  # need some history before probing
            for _ in range(min(n_landmarks_per_traj, len(all_lm))):
                lm = all_lm[int(rng.randint(0, len(all_lm)))]
                gx, gy, lm_idx = lm
                # BFS optimal next action
                if (cur_x, cur_y) == (gx, gy): continue
                bfs_path = bfs_torus((cur_x, cur_y), (gx, gy), env.size)
                if not bfs_path: continue
                bfs_a = bfs_path[0]
                feats.append(h[0, obs_pos_in_seq].cpu())
                landmark_idxs.append(lm_idx)
                targets.append(bfs_a)
                was_visited.append(lm_idx in visited_so_far)

    feats = torch.stack(feats)
    landmark_idxs = torch.tensor(landmark_idxs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    was_visited = torch.tensor(was_visited, dtype=torch.bool)
    return feats, landmark_idxs, targets, was_visited


def train_probe(feats, lm_idxs, targets, n_landmarks=200, n_actions=4,
                lm_embed_dim=16, n_epochs=300, lr=1e-2, device="cuda"):
    d = feats.shape[1]
    # Landmark embedding + linear head: (hidden, landmark_id) -> action
    lm_embed = nn.Embedding(n_landmarks, lm_embed_dim).to(device)
    head = nn.Linear(d + lm_embed_dim, n_actions).to(device)
    opt = torch.optim.AdamW(list(lm_embed.parameters()) + list(head.parameters()),
                             lr=lr, weight_decay=1e-4)
    F_d = feats.to(device)
    LM = lm_idxs.to(device)
    T = targets.to(device)
    for ep in range(n_epochs):
        lm_e = lm_embed(LM)
        inp = torch.cat([F_d, lm_e], dim=-1)
        logits = head(inp)
        loss = F.cross_entropy(logits, T)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        preds = head(torch.cat([F_d, lm_embed(LM)], dim=-1)).argmax(-1)
        train_acc = (preds == T).float().mean().item()
    return lm_embed, head, train_acc, loss.item()


def evaluate_probe(lm_embed, head, feats, lm_idxs, targets, device="cuda"):
    F_d = feats.to(device); LM = lm_idxs.to(device); T = targets.to(device)
    with torch.no_grad():
        preds = head(torch.cat([F_d, lm_embed(LM)], dim=-1)).argmax(-1)
        acc = (preds == T).float().mean().item()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-md", default="VECTOR_NAV_RESULTS.md")
    ap.add_argument("--n-train-trajs", type=int, default=100)
    ap.add_argument("--n-eval-trajs", type=int, default=50)
    ap.add_argument("--n-landmarks-per-traj", type=int, default=5)
    args = ap.parse_args()

    env_train = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                          n_landmarks=200, seed=0)  # training env
    env_test = GridWorld(size=64, n_obs_types=16, p_empty=0.5,
                         n_landmarks=200, seed=1000)  # held-out

    results = {}
    for variant in ["RoPE", "Vanilla", "Level15", "Level15GSF_NoDrop"]:
        all_train_acc = []
        all_test_acc = []
        all_visited_acc = []
        all_unvisited_acc = []
        for seed in [0, 1, 2]:
            ckpt = Path(f"mapformer/runs/{variant}_lm200/seed{seed}/{variant}.pt")
            if not ckpt.exists(): continue
            try: model, cfg = build(variant, ckpt)
            except Exception as e:
                print(f"  skip {variant} s{seed}: {e}"); continue

            print(f"\n=== {variant} s{seed} ===")
            print("  collecting train features...")
            train_feats, train_lm, train_tgt, _ = collect_probe_data(
                model, env_train, n_trajectories=args.n_train_trajs,
                T=128, n_landmarks_per_traj=args.n_landmarks_per_traj,
                seed=100 + seed,
            )
            print(f"  {train_feats.shape[0]} train samples")

            print("  training probe...")
            lm_embed, head, train_acc, loss = train_probe(
                train_feats, train_lm, train_tgt,
                n_landmarks=200, n_actions=4,
            )
            print(f"  train acc: {train_acc:.3f}, loss: {loss:.4f}")

            print("  collecting test features...")
            test_feats, test_lm, test_tgt, test_visited = collect_probe_data(
                model, env_test, n_trajectories=args.n_eval_trajs,
                T=128, n_landmarks_per_traj=args.n_landmarks_per_traj,
                seed=200 + seed,
            )
            print(f"  {test_feats.shape[0]} test samples")
            test_acc = evaluate_probe(lm_embed, head, test_feats, test_lm, test_tgt)
            print(f"  test acc (all): {test_acc:.3f}")

            # Split by whether target was visited
            visited_mask = test_visited
            if visited_mask.sum() > 0:
                visited_acc = evaluate_probe(lm_embed, head,
                                              test_feats[visited_mask],
                                              test_lm[visited_mask],
                                              test_tgt[visited_mask])
            else: visited_acc = None
            unvisited_mask = ~test_visited
            if unvisited_mask.sum() > 0:
                unvisited_acc = evaluate_probe(lm_embed, head,
                                                test_feats[unvisited_mask],
                                                test_lm[unvisited_mask],
                                                test_tgt[unvisited_mask])
            else: unvisited_acc = None
            print(f"  test acc (visited landmarks): {visited_acc:.3f}" if visited_acc is not None else "  test acc (visited landmarks): N/A")
            print(f"  test acc (unvisited landmarks): {unvisited_acc:.3f}" if unvisited_acc is not None else "  test acc (unvisited landmarks): N/A")

            all_train_acc.append(train_acc)
            all_test_acc.append(test_acc)
            if visited_acc is not None: all_visited_acc.append(visited_acc)
            if unvisited_acc is not None: all_unvisited_acc.append(unvisited_acc)

            del model, lm_embed, head; torch.cuda.empty_cache()

        if all_test_acc:
            results[variant] = {
                "train_acc": np.mean(all_train_acc),
                "test_acc": np.mean(all_test_acc),
                "test_visited_acc": np.mean(all_visited_acc) if all_visited_acc else None,
                "test_unvisited_acc": np.mean(all_unvisited_acc) if all_unvisited_acc else None,
                "n_seeds": len(all_test_acc),
            }

    # Generate markdown
    lines = []
    lines.append("# Vector navigation probe — Tolman cognitive-map test\n")
    lines.append("Tests whether prediction-trained representations support navigation")
    lines.append("to ARBITRARY landmarks without any goal-directed training. The model")
    lines.append("has only seen random walks; we probe whether its hidden state encodes")
    lines.append("the spatial relationship between current position and any landmark.\n")
    lines.append("**Probe:** given (hidden_t, target_landmark_id), predict the BFS-optimal")
    lines.append("next action to reach that landmark. Linear head + learnable landmark")
    lines.append("embedding. Frozen backbone, all gradient flows through probe only.\n")
    lines.append("**Key split:** test accuracy separated by whether the target landmark")
    lines.append("was VISITED during the random walk so far. If accuracy is similar for")
    lines.append("visited and unvisited landmarks, the model has a TRUE cognitive map —")
    lines.append("it knows landmark positions even before encountering them.\n")
    lines.append("Multi-seed (n=3 prediction-trained lm200 checkpoints). Held-out env (seed=1000).\n")

    lines.append("## Probe accuracy\n")
    lines.append(f"| Variant | Train | Test (all) | Test (visited landmarks) | Test (unvisited landmarks) | n |")
    lines.append("|---|---|---|---|---|---|")
    chance = 1.0 / 4
    for v in ["RoPE", "Vanilla", "Level15", "Level15GSF_NoDrop"]:
        if v not in results:
            lines.append(f"| {v} | — | — | — | — | 0 |"); continue
        r = results[v]
        v_acc = r['test_visited_acc']
        u_acc = r['test_unvisited_acc']
        v_str = f"{v_acc:.3f}" if v_acc is not None else "N/A"
        u_str = f"{u_acc:.3f}" if u_acc is not None else "N/A"
        lines.append(f"| **{v}** | {r['train_acc']:.3f} | {r['test_acc']:.3f} | "
                     f"{v_str} | {u_str} | {r['n_seeds']} |")
    lines.append(f"\nChance ≈ {chance:.3f} (uniform over 4 actions).\n")

    lines.append("## Interpretation\n")
    lines.append("- If RoPE near chance: no cognitive map; the representation has no spatial info")
    lines.append("- If MapFormer family well above chance: cognitive map encodes spatial structure")
    lines.append("- If accuracy on UNVISITED landmarks ≈ visited landmarks: TRUE cognitive map (knows where")
    lines.append("  landmarks ARE even if it hasn't been there yet)")
    lines.append("- If accuracy on unvisited ≪ visited: the 'map' is more like 'memory of visited cells'\n")
    lines.append("*Auto-generated by probe_vector_nav.py*\n")

    with open(args.output_md, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[-30:]))


if __name__ == "__main__":
    main()

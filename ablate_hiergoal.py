"""Check A: are the hier-goal models actually navigating, or exploiting a shortcut?

Motivation. PoPE-Flat holds 0.936 +/- 0.005 from T_explore=256 out to 2048 -- a
32x length extrapolation with no measurable decay, while every other variant
decays. That is either the strongest result in the project or a length-invariant
shortcut, and accuracy alone cannot tell the two apart. This is the same class of
failure as the original hier-goal task, which turned out to be ~94% solvable by
copying the previous action.

Two ablations, each of which ANY genuine goal-directed navigator must fail:

  random_goal      replace the two goal tokens with a different random goal, but
                   score against the ORIGINAL BFS actions. A model that reads the
                   goal must now be wrong.
  shuffle_explore  randomly permute the explore-phase ACTION tokens. The agent's
                   true end-of-explore position is unchanged in the targets, but
                   the observed path no longer implies it. A model that
                   path-integrates must now be wrong.

Episode layout (environment_hier_goal.generate_hier_episode):
  index 0 = room goal, 1 = local goal, then (action, obs) pairs from index 2.
  explore actions at 2, 4, ..., 2+2*(T_explore-1).

Interpretation: a real navigator drops toward the copy-previous-action floor
(0.327 on the interleaved task). A model that barely moves is not using that
information, and its headline number is measuring something else.
"""
import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
import torch

from mapformer.environment_hier_goal import HierGoalGridWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.train_hier_goal import _run
from mapformer.agg_hiergoal import DISPLAY

_REPO = Path(__file__).resolve().parent


def perturb(tokens, cond, env, T_explore, rng):
    """Return a copy of `tokens` with the named ablation applied."""
    t = tokens.clone()
    if cond == "intact":
        return t
    if cond == "random_goal":
        # a DIFFERENT goal, so the original BFS targets become wrong
        for b in range(t.shape[0]):
            old_r, old_l = int(t[b, 0]), int(t[b, 1])
            while True:
                r = env.room_tok0 + int(rng.randint(0, env.n_rooms))
                l = env.local_tok0 + int(rng.randint(0, env.n_local))
                if (r, l) != (old_r, old_l):
                    break
            t[b, 0], t[b, 1] = r, l
        return t
    if cond == "shuffle_explore":
        idx = np.arange(2, 2 + 2 * T_explore, 2)          # explore action slots
        for b in range(t.shape[0]):
            perm = rng.permutation(len(idx))
            t[b, idx] = t[b, idx[perm]]
        return t
    if cond == "destroy_context":
        # Randomise EVERYTHING before the navigate phase: both goal tokens and
        # every explore action AND observation. Only the navigate-phase prefix
        # survives. Surviving accuracy here can ONLY come from continuing the
        # navigate action sequence itself.
        n_ctx = 2 + 2 * T_explore
        for b in range(t.shape[0]):
            t[b, 0] = env.room_tok0 + int(rng.randint(0, env.n_rooms))
            t[b, 1] = env.local_tok0 + int(rng.randint(0, env.n_local))
            for j in range(2, n_ctx, 2):
                t[b, j] = env.action_offset + int(rng.randint(0, env.N_ACTIONS))
                t[b, j + 1] = env.obs_offset + int(rng.randint(0, env.n_obs_types + 1))
        return t
    raise ValueError(cond)


@torch.no_grad()
def run_cond(model, env, cond, T_explore, T_navigate, n_trials, device, seed):
    rng = np.random.RandomState(seed)
    correct = total = 0
    for _ in range(n_trials):
        toks, _om, ams, _info = env.generate_hier_batch(1, T_explore, T_navigate, rng)
        toks = perturb(toks, cond, env, T_explore, rng)
        lp, tgt = _run(model, toks, ams, device)
        if tgt.numel() == 0:
            continue
        correct += (lp.argmax(-1) == tgt).sum().item()
        total += tgt.numel()
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_REPO / "runs/hiergoal_fixed"))
    ap.add_argument("--variants", nargs="+",
                    default=["Vanilla", "Hourglass_k2", "PlainFlat",
                             "PlainHourglass", "PoPE", "MapPoPE_Hier"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--T-explore", type=int, default=128)
    ap.add_argument("--T-navigate", type=int, default=64)
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(_REPO / "HIERGOAL_ABLATION.md"))
    args = ap.parse_args()

    CONDS = ["intact", "random_goal", "shuffle_explore", "destroy_context"]
    env = HierGoalGridWorld(size=64, room_size=8, seed=10000, interleave_path=True)
    res = {v: {c: [] for c in CONDS} for v in args.variants}

    for v in args.variants:
        for s in args.seeds:
            cp = Path(args.runs_dir) / f"seed{s}" / f"{v}_hiergoal.pt"
            if not cp.exists():
                print(f"MISSING {cp}"); continue
            c = torch.load(cp, map_location=args.device, weights_only=False)
            m = VARIANT_MAP[v](vocab_size=c["vocab_size"], d_model=c["d_model"],
                               n_heads=c["n_heads"], n_layers=c["n_layers"],
                               grid_size=64).to(args.device).eval()
            m.load_state_dict(c["model_state"])
            for cond in CONDS:
                a = run_cond(m, env, cond, args.T_explore, args.T_navigate,
                             args.n_trials, args.device, seed=3000 + s)
                res[v][cond].append(a)
                print(f"[{DISPLAY.get(v, v)} s{s}] {cond:16s} acc={a:.4f}", flush=True)

    def cell(xs):
        if not xs:
            return "n/a"
        return f"{st.mean(xs):.3f}" + (f" ± {st.stdev(xs):.3f}" if len(xs) > 1 else "")

    lines = [
        "# Check A: are hier-goal models navigating, or exploiting a shortcut?",
        "",
        f"T_explore={args.T_explore} (OOD), T_navigate={args.T_navigate}, "
        f"n={len(args.seeds)} seeds, interleaved task.",
        "Copy-previous-action floor on this task = **0.327**. A genuine navigator "
        "should fall toward it when the goal is randomised or the explore path is "
        "destroyed; a model that barely moves is not using that information.",
        "",
        "| variant | intact | random_goal | shuffle_explore | destroy_context |",
        "|---|---|---|---|---|",
    ]
    for v in args.variants:
        lines.append(f"| {DISPLAY.get(v, v)} | " +
                     " | ".join(cell(res[v][c]) for c in CONDS) + " |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

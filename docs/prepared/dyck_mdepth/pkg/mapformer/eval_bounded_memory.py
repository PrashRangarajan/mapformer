"""Bounded-memory eval: does PoPE's advantage depend on RETRIEVING the action history?

Hypothesis (from the Route-A/B analysis). Two ways a model can know its position:
  Route A  position lives in the attention geometry -- MapFormer's
           theta = omega*cumsum(Delta), computed in the POSITION pathway and
           carryable as O(1) state.
  Route B  position is re-derived as content -- attend over the past action
           tokens and sum them. Needs an O(T) KV cache.

PoPE repairs long-range retrieval, which is what makes Route B work at OOD
length. If that is the whole story, PoPE should break as soon as the action
history stops being retrievable, while MapFormer (Route A) should not.

Manipulation: a PREFIX-PRESERVING SLIDING WINDOW on attention. Each query may
attend to the first `prefix` tokens (the goal specification -- an attention sink,
as in StreamingLLM) plus the last `window` tokens. Nothing else. Applied
identically to every variant. This is the streaming-inference regime: a running
angle is free to carry, an unbounded KV cache is not.

Eval-only: the checkpoints were TRAINED with full attention, so this measures how
robust the learned representation is to losing history -- which may overstate the
gap relative to models trained under the budget. Retraining at fixed W is the
follow-up if the effect is large.
"""
import argparse
import json
import statistics as st
from pathlib import Path

import torch

from mapformer.environment_hier_goal import HierGoalGridWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.train_hier_goal import evaluate
from mapformer.agg_hiergoal import DISPLAY

_REPO = Path(__file__).resolve().parent


def windowed(mask, window, prefix):
    """Add a sliding-window + prefix constraint to a causal mask.

    mask: (L, L) bool, True = BLOCKED (torch.triu convention used by the models).
    Keeps: the first `prefix` keys, and keys within `window` of the query.
    """
    L = mask.shape[-1]
    idx = torch.arange(L, device=mask.device)
    too_old = (idx[:, None] - idx[None, :]) >= window      # key further back than window
    is_prefix = (idx[None, :] < prefix).expand(L, L)       # always-visible prefix
    return mask | (too_old & ~is_prefix)


def patch(model, window, prefix):
    """Wrap every layer's forward so the causal mask it receives is windowed.

    Touches no model file: layers take `causal_mask` (WM/PoPE/EM) or the
    hourglass's fine/coarse masks positionally, so we rewrite that argument.
    """
    handles = []
    for mod in model.modules():
        if not hasattr(mod, "forward") or not hasattr(mod, "q_proj") and not hasattr(mod, "q_content"):
            continue
        orig = mod.forward

        def make(orig):
            def fwd(*a, **kw):
                a = list(a)
                for i, v in enumerate(a):
                    if torch.is_tensor(v) and v.dtype == torch.bool and v.dim() == 2 \
                       and v.shape[0] == v.shape[1]:
                        a[i] = windowed(v, window, prefix)
                if "causal_mask" in kw and torch.is_tensor(kw["causal_mask"]):
                    kw["causal_mask"] = windowed(kw["causal_mask"], window, prefix)
                return orig(*a, **kw)
            return fwd

        mod.forward = make(orig)
        handles.append((mod, orig))
    return handles


def unpatch(handles):
    for mod, orig in handles:
        mod.forward = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_REPO / "runs/hiergoal_multiseed"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--windows", nargs="+", type=int, default=[100000, 256, 128, 64, 32])
    ap.add_argument("--prefix", type=int, default=2, help="always-visible goal tokens")
    ap.add_argument("--t-explore", type=int, default=64)
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(_REPO / "BOUNDED_MEMORY.md"))
    args = ap.parse_args()

    env = HierGoalGridWorld(size=64, room_size=8, seed=10000)
    acc = {v: {w: [] for w in args.windows} for v in args.variants}
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
            for w in args.windows:
                h = patch(m, w, args.prefix)
                a, _ = evaluate(m, env, args.t_explore, 64, args.n_trials,
                                args.device, seed=2000 + s)
                unpatch(h)
                acc[v][w].append(a)
                print(f"[{DISPLAY.get(v, v)} s{s}] W={w if w < 99999 else 'inf':>6}: acc={a:.3f}", flush=True)
            del m
            torch.cuda.empty_cache()

    def ms(xs):
        return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), 0.0)

    hdr = [("inf" if w > 99999 else str(w)) for w in args.windows]
    lines = ["# Bounded-memory eval: does PoPE need to RETRIEVE the action history?\n",
             f"Prefix-preserving sliding window: each query sees the first {args.prefix} tokens "
             f"(goal) plus the last W. T_explore={args.t_explore} (in-distribution), "
             f"n_trials={args.n_trials}, seeds={args.seeds}. Eval-only: models were TRAINED "
             f"with full attention.\n",
             "| variant | " + " | ".join(f"W={h}" for h in hdr) + " |",
             "|" + "---|" * (len(args.windows) + 1)]
    for v in args.variants:
        cells = [f"{ms(acc[v][w])[0]:.3f} ± {ms(acc[v][w])[1]:.3f}" for w in args.windows]
        lines.append(f"| {DISPLAY.get(v, v)} | " + " | ".join(cells) + " |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {DISPLAY.get(v, v): {str(w): ms(acc[v][w]) for w in args.windows} for v in args.variants},
        indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

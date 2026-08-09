"""Held-out revisit ACCURACY on the paper's task -- the quantity the paper reports.

PAPER_VALIDATION.md recorded training loss only ("held-out revisit acc: n/a"),
so our reproduction was never compared to the paper on its own metric. The paper
reports MapFormer-WM 0.955 and MapFormer-EM 0.999 next-observation accuracy at
revisited cells.

Two held-out settings are reported because `GridWorld` fixes `obs_map` once from
`seed`, so "held out" is ambiguous and the choice changes the meaning:

  same-map   new trajectories on the SAME obs_map the checkpoint trained on.
             In-distribution; the map itself may be memorised in the weights.
  fresh-map  new trajectories on an UNSEEN obs_map (seed=10000). Requires
             building the cognitive map in-context, which is what MapFormer
             claims to do.

Masking mirrors train.py exactly: predict tokens[:, 1:] from tokens[:, :-1],
scored only where revisit_mask[:, 1:] is True (revisited observation positions).
"""
import argparse
import json
import statistics as st
from pathlib import Path

import torch

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP

_REPO = Path(__file__).resolve().parent


@torch.no_grad()
def revisit_accuracy(model, env, n_batches, batch_size, n_steps, device):
    """Accuracy + NLL at revisited observation positions (train.py masking)."""
    model.eval()
    correct = total = 0
    nll_sum = 0.0
    for _ in range(n_batches):
        tokens, _obs_mask, revisit_mask, _locs = env.generate_batch(batch_size, n_steps)
        tokens = tokens.to(device)
        mask = revisit_mask[:, 1:].to(device)
        if mask.sum() == 0:
            continue
        logits = model(tokens[:, :-1])
        tgt = tokens[:, 1:][mask]
        lg = logits[mask]
        correct += (lg.argmax(-1) == tgt).sum().item()
        total += tgt.numel()
        nll_sum += torch.nn.functional.cross_entropy(
            lg, tgt, reduction="sum").item()
    return (correct / max(total, 1), nll_sum / max(total, 1), total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_REPO / "runs/paper_task"))
    ap.add_argument("--variants", nargs="+",
                    default=["Vanilla", "VanillaEM", "VanillaEM_P0"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--fresh-seed", type=int, default=10000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(_REPO / "PAPER_TASK_ACCURACY.md"))
    args = ap.parse_args()

    dev = torch.device(args.device)
    res = {v: {"same": [], "fresh": []} for v in args.variants}

    for v in args.variants:
        cls = VARIANT_MAP[v]
        for s in args.seeds:
            ckpt = Path(args.runs_dir) / f"{v}_s{s}" / f"{v}.pt"
            if not ckpt.exists():
                print(f"MISSING {ckpt} -- skipping")
                continue
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = blob["model_state_dict"]
            # Build from the checkpoint's own stored config rather than
            # hardcoding dims, so a config change can never silently mismatch.
            cfg = blob["config"]

            # obs_map identity is what distinguishes the two settings; the
            # checkpoint trained on GridWorld(seed=s) (see run_em_p0.sh: the
            # trainer passes --seed through to the env).
            for label, env_seed in (("same", s), ("fresh", args.fresh_seed)):
                env = GridWorld(size=cfg["grid_size"],
                                n_obs_types=cfg["n_obs_types"],
                                p_empty=cfg["p_empty"],
                                n_landmarks=cfg["n_landmarks"], seed=env_seed)
                assert env.unified_vocab_size == cfg["vocab_size"], (
                    f"vocab mismatch: env {env.unified_vocab_size} "
                    f"vs ckpt {cfg['vocab_size']}")
                model = cls(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                            grid_size=cfg["grid_size"]).to(dev)
                model.load_state_dict(sd)
                acc, nll, n = revisit_accuracy(model, env, args.n_batches,
                                               args.batch_size, args.n_steps, dev)
                res[v][label].append(acc)
                print(f"[{v} s{s}] {label:5s}-map acc={acc:.4f} nll={nll:.4f} (n={n})")

    def cell(xs):
        if not xs:
            return "n/a"
        if len(xs) == 1:
            return f"{xs[0]:.3f}"
        return f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f}"

    lines = [
        "# Paper-task held-out revisit ACCURACY",
        "",
        "Paper config: 1 layer, 2 heads, d=128, T=128, 200K sequences "
        "(16 epochs x 98 batches x 128).",
        "Paper reports MapFormer-WM **0.955**, MapFormer-EM **0.999**.",
        "",
        "`same-map` = new trajectories on the trained obs_map; `fresh-map` = "
        "unseen obs_map (in-context map learning).",
        "",
        f"| variant | same-map acc | fresh-map acc |",
        "|---|---|---|",
    ]
    for v in args.variants:
        lines.append(f"| {v} | {cell(res[v]['same'])} | {cell(res[v]['fresh'])} |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

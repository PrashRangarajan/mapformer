"""Where does an index-position model's likelihood advantage come from?

The puzzle (INDEX_BASELINE_PAPER_TASK.md): index models reach a training loss of
1.59-1.68 nats on the paper's task, comfortably below the 2.079-nat marginal
entropy of the observation distribution -- yet their revisit ACCURACY is 0.513,
exactly the always-predict-blank floor. They beat the marginal in likelihood
while sitting at it in accuracy.

Hypothesis to test: the paper's walk is DIRECTED with run lengths 1-10, so an
out-and-back run retraces cells in reverse order a few steps later. That is
detectable from the ACTION TOKENS AS CONTENT -- "I went east 6 then turned west"
-- with no position code at all. If so, index models should win only at very
short recurrence intervals and be at the floor everywhere else, while a
path-integrating model should hold up across the whole range.

Measurement: for every revisited observation position, record the RECURRENCE
INTERVAL -- how many steps since that cell was last visited -- and bucket
accuracy by it. The environment already returns per-step locations, so the
interval is read off the trajectory rather than inferred.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP

_REPO = Path(__file__).resolve().parent
BUCKETS = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 10**9)]


def _label(lo, hi):
    return f"{lo}-{hi}" if hi < 10**8 else f"{lo}+"


@torch.no_grad()
def by_distance(model, env, n_batches, batch_size, n_steps, device):
    model.eval()
    hits = defaultdict(int)
    tot = defaultdict(int)
    blank = defaultdict(int)
    for _ in range(n_batches):
        tokens, _om, revisit, locs = env.generate_batch(batch_size, n_steps)
        logits = model(tokens[:, :-1].to(device))
        pred = logits.argmax(-1).cpu()
        tgt = tokens[:, 1:]
        for b in range(tokens.shape[0]):
            last = {}
            for step in range(n_steps):
                cell = tuple(locs[b][step]) if not isinstance(locs[b][step], tuple) \
                    else locs[b][step]
                pos = 2 * step + 1                 # observation token index
                if pos - 1 < pred.shape[1] and cell in last:
                    d = step - last[cell]
                    for lo, hi in BUCKETS:
                        if lo <= d <= hi:
                            k = _label(lo, hi)
                            tot[k] += 1
                            hits[k] += int(pred[b, pos - 1] == tgt[b, pos - 1])
                            blank[k] += int(tgt[b, pos - 1] == env.unified_blank)
                            break
                last[cell] = step
    return {k: dict(acc=hits[k] / tot[k], blank=blank[k] / tot[k], n=tot[k])
            for k in tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_REPO / "runs/paper_task"))
    ap.add_argument("--variants", nargs="+",
                    default=["Vanilla", "RoPE", "PlainFlat"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-batches", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--env-seed", type=int, default=10000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=str(_REPO / "REVISIT_DISTANCE.md"))
    args = ap.parse_args()
    dev = torch.device(args.device)

    agg = {v: defaultdict(list) for v in args.variants}
    blankref = {}
    for v in args.variants:
        for s in args.seeds:
            ck = Path(args.runs_dir) / f"{v}_s{s}" / f"{v}.pt"
            if not ck.exists():
                print(f"MISSING {ck}", flush=True)
                continue
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            cfg = blob["config"]
            env = GridWorld(size=cfg["grid_size"], n_obs_types=cfg["n_obs_types"],
                            p_empty=cfg["p_empty"], n_landmarks=cfg["n_landmarks"],
                            seed=args.env_seed)
            m = VARIANT_MAP[v](vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                               n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                               grid_size=cfg["grid_size"]).to(dev)
            m.load_state_dict(blob["model_state_dict"])
            r = by_distance(m, env, args.n_batches, args.batch_size,
                            args.n_steps, dev)
            for k, d in r.items():
                agg[v][k].append(d["acc"])
                blankref[k] = d["blank"]
            print(f"[{v} s{s}] " + "  ".join(
                f"{k}:{d['acc']:.3f}" for k, d in sorted(
                    r.items(), key=lambda kv: int(kv[0].split('-')[0].rstrip('+')))),
                flush=True)
            del m
            torch.cuda.empty_cache()

    keys = sorted(blankref, key=lambda k: int(k.split("-")[0].rstrip("+")))
    lines = ["# Revisit accuracy by recurrence interval", "",
             "Index-position models beat the marginal in LIKELIHOOD (train loss "
             "1.59-1.68 vs 2.079 nats) while sitting at it in ACCURACY (0.513 vs "
             "a 0.506 blank floor). This asks where that likelihood goes.", "",
             "Recurrence interval = steps since the cell was last visited. The "
             "paper's walk is directed with run lengths 1-10, so an out-and-back "
             "run retraces cells a few steps later -- detectable from the ACTION "
             "TOKENS AS CONTENT, with no position code. If that is the source, "
             "index models win only in the leftmost buckets.", "",
             "| variant | " + " | ".join(keys) + " |",
             "|---" * (len(keys) + 1) + "|"]
    for v in args.variants:
        if not any(agg[v].values()):
            continue
        lines.append(f"| {v} | " + " | ".join(
            f"{np.mean(agg[v][k]):.3f}" if agg[v].get(k) else "—" for k in keys) + " |")
    lines.append("| *blank rate (floor)* | " +
                 " | ".join(f"*{blankref[k]:.3f}*" for k in keys) + " |")
    lines.append("| *n per seed* | " + " | ".join("" for k in keys) + " |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    json.dump({v: {k: agg[v][k] for k in agg[v]} for v in args.variants},
              open(str(args.out).replace(".md", ".json"), "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Context-destruction ablation on the PAPER'S OWN task.

Why this exists
---------------
Match-Query's headline rests on an argument I have been making without measuring
it: that the paper's task leaves a CONTENT route to localisation open, because
observations are revealed at every step, and that Match-Query is a stricter test
because it closes that route.

The arithmetic behind the argument: a 64x64 torus is 4096 cells = 12 bits, and
one observation carries H(0.5) + 0.5*log2(16) = 3 bits, so ~4-5 revealed
observations along a known action path are enough to pin the current cell with
no reliable path integration at all.

If that is right, destroying the ACTION stream should cost the paper's task much
less than it costs Match-Query, where the same manipulation is catastrophic:

    Match-Query, query-phase actions shuffled:   0.918 -> 0.076   (chance 0.0625)

If instead the paper's task collapses just as hard, then Match-Query is not
buying the isolation I have claimed for it, and the claim has to be withdrawn.
The test is set up so either answer is informative and neither is assumed.

Conditions
----------
  intact           baseline.
  shuffle_actions  permute the ACTION tokens among action slots, within each
                   sequence. The observation stream is untouched and the targets
                   are unchanged, so everything needed for content-matching
                   survives and only the action->position mapping dies.
  shuffle_obs      permute the OBSERVATION tokens among observation slots. This
                   destroys revisit consistency -- the same cell no longer emits
                   the same symbol -- so the task becomes unanswerable. Targets
                   follow the shuffle automatically, since they are read off the
                   same token tensor. This is the sanity end of the scale: it
                   MUST fall to the floor, and if it does not, the metric is
                   measuring something other than retrieval.

Floors, measured not assumed
----------------------------
The paper's task scores every revisited observation position, blanks included,
and p_empty = 0.5 makes blank the majority class. So the operative floor is the
always-predict-blank rate, NOT 1/n_obs. Both are reported, along with the
measured blank fraction among scored targets.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP

_REPO = Path(__file__).resolve().parent


def perturb(tokens, donor, cond, rng):
    """Apply the named ablation. Even indices are actions, odd are observations.

    SHUFFLE vs RESAMPLE, and why both are here. Permuting the action slots
    destroys the walk's own statistics as well as its meaning: the paper's walk
    is directed with run lengths 1..10, so its action stream is strongly
    autocorrelated, and a permutation produces a stream no valid trajectory could
    emit. A collapse under `shuffle` therefore conflates "position information
    destroyed" with "input off-manifold" -- and the first run of this file showed
    the models fail CONFIDENTLY off-manifold (NLL 4.5 against ln(21)=3.04 for
    uniform), which is exactly how accuracy ends up BELOW the floor.

    `resample` fixes that. The action (or observation) stream is taken wholesale
    from an INDEPENDENT episode of the same generator, so it is a perfectly valid
    walk with the right run-length statistics -- it simply does not correspond to
    the observations beside it. Any collapse under `resample` cannot be blamed on
    the sequence looking impossible.
    """
    t = tokens.clone()
    if cond == "intact":
        return t
    L = t.shape[1]
    act, obs = np.arange(0, L, 2), np.arange(1, L, 2)
    if cond == "shuffle_actions":
        for b in range(t.shape[0]):
            t[b, act] = t[b, act[rng.permutation(len(act))]]
    elif cond == "shuffle_obs":
        for b in range(t.shape[0]):
            t[b, obs] = t[b, obs[rng.permutation(len(obs))]]
    elif cond == "resample_actions":
        t[:, act] = donor[:, act]          # a real walk, wrong walk
    elif cond == "resample_obs":
        t[:, obs] = donor[:, obs]          # a real observation stream, wrong one
    else:
        raise ValueError(cond)
    return t


@torch.no_grad()
def score(model, env, cond, n_batches, batch_size, n_steps, device, seed):
    """Revisit accuracy under `cond`, plus the floors measured on the same events."""
    model.eval()
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    ok = tot = 0
    nll = 0.0
    tgt_counts = Counter()
    for _ in range(n_batches):
        tokens, _om, revisit, *_ = env.generate_batch(batch_size, n_steps)
        # An INDEPENDENT episode of the same generator, used as the donor stream
        # for the resample conditions. Drawn every batch so it is never reused.
        donor, _dm, _dr, *_ = env.generate_batch(batch_size, n_steps)
        tokens = perturb(tokens, donor, cond, rng).to(device)
        mask = revisit[:, 1:].to(device)
        if not mask.any():
            continue
        logits = model(tokens[:, :-1])
        tgt = tokens[:, 1:]
        lg, tg = logits[mask], tgt[mask]
        ok += int((lg.argmax(-1) == tg).sum().item())
        nll += float(F.cross_entropy(lg, tg, reduction="sum").item())
        tot += int(tg.numel())
        tgt_counts.update(tg.tolist())
    blank_rate = tgt_counts[env.unified_blank] / max(tot, 1)
    marg = max(tgt_counts.values()) / max(tot, 1) if tgt_counts else float("nan")
    return dict(acc=ok / max(tot, 1), nll=nll / max(tot, 1), n=tot,
                blank_rate=blank_rate, marginal=marg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_REPO / "runs/paper_task"))
    ap.add_argument("--variants", nargs="+", default=["Vanilla", "VanillaEM_P0"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--env-seed", type=int, default=10000, help="fresh obs_map")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=str(_REPO / "PAPER_TASK_ABLATION.md"))
    args = ap.parse_args()
    dev = torch.device(args.device)
    CONDS = ["intact", "shuffle_actions", "resample_actions",
             "shuffle_obs", "resample_obs"]

    res = {v: {c: [] for c in CONDS} for v in args.variants}
    meta = {}
    for v in args.variants:
        cls = VARIANT_MAP[v]
        for s in args.seeds:
            ck = Path(args.runs_dir) / f"{v}_s{s}" / f"{v}.pt"
            if not ck.exists():
                print(f"MISSING {ck} -- skipping", flush=True)
                continue
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            cfg = blob["config"]
            env = GridWorld(size=cfg["grid_size"], n_obs_types=cfg["n_obs_types"],
                            p_empty=cfg["p_empty"], n_landmarks=cfg["n_landmarks"],
                            seed=args.env_seed)
            assert env.unified_vocab_size == cfg["vocab_size"]
            model = cls(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                        grid_size=cfg["grid_size"]).to(dev)
            model.load_state_dict(blob["model_state_dict"])
            for c in CONDS:
                r = score(model, env, c, args.n_batches, args.batch_size,
                          args.n_steps, dev, seed=5000 + s)
                res[v][c].append(r["acc"])
                meta[f"{v}_s{s}_{c}"] = r
                print(f"[{v} s{s}] {c:16s} acc={r['acc']:.4f} nll={r['nll']:.4f} "
                      f"(n={r['n']}, blank={r['blank_rate']:.3f})", flush=True)
            del model
            torch.cuda.empty_cache()

    any_meta = next(iter(meta.values()))
    lines = ["# Context-destruction ablation on the paper's own task", "",
             f"T={args.n_steps}, fresh obs_map (env seed {args.env_seed}), "
             f"n={len(args.seeds)} seeds, {args.n_batches}x{args.batch_size} "
             f"sequences per cell. Scored at revisited observation positions -- "
             f"the paper's own target.", "",
             "**Floors, measured on these events, not assumed.** The paper's "
             "task scores every revisited observation including blanks, and "
             f"p_empty=0.5 makes blank the majority class: measured blank rate "
             f"**{any_meta['blank_rate']:.3f}**, so the always-predict-blank "
             f"floor is that, not 1/16. Compare Match-Query, which restricts to "
             "non-blank answers and therefore has a 0.0625 chance and a 0.0893 "
             "never-moved floor.", "",
             "**shuffle vs resample.** `shuffle` permutes the slots, which "
             "also destroys the walk's run-length autocorrelation and puts the "
             "input off-manifold. `resample` substitutes the corresponding stream "
             "from an INDEPENDENT episode -- a perfectly valid walk that simply "
             "does not match the observations beside it. `resample` is the "
             "trustworthy column; `shuffle` is kept because it is what was "
             "reported first.", "",
             "| variant | intact | shuffle actions | **resample actions** | "
             "shuffle obs | resample obs |",
             "|---|---|---|---|---|---|"]
    summ = {}
    for v in args.variants:
        if not res[v]["intact"]:
            continue
        cells, summ[v] = [], {}
        for c in CONDS:
            a = np.array(res[v][c])
            summ[v][c] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1))
                          if len(a) > 1 else 0.0, "per_seed": a.tolist()}
            cells.append(f"{a.mean():.4f} ± {a.std(ddof=1) if len(a)>1 else 0:.4f}")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")

    lines += ["", "## The comparison this was run to make", "",
              "| task | manipulation | intact | destroyed | drop | floor |",
              "|---|---|---|---|---|---|"]
    for v in args.variants:
        if v not in summ:
            continue
        i = summ[v]["intact"]["mean"]
        for c, lab in (("shuffle_actions", "actions shuffled"),
                       ("resample_actions", "actions resampled (on-manifold)")):
            d = summ[v][c]["mean"]
            lines.append(f"| paper task ({v}) | {lab} | {i:.3f} | {d:.3f} "
                         f"| **{d-i:+.3f}** | {any_meta['blank_rate']:.3f} (blank) |")
    lines.append("| Match-Query (MapWM-Flat) | query actions shuffled | 0.918 | "
                 "0.076 | **-0.842** | 0.089 (never-moved) |")
    lines += ["", "Per-seed:", ""]
    for v in summ:
        for c in CONDS:
            lines.append(f"- `{v}` {c}: " +
                         ", ".join(f"{x:.4f}" for x in summ[v][c]["per_seed"]))
    Path(args.out).write_text("\n".join(lines) + "\n")
    json.dump({"summary": summ, "detail": meta},
              open(str(args.out).replace(".md", ".json"), "w"), indent=2)
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

"""Context-destruction for Match-Query, INCLUDING the stochastic-transition variant.

The clean task passed this once (0.918 -> 0.074 with explore observations
shuffled, -> 0.076 with the query path shuffled), and that single check is what
separates it from the 48 files in archive/void -- hier-goal went 0.912 -> 0.913 on
the same manipulation and was voided.

The noisy variant has NEVER been through it. It is a different task: drift means
the recorded action stream no longer locates the agent, so a model could in
principle fall back on something that survives destruction -- observation
marginals, or a positional prior over which cells get scored. Assuming the clean
task's pass carries over is exactly the assumption that has gone wrong here before.

Three conditions, same checkpoints, same scored positions and answers throughout
(answers derive from the TRUE walk, which is never altered):

  control            unmodified episode
  explore-obs        observation tokens PERMUTED within the explore phase -- the
                     map is destroyed, the path is intact
  query-actions      action tokens PERMUTED within the query phase -- the map is
                     intact, the agent can no longer know where it is

A model doing what the task claims must collapse toward the trivial floor under
BOTH. Anything that survives is being scored off something other than a cognitive
map.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from mapformer.environment_match_query import MatchQueryGridWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.train_match_query import _match_logits


@torch.no_grad()
def run(model, env, TE, TQ, n_batches, bs, dev, seed, p_noise, mode):
    rng = np.random.RandomState(seed)
    gen = np.random.RandomState(seed + 1)
    ok = tot = 0
    for _ in range(n_batches):
        toks, _rev, sps, ans, _i = env.generate_match_batch(bs, TE, TQ, rng, p_noise)
        t = toks.clone()
        if mode == "explore-obs":
            # observation tokens sit at ODD indices inside the explore phase
            idx = torch.arange(1, 2 * TE, 2)
            for b in range(t.shape[0]):
                t[b, idx] = t[b, idx[torch.from_numpy(gen.permutation(len(idx)))]]
        elif mode == "query-actions":
            # action tokens sit at EVEN indices inside the query phase
            idx = torch.arange(2 * TE, t.shape[1], 2)
            for b in range(t.shape[0]):
                t[b, idx] = t[b, idx[torch.from_numpy(gen.permutation(len(idx)))]]
        logits = model(t[:, :-1].to(dev))
        for b in range(t.shape[0]):
            for p, a in zip(sps[b], ans[b]):
                if p >= logits.shape[1]:
                    continue
                sl = _match_logits(logits, env, b, p)
                ok += int(sl.argmax().item() == a - env.obs_offset)
                tot += 1
    return ok / max(tot, 1), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="mapformer/runs/mq_noise_c2")
    ap.add_argument("--variants", nargs="+", default=["Vanilla", "Looped"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--t-explore", type=int, default=512)
    ap.add_argument("--t-query", type=int, default=256)
    ap.add_argument("--n-batches", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="MATCH_QUERY_NOISE_ABLATION.md")
    a = ap.parse_args()

    env = MatchQueryGridWorld(size=a.size, n_obs_types=a.n_obs, seed=10000)
    FLOOR = 0.1042   # never-moved baseline, 600-episode gate run at p=0.10
    MODES = ["control", "explore-obs", "query-actions"]
    res = {}
    for tag, p in (("p0", 0.0), ("p010", 0.10)):
        for v in a.variants:
            for s in a.seeds:
                cp = Path(a.runs_dir) / tag / f"{v}_s{s}" / f"{v}_matchquery.pt"
                if not cp.exists():
                    print("MISSING", cp); continue
                c = torch.load(cp, map_location=a.device, weights_only=False)
                m = VARIANT_MAP[v](vocab_size=c["vocab_size"], d_model=c["d_model"],
                                   n_heads=c["n_heads"], n_layers=c["n_layers"],
                                   grid_size=a.size).to(a.device).eval()
                m.load_state_dict(c["model_state"])
                for mode in MODES:
                    acc, n = run(m, env, a.t_explore, a.t_query, a.n_batches,
                                 a.batch_size, a.device, 9000 + s, p, mode)
                    res.setdefault((tag, v, mode), []).append(acc)
                    print(f"[{tag} {v} s{s}] {mode:14s} {acc:.4f} (n={n})", flush=True)
                del m; torch.cuda.empty_cache()

    o = ["# Context destruction on Match-Query, clean AND noisy", "",
         f"{a.size}^2, T_explore={a.t_explore}, T_query={a.t_query}, "
         f"chance {1.0/a.n_obs:.4f}. Same checkpoints, same scored positions and",
         "answers in every condition -- answers derive from the TRUE walk, which is",
         "never altered.", "",
         "| condition | " + " | ".join(MODES) + " |",
         "|---" * (len(MODES) + 1) + "|"]
    for tag, _p in (("p0", 0.0), ("p010", 0.10)):
        for v in a.variants:
            cells = []
            for mode in MODES:
                r = res.get((tag, v, mode), [])
                cells.append(f"{np.mean(r):.3f}" if r else "—")
            o.append(f"| {tag} · {v} | " + " | ".join(cells) + " |")
    o.append("")
    bad = []
    for tag, _p in (("p0", 0.0), ("p010", 0.10)):
        for v in a.variants:
            c = res.get((tag, v, "control"), [])
            if not c:
                continue
            for mode in MODES[1:]:
                d = res.get((tag, v, mode), [])
                # Threshold against the FLOOR, not against a fraction of control: a
                # destroyed condition should land at the trivial baseline, and
                # "keeps under half its accuracy" is a line a real leak can clear.
                if d and np.mean(d) > FLOOR + 0.02:
                    bad.append(f"{tag} {v} {mode}: {np.mean(d):.3f} vs control "
                               f"{np.mean(c):.3f}")
    o += ["## Verdict", ""]
    if bad:
        o += ["**FAILS.** These conditions kept more than half their accuracy after "
              "the context was destroyed, which means something other than the map "
              "is carrying the score:", ""] + [f"- {x}" for x in bad] + [""]
    else:
        o += ["**PASSES.** Every arm collapses under both destructions, at both "
              "noise levels. The score depends on the explore observations AND on "
              "the query path, which is what the task claims to measure. This is "
              "the check hier-goal failed (0.912 -> 0.913) and it is the reason "
              "Match-Query results are citable where those are not.", ""]
    Path(a.out).write_text("\n".join(o) + "\n")
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()},
              open(a.out.replace(".md", ".json"), "w"), indent=2)
    print("\n".join(o))


if __name__ == "__main__":
    main()

"""How aliased is the compositional world at each n_templates? (CPU only.)

Why this exists
---------------
`DISSOCIATION_SWEEP.md` sweeps `n_templates` as a proxy for "how much
compositional structure is available to exploit". It is not a clean proxy, and
the sweep's own numbers exposed that: prediction P2 said path integration's
`exact_acc` advantage would be FLAT in n_templates, and instead it grew
monotonically (+0.060 -> +0.158 -> +0.250 at nt = 2, 4, 8).

The reason is that n_templates moves TWO things at once. Fewer templates means
fewer distinct motifs to compose (the intended axis) but also more grid cells
that look identical (an unintended one). Knowing precisely where you are pays
off less when every cell looks the same anyway, because content cannot confirm
or contradict the estimate either way.

So the growth in the path-integration effect may be tracking aliasing rather
than compositional structure, and P2 cannot be scored as stated. This measures
the confound directly so it can be reported as a covariate rather than
hand-waved. The world is constructed by the environment's own `_draw_world` /
`_obs_map_from`, not reimplemented here (standing rule 7).

Three measures, cheapest to most task-relevant
----------------------------------------------
  cells_per_obs   mean over cells of how many grid cells share that cell's
                  observation. The ambiguity of ONE observation.
  H(obs)          entropy of the observation marginal over cells, in bits.
                  Falls as aliasing rises.
  run_k           the one that matters. Mean number of grid positions consistent
                  with a straight k-step run of observations. This is exactly
                  the "content route" to localisation: how many observations in a
                  row must you see before content alone pins you down? If run_k
                  collapses to ~1 by k=4, content localises fast and explicit
                  path integration is less necessary.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from mapformer.environment_compositional import CompositionalGridWorld

_REPO = Path(__file__).resolve().parent


def obs_map_for(n_templates, seed):
    env = CompositionalGridWorld(size=64, room_size=8,
                                 n_templates=n_templates, seed=seed)
    rng = np.random.RandomState(seed)
    templates, room_tmpl = env._draw_world(rng)
    return env._obs_map_from(templates, room_tmpl), env


def measures(obs, ks=(1, 2, 4, 8)):
    N = obs.shape[0]
    flat = obs.ravel()
    cnt = Counter(flat.tolist())
    cells_per_obs = float(np.mean([cnt[v] for v in flat]))
    p = np.array([c / flat.size for c in cnt.values()])
    H = float(-(p * np.log2(p)).sum())

    # run_k: how many positions are consistent with a straight k-step run of
    # observations? Averaged over both axes and over a sample of start cells.
    out = {}
    for k in ks:
        tot = 0.0
        n = 0
        for axis in (0, 1):
            # every length-k window along `axis`, keyed by its observation tuple
            table = defaultdict(int)
            for i in range(N):
                for j in range(N):
                    if axis == 0:
                        w = tuple(obs[(i + d) % N, j] for d in range(k))
                    else:
                        w = tuple(obs[i, (j + d) % N] for d in range(k))
                    table[w] += 1
            # mean multiplicity as SEEN from a random cell (size-biased), which
            # is what an agent standing somewhere actually faces
            for w, c in table.items():
                tot += c * c
                n += c
        out[f"run_{k}"] = tot / max(n, 1)
    return {"cells_per_obs": cells_per_obs, "H_obs_bits": H, **out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", nargs="+", type=int, default=[2, 4, 8, 16])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=str(_REPO / "ALIASING_COVARIATE.md"))
    args = ap.parse_args()

    rows = {}
    for nt in args.templates:
        per = [measures(obs_map_for(nt, s)[0]) for s in args.seeds]
        rows[nt] = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        print(f"nt={nt:3d}  " + "  ".join(f"{k}={v:.2f}" for k, v in rows[nt].items()),
              flush=True)

    lines = ["# Aliasing covariate for the dissociation sweep", "",
             "`n_templates` moves two things at once: how many distinct motifs "
             "there are to compose (the intended axis) and how many grid cells "
             "look identical (an unintended one). This measures the second, so "
             "it can be reported beside the sweep instead of hand-waved.", "",
             "World built by the environment's own `_draw_world` / "
             "`_obs_map_from` (rule 7). 64x64 grid, room size 8, "
             f"n={len(args.seeds)} worlds averaged.", "",
             "| n_templates | cells sharing an observation | H(obs) bits | "
             "positions consistent with a 1-step run | 2-step | 4-step | 8-step |",
             "|---|---|---|---|---|---|---|"]
    for nt in args.templates:
        r = rows[nt]
        lines.append(f"| {nt} | {r['cells_per_obs']:.0f} | {r['H_obs_bits']:.2f} | "
                     f"{r['run_1']:.0f} | {r['run_2']:.1f} | {r['run_4']:.2f} | "
                     f"{r['run_8']:.2f} |")
    lines += ["", "## How to use it", "",
              "`run_k` is the load-bearing column: the number of grid positions "
              "still consistent after seeing k observations in a straight line — "
              "i.e. how fast **content alone** localises the agent, with no "
              "position code involved. Where run_k collapses toward 1 quickly, "
              "the content route is open and explicit path integration matters "
              "less; where it stays large, position is the only route.", "",
              "Read the sweep's two effect columns against this:", "",
              "- the **hierarchy** effect on `cross_nb` should track the "
              "n_templates axis itself (how much structure there is to compose), "
              "not this table;",
              "- the **path-integration** effect on `exact_acc` is the one at "
              "risk of tracking this table instead. P2 predicted it flat and it "
              "grew (+0.060 → +0.158 → +0.250 at nt = 2, 4, 8). If that growth "
              "lines up with falling aliasing here, P2's failure is a property "
              "of the sweep axis, not of the models, and must be reported that "
              "way.", "",
              "## What this cannot do", "",
              "It measures the confound; it does not remove it. Separating the "
              "two axes properly needs a design where motif count and aliasing "
              "move independently — e.g. holding the number of distinct "
              "observation types per template fixed while varying how many "
              "templates tile the grid. That is a different sweep, not a "
              "reanalysis of this one.", ""]
    Path(args.out).write_text("\n".join(lines) + "\n")
    json.dump(rows, open(str(args.out).replace(".md", ".json"), "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

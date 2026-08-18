"""Context-destruction ablation for the family-tree and compositional tasks.

Why these two, and why now
--------------------------
This check is the single thing separating the results this repo still cites from
the 48 in archive/void. It is what exposed hier-goal -- randomising the goal AND
the entire explore phase left accuracy at 0.912 -> 0.913, i.e. nothing
navigational had ever been learned -- and it is what Match-Query passed
(0.918 -> 0.074). Family tree and compositional are both in the citable set and
NEITHER has been through it.

The specific risk in each:

  FAMILY TREE  scored at revisited nodes on a tree where shallow nodes are
               revisited constantly. The gates caught the hub-node baseline
               (0.163, and that is the real floor, not the 0.125 chance), but
               nobody has checked whether the RELATION tokens are load-bearing.
               Relations invert -- mother then child is the identity -- so the
               walk oscillates, and a model could conceivably score by tracking
               local oscillation depth rather than identity of the person.

  COMPOSITIONAL cross_nb is the compositional-transfer metric. If the ACTION
               stream can be scrambled without cost, "transfer" is being read
               off something other than the path.

Conditions follow ablate_paper_task.py, including its lesson: `shuffle` permutes
slots and so ALSO destroys the walk's own autocorrelation, putting the input
off-manifold; `resample` substitutes the corresponding stream from an
INDEPENDENT episode of the same generator, which is a perfectly valid walk that
simply does not match the observations beside it. On the paper task the two
agreed (0.231 vs 0.178), which is what retired the off-manifold worry there.
Both are reported here rather than assumed to agree again.

Interpretation. A task whose scored metric is genuinely carried by the context
must COLLAPSE toward its floor. Surviving the destruction is the failure mode --
it means the number was coming from somewhere else.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.train_variant import VARIANT_MAP

_REPO = Path(__file__).resolve().parent
CONDS = ["intact", "shuffle_actions", "resample_actions",
         "shuffle_obs", "resample_obs"]


def perturb(tokens, donor, cond, rng):
    """Even indices are actions/relations, odd are observations."""
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
        t[:, act] = donor[:, act]
    elif cond == "resample_obs":
        t[:, obs] = donor[:, obs]
    else:
        raise ValueError(cond)
    return t


@torch.no_grad()
def score_family(model, env, cond, n_batches, bs, n_steps, dev, seed):
    from collections import Counter
    model.eval()
    rng = np.random.RandomState(seed)
    grng = np.random.RandomState(seed + 777)
    ok = tot = 0
    nll = 0.0
    cnt = Counter()
    for _ in range(n_batches):
        toks, rev, _ = env.generate_batch(bs, n_steps, rng)
        donor, _dr, _di = env.generate_batch(bs, n_steps, grng)
        toks = perturb(toks, donor, cond, rng).to(dev)
        m = rev[:, 1:].to(dev)
        if not m.any():
            continue
        lg = model(toks[:, :-1])
        tgt = toks[:, 1:]
        sl, tl = lg[m], tgt[m]
        ok += int((sl.argmax(-1) == tl).sum().item())
        nll += float(F.cross_entropy(sl, tl, reduction="sum").item())
        tot += int(tl.numel())
        cnt.update(tl.tolist())
    return dict(acc=ok / max(tot, 1), nll=nll / max(tot, 1), n=tot,
                marginal=(max(cnt.values()) / tot) if tot else float("nan"))


@torch.no_grad()
def score_comp(model, env, cond, n_traj, bs, T, dev, seed):
    model.eval()
    rng = np.random.RandomState(seed)
    blank = env.unified_blank
    agg = {k: [0, 0] for k in ("exact", "cross", "cross_nb")}
    done = 0
    while done < n_traj:
        b = min(bs, n_traj - done)
        batch = env.generate_batch(b, T)
        donor = env.generate_batch(b, T)[0]
        tokens = perturb(batch[0], donor, cond, rng).to(dev)
        exact_m = batch[2][:, 1:].to(dev)
        cross_m = batch[4][:, 1:].to(dev)
        tgt = tokens[:, 1:]
        pred = model(tokens[:, :-1]).argmax(-1)
        for name, m in (("exact", exact_m), ("cross", cross_m)):
            if m.sum() > 0:
                agg[name][0] += int((pred[m] == tgt[m]).sum().item())
                agg[name][1] += int(m.sum())
        nb = cross_m & (tgt != blank)
        if nb.sum() > 0:
            agg["cross_nb"][0] += int((pred[nb] == tgt[nb]).sum().item())
            agg["cross_nb"][1] += int(nb.sum())
        done += b
    return {k: (c / max(n, 1)) for k, (c, n) in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["family", "compositional"], required=True)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-batches", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--n-obs", type=int, default=8)
    ap.add_argument("--n-traj", type=int, default=128)
    ap.add_argument("--env-seed", type=int, default=10000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)

    if args.task == "family":
        from mapformer.environment_family_tree import FamilyTreeWorld
        rd = args.runs_dir or str(_REPO / "runs/family_tree")
        vs = args.variants or ["MapEM_NC_NL", "VanillaEM_P0", "PlainFlat"]
        out = args.out or str(_REPO / "ABLATE_FAMILY_TREE.md")
    else:
        from mapformer.environment_compositional import CompositionalGridWorld
        rd = args.runs_dir or str(_REPO / "runs/comp_multiseed")
        vs = args.variants or ["Vanilla", "Hourglass_k2", "PlainFlat"]
        out = args.out or str(_REPO / "ABLATE_COMPOSITIONAL.md")

    res = {v: {c: [] for c in CONDS} for v in vs}
    extra = {}
    for v in vs:
        for s in args.seeds:
            suffix = "_familytree" if args.task == "family" else ""
            ck = Path(rd) / f"seed{s}" / f"{v}{suffix}.pt"
            if not ck.exists():
                print(f"MISSING {ck}", flush=True)
                continue
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            sd = blob.get("model_state") or blob.get("model_state_dict")
            if args.task == "family":
                env = FamilyTreeWorld(depth=args.depth, n_obs_types=args.n_obs,
                                      seed=args.env_seed)
                gs = 64
            else:
                env = CompositionalGridWorld(size=blob.get("grid_size", 64),
                                             seed=args.env_seed)
                gs = blob.get("grid_size", 64)
            m = VARIANT_MAP[v](vocab_size=blob["vocab_size"],
                               d_model=blob["d_model"], n_heads=blob["n_heads"],
                               n_layers=blob["n_layers"], grid_size=gs).to(dev)
            m.load_state_dict(sd)
            for c in CONDS:
                if args.task == "family":
                    r = score_family(m, env, c, args.n_batches, args.batch_size,
                                     args.n_steps, dev, 5000 + s)
                    res[v][c].append(r["acc"])
                    extra[f"{v}_s{s}_{c}"] = r
                    print(f"[{v} s{s}] {c:17s} acc={r['acc']:.4f} "
                          f"nll={r['nll']:.3f} n={r['n']}", flush=True)
                else:
                    r = score_comp(m, env, c, args.n_traj, args.batch_size,
                                   args.n_steps, dev, 5000 + s)
                    res[v][c].append(r["cross_nb"])
                    extra[f"{v}_s{s}_{c}"] = r
                    print(f"[{v} s{s}] {c:17s} cross_nb={r['cross_nb']:.4f} "
                          f"exact={r['exact']:.4f}", flush=True)
            del m
            torch.cuda.empty_cache()

    metric = ("revisit accuracy" if args.task == "family" else "cross_nb_acc")
    floor = ("**0.163** (hub-node baseline; the 0.125 chance is NOT the floor)"
             if args.task == "family" else
             "no analytic floor; the destroyed rows ARE the empirical floor")
    lines = [f"# Context-destruction ablation -- {args.task} task", "",
             "The check that exposed hier-goal (0.912 -> 0.913, i.e. nothing was "
             "learned) and that Match-Query passed (0.918 -> 0.074). Both tasks "
             "here are in the citable set and neither had been through it.", "",
             f"Metric: **{metric}**. Floor: {floor}.", "",
             "`shuffle` permutes slots, which also destroys the walk's "
             "autocorrelation; `resample` substitutes the stream from an "
             "INDEPENDENT episode -- a valid walk that simply does not match the "
             "observations beside it, so it stays on-manifold. On the paper task "
             "these agreed (0.231 vs 0.178).", "",
             "**A genuine result COLLAPSES here. Surviving destruction is the "
             "failure mode.**", "",
             "| variant | " + " | ".join(CONDS) + " |",
             "|---" * (len(CONDS) + 1) + "|"]
    summ = {}
    for v in vs:
        if not res[v]["intact"]:
            continue
        summ[v] = {}
        cells = []
        for c in CONDS:
            a = np.array(res[v][c])
            summ[v][c] = {"mean": float(a.mean()),
                          "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                          "per_seed": a.tolist()}
            cells.append(f"{a.mean():.4f} ± {a.std(ddof=1) if len(a)>1 else 0:.4f}")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    Path(out).write_text("\n".join(lines) + "\n")
    json.dump({"summary": summ, "detail": extra},
              open(str(out).replace(".md", ".json"), "w"), indent=2, default=str)
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

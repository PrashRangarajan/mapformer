"""Gates for the Dyck matched-depth control (DYCK_MDEPTH_PREREG.md), run BEFORE any training.

Rule 7: every sequence comes from environment_dyck.DyckWorld (the trainer's own sampler, with
the trainer's own per-batch D rule for the mixture); the metric is eval_dyck_literature.metrics
on eval_dyck_literature.cell_data -- the exact 512 evaluation sequences per cell that the
ladder (DYCK_LADDER_RESULTS.md) was scored on.

G1  every training-distribution sequence is valid Dyck-2 with max depth exactly its D.
G2  the evaluation cell L32 D12 is INSIDE the T12 training distribution: identical sampler
    (L, D), so its closer-depth / closing-distance / forced-token statistics must match the
    training batches' to sampling error; and under T4 (the ladder) it is NOT (closers at stack
    depth > 4 never occur in training).
G3  metric sanity: the sampler's exact next-token distribution scores A2f = 1 on every cell
    (A2f, dyck_mdepth_common: A2 restricted to prefixes where the sampler can emit a closer).
    Its plain A2 is reported too: 0.940 at L32 D12, because at forced-open prefixes it puts
    zero mass on both closers -- the reason A2f, not A2, is the registered primary.
G4  floors on A2 and A2f (chance 0.500) per training distribution: n-grams of order 1..6 FITTED ON
    THAT DISTRIBUTION, and the stack-free heuristic (exact depth + closer of the most recent
    open). Reported beside every cell of the results.
Exit status non-zero if G1 or G3 fails, or if any floor reaches 0.95 at L32 D12 (no dynamic
range left for the contrast).

Run from /home/prashr:  python3 -m mapformer.validate_dyck_mdepth --out <path.md>
"""
import argparse, json, sys
from collections import defaultdict
import numpy as np
import torch

from mapformer.environment_dyck import DyckWorld, check_dyck, VOCAB, BOS, OPEN_P, OPEN_B, CLOSE_P, CLOSE_B
from mapformer.eval_dyck_literature import metrics
from mapformer.dyck_mdepth_common import CELLS, cell_with_mask, a2_masked
from mapformer.validate_dyck import ngram_probs, nostack_probs, sampler_probs

TRAIN = {"T4 (ladder)": [4], "T12": [12], "Tmix 4-12": list(range(4, 13))}
N_FIT = 20000     # sequences per n-gram fit, as validate_dyck.ngram_fit


def train_batches(Ds, n_seq, seed, L=32, bs=128):
    """The trainer's rule: one D per batch of `bs`, drawn uniformly from Ds by a separate RNG
    (seed + 7,000,003) when len(Ds) > 1; sequences from DyckWorld.batch with rng(seed)."""
    w = DyckWorld(); rng = np.random.default_rng(seed)
    drng = np.random.default_rng(seed + 7_000_003) if len(Ds) > 1 else None
    out = []
    for _ in range(n_seq // bs):
        D = int(drng.choice(Ds)) if drng is not None else Ds[0]
        inp, tgt, valid, ent = w.batch(bs, L, D, rng)
        out.append((D, inp, tgt, valid, ent))
    return out


def stats(inp, tgt):
    """Stack depth before each closer, distance to the matching open, forced-token fraction."""
    depth, dist = [], []
    for i in range(tgt.shape[0]):
        st = []
        for t in range(tgt.shape[1]):
            x = int(tgt[i, t])
            if x in (OPEN_P, OPEN_B):
                st.append(t)
            else:
                depth.append(len(st)); dist.append(t - st.pop())
    depth, dist = np.array(depth), np.array(dist)
    return {"closers": int(depth.size), "frac_closer_depth_gt4": float((depth > 4).mean()),
            "frac_closer_depth_gt8": float((depth > 8).mean()), "max_closer_depth": int(depth.max()),
            "dist_median": float(np.median(dist)), "dist_p90": float(np.percentile(dist, 90)),
            "dist_max": int(dist.max())}


def ngram_fit_on(batches, order):
    """validate_dyck.ngram_fit, fitted on the given training batches instead of L32 D4."""
    cnt = defaultdict(lambda: np.zeros(VOCAB))
    for _, inp, tgt, _, _ in batches:
        seq = np.concatenate([np.full((inp.shape[0], order - 1), BOS), inp.numpy()], 1)
        L = inp.shape[1]
        for i in range(seq.shape[0]):
            for t in range(L):
                cnt[tuple(seq[i, t:t + order])][tgt[i, t].item()] += 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="markdown path; a .json is written beside it")
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    a = ap.parse_args()
    w = DyckWorld()
    data = {c: cell_with_mask(w, *c) for c in CELLS}
    res = {"cells": {}, "train": {}}
    ok = True

    # G1 / G2 on the training distributions
    for lab, Ds in TRAIN.items():
        bt = train_batches(Ds, N_FIT, seed=100)
        valid = all(check_dyck(tgt.numpy(), 32, D) for D, _, tgt, _, _ in bt)
        ok &= valid
        tgt_all = torch.cat([b[2] for b in bt]); inp_all = torch.cat([b[1] for b in bt])
        s = stats(inp_all, tgt_all)
        s["G1_valid"] = bool(valid)
        s["ce_floor"] = float(np.mean([b[4].mean().item() for b in bt]))
        s["D_counts"] = {str(D): sum(1 for b in bt if b[0] == D) for D in Ds}
        res["train"][lab] = s
        print(f"train {lab}: {s}", flush=True)
    for c in CELLS:
        s = stats(data[c]["inp"], data[c]["tgt"]); s["ce_floor"] = data[c]["ent"]
        # G3: the sampler's exact distribution is the CE-optimal predictor
        P = sampler_probs(data[c]["tgt"].numpy(), *c)
        s["G3_sampler_A2"] = a2_masked(P, data[c])
        s["G3_sampler_A2f"] = a2_masked(P, data[c], data[c]["close_ok"])
        ok &= abs(s["G3_sampler_A2f"] - 1.0) < 1e-9
        s["forced_open_share_of_scored"] = float((data[c]["dp"] & ~data[c]["close_ok"]).sum() / data[c]["dp"].sum())
        ns = nostack_probs(data[c]["inp"])
        s["floor_nostack_A2"] = a2_masked(ns, data[c])
        s["floor_nostack_A2f"] = a2_masked(ns, data[c], data[c]["close_ok"])
        res["cells"][f"L{c[0]}D{c[1]}"] = s
        print(f"cell L{c[0]}D{c[1]}: {s}", flush=True)

    # G4: n-gram floors fitted on each training distribution
    res["ngram_A2"] = {}; res["ngram_A2f"] = {}
    for lab, Ds in TRAIN.items():
        bt = train_batches(Ds, N_FIT, seed=200)
        for k in a.orders:
            fit = ngram_fit_on(bt, k)
            for c in CELLS:
                P = ngram_probs(fit, k, data[c]["inp"])
                res["ngram_A2"][f"{lab}|k{k}|L{c[0]}D{c[1]}"] = a2_masked(P, data[c])
                res["ngram_A2f"][f"{lab}|k{k}|L{c[0]}D{c[1]}"] = a2_masked(P, data[c], data[c]["close_ok"])
            print(f"n-gram {lab} k={k}: " + " ".join(
                f"L{c[0]}D{c[1]} A2 {res['ngram_A2'][f'{lab}|k{k}|L{c[0]}D{c[1]}']:.3f} "
                f"A2f {res['ngram_A2f'][f'{lab}|k{k}|L{c[0]}D{c[1]}']:.3f}" for c in CELLS), flush=True)

    best = {lab: {f"L{c[0]}D{c[1]}": max(max(res["ngram_A2f"][f"{lab}|k{k}|L{c[0]}D{c[1]}"] for k in a.orders),
                                        res["cells"][f"L{c[0]}D{c[1]}"]["floor_nostack_A2f"]) for c in CELLS}
            for lab in TRAIN}
    res["best_floor_A2f"] = best
    ok &= all(best[lab]["L32D12"] < 0.95 for lab in TRAIN)
    res["PASS"] = bool(ok)

    json.dump(res, open(a.out.replace(".md", ".json"), "w"), indent=1)
    L = ["# Dyck matched-depth gates (validate_dyck_mdepth.py)", "",
         f"**PASS: {ok}**  (G1 validity, G3 sampler A2f = 1, best A2f floor at L32 D12 < 0.95)", "",
         "## Training distributions (20,000 sequences each, the trainer's per-batch D rule)", "",
         "| train | G1 | CE floor | closers at depth >4 | >8 | max depth | closing dist median / p90 / max |",
         "|---|---|---|---|---|---|---|"]
    for lab, s in res["train"].items():
        L.append(f"| {lab} | {s['G1_valid']} | {s['ce_floor']:.3f} | {s['frac_closer_depth_gt4']:.3f} | "
                 f"{s['frac_closer_depth_gt8']:.3f} | {s['max_closer_depth']} | "
                 f"{s['dist_median']:.0f} / {s['dist_p90']:.0f} / {s['dist_max']} |")
    L += ["", "## Evaluation cells (the ladder's exact 512 sequences per cell)", "",
          "| cell | CE floor | closers at depth >4 | >8 | closing dist median / p90 / max | forced-open share of scored | G3 sampler A2f / A2 | stack-free A2f / A2 |",
          "|---|---|---|---|---|---|---|---|"]
    for c, s in res["cells"].items():
        L.append(f"| {c} | {s['ce_floor']:.3f} | {s['frac_closer_depth_gt4']:.3f} | {s['frac_closer_depth_gt8']:.3f} | "
                 f"{s['dist_median']:.0f} / {s['dist_p90']:.0f} / {s['dist_max']} | {s['forced_open_share_of_scored']:.3f} | "
                 f"{s['G3_sampler_A2f']:.3f} / {s['G3_sampler_A2']:.3f} | {s['floor_nostack_A2f']:.3f} / {s['floor_nostack_A2']:.3f} |")
    L += ["", "## Floors: n-grams fitted on each training distribution, A2f / A2 (chance 0.500)", "",
          "| train | order | " + " | ".join(f"L{c[0]}D{c[1]}" for c in CELLS) + " |", "|---|---|" + "---|" * len(CELLS)]
    for lab in TRAIN:
        for k in a.orders:
            L.append(f"| {lab} | {k} | " + " | ".join(
                f"{res['ngram_A2f'][f'{lab}|k{k}|L{c[0]}D{c[1]}']:.3f} / {res['ngram_A2'][f'{lab}|k{k}|L{c[0]}D{c[1]}']:.3f}"
                for c in CELLS) + " |")
    L += ["", "Best A2f floor (max over n-gram orders and the stack-free heuristic), to quote beside every cell:", ""]
    for lab in TRAIN:
        L.append(f"- {lab}: " + ", ".join(f"{c} {v:.3f}" for c, v in best[lab].items()))
    open(a.out, "w").write("\n".join(L) + "\n")
    print("PASS" if ok else "FAIL", "->", a.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Gates for the Dyck-2 task, run BEFORE training (rule 7: calls environment_dyck, does not
reimplement it). Writes DYCK_GATES.md / .json.

G1 every sampled sequence is well nested, returns to 0, has max depth exactly D (loop check).
G2 metric sanity: uniform over Val(s) must score F1 = 1 exactly.
G3 the sampler's own next-token distribution (the CE-optimal predictor) -- what a model that
   fits the training loss perfectly scores under the paper's metric.
G4 shortcut floors: n-gram predictors (orders 1..6, fitted on L=32 D=4), and a stack-free
   heuristic (exact depth counter + closer of the most recent OPEN token).
"""
import json, sys
from collections import defaultdict
import numpy as np
import torch

from mapformer.environment_dyck import (DyckWorld, f1_valid, check_dyck, VOCAB, BOS,
                                        OPEN_P, OPEN_B, CLOSER)

REPO = "/home/prashr/mapformer"
LS, DS = [32, 64, 96, 128], [4, 6, 8, 12]
N = 1024


def sampler_probs(tok, L, D):
    """Replay the sampler's rule to get its exact next-token distribution per prefix."""
    n = tok.shape[0]
    P = np.zeros((n, L, VOCAB))
    for i in range(n):
        st, reached = [], False
        for t in range(L):
            d, rem = len(st), L - t - 1
            def feas(da, ra): return 0 <= da <= D and rem >= (da if ra else 2 * D - da)
            co = feas(d + 1, reached or d + 1 == D)
            cc = d > 0 and feas(d - 1, reached)
            po = 0.5 if (co and cc) else (1.0 if co else 0.0)
            P[i, t, OPEN_P] = P[i, t, OPEN_B] = po / 2
            if cc:
                P[i, t, CLOSER[st[-1]]] = 1 - po
            x = tok[i, t]
            if x in (OPEN_P, OPEN_B): st.append(x)
            else: st.pop()
            reached = reached or len(st) == D
    return torch.from_numpy(P)


def ngram_fit(world, order, rng):
    cnt = defaultdict(lambda: np.zeros(VOCAB))
    inp, tgt, _, _ = world.batch(20000, 32, 4, rng)
    seq = np.concatenate([np.full((inp.shape[0], order - 1), BOS), inp.numpy()], 1)
    for i in range(seq.shape[0]):
        for t in range(32):
            cnt[tuple(seq[i, t:t + order])][tgt[i, t].item()] += 1
    return cnt


def ngram_probs(cnt, order, inp):
    n, L = inp.shape
    seq = np.concatenate([np.full((n, order - 1), BOS), inp.numpy()], 1)
    P = np.zeros((n, L, VOCAB))
    for i in range(n):
        for t in range(L):
            c = cnt.get(tuple(seq[i, t:t + order]))
            P[i, t] = (c + 1e-3) / (c.sum() + VOCAB * 1e-3) if c is not None else 1.0 / VOCAB
    return torch.from_numpy(P)


def nostack_probs(inp):
    """Exact depth, but the closer offered is that of the most recent OPEN token."""
    n, L = inp.shape
    P = np.zeros((n, L, VOCAB))
    for i in range(n):
        d, last = 0, None
        for t in range(L):
            x = inp[i, t].item()
            if x in (OPEN_P, OPEN_B): d += 1; last = x
            elif x != BOS: d -= 1
            allowed = [OPEN_P, OPEN_B] + ([CLOSER[last]] if d > 0 else [])
            P[i, t, allowed] = 1.0 / len(allowed)
    return torch.from_numpy(P)


def main():
    w = DyckWorld()
    out = {"grid": {}, "notes": {}}
    fits = {k: ngram_fit(w, k, np.random.default_rng(100 + k)) for k in range(1, 7)}
    g1_ok = True
    for L in LS:
        for D in DS:
            rng = np.random.default_rng(99991 + 7 * L + D)
            inp, tgt, valid, ent = w.batch(N, L, D, rng)
            g1_ok &= check_dyck(tgt.numpy(), L, D)
            row = {"ce_floor_nats": float(ent.mean())}
            uv = valid.double() / valid.sum(-1, keepdim=True)
            row["G2_uniform_valid"] = float(f1_valid(uv, valid)[0].mean())
            row["uniform_4brackets"] = float(f1_valid(
                torch.tensor([.25, .25, .25, .25, 0.]).expand(N, L, VOCAB), valid)[0].mean())
            sp = sampler_probs(tgt.numpy(), L, D)
            f, pv, bt = f1_valid(sp, valid)
            row["G3_sampler_exact"] = float(f.mean())
            row["G3_sampler_PV"] = float(pv.mean()); row["G3_sampler_BT"] = float(bt.mean())
            sm = 0.99 * sp + 0.01 * uv                     # sampler, smoothed onto Val(s) only
            row["G3_sampler_smoothed_valid"] = float(f1_valid(sm, valid)[0].mean())
            sm2 = 0.99 * sp + 0.01 / VOCAB                 # sampler, smoothed onto all tokens
            row["G3_sampler_smoothed_all"] = float(f1_valid(sm2, valid)[0].mean())
            for k, c in fits.items():
                row[f"G4_ngram{k}"] = float(f1_valid(ngram_probs(c, k, inp), valid)[0].mean())
            row["G4_nostack"] = float(f1_valid(nostack_probs(inp), valid)[0].mean())
            out["grid"][f"L{L}_D{D}"] = row
            print(f"L{L} D{D} " + " ".join(f"{k}={v:.3f}" for k, v in row.items()), flush=True)
    out["G1_all_valid"] = bool(g1_ok)
    json.dump(out, open(f"{REPO}/DYCK_GATES.json", "w"), indent=1)
    keys = list(next(iter(out["grid"].values())).keys())
    with open(f"{REPO}/DYCK_GATES.md", "w") as fh:
        fh.write(f"# Dyck-2 gates (validate_dyck.py, n={N} per cell)\n\n")
        fh.write(f"G1 all sampled sequences valid, back to 0, max depth exactly D: **{g1_ok}**\n\n")
        fh.write("| cell | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 1) + "\n")
        for c, r in out["grid"].items():
            fh.write(f"| {c} | " + " | ".join(f"{r[k]:.3f}" for k in keys) + " |\n")
    print("G1", g1_ok)


if __name__ == "__main__":
    main()

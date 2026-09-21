"""Stratified closing-delimiter accuracy on the code corpus.

The registered primary readout (CODE_PREREG.md). Given that the next byte is a
closing bracket, does the model put most mass on the RIGHT one of ) ] } ?
Probability is renormalised over the three closers, so the model gets no credit
for knowing merely that a bracket closes -- only for knowing which. That is the
part a stack is needed for, and it is the same construction as Hewitt et al.'s
bracket-closing memory used on Dyck-2.

Two filters matter and both are registered:
  - a closer is scored only if its matching opener is inside the SAME crop. A
    closer whose opener the model never saw is not a memory test.
  - cells whose measured no-stack floor exceeds 0.95 are uninformative and are
    marked as such in the output rather than being read.

Usage:
  python3 -m mapformer.eval_code_delims --runs-dir runs/code --split val
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .train_hourglass_enwik8 import build

_REPO = os.path.dirname(os.path.abspath(__file__))
LN2 = 0.6931471805599453
CLOSER_BYTES = [ord(")"), ord("]"), ord("}")]
DEPTH_BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 10 ** 9)]
DIST_BINS = [(0, 2), (3, 8), (9, 32), (33, 128), (129, 10 ** 9)]


def bin_of(v, bins):
    for i, (lo, hi) in enumerate(bins):
        if lo <= v <= hi:
            return i
    return len(bins) - 1


def label(bins, i):
    lo, hi = bins[i]
    return f"{lo}+" if hi > 10 ** 8 else f"{lo}-{hi}"


@torch.no_grad()
def score(model, data, pos, kind, open_pos, depth, device, seq_len, batch=24, align=None):
    """Returns per-closer correctness plus overall val bpc."""
    n = data.shape[0]
    starts = np.arange(0, n - seq_len - 1, seq_len)
    order = np.argsort(pos)
    pos, kind, open_pos, depth = pos[order], kind[order], open_pos[order], depth[order]

    closer_ids = torch.tensor(CLOSER_BYTES, device=device)
    got, lab, dep, dis = [], [], [], []
    nll_sum, nll_n = 0.0, 0

    for b0 in range(0, len(starts), batch):
        chunk = starts[b0:b0 + batch]
        x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64))
                         for i in chunk]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64))
                         for i in chunk]).to(device)
        logits = model(x)
        if align is not None:
            # Alignment check. logits[t] must predict data[i+t+1]. An off-by-one
            # in t would still produce a clean, plausible, WRONG accuracy, so
            # score the same logits against the byte before, at, and after the
            # claimed target: the claimed one must have the lowest CE by a wide
            # margin. A readout that is not verified against a shifted control
            # can report a solution that is not there.
            for sh in (-1, 0, 1):
                yy = torch.stack([torch.from_numpy(
                    data[i + 1 + sh:i + 1 + sh + seq_len].astype(np.int64))
                    for i in chunk]).to(device)
                align[sh] = align.get(sh, 0.0) + float(F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), yy.reshape(-1),
                    reduction="sum"))
        nll_sum += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                         y.reshape(-1), reduction="sum"))
        nll_n += y.numel()

        for r, i in enumerate(chunk):
            # logits[t] predicts data[i+t+1], so an absolute position p is
            # predictable from this crop iff i+1 <= p <= i+seq_len.
            lo = np.searchsorted(pos, i + 1, "left")
            hi = np.searchsorted(pos, i + seq_len, "right")
            if hi <= lo:
                continue
            p = pos[lo:hi]
            keep = open_pos[lo:hi] >= i          # opener visible in this crop
            if not keep.any():
                continue
            p, k = p[keep], kind[lo:hi][keep]
            # Direct check that the offsets address what they claim: the byte at
            # every scored position must BE the closer we are about to score.
            assert np.all(data[p] == np.array(CLOSER_BYTES)[k]), \
                "scored position is not the closer it is labelled as"
            d, dd = depth[lo:hi][keep], (p - open_pos[lo:hi][keep])
            t = torch.from_numpy((p - i - 1).astype(np.int64)).to(device)
            sub = logits[r].index_select(0, t).index_select(1, closer_ids)
            pred = sub.argmax(dim=1).cpu().numpy()
            got.append(pred == k)
            lab.append(k); dep.append(d); dis.append(dd)

    got = np.concatenate(got); dep = np.concatenate(dep); dis = np.concatenate(dis)
    return got, dep, dis, (nll_sum / nll_n) / LN2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=os.path.join(_REPO, "runs", "code"))
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "data"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--which", default="best", choices=["best", "final", "both"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--pattern", default="*",
                    help="restrict to checkpoints whose stem matches, e.g. '*_s0.*'")
    ap.add_argument("--check-alignment", action="store_true",
                    help="verify logits[t] predicts data[i+t+1] against +/-1 shifts")
    ap.add_argument("--out", default=os.path.join(_REPO, "CODE_DELIMS.json"))
    args = ap.parse_args()

    dd = Path(args.data_dir)
    data = np.fromfile(dd / f"code_{args.split}.bin", dtype=np.uint8)
    z = np.load(dd / f"code_{args.split}_brackets.npz")
    pos, kind, open_pos, depth = z["pos"], z["kind"], z["open_pos"], z["depth"]

    floors = json.load(open(os.path.join(_REPO, "CODE_GATES.json")))["floors"]
    wanted = ["best", "final"] if args.which == "both" else [args.which]

    out = {}
    for ck in sorted(Path(args.runs_dir).glob(f"{args.pattern}.pt")):
        stem = ck.name.replace(".pt", "")
        base, _, kindk = stem.rpartition(".")
        if kindk not in wanted:
            continue
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = blob["cfg"]
        model = build(cfg["model"], shorten=cfg["shorten"], dim=cfg["dim"],
                      heads=cfg["heads"], n_layers=cfg["n_layers"],
                      grid_size=cfg["grid_size"],
                      bottleneck_r=cfg["bottleneck_r"]).to(args.device)
        model.load_state_dict(blob["state_dict"])
        model.eval()

        align = {} if args.check_alignment else None
        got, dep, dis, bpc = score(model, data, pos, kind, open_pos, depth,
                                   args.device, args.seq_len, align=align)
        if align:
            best_sh = min(align, key=align.get)
            print(f"   alignment CE by shift: " +
                  "  ".join(f"{k:+d}:{v/1e6:.3f}" for k, v in sorted(align.items())) +
                  f"   -> best shift {best_sh:+d}" +
                  ("  OK" if best_sh == 0 else "  *** MISALIGNED ***"))
            if best_sh != 0:
                raise SystemExit("logit alignment is wrong; the metric is invalid")
        db = np.array([bin_of(int(v), DEPTH_BINS) for v in dep])
        xb = np.array([bin_of(int(v), DIST_BINS) for v in dis])
        cells = {}
        for i in range(len(DEPTH_BINS)):
            for j in range(len(DIST_BINS)):
                m = (db == i) & (xb == j)
                if m.sum() < 200:
                    continue
                key = f"d{label(DEPTH_BINS,i)}/x{label(DIST_BINS,j)}"
                cells[key] = {"acc": float(got[m].mean()), "n": int(m.sum()),
                              "floor": floors.get(key),
                              "informative": (floors.get(key) or 0) <= 0.95}
        out[f"{base}.{kindk}"] = {
            "overall_closer_acc": float(got.mean()),
            "n_scored": int(got.size),
            "val_bpc": bpc,
            "ckpt_val_bpc": blob.get("val_bpc"),
            "iter": blob.get("iter"),
            "cells": cells,
        }
        prim = cells.get("d5-8/x33-128", {})
        print(f"{base:28s} {kindk:5s} bpc={bpc:.4f} closer={got.mean():.3f} "
              f"PRIMARY d5-8/x33-128={prim.get('acc', float('nan')):.3f} "
              f"(floor {prim.get('floor')})")
        del model
        torch.cuda.empty_cache()

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

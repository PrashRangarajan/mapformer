"""Out-of-distribution readout for the code arm. Registered in CODE_PREREG.md
Amendment 1, before any OOD number was computed.

Same checkpoints, trained at seq 512, evaluated at crop length 2048. Two axes,
reported separately because they are different claims:

  O-A  val bpc by ABSOLUTE POSITION bucket (0-512 / 512-1024 / 1024-2048),
       mirroring JSB_LENGTH_RESULTS.md so the two tasks are comparable.
  O-B  closer-identity accuracy by BRACKET DISTANCE, with bins that run past the
       training context (129-512 / 513-1024 / 1025+). No arm ever saw a distance
       above 512.

The floor is RE-MEASURED on exactly the scored set, using validate_code's own
n-gram so there is one implementation, not two.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .train_hourglass_enwik8 import build
from .validate_code import fit_ngram, predict_ngram

_REPO = os.path.dirname(os.path.abspath(__file__))
LN2 = 0.6931471805599453
CLOSER_BYTES = [ord(")"), ord("]"), ord("}")]
POS_BUCKETS = [(0, 511), (512, 1023), (1024, 2047)]
DIST_BINS = [(0, 32), (33, 128), (129, 512), (513, 1024), (1025, 10 ** 9)]


def bin_of(v, bins):
    for i, (lo, hi) in enumerate(bins):
        if lo <= v <= hi:
            return i
    return len(bins) - 1


def lab(bins, i):
    lo, hi = bins[i]
    return f"{lo}+" if hi > 10 ** 8 else f"{lo}-{hi}"


@torch.no_grad()
def run(model, data, pos, kind, open_pos, device, seq_len, batch=6):
    n = data.shape[0]
    starts = np.arange(0, n - seq_len - 1, seq_len)
    order = np.argsort(pos)
    pos, kind, open_pos = pos[order], kind[order], open_pos[order]
    ids = torch.tensor(CLOSER_BYTES, device=device)

    nll = np.zeros(len(POS_BUCKETS)); cnt = np.zeros(len(POS_BUCKETS))
    got, dis, spos = [], [], []
    for b0 in range(0, len(starts), batch):
        chunk = starts[b0:b0 + batch]
        x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64))
                         for i in chunk]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64))
                         for i in chunk]).to(device)
        logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                             reduction="none").view(y.shape)          # (B, L)
        for bi, (lo, hi) in enumerate(POS_BUCKETS):
            hi = min(hi, seq_len - 1)
            if lo > hi:
                continue
            nll[bi] += float(ce[:, lo:hi + 1].sum()); cnt[bi] += ce[:, lo:hi + 1].numel()

        for r, i in enumerate(chunk):
            lo_i = np.searchsorted(pos, i + 1, "left")
            hi_i = np.searchsorted(pos, i + seq_len, "right")
            if hi_i <= lo_i:
                continue
            p = pos[lo_i:hi_i]
            keep = open_pos[lo_i:hi_i] >= i
            if not keep.any():
                continue
            p, k = p[keep], kind[lo_i:hi_i][keep]
            d = p - open_pos[lo_i:hi_i][keep]
            assert np.all(data[p] == np.array(CLOSER_BYTES)[k])
            t = torch.from_numpy((p - i - 1).astype(np.int64)).to(device)
            sub = logits[r].index_select(0, t).index_select(1, ids)
            got.append(sub.argmax(1).cpu().numpy() == k)
            dis.append(d); spos.append(p)
    return (nll / np.maximum(cnt, 1)) / LN2, np.concatenate(got), \
        np.concatenate(dis), np.concatenate(spos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=os.path.join(_REPO, "runs", "code"))
    ap.add_argument("--data-dir", default=os.path.join(_REPO, "data"))
    ap.add_argument("--pattern", default="*_s0.best")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(_REPO, "CODE_LONG.json"))
    args = ap.parse_args()

    dd = Path(args.data_dir)
    data = np.fromfile(dd / "code_val.bin", dtype=np.uint8)
    z = np.load(dd / "code_val_brackets.npz")
    pos, kind, open_pos = z["pos"], z["kind"], z["open_pos"]
    tdata = np.fromfile(dd / "code_train.bin", dtype=np.uint8)
    tz = np.load(dd / "code_train_brackets.npz")

    out = {}
    scored_ref = None
    for ck in sorted(Path(args.runs_dir).glob(f"{args.pattern}.pt")):
        base = ck.name.replace(".best.pt", "").replace(".final.pt", "")
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = blob["cfg"]
        model = build(cfg["model"], shorten=cfg["shorten"], dim=cfg["dim"],
                      heads=cfg["heads"], n_layers=cfg["n_layers"],
                      grid_size=args.seq_len,
                      bottleneck_r=cfg["bottleneck_r"]).to(args.device)
        miss = model.load_state_dict(blob["state_dict"], strict=False)
        assert not miss.missing_keys and not miss.unexpected_keys, miss
        model.eval()
        bpc, got, dis, spos = run(model, data, pos, kind, open_pos,
                                  args.device, args.seq_len)
        scored_ref = (dis, spos)
        db = np.array([bin_of(int(v), DIST_BINS) for v in dis])
        cells = {}
        for j in range(len(DIST_BINS)):
            m = db == j
            if m.sum() < 100:
                continue
            cells[lab(DIST_BINS, j)] = {"acc": float(got[m].mean()), "n": int(m.sum())}
        out[base] = {"bpc_by_position": {f"{lo}-{hi}": float(v)
                                         for (lo, hi), v in zip(POS_BUCKETS, bpc)},
                     "acc_by_distance": cells}
        print(f"{base:16s} bpc " + " ".join(f"{v:.4f}" for v in bpc) + "   acc " +
              " ".join(f"{lab(DIST_BINS,j)}={cells[lab(DIST_BINS,j)]['acc']:.3f}"
                       for j in range(len(DIST_BINS)) if lab(DIST_BINS, j) in cells))
        del model
        torch.cuda.empty_cache()

    # ---- the floor, re-measured on exactly the scored set ----
    if scored_ref is not None:
        dis, spos = scored_ref
        tables = fit_ngram(tdata, tz["pos"], tz["kind"], 8)
        preds = predict_ngram(tables, data, spos, 8)
        true = kind[np.searchsorted(np.sort(pos), spos)] if False else None
        # map each scored position back to its label
        order = np.argsort(pos)
        spos_idx = np.searchsorted(pos[order], spos)
        true = kind[order][spos_idx]
        maj = np.bincount(kind, minlength=3).argmax()
        db = np.array([bin_of(int(v), DIST_BINS) for v in dis])
        floors = {}
        for j in range(len(DIST_BINS)):
            m = db == j
            if m.sum() < 100:
                continue
            f = max(float((preds[m] == true[m]).mean()),
                    float((true[m] == maj).mean()))
            floors[lab(DIST_BINS, j)] = f
        out["_floor_by_distance"] = floors
        print("\nno-stack floor by distance: " +
              "  ".join(f"{k}={v:.3f}" for k, v in floors.items()))
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

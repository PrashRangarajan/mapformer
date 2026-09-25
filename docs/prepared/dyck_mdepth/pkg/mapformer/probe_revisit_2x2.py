"""Revisit accuracy by recurrence interval for the PAPER2X2 checkpoints (eval only).

WHY. In PAPER2X2 the index-code arms cross over: RoPE 0.805 vs PoPE-Flat 0.679 at T=128, but
0.374 vs 0.607 at T=1024. Hypothesis (not established): RoPE's content-dependent phase lets a query
pick a content-chosen relative offset, which an index model needs in order to answer revisits
from action tokens in context (short-range out-and-back retraces and local path integration).
PoPE deletes that term (magnitudes only, fixed per-(head, frequency) phase offsets), so it cannot
build those lookups: lower at training length, but nothing to misfire beyond it.

PREDICTIONS, written before this script was run (see REVISIT_2X2.md header):
  P1  At T=128, RoPE - PoPE-Flat is concentrated at SHORT recurrence intervals: detectably positive
      in at least one bucket <= 16 steps, its largest value in a bucket <= 16, and not detectably
      positive in any bucket >= 33.
  Q2  (descriptive, no direction registered) At T=1024, does RoPE keep its short-interval (<= 16)
      accuracy for revisits that occur LATE in the sequence (step > 128)? If yes, the length
      collapse is about long intervals; if no, it is about position beyond the training range.

Design: every arm at seed s sees the SAME trajectories (numpy RNG seeded per seed), on the
held-out map (env-seed 10000), so contrasts pair both the model seed and the evaluation data.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.stats_guard import paired

REPO = Path("/home/prashr/mapformer")
BUCKETS = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256), (257, 10**9)]
LAB = [f"{a}-{b}" if b < 10**8 else f"{a}+" for a, b in BUCKETS]
ARMS = ["RoPE", "PoPE-Flat", "Vanilla", "MapPoPE-Flat", "Vanilla_r4", "MapPoPE_r4"]


def bucket(d):
    for i, (a, b) in enumerate(BUCKETS):
        if a <= d <= b:
            return LAB[i]


@torch.no_grad()
def run(model, env, T, n_traj, bs, seed, dev):
    """-> {(phase, bucket): [hits, total, blanks]}; phase 'early' = step <= 128, 'late' = step > 128."""
    np.random.seed(50000 + seed)
    out = defaultdict(lambda: [0, 0, 0])
    done = 0
    while done < n_traj:
        b = min(bs, n_traj - done)
        tokens, _om, _rev, locs = env.generate_batch(b, T)
        pred = model(tokens[:, :-1].to(dev)).argmax(-1).cpu().numpy()
        tgt = tokens[:, 1:].numpy()
        for i in range(b):
            last = {}
            for step in range(T):
                cell = tuple(locs[i][step])
                j = 2 * step                      # prediction index for observation token 2*step+1
                if cell in last and j < pred.shape[1]:
                    key = ("early" if step < 128 else "late", bucket(step - last[cell]))
                    c = out[key]
                    c[0] += int(pred[i, j] == tgt[i, j]); c[1] += 1
                    c[2] += int(tgt[i, j] == env.unified_blank)
                last[cell] = step
        done += b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(REPO / "runs/paper2x2/p0"))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--lengths", nargs="+", type=int, default=[128, 1024])
    ap.add_argument("--n-traj", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    raw = {}  # (arm, T, seed) -> {phase|bucket: [h, n, blank]}
    for arm in ARMS:
        for s in a.seeds:
            blob = torch.load(Path(a.runs_dir) / f"{arm}_s{s}" / f"{arm}.pt", map_location="cpu", weights_only=False)
            cfg = blob["config"]
            m = VARIANT_MAP[arm](vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                                 n_layers=cfg["n_layers"], grid_size=cfg["grid_size"]).to(dev).eval()
            m.load_state_dict(blob["model_state_dict"])
            for T in a.lengths:
                env = GridWorld(size=cfg["grid_size"], n_obs_types=cfg["n_obs_types"], p_empty=cfg["p_empty"],
                                n_landmarks=cfg["n_landmarks"], seed=10000)
                r = run(m, env, T, a.n_traj if T <= 128 else max(64, a.n_traj // 4), 16 if T > 128 else 64, s, dev)
                raw[(arm, T, s)] = {f"{p}|{k}": v for (p, k), v in r.items()}
                tot = sum(v[1] for v in r.values()); hit = sum(v[0] for v in r.values())
                print(f"{arm:13s} s{s} T={T:5d} overall {hit / tot:.3f} (n={tot})", flush=True)
            del m; torch.cuda.empty_cache()
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in raw.items()}, open(REPO / "REVISIT_2X2.json", "w"))

    L = [open(REPO / "REVISIT_2X2.md").read().rstrip() + "\n"] if (REPO / "REVISIT_2X2.md").exists() else []
    L.append("## Results\n")
    for T in a.lengths:
        for phase in (["early"] if T <= 128 else ["early", "late"]):
            L.append(f"### T={T}, revisits at steps {'< 128 (inside the training length)' if phase == 'early' else '>= 128 (beyond it)'}\n")
            keys = [k for k in LAB if any(f"{phase}|{k}" in raw[(ARMS[0], T, s)] for s in a.seeds)]
            L.append("| arm | " + " | ".join(keys) + " |"); L.append("|---" * (len(keys) + 1) + "|")
            acc = {}
            for arm in ARMS:
                row = []
                for k in keys:
                    v = {s: raw[(arm, T, s)][f"{phase}|{k}"] for s in a.seeds if f"{phase}|{k}" in raw[(arm, T, s)]}
                    acc[(arm, k)] = {s: h / n for s, (h, n, _) in v.items() if n > 0}
                    row.append(f"{np.mean(list(acc[(arm, k)].values())):.3f}" if acc[(arm, k)] else "--")
                L.append(f"| {arm} | " + " | ".join(row) + " |")
            blank, share, nev = [], [], []
            totev = sum(raw[(ARMS[0], T, s)][f"{phase}|{k}"][1] for k in keys for s in a.seeds if f"{phase}|{k}" in raw[(ARMS[0], T, s)])
            for k in keys:
                v = [raw[(ARMS[0], T, s)][f"{phase}|{k}"] for s in a.seeds if f"{phase}|{k}" in raw[(ARMS[0], T, s)]]
                blank.append(f"*{sum(x[2] for x in v) / max(1, sum(x[1] for x in v)):.3f}*")
                nev.append(str(sum(x[1] for x in v)))
                share.append(f"{sum(x[1] for x in v) / max(1, totev):.1%}")
            L += ["| *blank rate (floor)* | " + " | ".join(blank) + " |",
                  "| *events (all seeds)* | " + " | ".join(nev) + " |",
                  "| *share of events* | " + " | ".join(share) + " |", ""]
            L.append("RoPE - PoPE-Flat, paired by seed (same trajectories):\n")
            L.append("| bucket | delta | sd | MDE | seeds + | verdict |"); L.append("|---|---|---|---|---|---|")
            for k in keys:
                A, B = acc[("RoPE", k)], acc[("PoPE-Flat", k)]
                common = sorted(set(A) & set(B))
                if len(common) < 3:
                    L.append(f"| {k} | too few seeds | | | | |"); continue
                c = paired({s: A[s] for s in common}, {s: B[s] for s in common})
                L.append(f"| {k} | {c.delta:+.3f} | {c.sd:.3f} | {c.mde:.3f} | {c.n_pos}/{c.n} | {c.verdict} |")
            L.append("")
    (REPO / "REVISIT_2X2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

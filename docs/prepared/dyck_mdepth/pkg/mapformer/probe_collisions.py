"""T2's decisive test: does length act ONLY through kernel collisions?

`THEORY_SEARCH_AND_LENGTH.md` T2 says OOD-length failure is pigeonhole exhaustion of the position
code: as the sequence lengthens, more prior keys compete for the same resolvable phase cells, and
some distractor lands at least as high as the answer.

For MapFormer-EM the position kernel is a clean position-only channel, so the collision count
needs no free parameter:

    N_coll(query p) = #{ prior observation keys j at a DIFFERENT cell with A_P[p, j] >= A_P[p, true] }

Measured across lengths on the paper task, together with accuracy at the same scored events.

**The test.** If length acts only through collisions, accuracy plotted against N_coll must fall on
ONE curve for every length. If, at matched N_coll, longer sequences are still worse, something
else drives the length axis and T2 is incomplete.

    python3 -m mapformer.probe_collisions
"""
from __future__ import annotations

import json

import numpy as np
import torch

from mapformer.ckpt_guard import REPO
from mapformer.environment import GridWorld
from mapformer.probe_anatomy import ap_from_delta
from mapformer.train_variant import VARIANT_MAP

ARM = "VanillaEM_P0"
RUNS = REPO / "runs/paper_task_rerun"
LENGTHS = [256, 512, 1024, 2048]
GRID, PE, N_EP, BATCH = 128, 0.8, 8, 4


@torch.no_grad()
def one(model, env, L, n_ep, dev):
    """L = n_steps. The stream is interleaved: 2L tokens, observations at odd indices, one loc
    per STEP -- token 2t+1 carries the observation of step t. Getting this wrong indexes past
    the end of locs, which is how it was caught."""
    batch = max(1, min(4, 4096 // L))
    rows = []
    for _ in range(max(1, n_ep // batch)):
        tokens, _om, revisit, locs = env.generate_batch(batch, L)
        x = tokens[:, :-1].to(dev)
        e = model.token_emb(x)
        AP = ap_from_delta(model, model.action_to_lie(e), e).sum(1)   # (B, tok, tok)
        logits = model(x)
        mask = revisit[:, 1:]
        for b in range(x.shape[0]):
            pos = np.asarray(locs[b])
            for p in torch.nonzero(mask[b]).flatten().tolist():
                if p >= x.shape[1] or p % 2 != 0:      # scored token p+1 is an odd (obs) index
                    continue
                t_step = p // 2
                if t_step >= len(pos):
                    continue
                cell = tuple(pos[t_step])
                same = [2 * u + 1 for u in range(t_step) if tuple(pos[u]) == cell]
                diff = [2 * u + 1 for u in range(t_step) if tuple(pos[u]) != cell]
                if not same or not diff:
                    continue
                ap = AP[b, p]
                best_true = float(ap[same].max())
                n_coll = int((ap[diff] >= best_true).sum())
                ok = int(logits[b, p].argmax().item() == int(tokens[b, p + 1]))
                rows.append((n_coll, ok, len(same) + len(diff)))
    return np.array(rows, dtype=float)


def main():
    dev = "cuda:1" if torch.cuda.is_available() else "cpu"
    out, table = {}, []
    for s in range(4):
        pt = RUNS / f"{ARM}_s{s}" / f"{ARM}.pt"
        blob = torch.load(pt, map_location="cpu", weights_only=False)
        cfg = blob["config"]
        m = VARIANT_MAP[ARM](vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                             n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                             grid_size=cfg["grid_size"]).to(dev).eval()
        m.load_state_dict(blob["model_state_dict"])
        for L in LENGTHS:
            env = GridWorld(size=GRID, n_obs_types=cfg["n_obs_types"], p_empty=PE,
                            n_landmarks=0, seed=10000 + s)
            r = one(m, env, L, N_EP, dev)
            out.setdefault(L, []).append(r)
            print(f"s{s} L={L:5d}: acc {r[:,1].mean():.3f}  mean collisions {r[:,0].mean():8.1f} "
                  f"(median {np.median(r[:,0]):.0f})  keys {r[:,2].mean():.0f}  n={len(r)}", flush=True)
    L_ = ["## T2 -- does length act only through kernel collisions? (VanillaEM_P0, paper task)\n",
          "`N_coll` = prior keys at a different cell that the model's own position kernel ranks "
          "at least as high as the correct key. No free parameters.\n",
          "| length | accuracy | mean collisions | median | prior keys |", "|---|---|---|---|---|"]
    for L in LENGTHS:
        r = np.concatenate(out[L])
        L_.append(f"| {L} | {r[:,1].mean():.3f} | {r[:,0].mean():.1f} | {np.median(r[:,0]):.0f} | "
                  f"{r[:,2].mean():.0f} |")
    L_ += ["", "### The test: accuracy against collisions, per length\n",
           "If length acts ONLY through collisions, every length falls on one curve.\n",
           "| collisions | " + " | ".join(f"L={L}" for L in LENGTHS) + " |",
           "|---" * (len(LENGTHS) + 1) + "|"]
    bins = [(0, 0), (1, 2), (3, 8), (9, 32), (33, 128), (129, 10**9)]
    for lo, hi in bins:
        cells = []
        for L in LENGTHS:
            r = np.concatenate(out[L])
            sel = (r[:, 0] >= lo) & (r[:, 0] <= hi)
            cells.append(f"{r[sel,1].mean():.3f} (n={int(sel.sum())})" if sel.sum() >= 20 else "-")
        lab = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")
        L_.append(f"| {lab} | " + " | ".join(cells) + " |")
    txt = "\n".join(L_)
    print("\n" + txt)
    (REPO / "runs/search/T2_COLLISIONS.md").write_text(txt + "\n")
    json.dump({str(L): np.concatenate(out[L]).tolist() for L in LENGTHS},
              open(REPO / "_COLLISIONS.json", "w"))


if __name__ == "__main__":
    main()

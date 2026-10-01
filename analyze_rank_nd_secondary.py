"""Declared secondaries for RANK_ND_PREREG.md (Amendment 1, written before any result of the batch was read).

Per arm and seed, at T=1024:
  S2  held-out-map accuracy split into WRAP-ONLY revisits (the cell was seen before only at a different
      unwrapped position, so the phase must be periodic in N) and the rest;
  S3  accuracy on the run's own TRAINING map (seed = run seed) beside the held-out map (seed 10000) --
      a gap means the map was memorised rather than read from context; SOLVED runs whose held-out
      accuracy is below 0.95 are flagged;
plus S4, the 2D control's B2 - A2 contrast, printed by analyze_rank_nd.py already.
Floors (S1) are in docs/audits/2026-09-27/nd_floor_wrap.py (retrace 0.750 at D=2 N=32, 0.653 at D=3 N=10).
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_nd import GridWorldND
from mapformer.stats_core import classify_run
from mapformer.train_variant import VARIANT_MAP

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/rank_nd"; S = list(range(8))
CELLS = {"A2": (2, 32, "Vanilla_r2ph"), "B2": (2, 32, "Vanilla_r3ph"), "A3": (3, 10, "Vanilla_r3ph"),
         "B3": (3, 10, "Vanilla_r4ph")}


@torch.no_grad()
def strat_acc(m, env, T, n, seed, dev):
    np.random.seed(seed)
    ok = {"wrap": 0, "near": 0}; tot = {"wrap": 0, "near": 0}
    for _ in range(n):
        tok, _o, rev = env.generate_trajectory(T)
        a = tok[0::2].numpy(); r = rev[1::2].numpy()
        U = np.cumsum(env.action_deltas[a], axis=0); P = [tuple(x) for x in env.visited_locations]
        seen_unw = set()
        wrap = np.zeros(T, bool)
        for t in range(T):
            if r[t]:
                wrap[t] = tuple(U[t]) not in seen_unw
            seen_unw.add(tuple(U[t]))
        x = tok[None].to(dev)
        pred = F.log_softmax(m(x[:, :-1]).float(), -1).argmax(-1)[0].cpu().numpy()
        for t in np.nonzero(r)[0]:
            i = 2 * t + 1; k = "wrap" if wrap[t] else "near"
            tot[k] += 1; ok[k] += int(pred[i - 1] == int(tok[i]))
    return {k: ok[k] / max(tot[k], 1) for k in ok}, tot


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    out = {}
    print("== S2/S3 per arm, T=1024: held-out map (wrap-only / other), own training map (all) ==")
    for key, (D, N, v) in CELLS.items():
        held = GridWorldND(dims=D, size=N, seed=10000)
        rows = []
        for s in S:
            b = torch.load(f"{R}/D{D}/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False); c = b["config"]
            m = VARIANT_MAP[v](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                               n_layers=c["n_layers"], grid_size=c["grid_size"])
            m.load_state_dict(b["model_state_dict"]); m.to(dev).eval()
            h, ht = strat_acc(m, held, 1024, 60, 10**6 + s, dev)
            own = GridWorldND(dims=D, size=N, seed=s)
            o, ot = strat_acc(m, own, 1024, 60, 10**6 + s, dev)
            hall = (h["wrap"] * ht["wrap"] + h["near"] * ht["near"]) / (ht["wrap"] + ht["near"])
            oall = (o["wrap"] * ot["wrap"] + o["near"] * ot["near"]) / (ot["wrap"] + ot["near"])
            sol = classify_run(b["losses"])["registered"] == "SOLVED"
            rows.append({"seed": s, "held_wrap": h["wrap"], "held_other": h["near"], "held_all": hall, "own_all": oall,
                         "wrap_share": ht["wrap"] / (ht["wrap"] + ht["near"]), "solved": sol,
                         "flag": sol and hall < 0.95})
        out[key] = rows
        mean = lambda f: np.mean([r[f] for r in rows])
        print(f"  {key} D={D} {v:13s} held wrap {mean('held_wrap'):.3f} other {mean('held_other'):.3f} all {mean('held_all'):.3f} | "
              f"own map {mean('own_all'):.3f} (gap {mean('own_all') - mean('held_all'):+.3f}) | wrap share {mean('wrap_share'):.2f} | "
              f"flagged SOLVED-but-held<0.95: {[r['seed'] for r in rows if r['flag']]}")
    json.dump(out, open(f"{REPO}/RANK_ND_SECONDARY.json", "w"), indent=1)


if __name__ == "__main__":
    main()

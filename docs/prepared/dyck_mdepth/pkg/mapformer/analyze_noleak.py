"""Analysis for the leakage test (NOLEAK_PREREG.md), built on the permanent guards.

2x2: {install 1/64, 1/8} x {leak open (runs/unfreeze), leak closed (runs/noleak)},
seeds 0-7. Reads exactly the registered quantities: the determinism re-check, the
manipulation check (traj_leak == 0 and traj_slope == traj_latpath at every epoch in the
closed arms), L1, L1b, L2, and L3. Endpoint slopes are recomputed from FINAL weights for
all four cells, so open and closed cells are measured identically.

Run: python3 -m mapformer.analyze_noleak   (from /home/prashr)
"""
import json
from pathlib import Path

import numpy as np
import torch

from mapformer.stats_guard import paired, table, interaction
from mapformer.ckpt_guard import compare_checkpoints
from mapformer.train_variant import VARIANT_MAP
from mapformer.model_em_warm import N_SYM, Q0, K_MAX

REPO = Path("/home/prashr/mapformer")
SEEDS = range(8)
CELLS = {"e64_open": ("unfreeze", "EMUnf_0"), "e8_open": ("unfreeze", "EMUnf_0_e8"),
         "e64_closed": ("noleak", "EMNoLeak_e64"), "e8_closed": ("noleak", "EMNoLeak_e8")}


def _slope(dd):
    k = np.arange(K_MAX); sym = dd[:N_SYM].mean(0); y = dd[Q0:Q0 + K_MAX] @ sym / (sym @ sym)
    return float(((k - k.mean()) * (y - y.mean())).sum() / ((k - k.mean()) ** 2).sum())


def _endpoint(variant, sd):
    m = VARIANT_MAP[variant](vocab_size=89, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    m.load_state_dict(sd); m.eval()
    with torch.no_grad():
        x = m.token_emb(torch.arange(89)); wi = m.action_to_lie.w_in.weight; wo = m.action_to_lie.w_out.weight
        dl = ((x[:, :2] @ wi[:, :2].T) @ wo.T).numpy(); dc = ((x[:, 2:] @ wi[:, 2:].T) @ wo.T).numpy()
    leak = float((np.linalg.norm(dc[:N_SYM], axis=1) / np.linalg.norm(dl[:N_SYM], axis=1)).mean())
    return _slope(dl + dc), _slope(dl), leak


def main():
    missing = [f"{d}/{v}_s{s}" for d, v in CELLS.values() for s in SEEDS
               if not (REPO / f"runs/{d}/{v}_s{s}/{v}_recency.json").exists()]
    if missing:
        print(f"INCOMPLETE: {len(missing)} runs missing, e.g. {missing[:3]}"); return
    R = {}
    for cell, (d, v) in CELLS.items():
        r = {k: {} for k in ("acc", "acc2", "loss", "eff", "latpath", "leak")}
        for s in SEEDS:
            j = json.load(open(REPO / f"runs/{d}/{v}_s{s}/{v}_recency.json"))
            ck = torch.load(REPO / f"runs/{d}/{v}_s{s}/{v}_recency.pt", map_location="cpu", weights_only=False)
            r["acc"][s], r["acc2"][s], r["loss"][s] = j["1024"]["acc"], j["2048"]["acc"], ck["losses"][-1]
            r["eff"][s], r["latpath"][s], r["leak"][s] = _endpoint(v, ck["model_state"])
            if cell.endswith("closed"):
                st = ck["model_state"]; lk = st["traj_leak"]; ts, tl = st["traj_slope"], st["traj_latpath"]
                ok = torch.isfinite(ts)
                r.setdefault("manip", {})[s] = (float(lk[torch.isfinite(lk)].abs().max()),
                                                float((ts[ok] - tl[ok]).abs().max()))
        R[cell] = r

    print("== determinism re-check")
    print((REPO / "runs/noleak_repro/DETERMINISM.txt").read_text().strip())
    print(compare_checkpoints(REPO / "runs/noleak_repro/EMUnf_0_e8_s0/EMUnf_0_e8_recency.pt",
                              REPO / "runs/unfreeze/EMUnf_0_e8_s0/EMUnf_0_e8_recency.pt").report())

    print("\n== manipulation check (closed arms): max |traj_leak|, max |traj_slope - traj_latpath| over all epochs")
    for cell in ("e8_closed", "e64_closed"):
        mm = R[cell]["manip"]; print(f"  {cell:11s} leak {max(v[0] for v in mm.values()):.2e}   identity {max(v[1] for v in mm.values()):.2e}")

    print("\n== cells (seeds 0-7): acc T=1024 | T=2048 | >=0.95 | final loss | eff slope | latent pathway | leak")
    for cell, r in R.items():
        a = np.array(list(r["acc"].values()))
        f = lambda k: np.mean(list(r[k].values()))
        print(f"  {cell:11s} {a.mean():.3f}+/-{a.std(ddof=1):.3f} | {f('acc2'):.3f} | {int((a >= 0.95).sum())}/8 | "
              f"{f('loss'):.3f} | {f('eff'):+.3f} | {f('latpath'):+.3f} | {f('leak'):.3f}")

    A = {c: R[c]["acc"] for c in R}
    L1 = paired(A["e8_closed"], A["e8_open"], "L1  e8: closed - open")
    c64 = paired(A["e64_closed"], A["e64_open"], "     e64: closed - open")
    sc = paired(A["e8_closed"], A["e64_closed"], "L3  scale, leak closed (e8 - e64)")
    so = paired(A["e8_open"], A["e64_open"], "L3  scale, leak open (e8 - e64)")
    lp = paired(R["e8_closed"]["latpath"], R["e8_open"]["latpath"], "L1b latent pathway e8: closed - open")
    print("\n== contrasts (paired by seed, accuracy at T=1024 unless labelled)")
    print(table([L1, c64, sc, so, lp]))
    print(interaction(sc, so, "L3  interaction (scale x leak)"))

    e8c = np.array(list(A["e8_closed"].values())); e8eff = np.array(list(R["e8_closed"]["eff"].values()))
    e64c = np.array(list(A["e64_closed"].values())); e64lp = np.array(list(R["e64_closed"]["latpath"].values()))
    print("\n== registered verdicts")
    print(f"  L1  e8_closed >=0.95 on {int((e8c >= 0.95).sum())}/8 (need >=7), mean final eff slope {e8eff.mean():+.3f} (need <= -0.95)")
    print(f"  L2  e64_closed mean acc {e64c.mean():.3f} (collapse if <= 0.75), mean final latent pathway {e64lp.mean():+.3f} (collapse if > -0.5)")
    json.dump({c: {k: v for k, v in r.items() if k != "manip"} for c, r in R.items()},
              open(REPO / "_NOLEAK.json", "w"), indent=1)


if __name__ == "__main__":
    main()

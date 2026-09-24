"""Readouts for RANK_PERHEAD_PREREG.md (pilot): per-head r=2 vs stored r=2 / r=4 at T=1024."""
import json
import numpy as np
import torch

from mapformer.analyze_rank_matched import classify

REPO = "/home/prashr/mapformer"; S = [0, 1]


def acc(js, v, T):
    J = json.load(open(js)); d = {x[0]: x for x in J[f"0.0|{v}|{T}"]}
    return {s: d[s][1] for s in d}


def main():
    new, old = f"{REPO}/runs/rank_perhead_pilot/p0", f"{REPO}/runs/rank_matched_e900/p0"
    print("== run classes at 900 epochs ==")
    for v, root in (("Vanilla_r2ph", new), ("Vanilla", old), ("Vanilla_r4", old)):
        for s in S:
            b = torch.load(f"{root}/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)
            c, tail, ratio = classify(b["losses"])
            print(f"  {v:13s} s{s}  {c:10s} tail loss {tail:.4f}" + ("" if ratio is None else f"  last10%/prev10% {ratio:.3f}"))
    print("\n== reproduction: our r=2 seed 0 retrained vs stored ==")
    a = np.array(torch.load(f"{new}/Vanilla_s0/Vanilla.pt", map_location="cpu", weights_only=False)["losses"])
    b = np.array(torch.load(f"{old}/Vanilla_s0/Vanilla.pt", map_location="cpu", weights_only=False)["losses"])
    print(f"  max |per-epoch loss diff| over {len(a)} epochs: {np.abs(a - b).max():.2e}")
    print("\n== T=1024 / 512 / 2048 accuracy ==")
    for T in (512, 1024, 2048):
        n = acc(f"{REPO}/RANK_PERHEAD_PILOT.json", "Vanilla_r2ph", T)
        o2 = acc(f"{REPO}/RANK_MATCHED_e900.json", "Vanilla", T); o4 = acc(f"{REPO}/RANK_MATCHED_e900.json", "Vanilla_r4", T)
        print(f"  T={T}: per-head r2 " + " ".join(f"{n[s]:.3f}" for s in S) +
              " | our r2 " + " ".join(f"{o2[s]:.3f}" for s in S) + " | our r4 " + " ".join(f"{o4[s]:.3f}" for s in S))
    st = json.load(open(f"{REPO}/RANK_PERHEAD_PILOT_STRATA.json")); so = json.load(open(f"{REPO}/RANK_MATCHED_e900_STRATA.json"))
    print("\n== T=1024 strata (per-head | our r2 | our r4), per seed ==")
    for k in ("plain_lag<128", "plain_lag>=128", "wrap"):
        print(f"  {k:15s} " + "  ".join(
            f"s{s}: {st[f'Vanilla_r2ph|{s}|1024'][k]['acc']:.3f} | {so[f'Vanilla|{s}|1024'][k]['acc']:.3f} | "
            f"{so[f'Vanilla_r4|{s}|1024'][k]['acc']:.3f}" for s in S))


if __name__ == "__main__":
    main()

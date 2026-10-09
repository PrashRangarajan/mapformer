"""Power of TW_AMBIG's registered primary A, by simulation with the registered decision function
(analyze_tw_ambig.verdict_a, unchanged), on per-seed data from the committed text-world batches. CPU. Output:
tw_ambig_power_out.txt.

Per-seed pools (accuracy at T=1024 and the registered SOLVED class), the path arms of the two committed text-world
batches, which are bimodal in eval mode (the 1/(1-p) attention-scale under-report of runs below ceiling):
  ORACLE pool, eval mode: TEXTWORLD path s0-s7 (runs/textworld) + TW_NORMSTEP MapWM / NormStep / NormStepNB s10-s17
    (TW_NORMSTEP.json): 32 (acc, class) pairs -- RoleTag is modelled as a text-world path model;
  ORACLE pool, train mode: TW_NORMSTEP acc40_train of the same 24 runs.
Context-free arms (MapWM, CF2; never SOLVED): CF-LOW N(0.62, 0.03) (the gate's contaminated-path oracle, 0.600) and
CF-HIGH N(0.80, 0.06) (ctx3 pilot: a trained context-free model reached 0.81 lead/far, 0.61 trail/far).
HSR scenarios: a seed learns the context step with probability p (then drawn from the oracle pool), else it is a
context-free seed; PARTIAL = oracle pool - 0.05 on every seed.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr")
from mapformer.analyze_tw_ambig import verdict_a
from mapformer.stats_core import classify_run
import json

REPO = "/home/prashr/mapformer"


def pools():
    ev, tm = [], []
    for s in range(8):
        e = json.load(open(f"{REPO}/runs/textworld/p0/Vanilla_r4_L1_s{s}/eval.json"))
        c = classify_run(torch.load(f"{REPO}/runs/textworld/p0/Vanilla_r4_L1_s{s}/Vanilla_r4.pt", map_location="cpu",
                                    weights_only=False)["losses"])["registered"]
        ev.append((e["eval"]["1024"]["acc"], c))
    d = json.load(open(f"{REPO}/TW_NORMSTEP.json"))
    for a in ("MapWM", "NormStep", "NormStepNB"):
        for s in range(10, 18):
            r = d[f"{a}_s{s}"]; ev.append((r["acc"], r["cls"])); tm.append((r["acc40_train"], r["cls"]))
    return ev, tm


def sim(n, pool, cf, p_learn, partial=False, n_sim=300, seed=0):
    rng = np.random.default_rng(seed); labs = []; fires = {"N": 0, "R": 0, "G": 0, "NI": 0}
    seeds = list(range(n))
    for _ in range(n_sim):
        res = {}
        o = [pool[i] for i in rng.integers(len(pool), size=n)]
        for j in seeds:
            res[f"RoleTag_s{j}"] = {"acc": o[j][0], "cls": o[j][1]}
            for a in ("MapWM", "CF2"):
                res[f"{a}_s{j}"] = {"acc": float(np.clip(rng.normal(*cf), 0, 1)), "cls": "STALLED"}
            if partial:
                h = pool[rng.integers(len(pool))]; res[f"HSR_s{j}"] = {"acc": h[0] - 0.05, "cls": h[1]}
            elif rng.random() < p_learn:
                h = pool[rng.integers(len(pool))]; res[f"HSR_s{j}"] = {"acc": h[0], "cls": h[1]}
            else:
                res[f"HSR_s{j}"] = {"acc": float(np.clip(rng.normal(*cf), 0, 1)), "cls": "STALLED"}
            res[f"RoPE1_s{j}"] = {"acc": 0.51, "cls": "STALLED"}
        A = verdict_a(res, seeds)
        labs.append(A["label"].split(" (")[0])
        fires["N"] += A["N"][2]; fires["R"] += A["R"][2] and A["R"][0] > 0; fires["G"] += A["G"][2]; fires["NI"] += A["NI_HSR"]
    u, c = np.unique(labs, return_counts=True)
    return {k: v / n_sim for k, v in fires.items()}, {a: b / n_sim for a, b in zip(u, c)}


def main():
    ev, tm = pools()
    print(f"oracle pool, eval mode (n {len(ev)}): mean {np.mean([a for a, _ in ev]):.4f}, sd {np.std([a for a, _ in ev], ddof=1):.4f}, "
          f"SOLVED {sum(c == 'SOLVED' for _, c in ev)}/{len(ev)}; train mode (n {len(tm)}): mean {np.mean([a for a, _ in tm]):.4f}, "
          f"sd {np.std([a for a, _ in tm], ddof=1):.4f}")
    scen = [("HSR learns, p=1.00", 1.0, False), ("p=0.75", 0.75, False), ("p=0.50", 0.5, False),
            ("p=0.25", 0.25, False), ("p=0 (HSR = context-free)", 0.0, False), ("PARTIAL (oracle - 0.05)", 1.0, True)]
    for cfname, cf in (("CF-LOW 0.62", (0.62, 0.03)), ("CF-HIGH 0.80", (0.80, 0.06))):
        print(f"\n== context-free arms {cfname}; eval mode; P(fires) for N (need), R (recovery, HSR > CF2), G (HSR vs oracle),"
              " NI (HSR non-inferior at 0.03); branch shares ==")
        for n in (6, 8, 10, 12):
            for name, p, part in scen:
                if n in (6, 12) and name not in ("p=0.75", "p=0.50"):
                    continue
                f, lab = sim(n, ev, cf, p, part, n_sim=300 if n <= 10 else 150, seed=n)
                print(f"  n={n:2d} {name:26s} N {f['N']:.2f} R {f['R']:.2f} G {f['G']:.2f} NI {f['NI']:.2f} | "
                      + "; ".join(f"{k.replace('CONTEXT STEP NEEDED; ', '')} {v:.2f}" for k, v in sorted(lab.items(), key=lambda x: -x[1])))
    print("\n== train mode (the registered mode check), CF-LOW, n=8 ==")
    for name, p, part in scen:
        f, lab = sim(8, tm, (0.62, 0.03), p, part, n_sim=300, seed=99)
        print(f"  n= 8 {name:26s} N {f['N']:.2f} R {f['R']:.2f} G {f['G']:.2f} NI {f['NI']:.2f} | "
              + "; ".join(f"{k.replace('CONTEXT STEP NEEDED; ', '')} {v:.2f}" for k, v in sorted(lab.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()

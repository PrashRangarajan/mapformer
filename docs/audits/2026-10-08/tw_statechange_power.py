"""Power for TW_STATECHANGE_PREREG.md (CPU, before any run of the batch). The registered decision functions of
analyze_tw_statechange.py are applied unchanged to resampled per-seed values from the STORED text-world runs
(tw_statechange_calib.json: accuracy on the batch's eval stream, aside field shifts; run classes from the checkpoints'
loss curves). Scenarios span the unknowns (state-clause shift relative to asides; T2drop accuracy). 2000 draws per cell.
Pools: MapWM-like = runs/textworld Vanilla_r4 s0-7 + runs/tw_normstep MapWM s10-17 (16); NormStep-like = tw_normstep
NormStep s10-17; DirOnly-like = tw_normstep DirOnly s10-17; RoPE-like = runs/textworld RoPE 1L s0-7.
Floors F1 = 0.2108 (revcopy_state), F2 = 0.6486 (last_dropped): gate_tw_statechange_out.txt."""
import json

import numpy as np
import torch

import mapformer.analyze_tw_statechange as A
from mapformer.stats_core import classify_run

D = "/home/prashr/mapformer/docs/audits/2026-10-08"
cal = json.load(open(f"{D}/tw_statechange_calib.json"))
F1, F2 = 0.2108, 0.6486


def ck_path(key):
    b, rest = key.split(":"); arm, s = rest.rsplit("_s", 1)
    if b == "TW":
        v = "Vanilla_r4" if arm == "MapWM" else "RoPE"
        return f"/home/prashr/mapformer/runs/textworld/p0/{v}_L1_s{s}/{v}.pt"
    return f"/home/prashr/mapformer/runs/tw_normstep/p0/{arm}_s{s}/{arm}.pt"


pool = {}
for key, r in cal.items():
    arm = key.split(":")[1].rsplit("_s", 1)[0]
    if arm == "NormStepNB":
        continue
    cls = classify_run(torch.load(ck_path(key), map_location="cpu", weights_only=False)["losses"])["registered"]
    pool.setdefault(arm, []).append({"acc": r["acc"]["intact"]["all"], "acc_rs": r["acc"]["intact_rs"]["all"],
                                     "solved": cls == "SOLVED", "shift_aside": r.get("geom", {}).get("shift_aside")})
print("pools:", {a: len(v) for a, v in pool.items()})
for a, v in pool.items():
    print(f"  {a}: acc {np.round([x['acc'] for x in v], 4).tolist()}  SOLVED {sum(x['solved'] for x in v)}/{len(v)}"
          + (f"  aside shift {np.round([x['shift_aside'] for x in v], 3).tolist()}" if v[0]["shift_aside"] is not None else ""))
rng = np.random.default_rng(0)
DRAWS = 2000


def draw(arm, n):
    return [pool[arm][i] for i in rng.integers(0, len(pool[arm]), n)]


def tally(f, n):
    out = {}
    for _ in range(DRAWS):
        lab = f(n)
        out[lab] = out.get(lab, 0) + 1
    return {k: v / DRAWS for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


def fmt(t):
    return "; ".join(f"{k[:58]} {v:.2f}" for k, v in t.items())


for n in (6, 8, 10):
    print(f"\n==================== n = {n} ====================")
    # A
    def fA(n):
        w, i = draw("MapWM", n), draw("RoPE", n)
        return A.contrast_state([x["acc_rs"] for x in i], [x["acc_rs"] for x in w], sum(x["solved"] for x in i),
                                sum(x["solved"] for x in w), n)[0].split(" (")[0]
    print("A (MapWM-like vs RoPE-like, T1 ~ all; re-scored = registered):", fmt(tally(fA, n)))
    # B geometry + aside comparison, by scenario (state shift = factor x an independent aside draw)
    for arm in ("MapWM", "NormStep"):
        for fac in (1.0, 2.0, 3.0):
            def fB(n, arm=arm, fac=fac):
                own, st = draw(arm, n), draw(arm, n)
                sh = [fac * x["shift_aside"] for x in st]; sa = [x["shift_aside"] for x in own]
                g = A.geom_label(sh)[0]; al = A.aside_label(sh, sa)[0]
                return f"{g} / {al}"
            print(f"B {arm}, state shift = {fac:.0f} x aside-like:", fmt(tally(fB, n)))
    # C
    for name, gen in (("bound (T2drop ~ the seed's accuracy - U(0, .05))", lambda x: x["acc"] - rng.uniform(0, .05)),
                      ("partial (U(.65, .85))", lambda x: rng.uniform(.65, .85)),
                      ("bound on 5 of 8 seeds, stale on the rest", None),
                      ("not used (U(0, .3))", lambda x: rng.uniform(0, .3))):
        def fC(n, gen=gen):
            w = draw("MapWM", n)
            if gen is None:
                k = round(5 * n / 8); a = [x["acc"] - .02 for x in w[:k]] + [rng.uniform(0, .3) for _ in w[k:]]
            else:
                a = [gen(x) for x in w]
            return A.c_label(a, F1, F2)[0]
        print(f"C {name}:", fmt(tally(fC, n)))
    # D
    for tag, y, shift in (("D1 NormStep-like vs MapWM-like (as stored)", "NormStep", 0.0),
                          ("D2 DirOnly-like vs MapWM-like (as stored)", "DirOnly", 0.0),
                          ("D2 DirOnly-like - 0.03 (state binding costs the oracle)", "DirOnly", -0.03),
                          ("D1 NormStep-like - 0.03", "NormStep", -0.03),
                          ("null: MapWM-like vs MapWM-like", "MapWM", 0.0)):
        def fD(n, y=y, shift=shift):
            w, o = draw("MapWM", n), draw(y, n)
            return A.contrast_state([x["acc_rs"] for x in w], [x["acc_rs"] + shift for x in o], sum(x["solved"] for x in w),
                                    sum(x["solved"] for x in o), n)[0].split(" (")[0]
        print(f"{tag}, re-scored (registered):", fmt(tally(fD, n)))
        if shift == 0.0:
            def fDe(n, y=y):
                w, o = draw("MapWM", n), draw(y, n)
                return A.contrast_state([x["acc"] for x in w], [x["acc"] for x in o], sum(x["solved"] for x in w),
                                        sum(x["solved"] for x in o), n)[0].split(" (")[0]
            print(f"   same in eval mode (the FLAG line):", fmt(tally(fDe, n)))

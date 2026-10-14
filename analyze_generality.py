"""Registered analysis for GENERALITY_PREREG.md (A: P0, E1, E2 primary at lr 1e-4 + descriptive at 1e-3, E3; B: monotone
methods vs Signed_r4 on ring and torus; then declared secondaries). CPU.
  python3 -m mapformer.analyze_generality [--root runs/generality] [--seeds 700-707] [--no-json]
Layout: <root>/torus/<ARM>/p0/<V>_s<S>/<V>.pt, evals GENERALITY_TORUS_<ARM>.json / _STRATA.json in --eval-dir;
constructions <root>/torus/init/s<S>/<AO_bal|SO_bal>/construct.json; ring <root>/ring/<ARM>_s<S>/{<ARM>.pt, eval.json}.
"""
import argparse, json, os

import numpy as np
import torch

from mapformer.model_wrap_exist import wrap_share
from mapformer.stats_core import perm2_p, signflip_p, fisher_solved, classify_run
from mapformer.train_variant import VARIANT_MAP

REPO = "/home/prashr/mapformer"
TORUS_ARMS = {"SO4": "SignedAO4_r4", "AO4": "AbsAO4_r4", "SO_bal": "SignedAO4_r4", "AO_bal": "AbsAO4_r4",
              "SO_bal_hi": "SignedAO4_r4", "AO_bal_hi": "AbsAO4_r4", "Signed_r4": "Signed_r4", "CARoPEF": "CARoPEF"}
RING_ARMS = ("Signed_r4", "CARoPEF", "CoPEF")
PAIRS = ((0, 1), (2, 3))
LOST_ABS, LOST_REL, LOST_MIN = 0.3, 10.0, 0.05


def torus_run(root, ev_dir, arm, s, model=False):
    V = TORUS_ARMS[arm]
    b = torch.load(f"{root}/torus/{arm}/p0/{V}_s{s}/{V}.pt", map_location="cpu", weights_only=False)
    ev = json.load(open(f"{ev_dir}/GENERALITY_TORUS_{arm}.json"))
    acc = {T: dict((x[0], x[1]) for x in ev[f"0.0|{V}|{T}"])[s] for T in (1024, 2048)}
    out = {"losses": np.asarray(b["losses"], float), "acc": acc[1024], "acc2048": acc[2048]}
    c = classify_run(b["losses"]); out["solved"] = c["registered"] == "SOLVED"; out["cls"] = c["cls"]
    if model:
        c2 = b["config"]
        m = VARIANT_MAP[V](vocab_size=c2["vocab_size"], d_model=c2["d_model"], n_heads=c2["n_heads"],
                           n_layers=c2["n_layers"], grid_size=c2["grid_size"])
        m.load_state_dict(b["model_state_dict"]); out["model"] = m
    return out


def e2(R, x, ctl, src, status):
    ns = len(src)
    aw = np.array([R[x, s]["acc"] for s in src]); sc = np.array([R[ctl, s]["acc"] for s in src])
    k_aw = sum(R[x, s]["solved"] for s in src); k_sc = sum(R[ctl, s]["solved"] for s in src)
    mx = lambda arm, s: R[arm, s]["losses"].max()
    lost = lambda arm, s: mx(arm, s) > LOST_ABS or (mx(arm, s) > LOST_REL * mx(ctl, s) and mx(arm, s) > LOST_MIN)
    l_aw = sum(lost(x, s) for s in src); l_sc = sum(lost(ctl, s) for s in src)
    pf = fisher_solved(k_aw, ns, k_sc, ns); pl = fisher_solved(ns - l_aw, ns, ns - l_sc, ns)
    d = aw - sc; sf = signflip_p(d)
    if k_aw >= ns - 1 and aw.mean() >= sc.mean() - 0.01 and l_aw <= 1:
        lab = "HELD"
    elif (pf < 0.05 and k_sc > k_aw) or (sf["p"] < 0.05 and d.mean() <= -0.05) or (pl < 0.05 and l_aw > l_sc):
        lab = "NOT HELD"
    else:
        lab = "PARTIAL"
    print(f"E2 [{x} vs {ctl}, {status}]: {lab} -- {x} {aw.mean():.4f} SOLVED {k_aw}/{ns} LOST {l_aw}/{ns} vs {ctl} "
          f"{sc.mean():.4f} SOLVED {k_sc}/{ns} LOST {l_sc}/{ns}; paired d {d.mean():+.4f} (sign-flip p {sf['p']:.4f}, min p "
          f"{sf['min_p']:.4f}); Fisher SOLVED p {pf:.4f}, Fisher not-LOST p {pl:.4f}")
    print(f"   max per-epoch loss {x}: {' '.join(f'{mx(x, s):.3f}' for s in src)} | {ctl}: {' '.join(f'{mx(ctl, s):.3f}' for s in src)}")
    return lab


def mono(task, ref, arm, acc, solved, n):
    a, r = np.array([acc[arm, s] for s in range(n)]), np.array([acc[ref, s] for s in range(n)])
    ka, kr = int(sum(solved[arm, s] for s in range(n))), int(sum(solved[ref, s] for s in range(n)))
    pf = fisher_solved(ka, n, kr, n); pp = perm2_p(r, a)["p"]; d = a.mean() - r.mean()
    if kr < 4:
        lab = f"UNREADABLE (Signed_r4 SOLVED {kr} < 4)"
    elif (pf < 0.05 and kr > ka) or (pp < 0.05 and d <= -0.05):
        lab = "MONOTONE FAILS"
    elif ka >= kr - 1 and a.mean() >= r.mean() - 0.01:
        lab = "NO DIFFERENCE"
    else:
        lab = "UNMEASURED"
    print(f"B [{task}] {arm} vs {ref}: {lab} -- {arm} {a.mean():.4f} SOLVED {ka}/{n} vs {ref} {r.mean():.4f} SOLVED {kr}/{n}; "
          f"d {d:+.4f} (perm p {pp:.4f}); Fisher p {pf:.4f}")
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{REPO}/runs/generality")
    ap.add_argument("--eval-dir", default=REPO)
    ap.add_argument("--seeds", default="700-707")
    ap.add_argument("--no-json", action="store_true")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-"); seeds = list(range(int(lo), int(hi or lo) + 1)); n = len(seeds)
    R = {(arm, s): torus_run(a.root, a.eval_dir, arm, s, model=arm in ("SO4", "AO4", "SO_bal", "AO_bal"))
         for arm in TORUS_ARMS for s in seeds}
    C = {(arm, s): json.load(open(f"{a.root}/torus/init/s{s}/{arm}/construct.json")) for arm in ("SO_bal", "AO_bal")
         for s in seeds if os.path.exists(f"{a.root}/torus/init/s{s}/{arm}/construct.json")}
    V = {}

    print("== A: torus, per seed: accuracy T = 1024 (held-out map) and class ==")
    print("seed | SO4 | AO4 | SO_bal | AO_bal | SO_bal_hi | AO_bal_hi")
    for s in seeds:
        print(f"{s} | " + " | ".join(f"{R[x, s]['acc']:.4f} {R[x, s]['cls']}" for x in
                                     ("SO4", "AO4", "SO_bal", "AO_bal", "SO_bal_hi", "AO_bal_hi")))
    src = [s for s in seeds if R["SO4", s]["solved"]]; ns = len(src)
    print(f"\nP0: SO4 SOLVED {ns}/{n} -> E1 / E2 on seeds {src}")
    V["P0"] = {"n_solved_sources": ns, "sources": src}
    V["E1"] = {}
    for x in ("SO_bal", "AO_bal"):
        if ns == 0:
            V["E1"][x] = "UNREADABLE (no solved source)"; print(f"E1 [{x}]: {V['E1'][x]}"); continue
        bad = [s for s in src if not ((x, s) in C and abs(C[x, s]["frozen_acc"] - C[x, s]["source_acc"]) <= 0.005
                                      and C[x, s]["construction"]["angle_err_max"] < 1e-2)]
        V["E1"][x] = "EXISTS" if not bad else f"CONSTRUCTION FAILED on {bad}"
        gaps = [abs(C[x, s]["frozen_acc"] - C[x, s]["source_acc"]) for s in src if (x, s) in C]
        errs = [C[x, s]["construction"]["angle_err_max"] for s in src if (x, s) in C]
        print(f"E1 [{x}]: {V['E1'][x]} -- max |frozen - source| {max(gaps):.4f}, max angle error {max(errs):.1e} rad")
    V["E2"] = {}
    for x, ctl, st in (("AO_bal", "SO_bal", "PRIMARY, lr 1e-4"), ("AO_bal_hi", "SO_bal_hi", "descriptive, lr 1e-3")):
        if ns < 4:
            lab = f"UNREADABLE (n_s = {ns} < 4)"; print(f"E2 [{x}]: {lab}")
        elif V["E1"]["AO_bal"] != "EXISTS" or V["E1"]["SO_bal"] != "EXISTS":
            lab = "UNREADABLE (E1 not EXISTS)"; print(f"E2 [{x}]: {lab}")
        else:
            lab = e2(R, x, ctl, src, st)
        V["E2"][x] = {"label": lab, "control": ctl, "status": st}
    so = np.array([R["SO4", s]["acc"] for s in seeds]); ao = np.array([R["AO4", s]["acc"] for s in seeds])
    k_so = sum(R["SO4", s]["solved"] for s in seeds); k_ao = sum(R["AO4", s]["solved"] for s in seeds)
    pf3 = fisher_solved(k_ao, n, k_so, n); d3 = ao.mean() - so.mean(); pp3 = perm2_p(so, ao)["p"]
    if k_so < 4:
        lab3 = f"UNREADABLE (SO4 SOLVED {k_so} < 4)"
    elif (pf3 < 0.05 and k_so > k_ao) or (pp3 < 0.05 and d3 <= -0.05):
        lab3 = "SIGNED LEARNS BETTER"
    elif k_ao >= k_so - 1 and ao.mean() >= so.mean() - 0.01:
        lab3 = "NO SIGN COST"
    else:
        lab3 = "UNMEASURED"
    V["E3"] = lab3
    print(f"E3: {lab3} -- AO4 {ao.mean():.4f} SOLVED {k_ao}/{n} vs SO4 {so.mean():.4f} SOLVED {k_so}/{n}; d {d3:+.4f} "
          f"(perm p {pp3:.4f}); Fisher p {pf3:.4f}")

    print("\n== B: published monotone methods vs Signed_r4 ==")
    V["B"] = {}
    acc = {(x, i): R[x, s]["acc"] for x in ("Signed_r4", "CARoPEF") for i, s in enumerate(seeds)}
    sol = {(x, i): R[x, s]["solved"] for x in ("Signed_r4", "CARoPEF") for i, s in enumerate(seeds)}
    for arm in ("CARoPEF",):                                        # CoPEF: ring only (prereg, Process and cost)
        V["B"][f"torus|{arm}"] = mono("torus T 1024", "Signed_r4", arm, acc, sol, n)
    racc, rsol = {}, {}
    for x in RING_ARMS:
        for i, s in enumerate(seeds):
            d = f"{a.root}/ring/{x}_s{s}"
            b = torch.load(f"{d}/{x}.pt", map_location="cpu", weights_only=False)
            racc[x, i] = json.load(open(f"{d}/eval.json"))["eval"]["128"]["acc"]
            rsol[x, i] = classify_run(b["losses"])["registered"] == "SOLVED"
    for arm in ("CARoPEF", "CoPEF"):
        V["B"][f"ring|{arm}"] = mono("ring p_plus 0.5", "Signed_r4", arm, racc, rsol, n)

    print("\n== secondaries (declared; no verdict) ==")
    for x in ("SO4", "AO4", "SO_bal", "AO_bal"):
        ws = [wrap_share(R[x, s]["model"], pairs=PAIRS)[1] for s in seeds]
        print(f"wrap share {x:7s}: {np.nanmean(ws):.2f} (per seed {' '.join(f'{w:.2f}' for w in ws)})")
    for x in ("SO_bal", "AO_bal"):
        fw = [C[x, s]["frozen_wrap"][1] for s in src if (x, s) in C]
        print(f"{x} wrap share at construction {np.nanmean(fw) if fw else float('nan'):.2f} -> after training "
              f"{np.nanmean([wrap_share(R[x, s]['model'], pairs=PAIRS)[1] for s in src]) if src else float('nan'):.2f}")
        for s in src:
            if (x, s) in C:
                c = C[x, s]["construction"]
                print(f"  {x} s{s}: |W_out| {c['w_out_absmax_signed']:.2f} -> {c['w_out_absmax_built']:.2f}; |omega| max "
                      f"{c['omega_absmax_signed']:.2f} -> {c['omega_absmax_built']:.2f}; latent cond {c['latent_cond_signed']:.0f}")
    for x in TORUS_ARMS:
        print(f"acc T = 2048 (OOD) {x:9s}: {np.mean([R[x, s]['acc2048'] for s in seeds]):.4f}; SOLVED "
              f"{sum(R[x, s]['solved'] for s in seeds)}/{n}; classes {' '.join(R[x, s]['cls'][:4] for s in seeds)}")
    for x in TORUS_ARMS:
        f = f"{a.eval_dir}/GENERALITY_TORUS_{x}_STRATA.json"
        if os.path.exists(f):
            st = json.load(open(f)); rows = [v for k, v in st.items() if k.endswith("|1024")]
            print(f"strata {x:9s}: " + "; ".join(f"{k} {np.mean([r[k]['acc'] for r in rows]):.3f}"
                                                 for k in ("plain_lag<128", "plain_lag>=128", "wrap")))
    auc = np.array([R["AO4", s]["losses"].mean() - R["SO4", s]["losses"].mean() for s in seeds])
    print(f"loss AUC AO4 - SO4 (paired): {auc.mean():+.4f} (sign-flip p {signflip_p(auc)['p']:.4f}; AO4 slower on "
          f"{int((auc > 0).sum())}/{n})")
    print("cross-batch, labelled: SIGN_MATCHED Abs_r4 (observation steps free, no fast-attn) 0/8 SOLVED, Signed_r4 8/8; "
          "FACTORIAL RoPE 1 layer 0.719 (0/4)")
    if not a.no_json:
        json.dump(V, open(f"{REPO}/GENERALITY_VERDICTS.json", "w"), indent=2)


if __name__ == "__main__":
    main()

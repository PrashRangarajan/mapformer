"""Registered analysis for ROBUST_PREREG.md: is the sign learnability cost a one-recipe / one-scale artefact? CPU.
  python3 -m mapformer.analyze_robust [--root runs/robust] [--eval-dir REPO] [--seeds 1400-1407] [--json-out PATH | --no-json]
  python3 -m mapformer.analyze_robust --selftest      # every label and reading from synthetic inputs (no run dirs)
Layout: ring <root>/ring/<recipe>/<arm>_s<S>/{<arm>.pt, eval.json} (train_gauge, deltas +1/-1), recipes def / wd0 / lrx3 /
lrd3; torus <root>/torus/<arm>/p0/<V>_s<S>/<V>.pt (train_variant, d 256, 2 layers), evals ROBUST_TORUS_<arm>.json and
ROBUST_TORUS_<arm>_STRATA.json in --eval-dir. Labels as GAUGE G1-G2 / MECHANISM M4: Fisher on SOLVED (classify_run, final 5%
of training loss < 0.05), unpaired permutation on accuracy; gate UNREADABLE if the signed arm solves < 4/n.
"""
import argparse, json, os

import numpy as np
import torch

from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4, MapFormerWM_AbsAO_r4, action_angles, wrap_share
from mapformer.stats_core import perm2_p, perm2_ci, fisher_solved, classify_run
from mapformer.train_variant import VARIANT_MAP

REPO = "/home/prashr/mapformer"
# recipe -> (lr, weight decay); the in-batch default is the RING_DIAL / GAUGE m-cell recipe
RECIPES = {"def": (1e-3, 0.05), "wd0": (1e-3, 0.0), "lrx3": (3e-3, 0.05), "lrd3": (3.333333e-4, 0.05)}
RING_ARMS = ("SO", "AO")
TORUS_V = {"SO": "SignedAO4_r4", "AO": "AbsAO4_r4"}
TORUS_CFG = {"d_model": 256, "n_layers": 2, "n_heads": 2, "n_steps": 1024, "lr": 1e-3, "batch_size": 16,
             "n_batches": 98, "schedule": "cosine", "fast_attn": True}
PAIRS_TORUS = ((0, 1), (2, 3))
CELLS = ("ring|def", "ring|wd0", "ring|lrx3", "ring|lrd3", "torus|d256L2")
SLB, NSC, UNM = "SIGNED LEARNS BETTER", "NO SIGN COST", "UNMEASURED"


def cell_label(ao_acc, so_acc, k_ao, k_so, n):
    """The GAUGE G1/G2 rule verbatim (analyze_gauge.py), with the signed-arm gate first."""
    ao_acc, so_acc = np.asarray(ao_acc, float), np.asarray(so_acc, float)
    pf = fisher_solved(k_ao, n, k_so, n); pp = perm2_p(so_acc, ao_acc)["p"]; d = float(ao_acc.mean() - so_acc.mean())
    if k_so < 4:
        lab = f"UNREADABLE (SO SOLVED {k_so} < 4)"
    elif (pf < 0.05 and k_so > k_ao) or (pp < 0.05 and d <= -0.05):
        lab = SLB
    elif k_ao >= k_so - 1 and ao_acc.mean() >= so_acc.mean() - 0.01:
        lab = NSC
    else:
        lab = UNM
    return lab, {"d": d, "perm_p": pp, "fisher_p": pf}


def reading(labels):
    """Overall reading, fixed in ROBUST_PREREG.md. labels: {cell: label} over CELLS."""
    ring = [c for c in CELLS if c.startswith("ring")]
    if labels["ring|def"].startswith("UNREADABLE"):
        return "UNREADABLE (the in-batch default ring reference failed its gate)"
    nsc = [c for c in CELLS if labels[c] == NSC]
    if nsc:
        r, t = [c for c in nsc if c in ring], [c for c in nsc if c not in ring]
        kind = " AND ".join(x for x, y in (("RECIPE-DEPENDENT", r), ("SCALE-DEPENDENT", t)) if y)
        return f"{kind} (NO SIGN COST in {', '.join(nsc)})"
    if all(labels[c] == SLB for c in CELLS):
        return "ROBUST"
    unr = [c for c in CELLS if labels[c].startswith("UNREADABLE")]
    unm = [c for c in CELLS if labels[c] == UNM]
    slb = [c for c in CELLS if labels[c] == SLB]
    if len(unr) >= 2:
        return (f"INSUFFICIENT (UNREADABLE {', '.join(unr)}; cost in {', '.join(slb) or 'none'}"
                + (f"; UNMEASURED {', '.join(unm)}" if unm else "") + ")")
    if not unm:
        return f"ROBUST WHERE READABLE (cost in {', '.join(slb)}; UNREADABLE {', '.join(unr)})"
    return (f"PARTLY UNMEASURED (cost in {', '.join(slb) or 'none'}; UNMEASURED {', '.join(unm)}"
            + (f"; UNREADABLE {', '.join(unr)}" if unr else "") + ")")


def selftest():
    n = 8
    hi, lo = [1.0] * n, [0.9] * n
    mid = [0.97, 0.99, 1.0, 0.95, 1.0, 0.98, 0.96, 1.0]
    cases = {"SLB": (lo, hi, 1, 8), "NSC": (hi, hi, 8, 8), "UNM": (mid, hi, 5, 8), "UNR": (lo, lo, 0, 3)}
    got = {k: cell_label(*v, n)[0] for k, v in cases.items()}
    for k, v in got.items():
        print(f"selftest cell {k}: {v}")
    assert got["SLB"] == SLB and got["NSC"] == NSC and got["UNM"] == UNM and got["UNR"].startswith("UNREADABLE")
    S, N, U, X = SLB, NSC, UNM, got["UNR"]
    for labs in ((S, S, S, S, S), (S, N, S, S, S), (S, S, S, S, N), (S, N, S, S, N), (S, S, S, X, S), (S, U, S, X, S),
                 (X, S, S, S, S), (S, S, U, S, U), (S, S, X, X, S)):
        L = dict(zip(CELLS, labs))
        print(f"selftest reading {[l[:4] for l in labs]}: {reading(L)}")
    print("selftest passed: every cell label and every reading reached")


def ring_runs(root, recipe, seeds, epochs):
    lr, wd = RECIPES[recipe]; R = {}
    for arm in RING_ARMS:
        for s in seeds:
            d = f"{root}/ring/{recipe}/{arm}_s{s}"
            ev = json.load(open(f"{d}/eval.json")); b = torch.load(f"{d}/{arm}.pt", map_location="cpu", weights_only=False)
            A = b["config"]["args"]
            assert (A["arm"], A["seed"], tuple(b["config"]["deltas"])) == (arm, s, (1, -1)), (d, A)
            assert abs(A["lr"] - lr) < 1e-9 and abs(A.get("weight_decay", 0.05) - wd) < 1e-12, (d, A["lr"], A.get("weight_decay"))
            assert A["epochs"] == epochs and len(b["losses"]) == epochs, (d, A["epochs"], len(b["losses"]))
            k = classify_run(b["losses"]); k10 = classify_run(b["losses"], solved_loss=0.1)
            R[arm, s] = {"acc": ev["eval"]["128"]["acc"], "acc4": ev["eval"]["512"]["acc"], "solved": k["registered"] == "SOLVED",
                         "solved10": k10["registered"] == "SOLVED", "cls": k["cls"], "auc": float(np.mean(b["losses"])),
                         "tail": float(k["tail"]), "wrap": ev.get("wrap"), "blob": b}
    return R


def torus_runs(root, ev_dir, seeds, epochs, n_batches=98):
    R = {}; want = {**TORUS_CFG, "n_batches": n_batches}
    for arm, V in TORUS_V.items():
        ev = json.load(open(f"{ev_dir}/ROBUST_TORUS_{arm}.json"))
        st_f = f"{ev_dir}/ROBUST_TORUS_{arm}_STRATA.json"; st = json.load(open(st_f)) if os.path.exists(st_f) else {}
        for s in seeds:
            b = torch.load(f"{root}/torus/{arm}/p0/{V}_s{s}/{V}.pt", map_location="cpu", weights_only=False)
            c = b["config"]
            bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
            assert not bad and b["variant"] == V and b["seed"] == s and c["epochs"] == epochs and len(b["losses"]) == epochs, \
                (arm, s, bad, c["epochs"], len(b["losses"]))
            acc = {T: dict((x[0], x[1]) for x in ev[f"0.0|{V}|{T}"])[s] for T in (1024, 2048)}
            k = classify_run(b["losses"]); k10 = classify_run(b["losses"], solved_loss=0.1)
            m = VARIANT_MAP[V](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"], n_layers=c["n_layers"],
                               grid_size=c["grid_size"])
            m.load_state_dict(b["model_state_dict"])
            R[arm, s] = {"acc": acc[1024], "acc2": acc[2048], "solved": k["registered"] == "SOLVED",
                         "solved10": k10["registered"] == "SOLVED", "cls": k["cls"], "auc": float(np.mean(b["losses"])),
                         "tail": float(k["tail"]), "model": m, "strata": st.get(f"{V}|{s}|1024")}
    return R


def opp_share(m, pairs):
    """Share of channels (|wrapped angle of the first move| > 0.05) whose two paired move angles, wrapped to (-pi, pi], have
    OPPOSITE signs (GAUGE's same-sign readout, complemented; gauge- and 2 pi-invariant)."""
    A = action_angles(m).double().numpy(); W = np.angle(np.exp(1j * A)); out = []
    for i, j in pairs:
        sel = np.abs(W[i]) > 0.05
        out.append((np.sign(W[i]) != np.sign(W[j]))[sel])
    o = np.concatenate(out)
    return float(o.mean()) if o.size else float("nan")


def contrast(name, R, seeds, n, acc_key="acc"):
    ao = np.array([R["AO", s][acc_key] for s in seeds]); so = np.array([R["SO", s][acc_key] for s in seeds])
    ka = sum(R["AO", s]["solved"] for s in seeds); ks = sum(R["SO", s]["solved"] for s in seeds)
    lab, st = cell_label(ao, so, ka, ks, n)
    ci = perm2_ci(so, ao, step=0.005, grid=(-1.0, 1.0))
    print(f"{name}: {lab} -- AO {ao.mean():.4f} SOLVED {ka}/{n} vs SO {so.mean():.4f} SOLVED {ks}/{n}; d {st['d']:+.4f} "
          f"[95% CI {ci['lo']:+.3f}, {ci['hi']:+.3f}] (perm p {st['perm_p']:.4f}); Fisher p {st['fisher_p']:.4f}")
    return {"label": lab, "AO_solved": int(ka), "SO_solved": int(ks), "AO_acc": float(ao.mean()), "SO_acc": float(so.mean()),
            **st, "ci": [ci["lo"], ci["hi"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{REPO}/runs/robust")
    ap.add_argument("--eval-dir", default=REPO)
    ap.add_argument("--seeds", default="1400-1407")
    ap.add_argument("--ring-epochs", type=int, default=300)
    ap.add_argument("--torus-epochs", type=int, default=900)
    ap.add_argument("--torus-n-batches", type=int, default=98, help="smoke only")
    ap.add_argument("--json-out", default=f"{REPO}/ROBUST_VERDICTS.json")
    ap.add_argument("--no-json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    lo, _, hi = a.seeds.partition("-"); seeds = list(range(int(lo), int(hi or lo) + 1)); n = len(seeds)
    RR = {r: ring_runs(a.root, r, seeds, a.ring_epochs) for r in RECIPES}
    TR = torus_runs(a.root, a.eval_dir, seeds, a.torus_epochs, a.torus_n_batches)

    print(f"== ring (+1/-1, 32 cells, T = 128, {a.ring_epochs} epochs), per seed: accuracy T = 128 and class ==")
    print("seed | " + " | ".join(f"{r}:{arm}" for r in RECIPES for arm in RING_ARMS))
    for s in seeds:
        print(f"{s} | " + " | ".join(f"{RR[r][arm, s]['acc']:.4f} {RR[r][arm, s]['cls'][:4]}" for r in RECIPES for arm in RING_ARMS))
    print(f"\n== torus (d 256, 2 layers, T = 1024, {a.torus_epochs} epochs), per seed: accuracy T = 1024 and class ==")
    for s in seeds:
        print(f"{s} | SO {TR['SO', s]['acc']:.4f} {TR['SO', s]['cls'][:4]} | AO {TR['AO', s]['acc']:.4f} {TR['AO', s]['cls'][:4]}")

    print("\n== registered contrasts (AO vs SO per cell) ==")
    V = {"cells": {}}
    for r in RECIPES:
        lr, wd = RECIPES[r]
        V["cells"][f"ring|{r}"] = contrast(f"ring|{r} (lr {lr:g}, wd {wd:g})", RR[r], seeds, n)
    V["cells"]["torus|d256L2"] = contrast("torus|d256L2 (d 256, 2 layers, 2 heads)", TR, seeds, n)
    V["reading"] = reading({c: V["cells"][c]["label"] for c in CELLS})
    print(f"\nReading: {V['reading']}")
    if not a.no_json:                                       # verdicts written before any secondary can fail
        json.dump(V, open(a.json_out, "w"), indent=2)

    print("\n== secondaries (declared; no verdict) ==")
    print("floor: predict blank ~0.50 (p_empty 0.5); CANCEL's measured +1/-1 ring floor 0.554; torus always-blank 0.512")
    for r in RECIPES:
        R = RR[r]
        for arm in RING_ARMS:
            w = [R[arm, s]["wrap"][1] for s in seeds if R[arm, s]["wrap"]]
            m = []
            for s in seeds:
                b = R[arm, s]["blob"]
                cls = MapFormerWM_SignedAO_r4 if arm == "SO" else MapFormerWM_AbsAO_r4
                mm = cls(vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=32,
                         action_ids=(b["config"]["action_offset"], b["config"]["action_offset"] + 1))
                mm.load_state_dict(b["model_state_dict"]); m.append(opp_share(mm, ((0, 1),)))
            print(f"ring|{r:4s} {arm}: acc 4T {np.mean([R[arm, s]['acc4'] for s in seeds]):.4f}; loss AUC "
                  f"{np.mean([R[arm, s]['auc'] for s in seeds]):.3f}; SOLVED at loss 0.1 {sum(R[arm, s]['solved10'] for s in seeds)}/{n}; "
                  f"classes {' '.join(R[arm, s]['cls'][:4] for s in seeds)}; wrap-cancel share {np.nanmean(w) if w else float('nan'):.2f}; "
                  f"opposite-sign share {np.nanmean(m):.2f}")
        k10a = sum(R["AO", s]["solved10"] for s in seeds); k10s = sum(R["SO", s]["solved10"] for s in seeds)
        print(f"   ring|{r} threshold sensitivity: SOLVED at loss 0.1 AO {k10a}/{n} vs SO {k10s}/{n} (Fisher p "
              f"{fisher_solved(k10a, n, k10s, n):.4f})")
    for arm in ("SO", "AO"):
        R = TR
        k10 = sum(R[arm, s]["solved10"] for s in seeds)
        print(f"torus {arm}: acc 2T (2048, OOD) {np.mean([R[arm, s]['acc2'] for s in seeds]):.4f}; loss AUC "
              f"{np.mean([R[arm, s]['auc'] for s in seeds]):.3f}; SOLVED at loss 0.1 {k10}/{n}; classes "
              f"{' '.join(R[arm, s]['cls'][:4] for s in seeds)}; wrap-cancel share "
              f"{np.nanmean([wrap_share(R[arm, s]['model'], pairs=PAIRS_TORUS)[1] for s in seeds]):.2f}; opposite-sign share "
              f"{np.nanmean([opp_share(R[arm, s]['model'], PAIRS_TORUS) for s in seeds]):.2f}")
        rows = [R[arm, s]["strata"] for s in seeds if R[arm, s]["strata"]]
        if rows:
            print(f"   strata {arm}: " + "; ".join(
                f"{k} {np.mean([r[k]['acc'] for r in rows if r[k]['acc'] is not None]):.3f}"
                for k in ("plain_lag<128", "plain_lag>=128", "wrap") if any(r[k]['acc'] is not None for r in rows)))
    k10a = sum(TR["AO", s]["solved10"] for s in seeds); k10s = sum(TR["SO", s]["solved10"] for s in seeds)
    print(f"   torus threshold sensitivity: SOLVED at loss 0.1 AO {k10a}/{n} vs SO {k10s}/{n} (Fisher p "
          f"{fisher_solved(k10a, n, k10s, n):.4f})")
    print("descriptive, across recipes (same seeds, same initial weights and data stream per seed): AO SOLVED "
          + ", ".join(f"{r} {sum(RR[r]['AO', s]['solved'] for s in seeds)}/{n}" for r in RECIPES)
          + "; SO SOLVED " + ", ".join(f"{r} {sum(RR[r]['SO', s]['solved'] for s in seeds)}/{n}" for r in RECIPES))


if __name__ == "__main__":
    main()

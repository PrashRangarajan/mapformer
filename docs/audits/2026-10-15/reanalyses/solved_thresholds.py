"""Fix-14 R1 (post hoc, CPU, eval-only; PREDICTIONS.md): SOLVED counts at final-5% loss thresholds 0.02 / 0.05 (registered) /
0.1 and under an accuracy criterion (acc >= 0.99 at the training length, the batch's stored eval-mode accuracy), for every
batch the workshop draft quotes a SOLVED count from, with Fisher p for the registered contrasts at each criterion.
Check: at 0.05 every arm must reproduce its registered count (REG, read from the results files); the script exits 1 if not.
  cd /home/prashr && OMP_NUM_THREADS=4 PYTHONPATH=/home/prashr python3 \
      mapformer/docs/audits/2026-10-15/reanalyses/solved_thresholds.py [--test]"""
import json, os, sys

import numpy as np
import torch

from mapformer.stats_core import fisher_solved

REPO = "/home/prashr/mapformer"; RUNS = f"{REPO}/runs"
THR = (0.02, 0.05, 0.1); ACC = 0.99
HERE = os.path.dirname(os.path.abspath(__file__))


def tail5(losses):
    """Final-5% mean, exactly as stats_core.classify_run (k5 = max(1, round(0.05 E)))."""
    l = np.asarray(losses, float); k5 = max(1, round(0.05 * len(l)))
    return float(l[-k5:].mean())


def ev_acc(d, T):
    return json.load(open(f"{d}/eval.json"))["eval"][str(T)]["acc"]


def js_acc(path, V, T, s):
    return dict((x[0], x[1]) for x in json.load(open(path))[f"0.0|{V}|{T}"])[s]


def pt(path):
    return torch.load(path, map_location="cpu", weights_only=False)["losses"]


# (batch, arm label, seeds, loader(seed) -> (losses, acc), registered SOLVED count from the results file)
def spec():
    S = []
    for a, k in (("Signed_r4", 8), ("Abs_r4", 0), ("Pos_r4", 1)):                       # SIGN_MATCHED_RESULTS.md
        S.append(("SIGN_MATCHED torus", a, range(8), lambda s, a=a: (pt(f"{RUNS}/sign_matched/p0/{a}_s{s}/{a}.pt"),
                                                                       js_acc(f"{REPO}/SIGN_MATCHED.json", a, 1024, s)), k))
    for p, ks, ka in ((0.5, 8, 2), (0.75, 8, 1), (0.9, 8, 4)):                           # RING_DIAL_RESULTS.md
        for a, k in (("Signed_r4", ks), ("Abs_r4", ka)):
            d = lambda s, a=a, p=p: f"{RUNS}/ring_dial/p0/{a}_p{p}_s{s}"
            S.append((f"RING_DIAL p{p}", a, range(30, 38), lambda s, a=a, d=d: (pt(f"{d(s)}/{a}.pt"), ev_acc(d(s), 128)), k))
    for e in (300, 900):                                                                  # RING_BUDGET_RESULTS.md
        d = lambda s, e=e: f"{RUNS}/ring_budget/p0/Abs_r4_p0.5_e{e}_s{s}"
        S.append(("RING_BUDGET", f"Abs_r4_e{e}", range(40, 48), lambda s, d=d: (pt(f"{d(s)}/Abs_r4.pt"), ev_acc(d(s), 128)), 3))
    for a, k in (("SO", 8), ("AO", 0), ("SO_bal", 8), ("AO_bal", 8), ("SO_cont", 8)):   # WRAP_EXISTENCE_RESULTS.md
        d = lambda s, a=a: f"{RUNS}/wrap_existence/p0/{a}_s{s}"
        S.append(("WRAP_EXISTENCE ring", a, range(600, 608), lambda s, a=a, d=d: (pt(f"{d(s)}/{a}.pt"), ev_acc(d(s), 128)), k))
    for a, k in (("SO", 8), ("SO_same", 8), ("SO_absmag", 7), ("SO_high", 8), ("AO", 0), ("AO_absmag", 0), ("AO_mid", 2),
                 ("AO_high", 8)):                                                         # MECHANISM_RESULTS.md
        d = lambda s, a=a: f"{RUNS}/wrap_mech/p0/{a}_s{s}"
        S.append(("MECHANISM ring", a, range(800, 808), lambda s, a=a, d=d: (pt(f"{d(s)}/{a}.pt"), ev_acc(d(s), 128)), k))
    for a, k in (("SO_same", 8), ("SO_same_fz", 7), ("SO_absmag", 7), ("SO_absmag_fz", 1)):  # ACCOUNTS_RESULTS.md T2
        d = lambda s, a=a: f"{RUNS}/accounts/t2/{a}_s{s}"
        S.append(("ACCOUNTS T2 ring", a, range(1000, 1008), lambda s, a=a, d=d: (pt(f"{d(s)}/{a}.pt"), ev_acc(d(s), 128)), k))
    for a, V, k in (("SO4", "SignedAO4_r4", 8), ("AO4", "AbsAO4_r4", 0), ("SO_bal", "SignedAO4_r4", 8),
                    ("AO_bal", "AbsAO4_r4", 8), ("AO_bal_hi", "AbsAO4_r4", 6)):            # GENERALITY_RESULTS.md (torus)
        S.append(("GENERALITY torus", a, range(700, 708),
                  lambda s, a=a, V=V: (pt(f"{RUNS}/generality/torus/{a}/p0/{V}_s{s}/{V}.pt"),
                                       js_acc(f"{REPO}/GENERALITY_TORUS_{a}.json", V, 1024, s)), k))
    for a, k in (("Signed_r4", 8), ("CoPEF", 0), ("CARoPEF", 0), ("RoPE", 0)):          # GENERALITY_RESULTS.md (ring)
        d = lambda s, a=a: f"{RUNS}/generality/ring/{a}_s{s}"
        S.append(("GENERALITY ring", a, range(700, 708), lambda s, a=a, d=d: (pt(f"{d(s)}/{a}.pt"), ev_acc(d(s), 128)), k))
    for a, k in (("SO4", 8), ("AO4", 1), ("AO4_high", 0), ("SO4_high", 2), ("AO4_mid", 0)):  # TORUS_START_RESULTS.md
        d = lambda s, a=a: f"{RUNS}/torus_start/p0/{a}_s{s}"
        S.append(("TORUS_START torus", a, range(1200, 1208), lambda s, a=a, d=d: (pt(f"{d(s)}/{a}.pt"), ev_acc(d(s), 1024)), k))
    d = lambda s: f"{RUNS}/wrap_cope/p0/CoPEF_s{s}"                                       # WRAP_COPE_RESULTS.md
    S.append(("WRAP_COPE ring", "CoPEF", range(900, 908), lambda s, d=d: (pt(f"{d(s)}/CoPEF.pt"), ev_acc(d(s), 128)), 1))
    for a, seeds, k in (("CoPEF_ring", range(1100, 1116), 0), ("CoPEF_line", range(1100, 1116), 1),
                        ("Signed_r4_line", range(1100, 1108), 8), ("RoPE_line", range(1100, 1108), 0)):  # ACCOUNTS T4
        d = lambda s, a=a: f"{RUNS}/accounts/t4/{a}_s{s}"
        f = a.rsplit("_", 1)[0]                                                          # CoPEF_ring -> CoPEF.pt
        S.append(("ACCOUNTS T4", a, seeds, lambda s, d=d, f=f: (pt(f"{d(s)}/{f}.pt"), ev_acc(d(s), 128)), k))
    return S


# registered contrasts the draft quotes (batch, x, ref): read as "x vs ref"
CONTRASTS = [("SIGN_MATCHED torus", "Abs_r4", "Signed_r4"), ("SIGN_MATCHED torus", "Pos_r4", "Signed_r4"),
             ("RING_DIAL p0.5", "Abs_r4", "Signed_r4"), ("RING_DIAL p0.75", "Abs_r4", "Signed_r4"),
             ("RING_DIAL p0.9", "Abs_r4", "Signed_r4"), ("WRAP_EXISTENCE ring", "AO", "SO"),
             ("MECHANISM ring", "AO", "SO"), ("MECHANISM ring", "AO_absmag", "SO_absmag"),
             ("MECHANISM ring", "AO_high", "AO_absmag"), ("MECHANISM ring", "AO_high", "AO_mid"),
             ("MECHANISM ring", "AO_high", "SO_high"), ("ACCOUNTS T2 ring", "SO_absmag_fz", "SO_absmag"),
             ("ACCOUNTS T2 ring", "SO_same_fz", "SO_same"), ("GENERALITY torus", "AO4", "SO4"),
             ("GENERALITY ring", "CoPEF", "Signed_r4"), ("TORUS_START torus", "AO4", "SO4"),
             ("TORUS_START torus", "AO4_high", "AO4"), ("ACCOUNTS T4", "CoPEF_line", "Signed_r4_line"),
             ("ACCOUNTS T4", "CoPEF_line", "CoPEF_ring")]


def counts(tails, accs):
    t = np.asarray(tails); out = {f"loss<{x}": int((t < x).sum()) for x in THR}
    out[f"acc>={ACC}"] = None if any(a is None for a in accs) else int((np.asarray(accs) >= ACC).sum())
    return out


def test():
    """Constructed cases: the tail rule on synthetic curves, and the count logic at each threshold."""
    ok = True
    c = np.r_[np.full(285, 1.0), np.full(15, 0.03)]        # E = 300 -> k5 = 15: tail = 0.03
    ok &= abs(tail5(c) - 0.03) < 1e-12
    c2 = np.r_[np.full(286, 1.0), np.full(14, 0.0)]        # one high epoch inside the last 15 -> tail = 1/15
    ok &= abs(tail5(c2) - 1 / 15) < 1e-12
    ok &= abs(tail5(np.r_[np.full(855, 1.0), np.full(45, 0.08)]) - 0.08) < 1e-12   # E = 900 -> k5 = 45
    k = counts([0.01, 0.03, 0.07, 0.2], [0.999, 0.98, 0.995, None])
    ok &= k["loss<0.02"] == 1 and k["loss<0.05"] == 2 and k["loss<0.1"] == 3 and k["acc>=0.99"] is None
    k = counts([0.01, 0.2], [0.99, 0.989])
    ok &= k["acc>=0.99"] == 1
    from mapformer.stats_core import classify_run                       # the registered rule agrees with tail5 < 0.05
    for curve in (c, c2, np.r_[np.full(855, 1.0), np.full(45, 0.049)], np.r_[np.full(855, 1.0), np.full(45, 0.05)]):
        ok &= (classify_run(curve)["registered"] == "SOLVED") == (tail5(curve) < 0.05)
    print("TEST", "PASS" if ok else "FAIL"); return ok


def main():
    if not test():
        sys.exit(1)
    if "--test" in sys.argv:
        return
    rows, R, bad = [], {}, []
    for batch, arm, seeds, f, reg in spec():
        tails, accs = [], []
        for s in seeds:
            L, a = f(s); tails.append(tail5(L)); accs.append(a)
        c = counts(tails, accs); c["n"] = len(list(seeds)); c["reg"] = reg; R[batch, arm] = c
        if c["loss<0.05"] != reg:
            bad.append((batch, arm, c["loss<0.05"], reg))
        rows.append(f"| {batch} | {arm} | {c['n']} | {c['loss<0.02']} | {c['loss<0.05']} (reg {reg}) | {c['loss<0.1']} | "
                    f"{'--' if c[f'acc>={ACC}'] is None else c[f'acc>={ACC}']} | "
                    f"{' '.join(f'{t:.3f}' for t in sorted(tails))} |")
    out = ["# R1: SOLVED counts by criterion (post hoc; PREDICTIONS.md)", "",
           f"Check at 0.05 against the registered counts: {'ALL REPRODUCE' if not bad else 'MISMATCH ' + str(bad)}", "",
           "| batch | arm | n | loss < 0.02 | loss < 0.05 | loss < 0.1 | acc >= 0.99 | final-5% losses, sorted |",
           "|---|---|---|---|---|---|---|---|"] + rows
    out += ["", "## Registered contrasts at each criterion: x k/n vs ref k/n (Fisher two-sided p)", "",
            "| batch | x vs ref | loss < 0.02 | loss < 0.05 | loss < 0.1 | acc >= 0.99 |", "|---|---|---|---|---|---|"]
    for batch, x, ref in CONTRASTS:
        cells = []
        for key in [f"loss<{t}" for t in THR] + [f"acc>={ACC}"]:
            a, b = R[batch, x], R[batch, ref]
            if a[key] is None or b[key] is None:
                cells.append("--"); continue
            cells.append(f"{a[key]}/{a['n']} vs {b[key]}/{b['n']} (p {fisher_solved(a[key], a['n'], b[key], b['n']):.4f})")
        out.append(f"| {batch} | {x} vs {ref} | " + " | ".join(cells) + " |")
    txt = "\n".join(out); print(txt); open(f"{HERE}/solved_thresholds_out.md", "w").write(txt + "\n")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()

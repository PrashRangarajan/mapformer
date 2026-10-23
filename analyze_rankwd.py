"""Registered analysis for RANKWD_PREREG.md: is weight decay necessary for shared rank 4 to learn the T = 1024 torus?

  python3 -m mapformer.analyze_rankwd               # reads runs/rankwd{,_wd0}/p0 + the two eval JSONs
  python3 -m mapformer.analyze_rankwd --selftest    # G0 / P labels, X2 REMOVAL / PARKING / UNREADABLE, norm growth and the
                                                    # wrong-root rejection, from planted checkpoints + snapshots on disk
                                                    # (DILUTED / UNRESOLVED / X2b labels are tested in analyze_rankclock)

Two arms, n = 8 fresh seeds 2100-2107, the RANK_MI / RANKCLOCK recipe: `wd` = Vanilla_r4mi at weight decay 0.05 (the
default every rank-4 result in the repo used), `wd0` = the same at weight decay 0. Shared machinery imported from
analyze_rankclock (rule 5): SOLVED via stats_core.classify_run (final-5% loss < 0.05), the per-inverse-pair clock geometry
(`geometry_sd`), the park-vs-remove readout (`x2_label`) and its snapshot ordering (`x2b_label`).

G0  wd SOLVED >= 6/8, else UNREADABLE (the default recipe must reproduce rank 4's 8/8).
P   (primary)  k_wd0 <= 2 and Fisher(wd, wd0) < .05             -> DECAY NEEDED
               k_wd0 >= k_wd - 1 and acc_wd0 >= acc_wd - 0.01     -> DECAY NOT NEEDED   (the house no-cost closeness)
               otherwise                                          -> PARTIAL
S1  (secondary) the park-vs-remove readout X2 on solved wd0 runs (needs >= 4, else X2 UNREADABLE) and the P4 ordering X2b,
    both exactly as registered for RANKCLOCK; printed for the wd arm as the reference.
Descriptive: accuracy at T = 512 / 1024 / 2048 (eval_noise_refine) with the unpaired permutation p AND the paired sign-flip
p beside P (the arms share init, data and evaluation trajectories at a seed; Amendment 1 finding 5); run classes; norm growth
end / init of |W_in|, |W_out|, |token_emb| and omega (snapshots) and of every LayerNorm / out_norm gain (checkpoint vs its
init of ones) -- AdamW here decays EVERY parameter, omega and the gains included (Amendment 1 finding 1); geometry medians
over ALL runs per arm; r(log10 final loss, acc) per length (rule 2).
S1 caveat (Amendment 1 finding 2): if P reads DECAY NEEDED (k_wd0 <= 2), X2 is UNREADABLE by construction and X2b cannot
reach p < .05; S1 is readable only when P misses, and then on a selected subset. The wd reference is NOT diagnostic of
removal: passive decay alone shrinks an unheld parameter to `passive_decay_factor()` of its size (~0.09), below the 0.2 bar.
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

from mapformer import analyze_rankclock as RC
from mapformer.stats_core import fisher_solved, perm2_p, signflip_p

REPO = RC.REPO
ROOT, ROOT_WD0 = f"{REPO}/runs/rankwd", f"{REPO}/runs/rankwd_wd0"
EVAL, EVAL_WD0 = f"{REPO}/RANKWD.json", f"{REPO}/RANKWD_WD0.json"
SEEDS = list(range(2100, 2108))
# RC's arm keys for the two roots: "r4" -> ROOT layout, "r4wd0" -> ROOT_WD0 layout (same variant, Vanilla_r4mi)
MAP = {"wd": "r4", "wd0": "r4wd0"}


def _norms(sd):
    return {k: float(sd[k].double().norm()) for k in
            ("action_to_lie.w_in.weight", "action_to_lie.w_out.weight", "token_emb.weight", "path_integrator.omega")}


_NORM_NAME = {"action_to_lie.w_in.weight": "w_in", "action_to_lie.w_out.weight": "w_out",
              "token_emb.weight": "token_emb", "path_integrator.omega": "omega"}


def passive_decay_factor(wd=0.05, lr=1e-3, epochs=900, n_batches=98):
    """exp(-wd * sum of the lr schedule): what AdamW's decoupled decay alone does to a parameter the gradient does not hold,
    over train()'s cosine schedule (5% linear warm-up, then cosine from 1 to 0.1), computed from the schedule itself."""
    total = epochs * n_batches
    warm = max(1, int(0.05 * total))
    s = 0.0
    for step in range(total):
        if step < warm:
            f = (step + 1) / warm
        else:
            q = (step - warm) / max(1, total - warm)
            f = 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * min(q, 1.0)))
        s += lr * f
    return float(np.exp(-wd * s))


def load(root=None, root_wd0=None, eval_json=None, eval_wd0=None, seeds=SEEDS):
    """RANKCLOCK's loader restricted to the two r4 arms (same recipe asserts, layout, snapshots, eval-row checks)."""
    saved = dict(RC.ARMS)
    try:
        RC.ARMS.clear()
        RC.ARMS.update({"r4": "Vanilla_r4mi", "r4wd0": "Vanilla_r4mi"})
        R, missing = RC.load(root or ROOT, root_wd0 or ROOT_WD0, eval_json or EVAL, eval_wd0 or EVAL_WD0, seeds)
    finally:
        RC.ARMS.clear()
        RC.ARMS.update(saved)
    out = {}
    for (arm, s), rec in R.items():
        d = RC.ckpt_dir(arm, s, root or ROOT, root_wd0 or ROOT_WD0)
        n0 = _norms(torch.load(os.path.join(d, "snaps", "e0000.pt")))
        n1 = _norms(torch.load(os.path.join(d, "snaps", "e0900.pt")))
        rec["norm_growth"] = {_NORM_NAME[k]: n1[k] / max(n0[k], 1e-12) for k in n0}
        ck = torch.load(os.path.join(d, "Vanilla_r4mi.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
        gains = [v.double() for k, v in ck.items() if "norm" in k and k.endswith(".weight")]
        rec["norm_growth"]["ln_gains"] = float(np.median([float(g.norm() / np.sqrt(g.numel())) for g in gains]))
        out[("wd" if arm == "r4" else "wd0", s)] = rec
    return out, missing


def p_label(k_wd, k_wd0, acc_wd, acc_wd0):
    if k_wd < 6:
        return f"UNREADABLE (wd SOLVED {k_wd}/8 < 6)", None
    p = fisher_solved(k_wd0, 8, k_wd, 8)
    if k_wd0 <= 2 and p < .05:
        return "DECAY NEEDED", p
    if k_wd0 >= k_wd - 1 and np.mean(acc_wd0) >= np.mean(acc_wd) - 0.01:
        return "DECAY NOT NEEDED", p
    return f"PARTIAL (wd0 {k_wd0}/8 vs wd {k_wd}/8, Fisher p {p:.4f})", p


def analyse(R, seeds=SEEDS, out=print):
    k = {a: sum(R[(a, s)]["solved"] for s in seeds) for a in MAP}
    acc = {a: [R[(a, s)]["acc1024"] for s in seeds] for a in MAP}
    lab, p = p_label(k["wd"], k["wd0"], acc["wd"], acc["wd0"])
    pa = perm2_p(np.asarray(acc["wd0"]), np.asarray(acc["wd"]))["p"]
    psf = signflip_p(np.asarray(acc["wd"]) - np.asarray(acc["wd0"]))["p"]   # exact at n = 8 (min p 0.0078)
    v = {"solved": k, "P": lab, "reading": lab}
    out("== RANKWD: shared rank 4, weight decay 0.05 vs 0 (SOLVED = final-5% loss < 0.05; S/T/D/R/X) ==")
    for a in MAP:
        out(f"  {a:4s} SOLVED {k[a]}/8 | acc@512/1024/2048 "
            + " / ".join(f"{np.mean([R[(a, s)][f'acc{T}'] for s in seeds]):.4f}" for T in RC.LENGTHS) + " | "
            + " ".join(f"s{s}:{RC.CODE[R[(a, s)]['cls']]}({R[(a, s)]['final_loss']:.3f})" for s in seeds))
    out(f"\nP (primary): {lab}" + (f" -- Fisher p {p:.4f}; acc@1024 {np.mean(acc['wd0']):.4f} vs "
                                   f"{np.mean(acc['wd']):.4f}, perm p {pa:.4f}, paired sign-flip p {psf:.4f}"
                                   if p is not None else ""))
    v["X2"] = RC.x2_label([R[("wd0", s)] for s in seeds])
    v["X2b_wd0"] = RC.x2b_label([R[("wd0", s)] for s in seeds])
    v["X2b_wd"] = RC.x2b_label([R[("wd", s)] for s in seeds])
    out(f"S1 X2 (park vs remove, solved wd0 runs): {v['X2']}")
    out(f"   X2b ordering: wd0 {v['X2b_wd0']} | wd {v['X2b_wd']}")
    ref = [R[("wd", s)] for s in seeds if R[("wd", s)]["solved"]]
    if ref:
        out(f"   reference wd (0.05; NON-DIAGNOSTIC of removal: passive decay alone gives {passive_decay_factor():.3f}), "
            f"{len(ref)} solved: median shrink "
            f"{np.median([r['u_norm'] / max(r['g_init']['u_norm'], 1e-12) for r in ref]):.3f}, readgain/init "
            f"{np.median([r['readgain'] / max(r['g_init']['readgain'], 1e-12) for r in ref]):.3f}")
    out(f"\nREADING (registered primary): {lab}")
    out("   scope: torus T = 1024, 900 epochs, shared rank 4, AdamW, lr 1e-3 cosine; one host.")
    out("\n== geometry medians over ALL runs per arm (descriptive) ==")
    for a in MAP:
        rows = [R[(a, s)] for s in seeds]
        out(f"  {a:4s} u_ratio {np.median([r['u_ratio'] for r in rows]):.3f}  shrink "
            f"{np.median([r['u_norm'] / max(r['g_init']['u_norm'], 1e-12) for r in rows]):.3f}  readgain "
            f"{np.median([r['readgain'] for r in rows]):.3f}")
    out("\n== norm growth end / init (medians; descriptive; ln_gains = RMS gain, init 1) ==")
    for a in MAP:
        g = [R[(a, s)]["norm_growth"] for s in seeds]
        out(f"  {a:4s} " + "  ".join(f"{kk} {np.median([x[kk] for x in g]):.3f}" for kk in g[0]))
    out("\n== rule 2: r(log10 final loss, acc) per length ==")
    fin = [r for r in R.values() if np.isfinite(r["final_loss"])]
    Ls = np.log10(np.array([r["final_loss"] for r in fin]) + 1e-6)
    for T in RC.LENGTHS:
        A = np.array([r[f"acc{T}"] for r in fin])
        rr = float(np.corrcoef(Ls, A)[0, 1]) if len(A) > 2 and A.std() > 0 and Ls.std() > 0 else float("nan")
        out(f"   T={T}: r = {rr:+.3f} over {len(A)} runs")
    out("\nper seed acc@1024 (class):")
    for a in MAP:
        out(f"  {a:4s} " + " ".join(f"{R[(a, s)]['acc1024']:.3f}{RC.CODE[R[(a, s)]['cls']]}" for s in seeds))
    return v


def selftest():
    silent = lambda *a, **k: None
    saved_seeds = list(RC.SEEDS)
    RC.SEEDS[:] = SEEDS                         # RC._plant / _write_eval iterate RC.SEEDS
    ok = True
    tmp = tempfile.mkdtemp(prefix="rankwd_selftest_")
    try:
        cases = [("P", "DECAY NEEDED", 8, 1, None), ("P", "DECAY NOT NEEDED", 8, 8, None),
                 ("P", "PARTIAL", 8, 7, None),        # 7/8 but mean acc 0.9875 < 1.000 - 0.01: closeness fails
                 ("P", "PARTIAL", 8, 4, None),
                 ("P", "PARTIAL", 7, 3, None),        # 7 vs 3: Fisher p 0.12 -> neither
                 ("P", "UNREADABLE", 5, 1, None),
                 ("X2", "PARKING", 8, 8, [(0, 1.0, True), (100, 1.0, False), (900, 0.9, False)]),
                 ("X2", "REMOVAL", 8, 8, [(0, 1.0, True), (450, 0.5, True), (900, 0.05, True)]),
                 ("X2", "X2 UNREADABLE", 8, 2, [(0, 1.0, True), (900, 0.9, False)])]
        for i, (field, want, kw, kw0, geo) in enumerate(cases):
            root, root0 = os.path.join(tmp, f"c{i}"), os.path.join(tmp, f"c{i}w")
            for j, s in enumerate(SEEDS):
                RC._plant("r4", s, j < kw, root, root0)
                RC._plant("r4wd0", s, j < kw0, root, root0, geo)
            ej, ew = os.path.join(tmp, f"e{i}.json"), os.path.join(tmp, f"e{i}w.json")
            saved = dict(RC.ARMS)
            RC.ARMS.clear(); RC.ARMS.update({"r4": "Vanilla_r4mi", "r4wd0": "Vanilla_r4mi"})
            try:
                RC._write_eval(ej, {"r4": kw}, ["r4"]); RC._write_eval(ew, {"r4wd0": kw0}, ["r4wd0"])
            finally:
                RC.ARMS.clear(); RC.ARMS.update(saved)
            R, missing = load(root, root0, ej, ew)
            assert not missing, missing[:2]
            v = analyse(R, out=silent)
            got = v[field]
            good = got.startswith(want)
            ok &= good
            print(f"  {'ok ' if good else 'BAD'} {field:3s} want {want!r:20s} (wd {kw}/8, wd0 {kw0}/8) got {got[:80]!r}")
        # (Amendment 1 finding 4) the closeness clause of DECAY NOT NEEDED: wd0 7/8 with its unsolved seed at acc 0.995
        import json as _json
        root, root0 = os.path.join(tmp, "close"), os.path.join(tmp, "closew")
        for j, s in enumerate(SEEDS):
            RC._plant("r4", s, True, root, root0)
            RC._plant("r4wd0", s, j < 7, root, root0)
        ej, ew = os.path.join(tmp, "close.json"), os.path.join(tmp, "closew.json")
        for path, k_, lo in ((ej, 8, 0.9), (ew, 7, 0.995)):
            _json.dump({f"0.0|Vanilla_r4mi|{T}": [[s, 1.0 if i < k_ else lo, 0.0] for i, s in enumerate(SEEDS)]
                        for T in RC.LENGTHS}, open(path, "w"))
        R, _ = load(root, root0, ej, ew)
        got = analyse(R, out=silent)["P"]
        good = got == "DECAY NOT NEEDED"
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} P   want 'DECAY NOT NEEDED' via closeness (wd 8/8, wd0 7/8 @0.995) got {got!r}")
        # (finding 6) norm growth: the analyzer's w_in / omega ratios must equal ratios computed here from the planted files
        root, root0 = os.path.join(tmp, "ng"), os.path.join(tmp, "ngw")
        for s in SEEDS:
            RC._plant("r4", s, True, root, root0)
            RC._plant("r4wd0", s, True, root, root0, [(0, 1.0, True), (450, 0.5, True), (900, 0.05, True)])
        ej, ew = os.path.join(tmp, "ng.json"), os.path.join(tmp, "ngw.json")
        for path, a_ in ((ej, "r4"), (ew, "r4wd0")):
            _json.dump({f"0.0|Vanilla_r4mi|{T}": [[s, 1.0, 0.0] for s in SEEDS] for T in RC.LENGTHS}, open(path, "w"))
        R, _ = load(root, root0, ej, ew)
        d = RC.ckpt_dir("r4wd0", SEEDS[0], root, root0)
        a0, a1 = torch.load(os.path.join(d, "snaps", "e0000.pt")), torch.load(os.path.join(d, "snaps", "e0900.pt"))
        want_win = float(a1["action_to_lie.w_in.weight"].double().norm() / a0["action_to_lie.w_in.weight"].double().norm())
        got_win = R[("wd0", SEEDS[0])]["norm_growth"]["w_in"]
        good = abs(got_win - want_win) < 1e-9 and abs(want_win - 1.0) > 1e-3 and \
            abs(R[("wd0", SEEDS[0])]["norm_growth"]["omega"] - 1.0) < 1e-9
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} norm growth w_in {got_win:.4f} == hand-computed {want_win:.4f}; omega unchanged 1.0")
        # (finding 6) a weight-decay-0 checkpoint sitting in the wd root must be rejected, not read as the wd arm
        root, root0 = os.path.join(tmp, "bad"), os.path.join(tmp, "badw")
        for s in SEEDS:
            RC._plant("r4wd0", s, True, os.path.join(tmp, "unused"), root)     # wd-0 args, written into the WD root
            RC._plant("r4wd0", s, True, root, root0)
        try:
            load(root, root0, ej, ew)
            print("  BAD a weight-decay-0 checkpoint in the wd root was accepted"); ok = False
        except AssertionError as e:
            print(f"  ok  wrong weight decay in the wd root raises ({str(e)[:50]}...)")
    finally:
        RC.SEEDS[:] = saved_seeds
        shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    R, missing = load()
    if missing:
        print(f"MISSING {len(missing)} checkpoints, e.g. {missing[:3]}")
        sys.exit(2)
    v = analyse(R)
    import json
    json.dump(v, open(f"{REPO}/RANKWD_VERDICTS.json", "w"), indent=2)


if __name__ == "__main__":
    main()

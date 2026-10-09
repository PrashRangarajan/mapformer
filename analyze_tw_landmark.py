"""Registered verdicts for TW_LANDMARK_PREREG.md, from TW_LANDMARK.json (tw_landmark_eval.py over every run).

Cells (arm, layers, training name rate): P2 = MapWM 2 layers at r in {0, 0.5, 1}; R2 = RoPE 2 layers at r in {0, 0.5, 1};
P1 = MapWM 1 layer at r in {0, 1}. Run keys "<arm>_L<layers>_r<rate>_s<seed>".

  python3 -m mapformer.analyze_tw_landmark [--json TW_LANDMARK.json] [--rescored TW_LANDMARK_RESCORED.json]
"""
import argparse
import json
import sys

import numpy as np

from scipy import stats

from mapformer.stats_core import perm2_p, perm2_ci, fisher_solved, signflip_p

REPO = "/home/prashr/mapformer"
SEEDS = [50, 51, 52, 53, 54, 55]
CELLS = {"P2": ("MapWM", 2, (0.0, 0.5, 1.0)), "R2": ("RoPE", 2, (0.0, 0.5, 1.0)), "P1": ("MapWM", 1, (0.0, 1.0))}
O_MIN, ACC_MIN, GATE_REL, NAME_USE, EQUIV, CEIL = 0.05, 0.02, 0.2, 0.05, 0.10, 0.98
# Amendment 1 (N1): the O branches that carry the P1 attribution qualifier, by name
ATTRIB = ("NAMES OVERSHADOW THE MAP", "MAP LOST, NAMES NOT USED EITHER", "NAMES OVERSHADOW WHEN PRESENT",
          "MAP LOST ONLY WHEN NAMES ARE STRIPPED")
OOD_O1 = (" [OUT-OF-DISTRIBUTION PROBES ONLY: at r=1 every trained revisit had a matching name, so fresh or absent names "
          "are untrained situations and a misfiring name route can produce this with the path route intact; r=1 has no "
          "in-distribution path probe -- NOT readable as map loss] [training-signal confound: at r=1 a 1024-word "
          "sequence holds 104.7 vs 131.8 moves and ~23.0 vs ~30.6 revisit targets, 25% less path supervision; P1 "
          "controls it only partly]")


def mde2(sd, n, alpha=0.05, power=0.80):
    """Two-sample MDE with two-sample df (Amendment 1, N3): (t_{1-a/2, 2n-2} + t_{power, 2n-2}) * sd * sqrt(2/n)."""
    df = 2 * n - 2
    return float((stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * sd * np.sqrt(2.0 / n))


def key(cell, r, s):
    arm, L, _ = CELLS[cell]
    return f"{arm}_L{L}_r{r}_s{s}"


def check_void(res, seeds):
    miss = [key(c, r, s) for c, (_a, _l, rs) in CELLS.items() for r in rs for s in seeds if key(c, r, s) not in res]
    if miss:
        return f"VOID: {len(miss)} runs missing ({', '.join(miss[:6])}{' ...' if len(miss) > 6 else ''})"
    nt = {res[key(c, r, s)]["n_targets"] for c, (_a, _l, rs) in CELLS.items() for r in rs for s in seeds}
    if len(nt) != 1:
        return f"VOID: scored target sets differ across runs ({sorted(nt)})"
    for c, (arm, _l, rs) in CELLS.items():
        for r in rs:
            for s in seeds:
                d = res[key(c, r, s)]
                if arm == "MapWM" and not ("rel_uninf" in d and "rel_own_u05" in d):
                    return "VOID: reliance missing on a path run"
                if arm != "MapWM" and "rel_uninf" in d:
                    return "VOID: reliance present on an index run"
    return None


def contrast(x, y, thr):
    """mean(y) - mean(x); exact permutation p; fires (p < .05 and |d| >= thr); 95% permutation CI."""
    d = float(np.mean(y) - np.mean(x)); p = perm2_p(x, y)["p"]
    ci = perm2_ci(x, y)
    return {"d": d, "p": p, "fire": bool(p < 0.05 and abs(d) >= thr), "lo": ci["lo"], "hi": ci["hi"]}


def fmt(c):
    lo = "nan" if c["lo"] is None else f"{c['lo']:+.4f}"; hi = "nan" if c["hi"] is None else f"{c['hi']:+.4f}"
    return f"d {c['d']:+.4f} (perm p {c['p']:.4f}, 95% CI [{lo}, {hi}]){' FIRES' if c['fire'] else ''}"


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


def verdict_O(g, rstar, out):
    """Primary O at training rate rstar vs 0 (P2). Returns the label."""
    x_rel, y_rel = g("P2", 0.0, "rel_uninf"), g("P2", rstar, "rel_uninf")
    x_s, y_s = g("P2", 0.0, "acc_strip"), g("P2", rstar, "acc_strip")
    cr, cs = contrast(x_rel, y_rel, O_MIN), contrast(x_s, y_s, O_MIN)
    nb = float(np.median(g("P2", rstar, "name_benefit")))
    out(f"  reliance (names uninformative) P2(r={rstar}) - P2(r=0): {fmt(cr)}")
    out(f"  names-stripped accuracy        P2(r={rstar}) - P2(r=0): {fmt(cs)}")
    out(f"  name use of P2(r={rstar}): median name benefit (acc names informative - uninformative) {nb:+.4f}"
        f" (names used if >= {NAME_USE})")
    med0 = float(np.median(x_rel))
    if med0 < GATE_REL:
        return (f"O UNMEASURED: P2 trained without names does not path-integrate (median reliance {med0:.4f} < "
                f"{GATE_REL}; {sum(c == 'SOLVED' for c in g('P2', 0.0, 'cls'))}/{len(x_rel)} SOLVED)")
    rd, ru = cr["fire"] and cr["d"] < 0, cr["fire"] and cr["d"] > 0
    sd, su = cs["fire"] and cs["d"] < 0, cs["fire"] and cs["d"] > 0
    if (rd and su) or (ru and sd):
        return f"MIXED (reliance {cr['d']:+.4f}, stripped accuracy {cs['d']:+.4f}; reported as it falls)"
    if rd and sd:
        return ("NAMES OVERSHADOW THE MAP" if nb >= NAME_USE else
                "MAP LOST, NAMES NOT USED EITHER (a learning failure, not cue competition)")
    if rd:
        return "NAMES OVERSHADOW WHEN PRESENT (theta used less when names are in the text; map intact when stripped)"
    if sd:
        return "MAP LOST ONLY WHEN NAMES ARE STRIPPED (form shift; theta still used when names are present)"
    dom = float(np.median(np.array(g("P2", rstar, "conf_name")) - np.array(g("P2", rstar, "conf_path"))))
    q = (f"; cue conflict: names dominate (median name - path share {dom:+.3f})" if dom > 0 else
         f"; cue conflict: the path dominates (median name - path share {dom:+.3f})")
    if dom <= -0.8:
        q += " -- names essentially unused by the path model: no cue competition took place"
    if ru or su:
        return "NAMES STRENGTHEN THE MAP" + q
    ok = all(c["lo"] is not None and c["lo"] >= -EQUIV for c in (cr, cs))
    if ok:
        ceil = min(np.median(x_s), np.median(y_s)) >= CEIL
        return "PATH INTEGRATION PERSISTS" + (" (both cells at ceiling)" if ceil else "") + \
            f" (95% CIs above -{EQUIV})" + q
    return f"UNMEASURED (no shift detected; a 95% CI reaches below -{EQUIV})" + q


def _attrib_label(x, y):
    c = contrast(x, y, O_MIN)
    if np.median(x) < GATE_REL:
        return c, "P1 qualifier unmeasured (P1 trained without names does not path-integrate)"
    if c["fire"] and c["d"] < 0:
        return c, "names ALSO lower 1-layer path integration: not cue competition alone"
    return c, ("the 1-layer path model, which cannot read names, keeps its map: names in the stream do not by "
               "themselves block step learning")


def attribution(g, out, g2=None):
    """P1 reliance r=1 - r=0. Amendment 1 (D3): the label is read from the dropout re-scored readouts when given (the
    x1/(1-p) correction is valid for one-layer models); the eval-mode reading is printed beside it."""
    ce, le = _attrib_label(g("P1", 0.0, "rel_uninf"), g("P1", 1.0, "rel_uninf"))
    out(f"  attribution control, P1 (1 layer: cannot read names) reliance r=1 - r=0, eval mode: {fmt(ce)} -> {le}")
    if g2 is None:
        return le + " (eval mode; no re-scored readouts given)"
    cr, lr = _attrib_label(g2("P1", 0.0, "rel_uninf"), g2("P1", 1.0, "rel_uninf"))
    out(f"  attribution control, re-scored (adopted): {fmt(cr)} -> {lr}")
    return lr + (" (re-scored; eval mode agrees)" if lr == le else f" (re-scored; eval mode reads: {le})")


def verdict_OID(g, out):
    """Amendment 1 (D1), registered: the in-distribution probe. P2(r=0.5) - P2(r=0) on rel_own_u05, the reliance in
    each run's OWN rendering on revisits to cells that are not landmarks at rate 0.5 (the same targets in both cells;
    both renderings are in distribution there: unnamed arrivals occur in training at both rates)."""
    x, y = g("P2", 0.0, "rel_own_u05"), g("P2", 0.5, "rel_own_u05")
    c = contrast(x, y, O_MIN)
    ca = contrast(g("P2", 0.0, "acc_own_u05"), g("P2", 0.5, "acc_own_u05"), O_MIN)
    out(f"  reliance, own rendering, cells unnamed at rate 0.5: P2(r=0.5) - P2(r=0): {fmt(c)}")
    out(f"  companion (no verdict): accuracy on the same targets: {fmt(ca)}")
    med0 = float(np.median(x))
    if med0 < GATE_REL:
        return f"O-ID UNMEASURED: P2 trained without names does not path-integrate (median reliance {med0:.4f})"
    if c["fire"]:
        return ("NAMES REDUCE PATH INTEGRATION AT UNNAMED PLACES (in distribution)" if c["d"] < 0 else
                "NAMES INCREASE PATH INTEGRATION AT UNNAMED PLACES (in distribution)")
    if c["lo"] is not None and c["lo"] >= -EQUIV:
        return f"PATH INTEGRATION AT UNNAMED PLACES PERSISTS (in distribution; 95% CI above -{EQUIV})"
    return f"O-ID UNMEASURED (no shift detected; the 95% CI reaches below -{EQUIV})"


def verdict_I(g, r, field, out, tag):
    x, y = g("R2", r, field), g("P2", r, field)
    c = contrast(x, y, ACC_MIN)
    sx, sy = sum(v == "SOLVED" for v in g("R2", r, "cls")), sum(v == "SOLVED" for v in g("P2", r, "cls"))
    n = len(x); pf = fisher_solved(sx, n, sy, n)
    sd = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2) if n > 1 else 0.0
    m_ = max(mde2(sd, n), ACC_MIN)
    out(f"  [{tag}] P2 - R2 on {field}: {fmt(c)}; SOLVED P2 {sy}/{n} vs R2 {sx}/{n} (Fisher p {pf:.4f}); "
        f"MDE {m_:.4f}; medians P2 {np.median(y):.4f} R2 {np.median(x):.4f}")
    ceil = " (BOTH AT CEILING)" if min(np.median(x), np.median(y)) >= CEIL else ""
    if c["fire"]:
        lab = ("PATH WINS" if c["d"] > 0 else "INDEX WINS") + ceil
    elif pf < 0.05:
        lab = f"SOLVED RATE {'HIGHER' if sy > sx else 'LOWER'} FOR PATH (convergence; accuracy unmeasured below {m_:.4f})"
    else:
        lab = f"NO DETECTABLE DIFFERENCE{ceil} (unmeasured below {m_:.4f})"
    return lab, c["p"]


def mc_secondary(g, out):
    """Amendment 1 (D3): MC-dropout accuracies (plain pass only)."""
    out("  MC-dropout accuracy (train mode, 3 dropout seeds; valid at any depth; Amendment 1 D3), own / strip:")
    for cell, (arm, L, rs) in CELLS.items():
        for r in rs:
            out(f"    {cell} r={r}: {np.mean(g(cell, r, 'acc_own_mc')):.4f} / {np.mean(g(cell, r, 'acc_strip_mc')):.4f} "
                f"(eval mode {np.mean(g(cell, r, 'acc_own')):.4f} / {np.mean(g(cell, r, 'acc_strip')):.4f})")
    for r in (0.0, 0.5, 1.0):
        c = contrast(g("R2", r, "acc_own_mc"), g("P2", r, "acc_own_mc"), ACC_MIN)
        out(f"    P2 - R2 at r={r} on MC-dropout accuracy: {fmt(c)}")
    c = contrast(g("P2", 0.0, "acc_strip_mc"), g("P2", 1.0, "acc_strip_mc"), O_MIN)
    out(f"    P2 names-stripped MC-dropout accuracy r=1 - r=0: {fmt(c)}")


def analyse(res, seeds, out=print, res2=None):
    v = check_void(res, seeds)
    if v:
        out(v); return {"void": v}
    g = lambda cell, r, f: [res[key(cell, r, s)][f] for s in seeds]
    g2 = (lambda cell, r, f: [res2[key(cell, r, s)][f] for s in seeds]) if res2 is not None else None
    n = len(seeds)
    out(f"== per cell (n={n}): acc own / strip / uninf / named | reliance strip / uninf / named | name benefit | "
        "conflict path / name | SOLVED | final loss ==")
    for cell, (arm, L, rs) in CELLS.items():
        for r in rs:
            m = lambda f: float(np.nanmean(g(cell, r, f))) if f in res[key(cell, r, seeds[0])] else float("nan")
            out(f"  {cell} r={r}: {m('acc_own'):.4f} / {m('acc_strip'):.4f} / {m('acc_uninf'):.4f} / {m('acc_named'):.4f}"
                f" | {m('rel_strip'):+.4f} / {m('rel_uninf'):+.4f} / {m('rel_named'):+.4f} | {m('name_benefit'):+.4f}"
                f" | {m('conf_path'):.3f} / {m('conf_name'):.3f} | {sum(c == 'SOLVED' for c in g(cell, r, 'cls'))}/{n}"
                f" | {m('final_loss'):.4f}")
    V = {}
    out("\n== PRIMARY O-ID (Amendment 1): the in-distribution probe, r=0.5 vs r=0 (P2; fires: perm p < .05 and "
        f"|d| >= {O_MIN}) ==")
    V["O-ID"] = verdict_OID(g, out)
    out(f"  REGISTERED O-ID: {V['O-ID']}")
    out("\n== PRIMARY O: do names overshadow the map? OUT-OF-DISTRIBUTION probes at r>0 (names fresh / absent at test)"
        f" (P2, MapWM 2 layers; fires: perm p < .05 and |d| >= {O_MIN}; gate: median reliance of P2(r=0) >= "
        f"{GATE_REL}) ==")
    for rstar in (1.0, 0.5):
        out(f" O{'1' if rstar == 1.0 else '2'}: r={rstar} vs r=0")
        lab = verdict_O(g, rstar, out)
        if lab.startswith(ATTRIB):
            if rstar == 1.0:
                lab += OOD_O1 + "; " + attribution(g, out, g2)
            else:
                conf = V["O-ID"].startswith("NAMES REDUCE")
                lab += (" [OUT-OF-DISTRIBUTION probes; CONFIRMED IN DISTRIBUTION by O-ID]" if conf else
                        " [OUT-OF-DISTRIBUTION PROBES ONLY; NOT CONFIRMED IN DISTRIBUTION by O-ID -- not readable "
                        "as map loss]")
        V[f"O{'1' if rstar == 1.0 else '2'}"] = lab
        out(f"  REGISTERED O{'1' if rstar == 1.0 else '2'}: {lab}")
    out(f"\n== PRIMARY I: path (P2) vs index (R2) at matched depth (fires: perm p < .05 and |d| >= {ACC_MIN}; "
        "Fisher-only = convergence) ==")
    labs, ps = {}, []
    for tag, r, f in (("I0", 0.0, "acc_own"), ("I1", 1.0, "acc_own"), ("I05", 0.5, "acc_own"),
                      ("I05u", 0.5, "acc_own_unnamed")):
        labs[tag], p = verdict_I(g, r, f, out, tag); ps.append(p)
    adj = holm(np.array(ps))
    for (tag, lab), pa in zip(labs.items(), adj):
        fired = lab.startswith(("PATH WINS", "INDEX WINS"))
        V[tag] = lab + (" (does not survive Holm over the 4 I tests)" if fired and pa >= 0.05 else "")
        out(f"  REGISTERED {tag}: {V[tag]}  [Holm p {pa:.4f}]")
    w0, w1 = V["I0"].startswith("PATH WINS"), V["I1"].startswith("PATH WINS")
    comp = ("NAMES CLOSE THE PATH ADVANTAGE" if w0 and not w1 else "PATH ADVANTAGE SURVIVES NAMES" if w0 and w1
            else "NO COMPOSITE (I0 is not PATH WINS)")
    V["composite"] = comp
    out(f"  COMPOSITE (from I0 and I1, no new test; a PATH WINS that does not survive Holm counts as a win): {comp}")

    out("\n== declared secondaries (no verdict) ==")
    for r in (0.0, 0.5, 1.0):
        nb = g("R2", r, "name_benefit")
        out(f"  R2 r={r}: name benefit {np.mean(nb):+.4f}; acc own landmark / unnamed "
            f"{np.nanmean(g('R2', r, 'acc_own_land')):.4f} / {np.nanmean(g('R2', r, 'acc_own_unnamed')):.4f}; "
            f"P2 {np.nanmean(g('P2', r, 'acc_own_land')):.4f} / {np.nanmean(g('P2', r, 'acc_own_unnamed')):.4f}")
    for rstar in (0.5, 1.0):
        d = np.array(g("P2", rstar, "rel_uninf")) - np.array(g("P2", 0.0, "rel_uninf"))
        out(f"  paired by seed (same init, same walks) P2 reliance r={rstar} - r=0: {d.mean():+.4f}, sign-flip p "
            f"{signflip_p(d)['p']:.4f}")
    for cell, rs in (("P2", (0.0, 0.5, 1.0)), ("P1", (0.0, 1.0))):
        for r in rs:
            out(f"  {cell} r={r} step table: move {np.mean(g(cell, r, 'move')):.3f}, opposition minus common "
                f"{np.mean(g(cell, r, 'opp_minus_common')):.3f}, name step {np.mean(g(cell, r, 'name_step')):.3f} "
                f"(identity part {np.mean(g(cell, r, 'name_identity_step')):.3f}) of a direction step; drift "
                f"{np.mean(g(cell, r, 'drift')):.1f}/64 ({' '.join(str(x) for x in g(cell, r, 'drift'))}); "
                f"rel_all(uninf) {np.mean(g(cell, r, 'rel_all_uninf')):+.4f}; rel_own {np.mean(g(cell, r, 'rel_own')):+.4f}; "
                f"mark step {np.mean(g(cell, r, 'mark_step')):.3f}; common / direction {np.mean(g(cell, r, 'common_over_dir')):.3f}")
    for rstar in (1.0,):
        c = contrast(g("R2", rstar, "conf_name"), g("P2", rstar, "conf_name"), 0)
        out(f"  cue conflict at r={rstar}: share following the NAME, P2 {np.mean(g('P2', rstar, 'conf_name')):.3f} vs "
            f"R2 {np.mean(g('R2', rstar, 'conf_name')):.3f} ({fmt(c)}); following the PATH, P2 "
            f"{np.mean(g('P2', rstar, 'conf_path')):.3f}")
    if "acc_own_mc" not in res[key("P2", 0.0, seeds[0])]:
        out("  MC-dropout accuracy: not in this pass (the re-scored pass runs with --no-mc)")
    else:
        mc_secondary(g, out)
    x = sum((g(c, r, "final_loss") for c, (_a, _l, rs) in CELLS.items() for r in rs), [])
    y = sum((g(c, r, "acc_own") for c, (_a, _l, rs) in CELLS.items() for r in rs), [])
    out(f"  r(final loss, acc own) over {len(x)} runs: {np.corrcoef(x, y)[0, 1]:+.3f}")
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=f"{REPO}/TW_LANDMARK.json")
    ap.add_argument("--rescored", default=None)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--verdicts-out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    res = json.load(open(a.json))
    res2 = json.load(open(a.rescored)) if a.rescored else None
    V = analyse(res, seeds, res2=res2)
    if a.verdicts_out:
        json.dump(V, open(a.verdicts_out, "w"), indent=1)
    if a.rescored:
        print("\n== declared secondary: dropout re-score (x 1/(1-p) at eval, rescore_hook). A ONE-LAYER correction: in "
              "2-layer models it compounds and is not a better estimate (DROPOUT_RESCORE.md), so only flips are "
              "flagged ==")
        lines = []
        V2 = analyse(res2, seeds, out=lines.append, res2=res2)
        print("\n".join("  " + l for l in lines if l.lstrip().startswith(("REGISTERED", "COMPOSITE", "P1 r=", "P2 r=", "R2 r="))
                        and "step table" not in l))
        for k in V:
            if V2.get(k) != V[k]:
                print(f"  FLAG: {k} re-scored reads '{V2.get(k)}' (registered: '{V[k]}')")


if __name__ == "__main__":
    main()

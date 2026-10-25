"""Registered analysis of the image-observation torus (IMAGE_TORUS_PREREG.md).

  python3 -m mapformer.analyze_image_torus --blocks A            # the batch (blocks as declared before launch)
  python3 -m mapformer.analyze_image_torus --selftest            # synthetic batches through every label and void

Reads runs/image_torus/<cell>/<cell>_s<seed>/eval_image.json (eval_image_torus.py) and IMAGE_TORUS_GATE.json
(gate_image_torus.py: floors and perception ceilings on the same evaluation walks). Writes the report to stdout and
IMAGE_TORUS_VERDICTS.json.

REGISTERED (accuracy on revisit targets, held-out map, test-split images, T = 1024; n = 8 vs 8; exact two-sided
permutation test, stats_core.perm2_p, C(16, 8) = 12870 relabellings; Holm over P1 and P2 at family alpha 0.05).
AMENDMENT 2 (redesign after pilot 1's stop rule): P1's path arm is I_ActOnly (image MapWM at the init-matched gain never
learned in two pilots); P2 is unchanged in form but PILOT-INFORMED (not blind), confirmed on the fresh seeds 3100-3107.
  P1  I_ActOnly - I_RoPE     does path integration survive pixel observations at one layer?
      PATH WINS IN PIXELS      d >= 0.05 and p_Holm < 0.05
      INDEX WINS IN PIXELS     d <= -0.05 and p_Holm < 0.05
      WITHIN 0.05              the 95% permutation CI of d lies inside (-0.05, +0.05)
      UNMEASURED               otherwise
      (a WITHIN label gets the suffix [NOT LEARNED ...] when both arms' means are at or below the retrace floor; a
      directional label gets [winning arm at or below the retrace floor] when the winner's mean is <= the mean
      retrace floor -- Amendment 1, m2; the retrace floor assumes perfect recognition, so it is optimistic for pixels)
  P2  I_ActOnly - I_MapWM    does the observation-driven step (the what-to-where leak) cost accuracy?
      THE OBSERVATION STEP COSTS IN PIXELS d >= 0.01 and p_Holm < 0.05
      ACTION-ONLY HURTS        d <= -0.01 and p_Holm < 0.05
      WITHIN 0.01              the 95% permutation CI of d lies inside (-0.01, +0.01)
      UNMEASURED               otherwise
      I_MapWM is the image MapWM at the init-matched encoder gain (30.3729). AMENDMENT 3: label words name the
      observation STEP, not a mechanism; beside P2 a declared descriptive per-seed END STATE of each image MapWM seed:
      LEARNED (acc > that seed's retrace floor), else COLLAPSED (encoder probe test acc < 0.5), else NOISY
      (within-class step / action step >= 1 and probe >= 0.5), else OTHER (precedence in that order).
      P2 and G are printed with "PILOT-INFORMED, NOT BLIND"; P1 with "partly pilot-informed".
      ("IN PIXELS" names the condition; the pixel-specific attribution is the secondary S2b
       (I_ActOnly - I_MapWM) - (T_ActOnly - T_MapWM), Amendment 1, m3)
      Beside P2, a DECLARED SECONDARY (no verdict, outside Holm; Amendment 1, M3 as revised): seeds within 0.01 of
      their seed's ceil_vote, I_ActOnly vs I_MapWM, two-sided Fisher exact (stats_core.fisher_solved).
  G   REGISTERED SECONDARY (own label, outside Holm; Amendment 2): I_MapWM_g1 - I_MapWM, image MapWM with encoder
      gain 1.0 vs 30.3729 -- does the cost scale with how strongly images enter the step map?
      GAIN 1 RESCUES MAPWM     d >= 0.05 and perm p < 0.05 (gain 1 ahead)
      NO RESCUE >= 0.05        the 95% CI of d lies below +0.05
      (AMENDMENT 3: the gain scales the whole image embedding -- step-map input AND residual stream -- so G is a total
      effect of the encoder-output scale at two values, not a dose-response and not specific to the step map)
      AS IT FALLS              otherwise
      beside it, I_ActOnly - I_MapWM_g1 (the residual cost at gain 1), d and perm p, no label.
  Every label carries the run classes of both arms (SOLVED / STALLED / DESCENDING, stats_core.classify_run; TAGS
  only -- in image arms the loss has a recognition floor, so STALLED is not "stuck": read it with the ceiling count) and,
  per arm, how many seeds sit above the retrace floor and within 0.01 of the perception ceiling (ceil_vote).
VOID: a declared run missing; the gate's seeds differ from the batch seeds; an ActOnly cell's leak readout |zeroobj - test| > 0.002 on any seed (wiring); a training
run that drew a test-split image, or a registered readout that drew a train-split image; bank md5 differs across
runs or from the gate's.
Declared secondaries (no verdict): S1 cost of pixels; S2 leak readouts and step statistics; S3 perception (probe,
transfer gap); S4 strata vs retrace; S5 context destruction; S6 run classes, r(loss, acc), lengths, own map;
S7 block B contrasts (if declared); P1' I_MapWM - I_RoPE (the pre-Amendment-2 P1).
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

from mapformer.stats_core import perm2_p, perm2_ci, classify_run, signflip_p, fisher_solved

ENC_GAIN = 30.3729          # = model_image.ENC_GAIN (not imported: the analysis needs no torch)
PILOT_P2 = "[PILOT-INFORMED, NOT BLIND: direction known from pilots 1-2 (3 seeds at 150 epochs)]"
PILOT_G = "[PILOT-INFORMED, NOT BLIND: one pilot seed]"
PILOT_P1 = ("[partly pilot-informed: image ActOnly 0.9944 on one pilot seed, image RoPE's pilot training loss known; "
            "image RoPE accuracy unseen]")

REPO = "/home/prashr/mapformer"
RUNS = f"{REPO}/runs/image_torus"
GATE = f"{REPO}/IMAGE_TORUS_GATE.json"
SEEDS = list(range(3100, 3108))
BLOCKS = {"A": ["I_ActOnly", "I_MapWM", "I_RoPE", "I_MapWM_g1", "T_MapWM", "T_ActOnly"],     # Amendment 2: 48 runs
          "B": ["I_NormStep", "I_RoPE_L4", "T_RoPE"]}
IMAGE_CELLS = ("I_MapWM", "I_ActOnly", "I_RoPE", "I_MapWM_g1", "I_NormStep", "I_RoPE_L4")
PATH_CELLS = ("I_MapWM", "I_ActOnly", "I_MapWM_g1", "I_NormStep", "T_MapWM", "T_ActOnly")
STRATA = ("plain_lag<128", "plain_lag>=128", "wrap")
SNAME = {"plain_lag<128": "short-gap", "plain_lag>=128": "long-gap", "wrap": "wrap-only"}
KEY = "test|1024"


def cells_for(blocks):
    return [c for b in blocks for c in BLOCKS[b]]


def load(runs, cells, seeds):
    D, missing = {}, []
    for c in cells:
        D[c] = []
        for s in seeds:
            p = f"{runs}/{c}/{c}_s{s}/eval_image.json"
            if not os.path.exists(p):
                missing.append(p); continue
            r = json.load(open(p))
            assert r["cell"] == c and r["seed"] == s, (p, r["cell"], r["seed"])
            r["cls"] = classify_run(r["losses"])
            D[c].append(r)
    return D, missing


def voids(D, gate):
    v = []
    for c in ("I_ActOnly", "T_ActOnly"):
        for r in D.get(c, []):
            L = r["acc"]["zeroobj|1024"] - r["acc"][KEY]
            if abs(L) > 0.002:
                v.append(f"{c} s{r['seed']}: leak readout {L:+.4f} (wiring)")
    md5s = set()
    for c in IMAGE_CELLS:
        for r in D.get(c, []):
            td, ed = r.get("train_draws") or {}, r.get("draws_eval_main") or {}
            if td.get("test", 1) != 0 or td.get("train", 0) <= 0:
                v.append(f"{c} s{r['seed']}: training draws {td} (a test image in training, or none recorded)")
            if ed.get("train", 1) != 0 or ed.get("test", 0) <= 0:
                v.append(f"{c} s{r['seed']}: registered readouts drew {ed} (must be test split only)")
            md5s.add(r.get("bank_md5"))
    for c in IMAGE_CELLS:                       # gain void (Amendment 3, n4): belt and braces over the cell-name chain
        want = 1.0 if c == "I_MapWM_g1" else ENC_GAIN
        for r in D.get(c, []):
            g = r.get("enc_gain")
            if g is None or abs(g - want) > 1e-4:
                v.append(f"{c} s{r['seed']}: encoder gain {g} != {want}")
    if len(md5s) > 1 or (md5s and gate.get("bank_md5") not in md5s):
        v.append(f"bank md5 differs: runs {sorted(md5s)}, gate {gate.get('bank_md5')}")
    return v


def acc(D, c, key=KEY):
    return np.array([r["acc"][key] for r in D[c]], float)


def floor_of(gate, s, field, sub=None):
    """One seed's floor, looked up BY SEED (Amendment 1, m4)."""
    if field == "ceil_vote":
        return float(gate["ceilings"][str(s)]["ceil_vote"])
    return float(gate["floors"][str(s)][sub or "all"][field])


def by_seed(gate, field, sub=None, seeds=None):
    seeds = seeds if seeds is not None else sorted(int(k) for k in gate["floors"])
    return np.array([floor_of(gate, s, field, sub) for s in seeds], float)


def seeds_of(D, c):
    return [r["seed"] for r in D[c]]


def n_within_ceiling(D, c, gate):
    return int(sum(r["acc"][KEY] >= floor_of(gate, r["seed"], "ceil_vote") - 0.01 for r in D[c]))


def classes(D, c):
    k = {"SOLVED": 0, "STALLED": 0, "DESCENDING": 0}
    for r in D[c]:
        k[r["cls"]["registered"]] += 1
    return k


def tags(D, c, gate):
    a = acc(D, c); n = len(a)
    ret = by_seed(gate, "retrace", seeds=seeds_of(D, c))
    k = classes(D, c)
    ab = int((a > ret).sum()); at = n_within_ceiling(D, c, gate)
    return (f"{c}: SOLVED {k['SOLVED']}/{n}, STALLED {k['STALLED']}, DESCENDING {k['DESCENDING']}; above retrace "
            f"{ab}/{n}; within 0.01 of ceil_vote {at}/{n}")


END_STATES = ("LEARNED", "COLLAPSED", "NOISY", "OTHER")


def end_state(r, retrace):
    """Amendment 3 (M1), thresholds fixed before the batch: LEARNED if acc > the seed's retrace floor; else COLLAPSED if
    the encoder's linear probe test acc < 0.5; else NOISY if within-class (per-instance) step / action step >= 1 (probe
    >= 0.5); else OTHER."""
    if r["acc"][KEY] > retrace:
        return "LEARNED"
    pr = (r.get("probe") or {}).get("test_acc")
    if pr is not None and pr < 0.5:
        return "COLLAPSED"
    ia = (r.get("step") or {}).get("instance_over_action")
    if ia is not None and ia >= 1.0 and pr is not None:
        return "NOISY"
    return "OTHER"


def end_states(D, c, gate):
    return [(r["seed"], end_state(r, floor_of(gate, r["seed"], "retrace"))) for r in D.get(c, [])]


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j])); adj[j] = run
    return adj


def label_p1(d, p, ci):
    if d >= 0.05 and p < 0.05:
        return "PATH WINS IN PIXELS"
    if d <= -0.05 and p < 0.05:
        return "INDEX WINS IN PIXELS"
    if ci["lo"] is not None and ci["lo"] > -0.05 and ci["hi"] < 0.05:
        return "WITHIN 0.05"
    return "UNMEASURED"


def label_p2(d, p, ci):
    if d >= 0.01 and p < 0.05:
        return "THE OBSERVATION STEP COSTS IN PIXELS"
    if d <= -0.01 and p < 0.05:
        return "ACTION-ONLY HURTS"
    if ci["lo"] is not None and ci["lo"] > -0.01 and ci["hi"] < 0.01:
        return "WITHIN 0.01"
    return "UNMEASURED"


def contrast(D, a, b, key=KEY):
    """mean(b) - mean(a) with the exact permutation p and the 95% test-inversion CI."""
    xa, xb = acc(D, a, key), acc(D, b, key)
    return {"a": a, "b": b, "d": float(xb.mean() - xa.mean()), "p": perm2_p(xa, xb)["p"],
            "ci": perm2_ci(xa, xb, step=0.0005), "mean_a": float(xa.mean()), "mean_b": float(xb.mean()),
            "sd_a": float(xa.std(ddof=1)), "sd_b": float(xb.std(ddof=1))}


def registered(D, gate):
    P1 = contrast(D, "I_RoPE", "I_ActOnly"); P2 = contrast(D, "I_MapWM", "I_ActOnly")      # Amendment 2
    adj = holm(np.array([P1["p"], P2["p"]]))                             # Holm over two (Amendment 1, M3 revised)
    P1["p_holm"], P2["p_holm"] = float(adj[0]), float(adj[1])
    P1["label"] = label_p1(P1["d"], P1["p_holm"], P1["ci"]); P2["label"] = label_p2(P2["d"], P2["p_holm"], P2["ci"])
    # declared secondary beside P2, no verdict, outside Holm
    k_b, k_a = n_within_ceiling(D, "I_ActOnly", gate), n_within_ceiling(D, "I_MapWM", gate)
    P2["fisher_secondary"] = {"k_ActOnly": k_b, "k_MapWM": k_a, "n": len(D["I_MapWM"]),
                              "p": fisher_solved(k_a, len(D["I_MapWM"]), k_b, len(D["I_ActOnly"]))}
    ret = float(by_seed(gate, "retrace").mean())
    for C in (P1, P2):
        if C["label"].startswith("WITHIN") and max(C["mean_a"], C["mean_b"]) <= ret:
            C["label"] += " [NOT LEARNED: both arms at or below the retrace floor]"
        elif C["label"] != "UNMEASURED" and not C["label"].startswith("WITHIN"):
            win = C["mean_b"] if C["d"] > 0 else C["mean_a"]
            if win <= ret:                                              # Amendment 1, m2
                C["label"] += " [winning arm at or below the retrace floor]"
    return {"P1": P1, "P2": P2}


def fmt(C, name):
    ci = C["ci"]
    return (f"{name}: {C['b']} {C['mean_b']:.4f} +/- {C['sd_b']:.4f} vs {C['a']} {C['mean_a']:.4f} +/- {C['sd_a']:.4f}; "
            f"d {C['d']:+.4f}, perm p {C['p']:.4g}" + (f", Holm {C['p_holm']:.4g}" if "p_holm" in C else "")
            + (f", 95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]" if ci["lo"] is not None else ", 95% CI none"))


def report(D, gate, blocks, out=print):
    R = registered(D, gate); V = {"blocks": blocks, "registered": {}, "secondary": {}}
    out("== REGISTERED (T = 1024, test-split images, held-out map; n = 8 vs 8; Holm over P1, P2) ==")
    for k in ("P1", "P2"):
        C = R[k]
        out(f"{k} -> {C['label']}  " + (PILOT_P2 if k == "P2" else PILOT_P1)); out("   " + fmt(C, k))
        if k == "P2":
            es = end_states(D, "I_MapWM", gate)
            out("   (declared descriptive, no verdict) image MapWM end state per seed: "
                + ", ".join(f"s{s} {e}" for s, e in es) + "; counts "
                + ", ".join(f"{e} {sum(x == e for _, x in es)}" for e in END_STATES))
            V["secondary"]["P2_end_states_I_MapWM"] = dict((str(s), e) for s, e in es)
            F = C["fisher_secondary"]
            out(f"   (declared secondary, no verdict, outside Holm) seeds within 0.01 of ceil_vote: I_ActOnly "
                f"{F['k_ActOnly']}/{F['n']} vs I_MapWM {F['k_MapWM']}/{F['n']}, Fisher p {F['p']:.4g}")
        out("   [" + tags(D, C["b"], gate) + "]"); out("   [" + tags(D, C["a"], gate) + "]")
        V["registered"][k] = {x: C[x] for x in ("a", "b", "d", "p", "p_holm", "label", "mean_a", "mean_b")}
        V["registered"][k]["ci"] = C["ci"]
        V["registered"][k]["pilot_informed"] = True if k == "P2" else "partly"
        V["registered"][k]["disclosure"] = PILOT_P2 if k == "P2" else PILOT_P1
        if k == "P2":
            V["secondary"]["P2_within_ceiling_fisher"] = C["fisher_secondary"]
    floors = {"blank": float(by_seed(gate, "blank").mean()), "retrace": float(by_seed(gate, "retrace").mean()),
              "ngram_token": float(np.mean([gate["floors"][k]["ngram_token_best"] for k in gate["floors"]])),
              "ngram_action": float(np.mean([gate["floors"][k]["ngram_action_best"] for k in gate["floors"]])),
              "ceil_vote": float(by_seed(gate, "ceil_vote").mean()),
              "classifier_test": gate["classifier"]["test_acc"]}
    V["floors"] = floors
    out("")
    out(f"Floors on these walks (mean over seeds): blank {floors['blank']:.4f}, retrace {floors['retrace']:.4f}, "
        f"token n-gram {floors['ngram_token']:.4f}, action-only n-gram {floors['ngram_action']:.4f}; perception "
        f"ceiling ceil_vote {floors['ceil_vote']:.4f} (standalone classifier {floors['classifier_test']:.4f})")
    out("")
    out("== DECLARED SECONDARIES (no verdict) ==")
    out("Per cell (mean +/- sd over seeds): acc T=1024 | 512 | 2048 (OOD) | own map | train images | zeroobj | shuffle")
    for c in D:
        if not D[c]:
            continue
        row = []
        for k in (KEY, "test|512", "test|2048", "own|1024", "train|1024", "zeroobj|1024", "shuffle|1024"):
            x = [r["acc"][k] for r in D[c] if k in r["acc"]]
            row.append(f"{np.mean(x):.4f}+/-{np.std(x, ddof=1) if len(x) > 1 else 0:.4f}" if x else "--")
        out(f"  {c:11s} " + " | ".join(row) + f"   [{tags(D, c, gate)}]")
    # S1 cost of pixels
    out("S1 cost of pixels (b - a):")
    S1 = {}
    for a, b in (("T_MapWM", "I_MapWM"), ("T_ActOnly", "I_ActOnly"), ("T_MapWM", "I_ActOnly")):
        C = contrast(D, a, b); S1[f"{b}-{a}"] = {"d": C["d"], "p": C["p"]}
        out("   " + fmt(C, f"{b} - {a}"))
    for c in IMAGE_CELLS:
        if c in D and D[c]:
            g = acc(D, c) - by_seed(gate, "ceil_vote", seeds=seeds_of(D, c))
            S1[f"{c}-ceil"] = float(g.mean()); out(f"   {c} - ceil_vote (paired by seed): {g.mean():+.4f} (min {g.min():+.4f})")
    V["secondary"]["S1"] = S1
    # S2 leak
    out("S2 leak: L = zeroobj - test (object steps zeroed at eval); step statistics (ratios to the action rms step)")
    S2 = {}
    for c in PATH_CELLS:
        if c in D and D[c]:
            L = np.array([r["acc"]["zeroobj|1024"] - r["acc"][KEY] for r in D[c]])
            oa = [r["step"]["object_over_action"] for r in D[c]]; ia = [r["step"]["instance_over_action"] for r in D[c]]
            S2[c] = {"L_median": float(np.median(L)), "L_range": [float(L.min()), float(L.max())],
                     "object_over_action_median": float(np.median(oa)), "object_over_action_range": [min(oa), max(oa)],
                     "instance_over_action_median": float(np.median(ia)), "instance_over_action_range": [min(ia), max(ia)]}
            out(f"   {c:11s} L median {np.median(L):+.4f} [{L.min():+.4f}, {L.max():+.4f}]; object/action step median "
                f"{np.median(oa):.4f} [{min(oa):.4f}, {max(oa):.4f}]; within-class (instance)/action median "
                f"{np.median(ia):.4f} [{min(ia):.4f}, {max(ia):.4f}]")
    if D.get("I_MapWM") and D.get("T_MapWM"):
        Li = [r["acc"]["zeroobj|1024"] - r["acc"][KEY] for r in D["I_MapWM"]]
        Lt = [r["acc"]["zeroobj|1024"] - r["acc"][KEY] for r in D["T_MapWM"]]
        p = perm2_p(Lt, Li)["p"]; S2["L_image_minus_token"] = {"d": float(np.mean(Li) - np.mean(Lt)), "p": p}
        out(f"   L(I_MapWM) - L(T_MapWM) {np.mean(Li) - np.mean(Lt):+.4f}, perm p {p:.4g} ('leaks more with images')")
    # S2b: pixel attribution of the leak (Amendment 1, m3), per seed (cells at a seed share map, walks and base init)
    ia = {r["seed"]: r["acc"][KEY] for r in D["I_ActOnly"]}; im = {r["seed"]: r["acc"][KEY] for r in D["I_MapWM"]}
    ta = {r["seed"]: r["acc"][KEY] for r in D["T_ActOnly"]}; tm = {r["seed"]: r["acc"][KEY] for r in D["T_MapWM"]}
    inter = np.array([(ia[s] - im[s]) - (ta[s] - tm[s]) for s in sorted(ia)])
    pi = signflip_p(inter)["p"]
    S2["pixel_attribution"] = {"d": float(inter.mean()), "p_signflip": pi,
                               "token_ActOnly_minus_MapWM": float(np.mean([ta[s] - tm[s] for s in ta]))}
    out(f"   S2b (I_ActOnly - I_MapWM) - (T_ActOnly - T_MapWM) {inter.mean():+.4f} (paired by seed, sign-flip p {pi:.3g}); "
        f"token ActOnly - MapWM {S2['pixel_attribution']['token_ActOnly_minus_MapWM']:+.4f}")
    V["secondary"]["S2"] = S2
    # S3 perception
    out("S3 perception: linear probe of the trained encoder (test split); transfer gap = train-images - test-images acc")
    S3 = {}
    for c in IMAGE_CELLS:
        if c in D and D[c]:
            pr = [r["probe"]["test_acc"] for r in D[c] if "probe" in r]
            gap = np.array([r["acc"]["train|1024"] - r["acc"][KEY] for r in D[c]])
            S3[c] = {"probe_median": float(np.median(pr)) if pr else None,
                     "probe_range": [min(pr), max(pr)] if pr else None,
                     "gap_median": float(np.median(gap)), "gap_range": [float(gap.min()), float(gap.max())],
                     "gap_p": signflip_p(gap)["p"]}
            out(f"   {c:11s} probe median {np.median(pr) if pr else float('nan'):.4f} "
                f"[{min(pr) if pr else float('nan'):.4f}, {max(pr) if pr else float('nan'):.4f}]; gap median "
                f"{np.median(gap):+.4f} [{gap.min():+.4f}, {gap.max():+.4f}] (sign-flip p {signflip_p(gap)['p']:.3g})")
    V["secondary"]["S3"] = S3
    # S4 strata
    out("S4 strata (T = 1024): cell acc / retrace floor on the same walks")
    rs = {st: by_seed(gate, "retrace", st).mean() for st in STRATA}
    for c in D:
        if D[c]:
            out(f"   {c:11s} " + "  ".join(
                f"{SNAME[st]} {np.mean([r['strata'][KEY][st]['acc'] for r in D[c]]):.3f} / {rs[st]:.3f}" for st in STRATA))
    # S5 context destruction
    out(f"S5 context destruction (observations shuffled): vs blank floor {floors['blank']:.4f}")
    for c in D:
        if D[c]:
            out(f"   {c:11s} {acc(D, c, 'shuffle|1024').mean():.4f}")
    # S6 r(loss, acc)
    tails = [r["cls"]["tail"] for c in D for r in D[c]]; accs = [r["acc"][KEY] for c in D for r in D[c]]
    rr = float(np.corrcoef(tails, accs)[0, 1]) if len(set(tails)) > 1 else float("nan")
    out(f"S6 r(final-5% training loss, acc T=1024) over {len(tails)} runs: {rr:+.3f}")
    V["secondary"]["S6_r_loss_acc"] = rr
    # G: registered secondary, own label, outside Holm (Amendment 2)
    G = contrast(D, "I_MapWM", "I_MapWM_g1")
    if G["d"] >= 0.05 and G["p"] < 0.05:
        G["label"] = "GAIN 1 RESCUES MAPWM"
    elif G["ci"]["hi"] is not None and G["ci"]["hi"] < 0.05:
        G["label"] = "NO RESCUE >= 0.05"
    else:
        G["label"] = "AS IT FALLS"
    Gr = contrast(D, "I_MapWM_g1", "I_ActOnly")
    out(f"G (registered secondary, outside Holm) -> {G['label']}  {PILOT_G}"); out("   " + fmt(G, "I_MapWM_g1 - I_MapWM"))
    out("   [" + tags(D, "I_MapWM_g1", gate) + "]")
    es1 = end_states(D, "I_MapWM_g1", gate)
    out("   (declared descriptive) gain-1 MapWM end state per seed: " + ", ".join(f"s{s} {e}" for s, e in es1))
    out("   residual cost at gain 1: " + fmt(Gr, "I_ActOnly - I_MapWM_g1"))
    V["registered_secondary"] = {"G": {"label": G["label"], "d": G["d"], "p": G["p"], "ci": G["ci"],
                                       "pilot_informed": True, "disclosure": PILOT_G},
                                 "residual_g1": {"d": Gr["d"], "p": Gr["p"]}}
    # P1' and S7
    P1o = contrast(D, "I_RoPE", "I_MapWM")
    lab = label_p1(P1o["d"], P1o["p"], P1o["ci"])
    out(f"P1' -> {lab}  [original P1 rule on the RAW p, outside Holm, superseded by Amendment 2; the pre-pilot "
        f"prediction (I_MapWM 0.97-0.99) was contradicted by the pilots]")
    out("   " + fmt(P1o, "I_MapWM - I_RoPE"))
    V["secondary"]["P1_original"] = {"label": lab, "d": P1o["d"], "p_raw": P1o["p"], "superseded": True}
    if "B" in blocks:
        out("S7 block B:")
        S7 = {}
        # Amendment 3 (n7): the depth reference is read against the path arm that learns (I_ActOnly)
        for a, b in (("I_RoPE", "I_RoPE_L4"), ("I_RoPE_L4", "I_ActOnly"), ("I_MapWM", "I_NormStep"),
                     ("T_RoPE", "T_MapWM"), ("T_RoPE", "I_RoPE")):
            C = contrast(D, a, b); S7[f"{b}-{a}"] = {"d": C["d"], "p": C["p"]}; out("   " + fmt(C, f"{b} - {a}"))
        V["secondary"]["S7"] = S7
    return V


def run(runs, gate_path, blocks, seeds, verdicts_out, out=print):
    gate = json.load(open(gate_path))
    cells = cells_for(blocks)
    D, missing = load(runs, cells, seeds)
    if missing:
        out(f"VOID: {len(missing)} declared run(s) missing, e.g. {missing[0]}")
        return {"void": [f"missing {m}" for m in missing]}
    extra = [c for b in BLOCKS if b not in blocks for c in BLOCKS[b] if os.path.isdir(f"{runs}/{c}")]
    if extra:
        out(f"NOTE: runs of undeclared cells exist ({extra}); they are not read")
    gs = sorted(int(k) for k in gate["floors"])
    if gs != sorted(seeds) or sorted(int(k) for k in gate["ceilings"]) != sorted(seeds):
        out(f"VOID: the gate's seeds {gs} differ from the batch seeds {sorted(seeds)}")
        return {"void": [f"gate seeds {gs} != batch seeds"]}
    v = voids(D, gate)
    if v:
        out("VOID: " + "; ".join(v))
        res = {"void": v}
    else:
        res = report(D, gate, blocks, out)
    if verdicts_out:
        json.dump(res, open(verdicts_out, "w"), indent=1)
    return res


# ------------------------------------------------------------------------------------------------ selftest
def _fake_gate(path, seeds):
    fl = {str(s): {"all": {"blank": 0.51, "retrace": 0.845, "n": 32000},
                   **{st: {"blank": 0.51, "retrace": 0.9 if st == "plain_lag<128" else 0.51, "n": 9000} for st in STRATA},
                   "ngram_token_best": 0.53, "ngram_action_best": 0.51} for s in seeds}
    ce = {str(s): {"ceil_vote": 0.993, "ceil_last": 0.990} for s in seeds}
    json.dump({"floors": fl, "ceilings": ce, "bank_md5": "x", "classifier": {"test_acc": 0.985}}, open(path, "w"))


def _fake_runs(root, seeds, accs, blocks="A", actonly_leak=0.0, test_draw_in_train=False, train_draw_in_eval=False,
               drop=None, desc=None, bad_md5=False, no_train_draws=False, bad_gain=False, probe=None, inst=None):
    rng = np.random.default_rng(0)
    for c in cells_for(blocks):
        for i, s in enumerate(seeds):
            if drop == (c, s):
                continue
            a = float(accs[c][i] if hasattr(accs[c], "__len__") else accs[c] + rng.normal(0, 0.002))
            solved = a > 0.97
            losses = list(np.linspace(1.0, 0.01 if solved else 0.3, 900))
            if desc and c in desc:
                losses = list(np.linspace(1.0, 0.1, 900))
            img = c.startswith("I_")
            r = {"cell": c, "seed": s, "losses": losses,
                 "acc": {KEY: a, "test|512": a, "test|2048": a - 0.05, "own|1024": a, "train|1024": a + 0.002,
                         "shuffle|1024": 0.5},
                 "strata": {KEY: {st: {"acc": a, "n": 9000} for st in STRATA}}}
            if c in PATH_CELLS:
                r["acc"]["zeroobj|1024"] = a + (actonly_leak if c == "I_ActOnly" else (0.0 if c.endswith("ActOnly") else 0.004))
                r["step"] = {"object_over_action": 0.05, "instance_over_action": 0.02 if img else 0.0}
                if inst and c in inst:
                    r["step"]["instance_over_action"] = inst[c][i]
            if img:
                r["bank_md5"] = "y" if (bad_md5 and c == "I_RoPE") else "x"
                r["train_draws"] = ({} if no_train_draws else {"train": 10 ** 8, "test": 1 if test_draw_in_train else 0})
                r["draws_eval_main"] = {"train": 1 if train_draw_in_eval else 0, "test": 10 ** 6}
                r["probe"] = {"test_acc": probe[c][i] if (probe and c in probe) else 0.98}
                r["enc_gain"] = (1.0 if c == "I_MapWM_g1" else ENC_GAIN) + (0.5 if (bad_gain and c == "I_ActOnly") else 0.0)
            d = f"{root}/{c}/{c}_s{s}"; os.makedirs(d, exist_ok=True)
            json.dump(r, open(f"{d}/eval_image.json", "w"))


def selftest():
    S = SEEDS
    base = {"I_MapWM": 0.99, "I_ActOnly": 0.995, "I_RoPE": 0.72, "I_MapWM_g1": 0.99, "T_MapWM": 1.0, "T_ActOnly": 1.0,
            "I_NormStep": 0.99,
            "I_RoPE_L4": 0.85, "T_RoPE": 0.72}
    lin = lambda lo, hi: list(np.linspace(lo, hi, 8))
    cases = [
        ("P1 PATH WINS / P2 OBSERVATION STEP COSTS", dict(base, I_MapWM=lin(0.975, 0.985), I_ActOnly=lin(0.994, 0.998)),
         {"P1": "PATH WINS IN PIXELS", "P2": "THE OBSERVATION STEP COSTS IN PIXELS"}, "A", {}),
        ("P1 INDEX WINS / P2 ACTION-ONLY HURTS", dict(base, I_MapWM=lin(0.60, 0.62), I_RoPE=lin(0.72, 0.74),
                                                      I_ActOnly=lin(0.55, 0.57)),
         {"P1": "INDEX WINS IN PIXELS [winning arm at or below the retrace floor]",
          "P2": "ACTION-ONLY HURTS [winning arm at or below the retrace floor]"}, "A", {}),
        ("PATH WINS below retrace; P2 WITHIN NOT LEARNED", dict(base, I_MapWM=lin(0.800, 0.810), I_RoPE=lin(0.72, 0.73),
                                                               I_ActOnly=lin(0.8005, 0.8105)),
         {"P1": "PATH WINS IN PIXELS [winning arm at or below the retrace floor]",
          "P2": "WITHIN 0.01 [NOT LEARNED: both arms at or below the retrace floor]"}, "A", {}),
        ("Holm changes both labels (raw p 0.032 / 0.039 -> Holm 0.063 / 0.063)",
         dict(base, I_MapWM=lin(0.90, 0.94), I_RoPE=lin(0.716, 0.991), I_ActOnly=lin(0.916, 0.956)),
         {"P1": "UNMEASURED", "P2": "UNMEASURED"}, "A", {"raw_p2_below": 0.05}),
        ("P1 WITHIN / P2 WITHIN", dict(base, I_MapWM=lin(0.900, 0.904), I_RoPE=lin(0.899, 0.903),
                                       I_ActOnly=lin(0.9005, 0.9045)),
         {"P1": "WITHIN 0.05", "P2": "WITHIN 0.01"}, "A", {}),
        ("P1 UNMEASURED / P2 UNMEASURED", dict(base, I_MapWM=[0.6, 1.0] * 4, I_RoPE=[0.55, 0.95] * 4,
                                               I_ActOnly=[0.5, 1.0, 0.7, 0.9] * 2),
         {"P1": "UNMEASURED", "P2": "UNMEASURED"}, "A", {}),
        ("WITHIN, NOT LEARNED", dict(base, I_MapWM=lin(0.500, 0.504), I_RoPE=lin(0.499, 0.503),
                                     I_ActOnly=lin(0.5005, 0.5045)),
         {"P1": "WITHIN 0.05 [NOT LEARNED: both arms at or below the retrace floor]",
          "P2": "WITHIN 0.01 [NOT LEARNED: both arms at or below the retrace floor]"}, "A", {}),
        ("block B present", dict(base, I_MapWM=lin(0.975, 0.985), I_ActOnly=lin(0.994, 0.998)),
         {"P1": "PATH WINS IN PIXELS", "P2": "THE OBSERVATION STEP COSTS IN PIXELS"}, "AB", {}),
        ("tags with DESCENDING runs", dict(base, I_MapWM=lin(0.975, 0.985), I_ActOnly=lin(0.994, 0.998)),
         {"P1": "PATH WINS IN PIXELS", "P2": "THE OBSERVATION STEP COSTS IN PIXELS"}, "A", {"desc": {"I_MapWM"}}),
    ]
    ok = True
    for name, accs, want, blocks, kw in cases:
        kw = dict(kw); raw_below = kw.pop("raw_p2_below", None)
        with tempfile.TemporaryDirectory() as td:
            _fake_gate(f"{td}/gate.json", S); _fake_runs(f"{td}/runs", S, accs, blocks, **kw)
            lines = []
            res = run(f"{td}/runs", f"{td}/gate.json", blocks, S, f"{td}/v.json", out=lines.append)
            got = {k: res["registered"][k]["label"] for k in ("P1", "P2")}
            good = got == want and ("B" not in blocks or "S7" in res["secondary"])
            if "desc" in kw:
                good &= any("DESCENDING 8" in l for l in lines)
            if raw_below is not None:
                for k in ("P1", "P2"):
                    good &= res["registered"][k]["p"] < raw_below <= res["registered"][k]["p_holm"]
                    good &= res["registered"][k]["d"] >= (0.05 if k == "P1" else 0.01)
            ok &= good
            print(f"[{'ok' if good else 'FAIL'}] {name}: {got}")
    for name, kw, frag in (("void: ActOnly leak readout", {"actonly_leak": 0.01}, "leak readout"),
                           ("void: test image in training", {"test_draw_in_train": True}, "training draws"),
                           ("void: train image in a registered readout", {"train_draw_in_eval": True}, "registered readouts"),
                           ("void: missing run", {"drop": ("I_RoPE", 3103)}, "missing"),
                           ("void: bank md5 mismatch", {"bad_md5": True}, "bank md5"),
                           ("void: no train draws recorded", {"no_train_draws": True}, "none recorded"),
                           ("void: encoder gain mismatch", {"bad_gain": True}, "encoder gain")):
        with tempfile.TemporaryDirectory() as td:
            _fake_gate(f"{td}/gate.json", S); _fake_runs(f"{td}/runs", S, base, "A", **kw)
            res = run(f"{td}/runs", f"{td}/gate.json", "A", S, None, out=lambda *_: None)
            good = "void" in res and any(frag in v for v in res["void"])
            ok &= good
            print(f"[{'ok' if good else 'FAIL'}] {name}: {res.get('void', ['not void'])[0][:90]}")
    # G (registered secondary): each label
    for name, accs, want in (("G GAIN 1 RESCUES MAPWM", dict(base, I_MapWM=lin(0.50, 0.52), I_MapWM_g1=lin(0.97, 0.99)),
                              "GAIN 1 RESCUES MAPWM"),
                             ("G NO RESCUE", dict(base, I_MapWM=lin(0.50, 0.52), I_MapWM_g1=lin(0.50, 0.52)),
                              "NO RESCUE >= 0.05"),
                             ("G AS IT FALLS", dict(base, I_MapWM=[0.5, 0.9] * 4, I_MapWM_g1=[0.5, 1.0, 0.6, 1.0] * 2),
                              "AS IT FALLS")):
        with tempfile.TemporaryDirectory() as td:
            _fake_gate(f"{td}/gate.json", S); _fake_runs(f"{td}/runs", S, accs, "A")
            res = run(f"{td}/runs", f"{td}/gate.json", "A", S, None, out=lambda *_: None)
            got = res["registered_secondary"]["G"]["label"]; good = got == want; ok &= good
            print(f"[{'ok' if good else 'FAIL'}] {name}: {got}")
    # end states (Amendment 3, M1), disclosures (m1, m2) and P1' under the original rule (m3)
    with tempfile.TemporaryDirectory() as td:
        accs = dict(base, I_MapWM=[0.5] * 6 + [0.99, 0.99], I_ActOnly=lin(0.994, 0.998))
        pr = {"I_MapWM": [0.13, 0.13, 0.13, 0.87, 0.87, 0.87, 0.98, 0.98]}
        ins = {"I_MapWM": [0.0, 0.0, 0.0, 120.0, 120.0, 0.5, 0.02, 0.02]}
        _fake_gate(f"{td}/gate.json", S); _fake_runs(f"{td}/runs", S, accs, "A", probe=pr, inst=ins)
        lines = []
        res = run(f"{td}/runs", f"{td}/gate.json", "A", S, None, out=lines.append)
        es = res["secondary"]["P2_end_states_I_MapWM"]
        cnt = {e: sum(v == e for v in es.values()) for e in END_STATES}
        good = cnt == {"LEARNED": 2, "COLLAPSED": 3, "NOISY": 2, "OTHER": 1}
        good &= any(l.startswith("P2 -> ") and PILOT_P2 in l for l in lines)
        good &= any(l.startswith("P1 -> ") and PILOT_P1 in l for l in lines)
        good &= any(l.startswith("G (registered") and PILOT_G in l for l in lines)
        good &= (res["registered"]["P2"]["pilot_informed"] is True and res["registered_secondary"]["G"]["pilot_informed"]
                 is True and res["registered"]["P1"]["pilot_informed"] == "partly")
        good &= any(l.startswith("P1' -> ") and "superseded" in l for l in lines) and res["secondary"]["P1_original"]["superseded"]
        good &= any("median" in l and "[" in l for l in lines if l.strip().startswith("I_MapWM ") and "probe" in l)
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] end states {cnt}; disclosures on P1 / P2 / G; P1' original rule; medians")
    # gate seeds must equal the batch seeds
    with tempfile.TemporaryDirectory() as td:
        _fake_gate(f"{td}/gate.json", S[:-1] + [9999]); _fake_runs(f"{td}/runs", S, base, "A")
        res = run(f"{td}/runs", f"{td}/gate.json", "A", S, None, out=lambda *_: None)
        good = "void" in res and "gate seeds" in res["void"][0]; ok &= good
        print(f"[{'ok' if good else 'FAIL'}] void: gate seeds differ")
    # the within-ceiling Fisher count is a secondary only: a significant count difference (8/8 vs 1/8) with an accuracy
    # difference below the 0.01 threshold must NOT change P2's label, and the line must appear in the report and the verdicts' secondaries
    with tempfile.TemporaryDirectory() as td:
        acc_f = dict(base, I_MapWM=list(np.linspace(0.979, 0.981, 7)) + [0.986], I_ActOnly=lin(0.984, 0.986))
        _fake_gate(f"{td}/gate.json", S); _fake_runs(f"{td}/runs", S, acc_f, "A")
        lines = []
        res = run(f"{td}/runs", f"{td}/gate.json", "A", S, None, out=lines.append)
        F = res["secondary"]["P2_within_ceiling_fisher"]
        good = (res["registered"]["P2"]["label"] == "WITHIN 0.01" and F["k_ActOnly"] == 8 and F["k_MapWM"] == 1
                and F["p"] < 0.05
                and any("declared secondary, no verdict" in l for l in lines) and "fisher" not in res["registered"]["P2"])
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] Fisher count is a secondary only: P2 {res['registered']['P2']['label']} "
              f"(perm p {res['registered']['P2']['p']:.3g}); count {F['k_ActOnly']} vs {F['k_MapWM']}, Fisher p {F['p']:.3g}")
    # Holm: the adjusted p of the smaller is 2p, the larger max(2 p_min, p_max)
    h = holm(np.array([0.01, 0.03])); good = np.allclose(h, [0.02, 0.03]); ok &= good
    h2 = holm(np.array([0.04, 0.01])); good2 = np.allclose(h2, [0.04, 0.02]); ok &= good2
    print(f"[{'ok' if good and good2 else 'FAIL'}] Holm {h.tolist()} {h2.tolist()}")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", choices=["A", "AB"])
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--gate", default=GATE)
    ap.add_argument("--verdicts", default=f"{REPO}/IMAGE_TORUS_VERDICTS.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.blocks:
        ap.error("--blocks is required (as declared in IMAGE_TORUS_PREREG.md before launch)")
    res = run(a.runs, a.gate, a.blocks, SEEDS, a.verdicts)
    return 1 if "void" in res else 0


if __name__ == "__main__":
    sys.exit(main())

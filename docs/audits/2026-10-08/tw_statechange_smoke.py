"""Smoke test of analyze_tw_statechange.py (TW_STATECHANGE_PREREG.md): every branch of every state function on synthetic
data, the end-to-end analysis on synthetic batches (predicted, leak, state unused, index not beaten, re-score flip, VOID
cases), and collect() end to end (readouts + floors + replay check) on 4 untrained checkpoints in pilot mode. CPU only.
Prints PASS / FAIL per case; exits non-zero on any FAIL."""
import copy
import io
import json
import os
import sys
import tempfile

import numpy as np
import torch

import mapformer.analyze_tw_statechange as A

FAILS = []; N = 0


def ok(name, cond, detail=""):
    global N
    N += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)
    if not cond:
        FAILS.append(name)


rng = np.random.default_rng(0)
n = 8
hi = list(0.99 + 0.005 * rng.random(n)); lo = list(0.50 + 0.01 * rng.random(n))
# ---- contrast_state
ok("contrast BETTER", A.contrast_state(lo, hi, 0, 8, n)[0] == "BETTER")
ok("contrast WORSE", A.contrast_state(hi, lo, 8, 0, n)[0] == "WORSE")
ok("contrast CONFLICT", A.contrast_state(lo, hi, 8, 0, n)[0] == "CONFLICT")
x = [0.97 + 0.001 * i for i in range(n)]; y = [0.971 + 0.001 * i for i in range(n)]
ok("contrast SOLVED RATE HIGHER", A.contrast_state(x, y, 0, 8, n)[0].startswith("SOLVED RATE HIGHER"))
ok("contrast SOLVED RATE LOWER", A.contrast_state(x, y, 8, 0, n)[0].startswith("SOLVED RATE LOWER"))
ok("contrast CEILING", A.contrast_state([0.9995] * n, [0.9999] * n, 8, 8, n)[0] == "CEILING")
ok("contrast NO DIFFERENCE", A.contrast_state(x, y, 4, 4, n)[0].startswith("NO DIFFERENCE"))
ok("contrast 1% gap does not fire", A.contrast_state([0.98] * 4 + [0.981] * 4, [0.99] * 4 + [0.991] * 4, 8, 8, n)[0]
   .startswith("NO DIFFERENCE"))
# ---- geometry / asides
ok("geom OFF 8/8", A.geom_label([0.05] * 8) == ("OFF", 8))
ok("geom OFF 6/8 (boundary)", A.geom_label([0.05] * 6 + [0.3] * 2) == ("OFF", 6))
ok("geom MIXED 5/8", A.geom_label([0.05] * 5 + [0.3] * 3) == ("MIXED", 5))
ok("geom MIXED 3/8", A.geom_label([0.05] * 3 + [0.3] * 5) == ("MIXED", 3))
ok("geom ON 2/8 off", A.geom_label([0.05] * 2 + [0.3] * 6) == ("ON", 2))
ok("geom threshold inclusive", A.geom_label([A.SHIFT_OFF] * 8)[0] == "OFF")
ok("aside LARGER", A.aside_label([0.3] * 8, [0.05] * 8)[0] == "LARGER THAN ASIDES")
ok("aside SMALLER", A.aside_label([0.01] * 8, [0.05] * 8)[0] == "SMALLER THAN ASIDES")
ok("aside NOT DISTINGUISHED", A.aside_label([0.05, 0.06] * 4, [0.06, 0.05] * 4)[0] == "NOT DISTINGUISHED FROM ASIDES")
# ---- B verdict
for g_ in ("OFF", "ON", "MIXED"):
    for c_ in ("STATE BOUND TO PLACE", "STATE NOT SHOWN USED"):
        v = A.b_verdict(g_, 4, 8, "NOT DISTINGUISHED FROM ASIDES", c_)
        exp = {"OFF": "STATE VERBS STAY OFF THE MAP", "ON": "STATE VERBS MOVE THE MAP", "MIXED": "MIXED"}[g_]
        ok(f"B {g_} / {c_[:16]}", v.startswith(exp) and (("not shown used" in v) == c_.startswith("STATE NOT")), f"-> {v}")
# ---- functional (secondary)
ok("func COSTS", A.func_label([0.03 + 0.001 * i for i in range(8)])[0] == "COSTS")
ok("func USED", A.func_label([-0.03 - 0.001 * i for i in range(8)])[0] == "USED")
ok("func NO COST (small)", A.func_label([-0.003 - 0.0001 * i for i in range(8)])[0] == "NO COST")
# ---- C
ok("C BOUND", A.c_label([0.95] * 8, 0.21, 0.65)[0] == "STATE BOUND TO PLACE")
ok("C PARTLY", A.c_label([0.70] * 8, 0.21, 0.65)[0] == "STATE USED, NOT SHOWN BEYOND THE LAST-DROP RULE")
ok("C NOT SHOWN", A.c_label([0.25] * 8, 0.21, 0.65)[0] == "STATE NOT SHOWN USED")
ok("C NOT SHOWN (mixed signs)", A.c_label([0.9] * 4 + [0.0] * 4, 0.21, 0.65)[0] == "STATE NOT SHOWN USED")
ok("A headlines", A.a_headline("BETTER").startswith("PATH NEEDED") and A.a_headline("WORSE").startswith("INDEX")
   and A.a_headline("CEILING") == "LOCATION CONTRAST CEILING")

# ---- end to end on synthetic batches
FL = {"constant": {"all": 0.514, "T1": 0.512, "T2drop": 0.0}, "revcopy": {"all": 0.566, "T1": 0.607, "T2drop": 0.0},
      "revcopy_state": {"all": 0.610, "T1": 0.607, "T2drop": 0.211}, "last_dropped": {"all": 0.106, "T2drop": 0.649},
      "last_saw": {"all": 0.804, "T2drop": 0.0}, "first_saw": {"all": 0.754, "T2drop": 0.0},
      **{f"ngram{k}": {"all": 0.51, "T2drop": 0.0} for k in range(1, 6)}}
STR = ("all", "T1", "T1s", "T1a", "T1clean", "T2take", "T2drop", "T3")


def run(arm, accs, rs=None, shift=0.04, aside=0.05, cls="SOLVED", epochs=900, L=-0.003):
    a = {"intact": {k: accs.get(k, accs["all"]) for k in STR}}
    a["intact_rs"] = {k: (rs or accs).get(k, (rs or accs)["all"]) for k in STR}
    r = {"arm": arm, "acc": a, "cls": cls, "tail": 0.01 if cls == "SOLVED" else 0.3, "epochs": epochs, "acc2048": accs["all"]}
    if arm != "RoPE":
        z = arm == "DirOnly"
        r["geom"] = {"shift_sc": 0.0 if z else shift, "shift_take": 0.0 if z else shift, "shift_drop": 0.0 if z else shift,
                     "shift_aside": 0.0 if z else aside, "R_sc": 0.0 if z else 0.3, "R_aside": 0.0 if z else 0.4,
                     "carry_cos": 0.1, "class_step": {"verb": 0.1, "take_verb": 0.05}}
        r.update(L_sc=0.0 if z else L, L_sc_rs=0.0 if z else L, L_sc0=0.0 if z else -0.01, L_sc_all=0.0 if z else L,
                 L_aside=0.0 if z else -0.003, drift_sc=0.0 if z else 0.05, drift_aside=0.0 if z else 0.1,
                 drift_optword=0.03, clock_channels=2, reliance=0.0 if z else 0.48)
    return r


def batch(**kw):
    res = {}
    for i, s in enumerate(A.SEEDS):
        j = 0.001 * i
        res[f"MapWM_s{s}"] = run("MapWM", kw.get("W", {"all": 0.99 + j, "T1": 0.995 + j / 2, "T2drop": 0.95 - j}),
                                 shift=kw.get("Wshift", 0.04 + j), aside=0.05 + j / 2)
        res[f"NormStep_s{s}"] = run("NormStep", kw.get("N", {"all": 0.99 + j, "T1": 0.995, "T2drop": 0.94 - j}),
                                    shift=kw.get("Nshift", 0.05 + j), aside=0.05)
        res[f"DirOnly_s{s}"] = run("DirOnly", kw.get("D", {"all": 0.972 + j / 10, "T1": 0.975, "T2drop": 0.80 + j}),
                                   rs=kw.get("Drs"))
        res[f"RoPE_s{s}"] = run("RoPE", kw.get("I", {"all": 0.514 + j / 10, "T1": 0.512 + j / 10, "T2drop": 0.0}),
                                cls="STALLED")
    return res


def go(res, fl=FL):
    buf = io.StringIO()
    V = A.analyse(res, fl, A.SEEDS, out=lambda s: buf.write(s + "\n"))
    return V, buf.getvalue()


V, txt = go(batch())
ok("e2e predicted: A", V["A"].startswith("PATH NEEDED"), f"-> {V['A']}")
ok("e2e predicted: B MapWM", V["B_MapWM"].startswith("STATE VERBS STAY OFF THE MAP"), f"-> {V['B_MapWM']}")
ok("e2e predicted: C MapWM", V["C_MapWM"].startswith("STATE BOUND TO PLACE"), f"-> {V['C_MapWM']}")
ok("e2e predicted: D2 DirOnly worse", V["D2"].startswith("WORSE"), f"-> {V['D2']}")
ok("e2e predicted: SUMMARY holds", V["SUMMARY"].startswith("PREDICTION HOLDS"))
V, _ = go(batch(Wshift=0.3, Nshift=0.05))
ok("e2e leak: B MapWM MOVE + LARGER", V["B_MapWM"].startswith("STATE VERBS MOVE THE MAP") and "larger than asides" in V["B_MapWM"],
   f"-> {V['B_MapWM']}")
ok("e2e leak: B NormStep OFF", V["B_NormStep"].startswith("STATE VERBS STAY OFF"))
ok("e2e leak: SUMMARY does not hold", V["SUMMARY"].startswith("PREDICTION DOES NOT HOLD"))
V, _ = go(batch(W={"all": 0.9, "T1": 0.99, "T2drop": 0.01}))
ok("e2e state unused: C NOT SHOWN, B qualified", V["C_MapWM"].startswith("STATE NOT SHOWN USED") and "not shown used" in V["B_MapWM"],
   f"-> {V['B_MapWM']}")
ok("e2e state unused: SUMMARY does not hold", V["SUMMARY"].startswith("PREDICTION DOES NOT HOLD"))
V, _ = go(batch(I={"all": 0.99, "T1": 0.995, "T2drop": 0.9}))
ok("e2e index equal: A not PATH NEEDED", not V["A"].startswith("PATH NEEDED"), f"-> {V['A']}")
V, _ = go(batch(D={"all": 0.99, "T1": 0.995, "T2drop": 0.95}, Drs={"all": 0.93, "T1": 0.995, "T2drop": 0.95}))
ok("e2e eval-mode disagreement flagged on D2 (registered = re-scored)", "FLAG" in V["D2"] and V["D2"].startswith("WORSE")
   and "eval mode" in V["D2"], f"-> {V['D2']}")
r = batch(); del r["NormStep_s53"]
V, txt = go(r)
ok("e2e VOID missing run", V.get("VOID") and "missing ['NormStep_s53']" in txt)
r = batch(); r["RoPE_s50"]["epochs"] = 450
V, txt = go(r)
ok("e2e VOID short run", V.get("VOID") and "RoPE_s50" in txt)
r = batch(); r["DirOnly_s52"]["geom"]["shift_sc"] = 1e-6
try:
    go(r); ok("e2e VOID DirOnly step", False)
except AssertionError as e:
    ok("e2e VOID DirOnly step (assertion)", "VOID" in str(e))

# ---- collect() end to end, untrained checkpoints, pilot mode (n = 1)
from mapformer.train_tw_statechange import build, make_env
tmp = tempfile.mkdtemp(prefix="twsc_smoke_", dir=os.environ.get("TMPDIR", "/tmp"))
env = make_env(150); torch.set_num_threads(4)
for arm in A.ARMS:
    torch.manual_seed(150); m = build(arm, env)
    d = f"{tmp}/{arm}_s150"; os.makedirs(d)
    args = {"p_take": 0.4, "p_drop": 0.4, "no_state_vocab": False, "size": 64, "n_layers": 1, "seed": 150}
    torch.save({"model_state_dict": m.state_dict(), "losses": list(np.linspace(4, 3, 900)), "arm": arm, "seed": 150,
                "config": {"args": args, "vocab_size": env.unified_vocab_size}}, f"{d}/{arm}.pt")
    json.dump({"eval": {"1024": {"acc": 0.0, "nll": 0}, "2048": {"acc": 0.0, "nll": 0}}}, open(f"{d}/eval.json", "w"))
res, fl = A.collect(tmp, [150], f"{tmp}/out.json")
ok("collect end to end (4 untrained runs)", set(res) == {f"{a}_s150" for a in A.ARMS} and "T2drop" in fl["last_dropped"]
   and res["DirOnly_s150"]["geom"]["shift_sc"] == 0.0 and res["DirOnly_s150"]["L_sc"] == 0.0,
   f"(MapWM acc {res['MapWM_s150']['acc']['intact']['all']:.4f}, reliance {res['MapWM_s150']['reliance']:+.4f}, "
   f"shift_sc {res['MapWM_s150']['geom']['shift_sc']:.3f}; F2 {fl['last_dropped']['T2drop']:.4f})")
print(f"\n{N} cases: {'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)

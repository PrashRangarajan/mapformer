"""Smoke test of every registered branch of analyze_tw_landmark on synthetic per-run data (CPU, no models).
Each scenario builds a TW_LANDMARK.json-shaped dict, runs analyse(), and asserts the expected label prefix."""
import sys

import numpy as np

from mapformer.analyze_tw_landmark import analyse, key, CELLS, SEEDS

rng = np.random.default_rng(1)
FAIL = []


def run(c, r, s, mod=None):
    arm, L, _ = CELLS[c]
    d = {"n_targets": 4617, "cls": "SOLVED", "final_loss": 0.01, "conf_path": 0.9, "conf_name": 0.05}
    if arm == "MapWM":
        d.update(acc_own=0.99, acc_strip=0.99, acc_uninf=0.99, acc_named=0.99, rel_uninf=0.49, rel_strip=0.49,
                 rel_named=0.49, rel_own=0.49, rel_all_uninf=0.49, acc_own_land=0.99, acc_own_unnamed=0.99, move=0.1,
                 opp_minus_common=0.01, name_step=0.05, name_identity_step=0.02, mark_step=0.05, common_over_dir=0.05,
                 drift=2, drift_rad=0.1, channels=64)
    else:
        a = {0.0: 0.75, 0.5: 0.85, 1.0: 0.995}[r]
        d.update(acc_own=a, acc_strip=0.75, acc_uninf=0.70, acc_named=0.99, acc_own_land=0.99 if r else a,
                 acc_own_unnamed=0.65 if r < 1 else float("nan"), cls="STALLED" if r < 1 else "SOLVED", final_loss=0.6)
    d["name_benefit"] = d["acc_named"] - d["acc_uninf"]
    for k in list(d):
        if isinstance(d[k], float) and k not in ("final_loss",) and not np.isnan(d[k]):
            d[k] = d[k] + rng.normal(0, 0.003)
    if mod:
        mod(d, s)
    d["name_benefit"] = d["acc_named"] - d["acc_uninf"]
    return d


def build(mods=None, seeds=SEEDS):
    mods = mods or {}
    return {key(c, r, s): run(c, r, s, mods.get((c, r))) for c, (_a, _l, rs) in CELLS.items() for r in rs for s in seeds}


def lost(d, s=None):
    d.update(rel_uninf=rng.uniform(0, 0.02), acc_strip=rng.uniform(0.5, 0.61), acc_uninf=0.55, acc_named=0.99)


def expect(name, res, checks, seeds=SEEDS):
    lines = []
    V = analyse(res, seeds, out=lines.append)
    ok = all(str(V.get(k, V.get("void", ""))).startswith(v) if not v.startswith("~") else v[1:] in str(V.get(k, ""))
             for k, v in checks.items())
    print(f"{'PASS' if ok else 'FAIL'} {name}: " + "; ".join(f"{k} = {V.get(k, V.get('void'))}" for k in checks))
    if not ok:
        FAIL.append(name); print("\n".join(lines))


# --- O branches (O1 at r = 1; O2 at r = 0.5 left unchanged -> PERSISTS)
expect("baseline: nothing changes", build(), {"O1": "PATH INTEGRATION PERSISTS", "O2": "PATH INTEGRATION PERSISTS"})
expect("overshadow, P1 keeps its map", build({("P2", 1.0): lost}),
       {"O1": "NAMES OVERSHADOW THE MAP", "O2": "PATH INTEGRATION PERSISTS"})
expect("overshadow, attribution", build({("P2", 1.0): lost}), {"O1": "~keeps its map: names in the stream do not"})
expect("overshadow, P1 also loses", build({("P2", 1.0): lost, ("P1", 1.0): lost}), {"O1": "~names ALSO lower"})
expect("overshadow, P1 gate fails", build({("P2", 1.0): lost, ("P1", 0.0): lost, ("P1", 1.0): lost}),
       {"O1": "~P1 qualifier unmeasured"})
expect("learning failure", build({("P2", 1.0): lambda d, s: (lost(d), d.update(acc_named=0.55))}),
       {"O1": "MAP LOST, NAMES NOT USED EITHER"})
expect("overshadow when present", build({("P2", 1.0): lambda d, s: d.update(rel_uninf=rng.uniform(0, 0.02))}),
       {"O1": "NAMES OVERSHADOW WHEN PRESENT"})
expect("form shift", build({("P2", 1.0): lambda d, s: d.update(acc_strip=rng.uniform(0.5, 0.61))}),
       {"O1": "MAP LOST ONLY WHEN NAMES ARE STRIPPED"})
expect("strengthen", build({("P2", 0.0): lambda d, s: d.update(rel_uninf=0.40 + rng.normal(0, 0.003)),
                            ("P2", 0.5): lambda d, s: d.update(rel_uninf=0.40 + rng.normal(0, 0.003))}),
       {"O1": "NAMES STRENGTHEN THE MAP"})
expect("mixed", build({("P2", 1.0): lambda d, s: d.update(rel_uninf=0.3, acc_strip=1.10)}), {"O1": "MIXED"})
expect("gate fails", build({("P2", 0.0): lost}), {"O1": "O UNMEASURED", "O2": "O UNMEASURED"})


def noisy(d, s):
    d.update(rel_uninf=[0.49, 0.10, 0.49, 0.30, 0.49, 0.05][s - 50], acc_strip=[0.99, 0.6, 0.99, 0.8, 0.99, 0.55][s - 50])


def noisy0(d, s):
    d.update(rel_uninf=[0.10, 0.49, 0.30, 0.49, 0.05, 0.49][s - 50], acc_strip=[0.6, 0.99, 0.8, 0.99, 0.55, 0.99][s - 50])


expect("unmeasured (bimodal, no shift)", build({("P2", 1.0): noisy, ("P2", 0.0): noisy0}),
       {"O1": "UNMEASURED (no shift"})
expect("persists at ceiling", build(), {"O1": "~both cells at ceiling"})
expect("persists, names unused", build(), {"O1": "~no cue competition took place"})
expect("persists, names dominate in conflict",
       build({("P2", 1.0): lambda d, s: d.update(conf_name=0.9, conf_path=0.1)}), {"O1": "~cue conflict: names dominate"})

# --- I branches and composite
expect("I: path wins at 0 and 0.5u; ceiling at 1 -> names close the gap", build(),
       {"I0": "PATH WINS", "I05u": "PATH WINS", "I1": "NO DETECTABLE DIFFERENCE (BOTH AT CEILING)",
        "composite": "NAMES CLOSE THE PATH ADVANTAGE"})
expect("I: path wins everywhere -> survives",
       build({("R2", 1.0): lambda d, s: d.update(acc_own=0.80)}), {"I1": "PATH WINS", "composite": "PATH ADVANTAGE SURVIVES"})
expect("I: index wins at 1", build({("P2", 1.0): lambda d, s: d.update(acc_own=0.90)}), {"I1": "INDEX WINS"})
expect("I: Fisher-only (convergence)",
       build({("R2", 0.0): lambda d, s: d.update(acc_own=0.98)}), {"I0": "SOLVED RATE HIGHER FOR PATH"})
expect("I: no difference at 0 -> no composite",
       build({("R2", 0.0): lambda d, s: d.update(acc_own=0.99, cls="SOLVED")}),
       {"I0": "NO DETECTABLE DIFFERENCE", "composite": "NO COMPOSITE"})


def holm_p2(d, s):     # overlapping P2 vs R2 at r = 1 and on unnamed revisits at r = 0.5: p ~0.02-0.04 each
    d.update(acc_own=[0.80, 0.97, 0.78, 0.97, 0.80, 0.97][s - 50], cls="SOLVED",
             acc_own_unnamed=[0.80, 0.97, 0.78, 0.97, 0.80, 0.97][s - 50])


def holm_r2(d, s):
    d.update(acc_own=[0.75, 0.79, 0.75, 0.81, 0.75, 0.79][s - 50], cls="SOLVED",
             acc_own_unnamed=[0.75, 0.79, 0.75, 0.81, 0.75, 0.79][s - 50])


expect("I: fires unadjusted, not after Holm", build({("P2", 1.0): holm_p2, ("R2", 1.0): holm_r2,
                                                     ("P2", 0.5): holm_p2, ("R2", 0.5): holm_r2}),
       {"I1": "~does not survive Holm", "I05u": "~does not survive Holm"})
expect("mixed carries no attribution", build({("P2", 1.0): lambda d, s: d.update(rel_uninf=0.3, acc_strip=1.10)}),
       {"O1": "MIXED"})

# --- voids
res = build(); del res[key("R2", 0.5, 53)]
expect("void: missing run", res, {"O1": "VOID: 1 runs missing"})
res = build(); res[key("P2", 0.5, 50)]["n_targets"] = 4000
expect("void: target sets differ", res, {"O1": "VOID: scored target sets differ"})
res = build(); del res[key("P1", 1.0, 51)]["rel_uninf"]
expect("void: reliance missing", res, {"O1": "VOID: reliance missing"})

print("ALL BRANCHES PASS" if not FAIL else f"FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)

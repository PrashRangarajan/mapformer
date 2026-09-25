"""Tests for stats_guard, ckpt_guard and probe_rewind, against COMMITTED numbers.

    python3 -m mapformer.test_guards            (from /home/prashr, CPU only, ~1 min)

Each test recomputes something already published in a results file and asserts the
published value, so a regression in the utilities -- or a published number that the
data does not support -- fails here rather than in a paper.
"""
from __future__ import annotations

import json
import math
import sys
import time
import traceback
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from mapformer import ckpt_guard as CG
from mapformer import stats_guard as SG
from mapformer.ckpt_guard import REPO

R3 = lambda x: round(float(x), 3)


def _close(x, want, tol=5e-4, what=""):
    assert abs(x - want) <= tol, f"{what}: got {x:+.4f}, want {want:+.4f}"


# ============================================================================ stats_guard
def test_magonly_headline():
    """MAGONLY_RESULTS.md M1/M5/M6 from _MAGONLY.json."""
    acc, loss = SG.load_arms("_MAGONLY.json")
    c = SG.paired(acc["EMDoF_alignfree"], acc["EMDoF_magonly"], "AlignFree - MagOnly")
    assert (R3(c.delta), R3(c.sd), R3(c.mde), c.n_pos, c.n) == (0.146, 0.150, 0.086, 22, 24), str(c)
    assert c.verdict == "DETECTABLE"
    assert c.row() == "| AlignFree - MagOnly | +0.146 | 0.150 | 0.086 | 22/24 | DETECTABLE |", c.row()
    rep = SG.replication_split(c, first_k=8)
    assert (R3(rep.first.delta), rep.first.n_pos, R3(rep.first.mde)) == (0.213, 8, 0.110)
    assert (R3(rep.fresh.delta), rep.fresh.n_pos, R3(rep.fresh.mde)) == (0.113, 14, 0.111)
    assert rep.replicates and not rep.sign_flip          # "DETECTABLE (just)"
    # M2 / M3
    m2 = SG.paired(acc["EMDoF_magonly"], acc["EMDoF_alignlock"])
    assert (R3(m2.delta), R3(m2.mde), m2.n_pos, m2.verdict) == (0.018, 0.089, 13, "unmeasured")
    m3 = SG.paired(acc["EMDoF_magonly"], acc["VanillaEM_P0_r4"])
    assert (R3(m3.delta), R3(m3.mde), m3.n_pos) == (-0.015, 0.083, 12)
    # M6: rule 9 over 96 runs; loss AlignFree - MagOnly -0.410, 23/24 better fit
    r9 = SG.rule9(sum((acc[v] for v in acc), []), sum((loss[v] for v in loss), []))
    assert (R3(r9.r), r9.n) == (-0.978, 96) and not r9.mediator_warning
    lc = SG.paired(loss["EMDoF_alignfree"], loss["EMDoF_magonly"])
    assert (R3(lc.delta), R3(lc.mde), lc.n - lc.n_pos) == (-0.410, 0.233, 23)
    return f"{c}\n      fresh {rep.fresh}"


def test_d5_split():
    """D5_RESULTS.md E1/E2/E3/E5 and AUDIT finding 3, from _D5_N24.json."""
    acc, loss = SG.load_arms("_D5_N24.json")
    c = SG.paired(acc["EMDoF_alignlock"], acc["VanillaEM_P0_r4"], "AlignLock - P0")
    rep = SG.replication_split(c, first_k=8)
    assert (R3(rep.first.delta), rep.first.n_pos) == (0.120, 6), str(rep.first)
    assert (R3(rep.fresh.delta), rep.fresh.n_pos, rep.fresh.n) == (-0.109, 3, 16), str(rep.fresh)
    assert (R3(rep.pooled.delta), rep.pooled.n_pos, R3(rep.pooled.mde)) == (-0.033, 9, 0.116)
    assert rep.sign_flip and not rep.replicates and rep.pooled.verdict == "unmeasured"
    lrep = SG.replication_split(SG.paired(loss["EMDoF_alignlock"], loss["VanillaEM_P0_r4"]))
    assert (R3(lrep.first.delta), R3(lrep.fresh.delta), R3(lrep.pooled.mde)) == (-0.292, 0.290, 0.310)
    assert lrep.sign_flip
    # E5 table
    ph = SG.paired(acc["EMDoF_alignfree"], acc["EMDoF_alignlock"])
    assert (R3(ph.delta), R3(ph.sd), R3(ph.mde), ph.n_pos) == (0.165, 0.155, 0.088, 21)
    tot = SG.paired(acc["VanillaEM_r4"], acc["VanillaEM_P0_r4"])
    assert (R3(tot.delta), R3(tot.sd), R3(tot.mde), tot.n_pos) == (0.128, 0.190, 0.108, 17)
    # AUDIT finding 3: sep - P0 fresh +0.073 (9/16, MDE 0.130); phase fresh +0.173 (13/16, MDE 0.127)
    tr = SG.replication_split(tot)
    assert (R3(tr.first.delta), R3(tr.fresh.delta), tr.fresh.n_pos, R3(tr.fresh.mde)) == (0.237, 0.073, 9, 0.130)
    assert tr.fresh_unmeasured and not tr.sign_flip
    pr = SG.replication_split(ph)
    assert (R3(pr.fresh.delta), pr.fresh.n_pos, R3(pr.fresh.mde)) == (0.173, 13, 0.127) and pr.replicates
    # E3: rule 9 and the loss-matched residuals
    lm1, r9 = SG.loss_matched(acc, loss, "EMDoF_alignlock", "VanillaEM_P0_r4")
    lm2, _ = SG.loss_matched(acc, loss, "EMDoF_alignfree", "EMDoF_alignlock")
    assert (R3(r9.r), r9.n) == (-0.986, 96) and r9.mediator_warning
    assert (R3(lm1.delta), R3(lm1.mde)) == (0.001, 0.021), str(lm1)
    assert (R3(lm2.delta), R3(lm2.mde)) == (-0.009, 0.015), str(lm2)
    return rep.report()


def test_summary_index_is_seed():
    """The _*.json lists are read as index == seed; verify against the per-run JSON."""
    acc, _ = SG.load_arms("_MAGONLY.json")
    for arm in ("EMDoF_alignfree", "EMDoF_magonly"):
        for s in (0, 9, 23):
            r = json.load(open(REPO / f"runs/dof/recency/{arm}_s{s}/{arm}_recency.json"))
            assert r["1024"]["acc"] == acc[arm][s], (arm, s)
    return "index == seed for 2 arms x seeds {0, 9, 23}"


def test_stats_edge_cases():
    # unmeasured, never "null"
    c = SG.from_diffs([0.01, -0.02, 0.03, 0.0], "tiny")
    assert c.verdict == "unmeasured"
    # interaction MDE formula
    d1, d2 = [0.3, 0.5, 0.4, 0.6], [0.1, 0.0, 0.2, 0.1]
    it = SG.interaction(d1, d2, "x")
    want = 2.8 * math.sqrt(np.var(d1, ddof=1) / 4 + np.var(d2, ddof=1) / 4)
    _close(it.mde, want, 1e-12, "interaction MDE"); _close(it.delta, 0.35, 1e-12, "delta")
    itp = SG.interaction(d1, d2, "x", paired=True)
    assert itp.paired and abs(itp.delta - 0.35) < 1e-12
    # dropped seeds are reported, not silent
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = SG.paired({0: 1.0, 1: 2.0, 2: 3.0}, {0: 0.0, 1: 1.0, 3: 9.0})
        assert c.n == 2 and any("DROPPED" in str(x.message) for x in w)
    return "unmeasured / interaction / dropped-seed warning"


# ============================================================================ ckpt_guard
def test_determinism_pair():
    """AUDIT finding 10: dof/recency and recency_em P0 s0 are the same computation."""
    c = CG.compare_checkpoints("runs/dof/recency/VanillaEM_P0_r4_s0/VanillaEM_P0_r4_recency.pt",
                               "runs/recency_em/VanillaEM_P0_r4_s0/VanillaEM_P0_r4_recency.pt")
    assert c.identical and c.verdict == "DETERMINISM" and c.losses_equal, c.report()
    # MagOnly's in-batch determinism control, stored as DETERMINISM.txt "REUSE LICENSED"
    c2 = CG.compare_checkpoints("runs/magonly_repro/EMDoF_alignfree_s0/EMDoF_alignfree_recency.pt",
                                "runs/dof/recency/EMDoF_alignfree_s0/EMDoF_alignfree_recency.pt")
    assert c2.verdict == "DETERMINISM", c2.report()
    # a genuinely different pair must NOT be called determinism
    c3 = CG.compare_checkpoints("runs/dof/recency/VanillaEM_P0_r4_s0/VanillaEM_P0_r4_recency.pt",
                                "runs/dof/recency/VanillaEM_P0_r4_s1/VanillaEM_P0_r4_recency.pt")
    assert c3.verdict == "DIFFERENT" and c3.differing
    return c.report()


def test_u1_shared_tensors():
    """UNFREEZE U1: EMUnf_0 s0 == EMWarm_train s0 on all 27 shared tensors + loss curve."""
    c = CG.compare_checkpoints("runs/unfreeze/EMUnf_0_s0/EMUnf_0_recency.pt",
                               "runs/warm/EMWarm_train_s0/EMWarm_train_recency.pt",
                               shared_only=True)
    assert c.identical and c.n_compared == 27, c.report()
    assert sorted(c.only_a) == ["traj_lat", "traj_leak", "traj_slope"], c.only_a
    return f"{c.n_compared} shared tensors bitwise equal; extra in A: {c.only_a}"


def test_nan_aware():
    """Two identical NoLeak models at init: torch.equal says NO (NaN traj buffers)."""
    from mapformer.train_variant import VARIANT_MAP
    def mk():
        torch.manual_seed(0); np.random.seed(0)
        return VARIANT_MAP["EMNoLeak_e8"](vocab_size=89)
    a, b = mk().state_dict(), mk().state_dict()
    naive = all(torch.equal(a[k], b[k]) for k in a)
    assert not naive, "expected the naive check to false-alarm on NaN buffers"
    c = CG.compare_state_dicts(a, b)
    assert c.weights_equal, c.report()
    b2 = {k: v.clone() for k, v in b.items()}
    b2["traj_slope"][3] = 0.0                 # a NaN where A has NaN -> must differ
    b2["p0_pos"].view(-1)[0] += 1e-7
    c2 = CG.compare_state_dicts(a, b2)
    assert set(c2.differing) == {"traj_slope", "p0_pos"}, c2.differing
    z = torch.zeros(3); nz = z.clone(); nz[1] = -0.0
    assert not CG.tensors_equal(z, nz)[0], "-0.0 vs +0.0 must count as a bitwise difference"
    return "naive torch.equal: False (false alarm); NaN-aware: equal; perturbations detected"


def test_layout_guard():
    """eval_noise_refine's layout against a recency run dir: must fail LOUDLY."""
    try:
        CG.require_checkpoints("runs/dof/recency", "noise_refine", ["VanillaEM_P0_r4"], [0, 1],
                               noises=[0.0])
    except CG.CheckpointLayoutError as e:
        msg = str(e)
        assert "p0/VanillaEM_P0_r4_s0/VanillaEM_P0_r4.pt" in msg, msg
        assert "'recency'" in msg and "finds 2" in msg, msg
    else:
        raise AssertionError("wrong layout did not raise")
    f = CG.require_checkpoints("runs/dof/recency", "recency", ["VanillaEM_P0_r4"], range(24))
    assert len(f.found) == 24 and not f.missing
    assert CG.noise_tag(0.0) == "p0" and CG.noise_tag(0.1) == "p01" and CG.noise_tag(0.25) == "p025"
    return msg.splitlines()[0] + " ... (layouts that WOULD match: 'recency')"


def _recency_batch(T=192, B=2, seed=0):
    from mapformer.environment_recency import RecencyWorld
    env = RecencyWorld(k_max=64, seed=seed)
    toks, *_ = env.generate_batch(B, T, np.random.RandomState(seed))
    return toks


def _train_steps(m, toks, n=3):
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
    m.train()
    for _ in range(n):
        logits = m(toks[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), toks[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()


def test_manipulation_checks():
    from mapformer.train_variant import VARIANT_MAP
    from mapformer.environment_recency import RecencyWorld
    toks = _recency_batch()
    # (a) EMWarm_freeze: the position pathway is frozen, the rest trains
    torch.manual_seed(0)
    m = VARIANT_MAP["EMWarm_freeze"](vocab_size=89)
    before = CG.snapshot(m)
    _train_steps(m, toks)
    path = ["token_emb.latent", "action_to_lie.w_in.weight", "action_to_lie.w_out.weight",
            "path_integrator.omega", "p0_pos"]
    CG.assert_frozen_unchanged(before, m, path)
    CG.assert_moved(before, m, ["token_emb.base.weight", "layers.0.q_content.weight"])
    CG.assert_zero(m.token_emb.base.weight, (slice(None), slice(0, 2)), "base latent coords")
    try:
        CG.assert_frozen_unchanged(before, m, ["layers.0.q_content.weight"])
        raise RuntimeError("guard failed to detect a moved tensor")
    except AssertionError:
        pass
    # (b) EMNoLeak_e8: masked gradient keeps w_in's content columns exactly zero
    torch.manual_seed(0)
    n = VARIANT_MAP["EMNoLeak_e8"](vocab_size=89)
    b0 = CG.snapshot(n)
    _train_steps(n, toks)
    CG.assert_zero(n.action_to_lie.w_in.weight, (slice(None), slice(2, None)), "w_in content cols")
    assert not CG.tensors_equal(b0["action_to_lie.w_in.weight"][:, :2],
                                n.action_to_lie.w_in.weight[:, :2])[0], "latent cols must move"
    # (c) recording and env construction draw no global RNG; randn does
    CG.assert_no_rng_consumed(n._record, 0)
    CG.assert_no_rng_consumed(RecencyWorld, k_max=64, seed=3)
    for fn in (lambda: torch.randn(2), lambda: np.random.rand(), lambda: __import__("random").random()):
        try:
            CG.assert_no_rng_consumed(fn); raise RuntimeError("RNG use not detected")
        except AssertionError:
            pass
    g = np.random.RandomState(1)
    try:
        CG.assert_no_rng_consumed(g.rand, extra_generators=[g]); raise RuntimeError("missed")
    except AssertionError:
        pass
    # (d) same function at init: NoLeak_e8 == Unf_0_e8 (bitwise); MagOnly == AlignFree (atol)
    x = toks[:, :-1]
    def mk(v, s=0):
        torch.manual_seed(s); np.random.seed(s)
        return VARIANT_MAP[v](vocab_size=89)
    CG.assert_same_function_at_init(mk("EMNoLeak_e8"), mk("EMUnf_0_e8"), x)
    CG.assert_same_function_at_init(mk("EMUnf_0"), mk("EMWarm_train"), x)
    dm = CG.assert_same_function_at_init(mk("EMDoF_magonly"), mk("EMDoF_alignfree"), x, atol=1e-5)
    try:
        CG.assert_same_function_at_init(mk("EMUnf_0", 0), mk("EMUnf_0", 1), x)
        raise RuntimeError("different seeds passed")
    except AssertionError:
        pass
    return f"freeze held, mask held, RNG checks, init-equivalence (MagOnly vs AlignFree max|d| {dm:.1e})"


# ============================================================================ probe_rewind
def test_probe_rewind_named():
    from mapformer.probe_rewind import probe_checkpoint
    w = probe_checkpoint(REPO / "runs/warm/EMWarm_freeze_s0/EMWarm_freeze_recency.pt")
    _close(w["slope"], -1.0, 1e-6, "EMWarm_freeze s0 slope")
    _close(w["slope_latpath"], -1.0, 1e-6, "latpath")
    assert w["leak"] == 0.0 and w["form"] == "single_p0"
    assert w["sel_sum"] == 1.0 and all(h == 1.0 for h in w["sel_per_head"]), w["sel_per_head"]
    v = probe_checkpoint(REPO / "runs/dof/recency/VanillaEM_P0_r4_s0/VanillaEM_P0_r4_recency.pt")
    assert abs(v["slope"]) < 0.02, v["slope"]
    assert v["sel_sum"] < 0.5 and v["block_below_m05"] == 0
    return (f"EMWarm_freeze s0 slope {w['slope']:+.6f}, A_P sel {w['sel_sum']:.3f} (n={w['n_queries']});"
            f" VanillaEM_P0_r4 s0 slope {v['slope']:+.4f}, A_P sel {v['sel_sum']:.3f} "
            f"(floor {v['floor']:.3f})")


def test_probe_rewind_matches_stored():
    """Every slope in _REWIND_PROBE.json (7 arms x 8 seeds; three origin forms)."""
    from mapformer.probe_rewind import load_model, delta_table, rewind_slope
    stored = json.load(open(REPO / "_REWIND_PROBE.json"))
    where = {"EMWarm_freeze": "runs/warm", "EMWarm_train": "runs/warm"}
    worst, forms = 0.0, set()
    from mapformer.probe_rewind import position_origins
    for arm, rows in stored.items():
        for s, row in enumerate(rows):
            d = where.get(arm, "runs/dof/recency")
            m, _, env = load_model(REPO / f"{d}/{arm}_s{s}/{arm}_recency.pt")
            sl = rewind_slope(delta_table(m), env)
            worst = max(worst, abs(sl - row[0]))
            forms.add(position_origins(m)[2])
    assert worst < 1e-6, f"max |slope - stored| {worst:.2e}"
    assert forms == {"single_p0", "separate", "property"}, forms
    return f"56 slopes reproduced, max |diff| {worst:.1e}; origin forms {sorted(forms)}"


def test_probe_latent_pathway():
    """UNFREEZE correction table, EMUnf_0_e8: eff -0.602, latpath -0.866 (endpoint).

    Its 'coordinate 0 only' column (-0.989) is NOT an endpoint recompute like the other
    two: it is the RECORDED traj_lat at the last epoch boundary (mean -0.9888). The
    endpoint value is -0.9884 -> -0.988. Both are asserted, so the distinction is kept.
    """
    from mapformer.probe_rewind import probe_checkpoint
    paths = [REPO / f"runs/unfreeze/EMUnf_0_e8_s{s}/EMUnf_0_e8_recency.pt" for s in range(8)]
    rs = [probe_checkpoint(p, select=False) for p in paths]
    eff = np.mean([r["slope"] for r in rs]); lp = np.mean([r["slope_latpath"] for r in rs])
    c0 = np.mean([r["slope_coord0"] for r in rs])
    _close(eff, -0.602, 5e-4, "effective"); _close(lp, -0.866, 5e-4, "latent pathway")
    _close(c0, -0.988, 5e-4, "coordinate 0 (endpoint)")
    rec = np.mean([float(CG.load_checkpoint(p).state["traj_lat"][-1]) for p in paths])
    _close(rec, -0.989, 5e-4, "coordinate 0 (recorded traj_lat[-1])")
    lo, hi = min(r["slope_latpath"] for r in rs), max(r["slope_latpath"] for r in rs)
    assert (round(lo, 2), round(hi, 2)) == (-1.10, -0.43), (lo, hi)
    return (f"eff {eff:+.3f}, latpath {lp:+.3f} (seeds {lo:+.2f}..{hi:+.2f}), "
            f"coord0 endpoint {c0:+.4f} / recorded {rec:+.4f}")


def test_probe_leaves_rng():
    from mapformer.probe_rewind import probe_checkpoint
    CG.assert_no_rng_consumed(probe_checkpoint,
                              REPO / "runs/dof/recency/EMDoF_alignlock_s0/EMDoF_alignlock_recency.pt",
                              n_episodes=4)
    return "probe_checkpoint consumed no caller RNG"


# ============================================================================ runner
TESTS = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]


def main():
    fails = 0
    t00 = time.time()
    for t in TESTS:
        t0 = time.time()
        try:
            info = t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
            if info:
                print("      " + str(info).replace("\n", "\n      "))
        except Exception:
            fails += 1
            print(f"FAIL {t.__name__} ({time.time() - t0:.1f}s)")
            traceback.print_exc()
    print(f"\n{len(TESTS) - fails}/{len(TESTS)} passed in {time.time() - t00:.1f}s")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

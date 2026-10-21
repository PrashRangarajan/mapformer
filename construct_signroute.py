"""Rule 14 existence for the G-2 (SIGNROUTE) arms: construct a solution IN each class before
reading any training outcome.

Why this has to happen before the batch (the user's requirement, and rule 14): a failure is
only evidence about SEARCH if the class can represent a solution at all. `wrap_construct`
(`model_wrap_exist.py`) did this for the |.| class; this file does it for the two new arms, and
the two answers are not the same.

  hybrid     A solution EXISTS and is built here from a TRAINED signed source (pilot seed 1880):
             its content weights and omega are copied, and each move's signed step D(.) is split
             into positive and negative parts placed in separate non-negative readout columns,
                 |W|[:,0] = D(x)^+/c, |W|[:,1] = D(x)^-/c, z_x = c(e0 - e1)  ->  |W| z_x = D(x)
                 |W|[:,2] = D(y)^+/c, |W|[:,3] = D(y)^-/c, z_y = c(e2 - e3)  ->  |W| z_y = D(y)
             so every angle equals the source's while the readout matrix is non-negative. The
             latents here are ORTHOGONAL, not antipodal: the construction does not use the collective
             route, and does not need to -- existence is the question. (Amendment 1 finding 9: an
             earlier docstring described a 2-column antipodal build that the code no longer does.)

  abs_tied   NO solution exists, and this is a proof, not a measurement artefact. Under the tie
             the two move steps are EQUAL -- step(y) = |W . (-z_x)| = |W . z_x| = step(x), by
             evenness -- so after n_x moves east and n_y moves west the angle is
                 theta_c = omega_c * step(x)_c * (n_x + n_y),
             a function of the TOTAL number of moves only. Net displacement n_x - n_y does not
             appear, so no choice of (W_in, W_out, omega) represents the ring map: the class is a
             pure clock in the sense of S1. `test_abs_tied_cannot_represent_the_map` measures the
             consequence (two cells reached by different net displacement get an identical
             position code) rather than trusting the algebra.

             CONSEQUENCE FOR THE PREREGISTRATION, and it narrows what G-2 can claim: abs_tied's
             predicted failure is a REPRESENTABILITY failure, not a search failure. It is a
             construct-validity control that pins the floor -- it shows the collective route is
             what carries the map -- and it must NOT be reported as "the route is hard to find".
             SEARCH_THEORY Sec. 6 item 4 calls it "predicted to fail", which is right, but the
             runbook's "decisive" framing belongs to the hybrid arm, whose solution exists and
             can therefore be found or missed.

  cd /home/prashr && python3 -m mapformer.construct_signroute
"""
import math
import sys
import time
import traceback

import numpy as np
import torch

from mapformer.environment_steps import GridWorldSteps
from mapformer.model_signroute import ARMS
from mapformer.train_cancel import evaluate

SIZE, D, H, R = 32, 128, 2, 4
DELTAS = (1, -1)


def _env(seed=10000):
    return GridWorldSteps(size=SIZE, seed=seed, p_first=0.5, deltas=DELTAS)


def _mk(arm, vocab, seed):
    torch.manual_seed(seed)
    return ARMS[arm](vocab_size=vocab, d_model=D, n_heads=H, n_layers=1, grid_size=SIZE,
                     action_ids=(0, 1))


PILOT = _REPO_SRC = "runs/signroute_pilot/signed_lr1e-3_s1880/signed.pt"


def load_trained_signed(path=None):
    """A TRAINED SignedAO_r4 ring model. The content path has to be trained for an existence test
    to mean anything: a correct position code on a random transformer scores 0.03, so the first
    version of this file measured nothing (the construction held the angles to 0.00e+00 rad and
    still scored 0.0315). `wrap_construct` takes a trained signed model for the same reason."""
    import os
    p = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), PILOT)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"no trained signed ring source at {p}. Train one first (pilot seed, OUTSIDE the "
            f"registered batch, rule 29):\n  python3 -m mapformer.train_signroute --arm signed "
            f"--lr 1e-3 --seed 1880 --device cuda:1 --output-dir mapformer/{os.path.dirname(PILOT)}")
    b = torch.load(p, map_location="cpu", weights_only=False)
    c = b["config"]
    m = ARMS["signed"](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                       n_layers=c["n_layers"], grid_size=c["grid_size"],
                       action_ids=(c["action_offset"], c["action_offset"] + 1))
    m.load_state_dict(b["model_state_dict"])
    return m.eval(), b


def construct_hybrid(src=None):
    """A hybrid model with the trained signed model's EXACT angles and content, readout >= 0.

    The hybrid step is |W_out| z, so an exact reproduction needs non-negative columns whose
    signed combination gives each move's step. With r = 4 and two moves there is room to do it
    column by column: split each move's step into positive and negative parts,

        |W|[:, 0] = D(x)^+ / c   |W|[:, 1] = D(x)^- / c   z_x = c(e0 - e1)  ->  |W| z_x = D(x)
        |W|[:, 2] = D(y)^+ / c   |W|[:, 3] = D(y)^- / c   z_y = c(e2 - e3)  ->  |W| z_y = D(y)

    so every angle, omega and content weight equals the source's and the readout is confined to
    the non-negative orthant. This construction does NOT sit on the collective route (z_x and z_y
    are orthogonal here, not antipodal); it does not need to. Existence is the question -- whether
    training FINDS a solution is what the batch measures.
    """
    if src is None:
        src, _ = load_trained_signed()
    m = ARMS["hybrid"](vocab_size=src.token_emb.weight.shape[0], d_model=D, n_heads=H,
                       n_layers=1, grid_size=SIZE,
                       action_ids=tuple(int(i) for i in src.act))
    C = H * src.n_blocks
    with torch.no_grad():
        om = src.path_integrator.omega.detach().reshape(-1).double()
        Dm = src.step_table()[src.act].reshape(2, -1).double()         # (2, C) signed steps
        e = src.token_emb.weight.detach().double()
        c = float(e[src.act].norm(dim=1).mean())
        W_out = torch.zeros(C, R, dtype=torch.float64)
        W_out[:, 0] = Dm[0].clamp_min(0) / c
        W_out[:, 1] = (-Dm[0]).clamp_min(0) / c
        W_out[:, 2] = Dm[1].clamp_min(0) / c
        W_out[:, 3] = (-Dm[1]).clamp_min(0) / c
        L = torch.zeros(2, R, dtype=torch.float64)
        L[0, 0], L[0, 1] = c, -c
        L[1, 2], L[1, 3] = c, -c
        W_in = (torch.linalg.pinv(e[src.act]) @ L).T
        sd = {k: v.detach().clone() for k, v in src.state_dict().items()}
        sd["action_to_lie.w_in.weight"] = W_in.to(torch.float32)
        sd["action_to_lie.w_out.weight"] = W_out.to(torch.float32)
        m.load_state_dict(sd)
    return m, src


# ------------------------------------------------------------------------------------- tests
def test_hybrid_construction_reproduces_the_signed_angles():
    m, src = construct_hybrid()
    with torch.no_grad():
        a_h = (m.path_integrator.omega * m.step_table()[m.act]).reshape(2, -1).double()
        a_s = (src.path_integrator.omega * src.step_table()[src.act]).reshape(2, -1).double()
        err = float(torch.angle(torch.exp(1j * (a_h - a_s))).abs().max())
        clock_h = float(m.clock().abs().max())
        clock_s = float(src.clock().abs().max())
        wneg = float((m.action_to_lie.w_out.weight < 0).float().mean())
        wneg_s = float((src.action_to_lie.w_out.weight < 0).float().mean())
        blind = float(m.step_table()[m.act].abs().max())
    assert err < 1e-4, f"hybrid angles differ from the signed source by {err:.3e} rad"
    # The construction reproduces the SOURCE, so it inherits the source's clock rather than
    # sitting on u = 0; z_x and z_y are orthogonal here by design (see construct_hybrid). The
    # check is that the clock is inherited, not that it vanishes.
    assert abs(clock_h - clock_s) < 1e-3, (f"hybrid clock {clock_h:.3e} != source clock "
                                           f"{clock_s:.3e}: the angles were not inherited")
    assert blind > 1e-3, f"constructed hybrid is blind (max|step| {blind:.3e})"
    assert wneg == 0.0, f"the hybrid readout must be non-negative, {100*wneg:.1f}% entries are < 0"
    assert wneg_s > 0.1, (f"the signed source readout is already non-negative ({100*wneg_s:.1f}% "
                          f"negative) -- then the hybrid constraint is vacuous at this source")
    sc = {k: (float(getattr(m.action_to_lie, k).weight.abs().max()),
              float(getattr(src.action_to_lie, k).weight.abs().max())) for k in ("w_in", "w_out")}
    for k, (b, a) in sc.items():
        assert b < 20 * max(a, 1e-6), (f"constructed {k} scale {b:.3g} vs source {a:.3g}: a construction far "
                                       f"off the trained scale is not one training would reach (rule 14, 15)")
    return (f"weight scale max|w_in| {sc['w_in'][0]:.3f} (source {sc['w_in'][1]:.3f}), max|w_out| "
            f"{sc['w_out'][0]:.3f} (source {sc['w_out'][1]:.3f}); "
            f"angles match the trained source to {err:.2e} rad; clock inherited "
            f"{clock_h:.3f} vs source {clock_s:.3f}; readout {100*wneg:.0f}% negative vs source "
            f"{100*wneg_s:.0f}%; max|step| {blind:.3f}")


def test_hybrid_construction_scores_like_the_source():
    """Rule 14: the FROZEN construction must actually solve the task, not merely hold the angles."""
    m, src = construct_hybrid()
    out = {}
    for nm, mod in (("hybrid_built", m), ("signed_built", src)):
        acc, nll = evaluate(mod, _env(), 128, 200, "cpu", seed=10 ** 6)
        out[nm] = (acc, nll)
    ah, ash = out["hybrid_built"][0], out["signed_built"][0]
    assert ash > 0.95, (f"the TRAINED signed source scores {ash:.4f}; it must solve the ring before "
                        f"it can serve as the content for an existence test")
    assert abs(ah - ash) < 0.02, (f"frozen hybrid {ah:.4f} vs its trained signed source {ash:.4f}: "
                                  f"the construction does not inherit the source's solution")
    return (f"frozen hybrid {ah:.4f} (nll {out['hybrid_built'][1]:.4f}) == trained signed source "
            f"{ash:.4f} (nll {out['signed_built'][1]:.4f}) -- a solution EXISTS in the hybrid class")


def test_abs_tied_cannot_represent_the_map():
    """The tie makes the code a function of the MOVE COUNT only, so cells at different net
    displacement collide. Measured over many random readouts, in angle space."""
    V = _env().unified_vocab_size
    worst = 0.0
    for s in range(8):
        m = _mk("abs_tied", V, s)
        with torch.no_grad():
            st = m.step_table()[m.act].reshape(2, -1).double()
            assert torch.equal(st[0], st[1]), "the two tied move steps must be identical"
            om = m.path_integrator.omega.detach().reshape(-1).double()
            # cell +1 reached by (1 east, 0 west) vs cell -1 by (0 east, 1 west): different net
            # displacement, same move count -> identical angle, hence an identical position code.
            a_east = om * st[0]
            a_west = om * st[1]
            worst = max(worst, float(torch.angle(torch.exp(1j * (a_east - a_west))).abs().max()))
    assert worst < 1e-6, (f"east and west produce different angles (max {worst:.3e}); the tie "
                          f"is not collapsing them and this control does not do its job")
    return (f"east and west angles identical to {worst:.2e} rad over 8 seeds: the code depends on "
            f"n_east + n_west only, so NO solution exists in the abs_tied class (floor control, "
            f"not a search test)")


def report_abs_tied_untrained_accuracy():
    """REPORT, not a test (Amendment 1 finding 9: it could not fail, so it is not counted). Context only:
    Measured: train-free evaluation of the constructed-angle abs_tied model against the ring
    floors already on record (blank ~0.50, n-gram 0.554, retrace 0.860, LRSWEEP_RESULTS.md)."""
    V = _env().unified_vocab_size
    accs = []
    for s in range(3):
        m = _mk("abs_tied", V, s)
        acc, _ = evaluate(m, _env(), 128, 120, "cpu", seed=10 ** 6)
        accs.append(acc)
    a = float(np.mean(accs))
    assert a < 0.95, f"an untrained abs_tied model already scores {a:.4f}; the floor claim is wrong"
    return (f"untrained abs_tied mean accuracy {a:.4f} over 3 seeds (ring floors on record: blank "
            f"~0.50, n-gram 0.554, retrace 0.860) -- reported as context, the structural argument "
            f"above is what rules the class out")


TESTS = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]


def save_construction(path):
    """Write the hybrid construction as a checkpoint train_signroute --init-from accepts (the rule 14
    trainable warm start, SIGNROUTE_PREREG.md Sec. 7). Amendment 1 finding 9: this used to be an inline
    snippet; it is committed so the warm start is reproducible."""
    import os
    m, src = construct_hybrid()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_state_dict": m.state_dict(), "losses": [], "variant": "hybrid", "seed": 1880,
                "config": {"vocab_size": m.token_emb.weight.shape[0], "d_model": D, "n_heads": H,
                           "n_layers": 1, "grid_size": SIZE, "action_offset": int(m.act[0]),
                           "deltas": DELTAS}}, path)
    print("saved", path)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--save":
        save_construction(sys.argv[2])
        return
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
    try:
        print("REPORT report_abs_tied_untrained_accuracy:\n      " + report_abs_tied_untrained_accuracy())
    except Exception:
        traceback.print_exc()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

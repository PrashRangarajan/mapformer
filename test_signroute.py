"""Unit checks for the G-2 (SIGNROUTE) arms, before any GPU time.

Every claim `model_signroute.py` makes in prose is measured here, including the two that
decide the experiment -- that `hybrid` HAS the collective route of Prop. S2 and that
`abs_tied` does NOT. A test that cannot fail is worth nothing (rule 29 found "unfailable
tests" in an earlier batch), so each route test also checks the OPPOSITE arm fails it.

  cd /home/prashr && python3 -m mapformer.test_signroute
"""
import sys
import time
import traceback

import torch

from mapformer.model_signroute import ARMS, _signroute_variant
from mapformer import model_wrap_exist as MWE

V, D, H, R = 40, 128, 2, 4
ACT = (0, 1)


def _mk(name, seed=0, **kw):
    torch.manual_seed(seed)
    return ARMS[name](vocab_size=V, d_model=D, n_heads=H, n_layers=1, grid_size=32,
                      action_ids=ACT, **kw).eval()


# --------------------------------------------------------------------- wiring parity
def test_signed_matches_existing_AO():
    """The new `signed` arm must be bitwise-identical to model_wrap_exist.SignedAO_r4."""
    torch.manual_seed(7)
    new = ARMS["signed"](vocab_size=V, d_model=D, n_heads=H, n_layers=1, grid_size=32,
                         action_ids=ACT).eval()
    torch.manual_seed(7)
    old = MWE.MapFormerWM_SignedAO_r4(V, d_model=D, n_heads=H, n_layers=1, grid_size=32,
                                      action_ids=ACT).eval()
    ks_new, ks_old = set(new.state_dict()), set(old.state_dict())
    assert ks_new == ks_old, f"state-dict keys differ: {ks_new ^ ks_old}"
    for k in sorted(ks_new):
        assert torch.equal(new.state_dict()[k], old.state_dict()[k]), f"init differs at {k}"
    tok = torch.randint(0, V, (3, 16))
    with torch.no_grad():
        a, b = new(tok), old(tok)
        sa, sb = new.step_table(), old.step_table()
    assert torch.equal(sa, sb), f"step_table differs, max|d| {(sa - sb).abs().max():.3e}"
    assert torch.equal(a, b), f"forward differs, max|d| {(a - b).abs().max():.3e}"
    return "state dict, step_table and forward all bitwise-equal to SignedAO_r4"


def test_abs_matches_existing_AO():
    torch.manual_seed(11)
    new = ARMS["abs"](vocab_size=V, d_model=D, n_heads=H, n_layers=1, grid_size=32,
                      action_ids=ACT).eval()
    torch.manual_seed(11)
    old = MWE.MapFormerWM_AbsAO_r4(V, d_model=D, n_heads=H, n_layers=1, grid_size=32,
                                   action_ids=ACT).eval()
    tok = torch.randint(0, V, (3, 16))
    with torch.no_grad():
        assert torch.equal(new.step_table(), old.step_table()), "step_table differs"
        assert torch.equal(new(tok), old(tok)), "forward differs"
    return "bitwise-equal to AbsAO_r4"


def test_param_parity():
    """No arm may add or remove a parameter: a capacity difference would be a confound."""
    counts, shapes = {}, {}
    for nm in ARMS:
        m = _mk(nm)
        counts[nm] = sum(p.numel() for p in m.parameters())
        shapes[nm] = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    base = counts["signed"]
    bad = {k: v for k, v in counts.items() if v != base}
    assert not bad, f"parameter count differs from signed={base:,}: {bad}"
    for nm, sh in shapes.items():
        assert sh == shapes["signed"], f"{nm} has different tensor shapes"
    return f"all {len(ARMS)} arms at {base:,} params, identical shapes"


# ------------------------------------------------------- observation-step-free by construction
def test_observation_steps_are_exactly_zero():
    """Only the two move tokens may step. Everything else must be EXACTLY 0 (not merely small):
    S2/S3's inverse-pair algebra holds exactly only in the observation-step-free construction."""
    rows = []
    for nm in ARMS:
        m = _mk(nm, seed=3)
        with torch.no_grad():
            st = m.step_table()
        non_move = [i for i in range(V) if i not in ACT]
        mx = st[non_move].abs().max().item()
        assert mx == 0.0, f"{nm}: non-move step is {mx:.3e}, not exactly 0"
        mv = st[list(ACT)].abs().max().item()
        assert mv > 0, f"{nm}: move steps are all zero -- the arm is blind at init"
        rows.append(f"{nm} move max {mv:.3f}")
    return "non-move steps exactly 0 in every arm; " + ", ".join(rows)


# ---------------------------------------------------------------- the decisive route tests
def test_hybrid_has_the_collective_route():
    """S2(a) with W -> |W|: setting z_y = -z_x must zero the clock on EVERY channel while the
    steps themselves stay non-zero (non-blind). The same manipulation on `abs` must NOT zero it
    -- that is S3(a), and it is why this test can fail."""
    out = {}
    for nm in ("hybrid", "abs"):
        m = _mk(nm, seed=5)
        a2l = m.action_to_lie
        torch.manual_seed(99)
        z_x = torch.randn(1, R)
        with torch.no_grad():
            zz = torch.cat([z_x, -z_x], 0)            # the collective move u = z_x + z_y = 0
            a = a2l.from_latent(zz)                   # (2, H*nb)
            clock = (a[0] + a[1]).abs().max().item()  # pre-omega; omega != 0 so this decides
            blind = a[0].abs().max().item()
        out[nm] = (clock, blind)
    hc, hb = out["hybrid"]
    ac, ab = out["abs"]
    assert hc < 1e-6, f"hybrid clock at u=0 is {hc:.3e}, expected ~0 (the route must exist)"
    assert hb > 1e-3, f"hybrid is blind at u=0 (max|step| {hb:.3e}) -- a zero clock there is vacuous"
    assert ac > 1e-3, (f"abs clock at u=0 is {ac:.3e}: if abs ALSO cancels collectively, this "
                       f"test cannot distinguish the arms and S3(a) is not what is being tested")
    return (f"hybrid: clock {hc:.2e} with steps {hb:.3f} (route exists, non-blind) | "
            f"abs: clock {ac:.3f} at the same latents (S3(a): the route ends at a(y)=a(x))")


def test_abs_evenness_is_why_abs_fails():
    """S3(a) verbatim: under |.| the collective latent move ends at a(y) = a(x) in EVERY channel."""
    m = _mk("abs", seed=5)
    torch.manual_seed(100)
    z_x = torch.randn(1, R)
    with torch.no_grad():
        a = m.action_to_lie.from_latent(torch.cat([z_x, -z_x], 0))
    d = (a[0] - a[1]).abs().max().item()
    assert d < 1e-6, f"a(y) != a(x) under abs at z_y=-z_x (max|d| {d:.3e}) -- evenness broken"
    return f"a(y) == a(x) on all {a.shape[1]} channels (max|d| {d:.2e}): the route makes a clock"


def test_abs_tied_has_no_route():
    """abs_tied's clock is 2 omega |w . z_x|, so it vanishes only blindly. Checked over many
    random readouts/latents: the clock is zero only where the step itself is zero."""
    worst_ratio, n = 1e9, 0
    for s in range(12):
        m = _mk("abs_tied", seed=s)
        with torch.no_grad():
            st = m.step_table()[list(ACT)]                 # (2, H, nb)
            assert torch.equal(st[0], st[1]), "tied arm: the two move steps must be equal"
            clock = m.clock().abs()                        # (H, nb), = 2*omega*|w.z|
            step = st[0].abs()
            om = m.path_integrator.omega.abs()
            # exact identity clock == 2 * omega * |step|
            ident = (clock - 2 * om * step).abs().max().item()
            assert ident < 1e-5, f"seed {s}: clock != 2*omega*|step| (max|d| {ident:.3e})"
            live = step > 1e-6
            n += int(live.sum())
            if live.any():
                worst_ratio = min(worst_ratio, (clock[live] / (2 * om * step)[live]).min().item())
    assert worst_ratio > 0.99, f"a non-blind channel had a cancelling clock (ratio {worst_ratio:.3f})"
    return (f"clock == 2*omega*|step| exactly over 12 seeds; no non-blind channel cancels "
            f"({n} live channels checked, min ratio {worst_ratio:.3f})")


def test_tie_actually_ties():
    m = _mk("abs_tied", seed=2)
    with torch.no_grad():
        z = m.latents()
    d = (z[1] + z[0]).abs().max().item()
    assert d < 1e-6, f"latents not tied: |z_y + z_x| = {d:.3e}"
    u = _mk("abs", seed=2)
    with torch.no_grad():
        zu = u.latents()
    du = (zu[1] + zu[0]).abs().max().item()
    assert du > 1e-3, f"the UNtied arm is also tied at init ({du:.3e}) -- the contrast is empty"
    return f"tied |z_y+z_x| {d:.2e} vs untied {du:.3f}"


# ------------------------------------------------------------------------ gradient wiring
def test_grad_reaches_the_right_places():
    """Every arm: w_in and w_out both receive gradient. (Amendment 1 finding 14: an earlier docstring
    claimed this also measures that abs_tied's second move embedding gets no STEP-path gradient. It
    does not; that follows from step_table reading ea[:1] and is stated, not measured.)"""
    notes = []
    for nm in ("signed", "abs", "hybrid", "abs_tied", "exp", "softplus"):
        m = _mk(nm, seed=1).train()
        tok = torch.randint(0, V, (4, 24))
        m(tok).sum().backward()
        a2l = m.action_to_lie
        gi = a2l.w_in.weight.grad
        go = a2l.w_out.weight.grad
        assert gi is not None and gi.abs().sum() > 0, f"{nm}: w_in got no gradient"
        assert go is not None and go.abs().sum() > 0, f"{nm}: w_out got no gradient"
        notes.append(f"{nm} |g_in| {gi.abs().sum():.2e} |g_out| {go.abs().sum():.2e}")
    return "; ".join(notes)


def test_freeze_move_latents():
    m = _mk("hybrid", seed=4)
    before = m.step_table().detach().clone()
    m.freeze_move_latents()
    after = m.step_table().detach().clone()
    assert torch.allclose(before, after, atol=1e-6), "freezing changed the step table"
    assert not m.action_to_lie.w_in.weight.requires_grad, "w_in still requires grad after freeze"
    m.train()
    tok = torch.randint(0, V, (4, 24))
    m(tok).sum().backward()
    assert m.action_to_lie.w_in.weight.grad is None, "w_in received a gradient while frozen"
    assert m.token_emb.weight.grad is not None and m.token_emb.weight.grad.abs().sum() > 0, \
        "token embeddings got no gradient -- the content path must stay live"
    return "step table unchanged, w_in frozen with no grad, content path still trains"


# ------------------------------------------------------------ measured init behaviour (no assert on prose)
def test_init_step_statistics():
    """MEASURE the init confound rather than assert it: exp and softplus start away from 0, so at
    init every stepping token advances the clock by a near-constant amount (the model is RoPE with
    a rescaled omega), while signed/abs start near 0 (near NoPE). Reported, not judged."""
    rows = []
    for nm in ARMS:
        vals = []
        for s in range(8):
            m = _mk(nm, seed=s)
            with torch.no_grad():
                st = m.step_table()[list(ACT)]
            vals.append(st)
        v = torch.stack(vals)
        rows.append(f"{nm:9s} mean {v.mean():+.4f} sd {v.std():.4f} min {v.min():+.4f} max {v.max():+.4f}")
    return "per-arm move-step stats over 8 seeds:\n" + "\n".join(rows)


def test_sign_census_at_batch_vocab():
    """The prereg's sign table, at the batch's REAL vocabulary (GridWorldSteps ring 32) and with the
    EFFECTIVE readout (|W_out| for the hybrid; the stored W_out's sign is irrelevant there). Amendment 1
    finding 12: the first census used vocab 40 and printed the stored W_out. Reported, not judged."""
    from mapformer.environment_steps import GridWorldSteps
    env = GridWorldSteps(size=32, seed=0, p_first=0.5, deltas=(1, -1))
    Vb, act = env.unified_vocab_size, (env.action_offset, env.action_offset + 1)
    rows = [f"vocab {Vb}, action_ids {act}, 8 seeds; omega's sign is free, so 'row-coherent' is the share "
            f"of rows of diag(omega) W_eff with a single sign"]
    for nm in ARMS:
        neg, wneg, coh = [], [], []
        for s in range(8):
            torch.manual_seed(s)
            m = ARMS[nm](vocab_size=Vb, d_model=D, n_heads=H, n_layers=1, grid_size=32,
                         action_ids=act).eval()
            with torch.no_grad():
                st = m.step_table()[list(act)]
                W = m.action_to_lie.w_out.weight
                We = W.abs() if m.action_to_lie.mode == "hybrid" else W
                rw = m.path_integrator.omega.reshape(-1, 1) * We
            neg.append((st < 0).float().mean()); wneg.append((We < 0).float().mean())
            coh.append((((rw >= 0).all(1)) | ((rw <= 0).all(1))).float().mean())
        rows.append(f"{nm:9s} increment<0 {100*torch.stack(neg).mean():5.1f}%  effective readout<0 "
                    f"{100*torch.stack(wneg).mean():5.1f}%  row-coherent {100*torch.stack(coh).mean():5.1f}%")
    return "\n".join(rows)


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

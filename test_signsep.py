"""Unit checks for the SIGNSEP arms (SIGNSEP_PREREG.md), CPU, before any GPU time. Every route / barrier claim in
`model_signsep.py` is measured here, and each test also checks that the arm it should NOT hold for fails it (rule 29:
no unfailable tests).

  cd /home/prashr && python3 -m mapformer.test_signsep
"""
import math
import sys
import traceback

import numpy as np
import torch

from mapformer import model_wrap_exist as MWE
from mapformer.environment_steps import GridWorldSteps
from mapformer.model_signsep import ARMS, TAU_PAST, angles, build, default_angles, diagnostics
from mapformer.train_wrap_mech import set_move_angles

ENV = GridWorldSteps(size=32, seed=0, p_first=0.5, deltas=(1, -1))
V, ACT = ENV.unified_vocab_size, (ENV.action_offset, ENV.action_offset + 1)


def _tok(n=4, T=64, seed=3):
    np.random.seed(seed)
    return torch.stack([ENV.generate_trajectory(T)[0] for _ in range(n)])


def _b(arm, seed):
    return build(arm, V, ACT, seed=seed, grid_size=32)[0]


def test_anchors_equal_mechanism_start():
    """S and A at the common start are bitwise MECHANISM's SO_absmag / AO_absmag start (model_wrap_exist classes +
    train_wrap_mech.set_move_angles(|default angles|)) at the same seed."""
    for arm, old_cls in (("signed", MWE.MapFormerWM_SignedAO_r4), ("abs", MWE.MapFormerWM_AbsAO_r4)):
        new = _b(arm, 2390).eval()
        torch.manual_seed(2390)
        old = old_cls(V, d_model=128, n_heads=2, n_layers=1, grid_size=32, action_ids=ACT)
        with torch.no_grad():
            st = old.action_to_lie.w_out(old.action_to_lie.w_in(old.token_emb.weight[old.act]))
            ang0 = st * old.path_integrator.omega.reshape(-1)
        set_move_angles(old, ang0.abs()); old.eval()
        sd_n, sd_o = new.state_dict(), old.state_dict()
        assert set(sd_n) == set(sd_o), set(sd_n) ^ set(sd_o)
        for k in sd_n:
            assert torch.equal(sd_n[k], sd_o[k]), f"{arm}: init differs at {k}"
        tok = _tok()
        with torch.no_grad():
            assert torch.equal(new(tok), old(tok)), f"{arm}: forward differs"
    return "signed / abs = MECHANISM's SO_absmag / AO_absmag start, state dict and forward bitwise"


def test_cells_function_identical_at_start():
    tok = _tok()
    for s in (2390, 2391, 2392):
        ms = {a: _b(a, s).eval() for a in ARMS}
        with torch.no_grad():
            ref = ms["signed"](tok)
            for a in ("abs", "abs_wrap", "signed_frozen"):
                d = float((ms[a](tok) - ref).abs().max())
                assert d == 0.0, f"seed {s}: {a} differs from signed at the start by {d:.3e}"
            dp = float((ms["abs_wrap_past"](tok) - ref).abs().max())
            assert dp > 1e-3, "abs_wrap_past must differ from the common start (its tau starts at TAU_PAST / 2)"
        assert float(ms["abs_wrap"].wrap_tau.abs().max()) == 0.0
        assert torch.allclose(ms["abs_wrap_past"].wrap_tau, torch.full((2,), TAU_PAST / 2))
        A = angles(ms["abs"]).double()
        assert float(A.min()) >= 0.0, "the common start must have non-negative angles (balanced |.| start)"
    n = {a: sum(p.numel() for p in ms[a].parameters()) for a in ARMS}
    ntr = {a: sum(p.numel() for p in ms[a].parameters() if p.requires_grad) for a in ARMS}
    assert n["signed"] == n["abs"] == n["signed_frozen"] == n["abs_wrap"] - 2 == n["abs_wrap_past"] - 2, n
    assert ntr["signed_frozen"] == n["signed"] - ms["signed"].action_to_lie.w_in.weight.numel(), ntr
    return f"S, A, W, F logits identical at 3 seeds; W_past differs; params {n}; trainable {ntr}"


def test_abs_wrap_is_sign_coherent():
    """W is a |.| class: in every channel both move angles have the sign of omega_c, for ANY parameters (random W, z,
    tau of either sign, omega of either sign). Its counter-arm (signed) is not."""
    g = torch.Generator().manual_seed(0)
    for arm in ("abs_wrap", "signed"):
        m = _b(arm, 2390)
        with torch.no_grad():
            for p in (m.action_to_lie.w_in.weight, m.action_to_lie.w_out.weight, m.path_integrator.omega):
                p.copy_(torch.randn(p.shape, generator=g))
            if arm == "abs_wrap":
                m.wrap_tau.copy_(3 * torch.randn(2, generator=g))
        A = angles(m)
        share = float(((A[0] * A[1]) >= 0).double().mean())
        if arm == "abs_wrap":
            assert share == 1.0, f"abs_wrap must be sign-coherent per channel, share {share}"
            om = m.path_integrator.omega.reshape(-1)
            assert bool(((A * om) >= 0).all()), "abs_wrap angles must carry omega's sign"
        else:
            assert share < 0.9, f"control: random signed parameters should break coherence (share {share})"
    return "abs_wrap sign-coherent at random parameters (P3, S1 apply); signed at random parameters is not"


def test_collective_wrap_point():
    """On the route's end (z_y = -z_x, tau = (0, 2)) every abs_wrap channel's clock is 2 pi sign(omega) exactly (a wrap
    in EVERY channel through one coordinate); abs at the same latents has clock 2 |a(x)|, not 0 mod 2 pi (S3(a))."""
    m = _b("abs_wrap", 2391); a = _b("abs", 2391)
    for mm in (m, a):
        with torch.no_grad():
            e = mm.token_emb.weight[mm.act].double(); z = e @ mm.action_to_lie.w_in.weight.double().T
            Z = torch.stack([z[0], -z[0]])
            mm.action_to_lie.w_in.weight.copy_((torch.linalg.pinv(e) @ Z).T.float())
    with torch.no_grad():
        m.wrap_tau.copy_(torch.tensor([0.0, 2.0]))
    k = angles(m).double().sum(0)
    om = m.path_integrator.omega.reshape(-1).double()
    err = float((k - 2 * math.pi * torch.sign(om)).abs().max())
    assert err < 1e-5, f"abs_wrap collective point: clock differs from 2 pi by {err:.2e}"
    assert float(m.pre_steps().min()) >= 0, "the collective point must lie on the pre-|.| >= 0 region"
    ka = angles(a).double()
    wa = torch.atan2(torch.sin(ka.sum(0)), torch.cos(ka.sum(0))).abs()
    assert float(wa.median()) > 0.1, f"control: abs at the same latents should NOT cancel (median {wa.median():.3f})"
    return f"abs_wrap clock = 2 pi in every channel to {err:.1e} rad; abs at the same latents median |wrap k| {wa.median():.2f}"


def test_frozen_has_no_collective_direction():
    """F: W_in has no grad and the step path reads a frozen buffer, so after an optimiser step the latents are
    unchanged while W_out moves; each step-path parameter moves one channel's clock. S: the latents move."""
    out = {}
    for arm in ("signed_frozen", "signed"):
        m = _b(arm, 2392).train()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.05)
        z0 = m.latents().detach().clone(); w0 = m.action_to_lie.w_out.weight.detach().clone()
        tok = _tok(n=8, T=64)
        loss = torch.nn.functional.cross_entropy(m(tok[:, :-1]).reshape(-1, V), tok[:, 1:].reshape(-1))
        loss.backward(); opt.step()
        dz = float((m.latents().detach() - z0).abs().max())
        dw = float((m.action_to_lie.w_out.weight.detach() - w0).abs().max())
        out[arm] = (dz, dw)
        # a single channel's clock: does any W_in entry reach it, and does any W_out row other than its own?
        m.zero_grad()
        if arm == "signed":
            m.clock().reshape(-1)[5].backward()
            gin = m.action_to_lie.w_in.weight.grad
            assert gin is not None and float(gin.abs().max()) > 0, "signed: W_in must reach a channel's clock"
        else:
            m.clock().reshape(-1)[5].backward()
            assert m.action_to_lie.w_in.weight.grad is None, "signed_frozen: W_in must get no grad"
            gW = m.action_to_lie.w_out.weight.grad
            rows = (gW.abs().sum(1) > 0).nonzero().reshape(-1).tolist()
            assert rows == [5], f"signed_frozen: channel 5's clock must depend on W_out row 5 only, got {rows}"
    assert out["signed_frozen"][0] == 0.0 and out["signed_frozen"][1] > 0, out
    assert out["signed"][0] > 0, out
    return f"after one AdamW step: frozen latents moved {out['signed_frozen'][0]:.1e}, W_out {out['signed_frozen'][1]:.1e}; signed latents {out['signed'][0]:.1e}"


def test_tau_gets_gradient_and_moves_every_channel():
    m = _b("abs_wrap", 2393)
    m.clock().sum().backward()
    g = float(m.wrap_tau.grad.abs().min())
    assert g > 0, "tau must get gradient"
    m.zero_grad()
    om = m.path_integrator.omega.reshape(-1)
    J = torch.autograd.functional.jacobian(lambda t: _clock_tau(m, t), m.wrap_tau.detach().clone())
    # d k_c / d tau_y = pi sign(omega_c) on the >= 0 region (the start)
    err = float((J[:, 1].double() - math.pi * torch.sign(om).double()).abs().max())
    assert err < 1e-5, f"d clock / d tau_y must be pi sign(omega) in every channel, err {err:.2e}"
    a = _b("abs", 2393)
    assert not hasattr(a, "wrap_tau"), "control: abs has no tau"
    return f"d k_c / d tau_y = pi sign(omega_c) in all {J.shape[0]} channels (err {err:.1e})"


def _clock_tau(m, t):
    old = m.wrap_tau.data.clone()
    from torch.func import functional_call
    params = dict(m.named_parameters()); params["wrap_tau"] = t
    bufs = dict(m.named_buffers())

    class _C(torch.nn.Module):
        def __init__(s, mm):
            super().__init__(); s.mm = mm

        def forward(s):
            return s.mm.clock().reshape(-1)
    out = functional_call(_C(m), {f"mm.{k}": v for k, v in {**params, **bufs}.items()}, ())
    m.wrap_tau.data.copy_(old)
    return out


def test_existence_constructions():
    """Rule 14, frozen: each new arm's construction reproduces the source's angles mod 2 pi and its logits; W's
    construction wraps every channel through tau_y = 2 with every pre-|.| step >= 0."""
    from mapformer.construct_signsep import construct, load_source
    src, vocab, act = load_source()
    tok = _tok(n=4, T=128)
    a_src = angles(src).double()
    msg = []
    for arm in ("abs", "abs_wrap", "signed_frozen"):
        m = construct(arm, src, vocab, act)
        a = angles(m).double()
        err = float(torch.atan2(torch.sin(a - a_src), torch.cos(a - a_src)).abs().max())
        with torch.no_grad():
            ld = float((m(tok) - src(tok)).abs().max())
        assert err < 1e-5 and ld < 1e-2, f"{arm}: angle err {err:.2e}, logit diff {ld:.2e}"
        if arm == "abs_wrap":
            k = a.sum(0)
            assert float(m.pre_steps().min()) >= 0 and bool((k.abs() > math.pi).all()), "W construction must wrap all"
        msg.append(f"{arm} err {err:.1e} logit {ld:.1e}")
    return "; ".join(msg)


def test_diagnostics_run():
    for a in ARMS:
        d = diagnostics(_b(a, 2394))
        assert np.isfinite(d["u_ratio"]) and 0 <= d["cancel_share"] <= 1, (a, d)
        if a.startswith("abs_wrap"):
            assert "tau_sum" in d
    return "diagnostics finite on every arm"


def test_tau_decay_exemption_is_exact():
    """Amendment 2: train_signsep.exempt_from_decay makes AdamW skip tau's weight decay exactly (against a reference
    optimiser with a wd = 0 group for tau) and leaves every other parameter's decay alone; without it tau decays."""
    import copy
    from mapformer.train_signsep import exempt_from_decay
    base = _b("abs_wrap", 2395)
    with torch.no_grad():
        base.wrap_tau.fill_(1.5)
    tok = _tok(n=4, T=32)
    ref, hk, plain = copy.deepcopy(base), copy.deepcopy(base), copy.deepcopy(base)
    others = [p for n, p in ref.named_parameters() if n != "wrap_tau"]
    o_ref = torch.optim.AdamW([{"params": others}, {"params": [ref.wrap_tau], "weight_decay": 0.0}], lr=1e-2,
                              weight_decay=0.05)
    h = exempt_from_decay([hk.wrap_tau])
    try:
        o_hk = torch.optim.AdamW(hk.parameters(), lr=1e-2, weight_decay=0.05)
        for m, o in ((ref, o_ref), (hk, o_hk)):
            torch.manual_seed(0)
            for _ in range(5):
                o.zero_grad(); m(tok).pow(2).mean().backward(); o.step()
    finally:
        for x in h:
            x.remove()
    o_pl = torch.optim.AdamW(plain.parameters(), lr=1e-2, weight_decay=0.05)
    torch.manual_seed(0)
    for _ in range(5):
        o_pl.zero_grad(); plain(tok).pow(2).mean().backward(); o_pl.step()
    d_tau = float((ref.wrap_tau - hk.wrap_tau).abs().max())
    d_all = max(float((a - b).abs().max()) for a, b in zip(ref.parameters(), hk.parameters()))
    d_plain = float((plain.wrap_tau - ref.wrap_tau).abs().max())
    # float32 reassociation (foreach grouping differs between the two optimisers) leaves ~1e-6; the decay effect is ~1e-3
    assert d_plain > 1e-4, f"control: with decay tau should differ ({d_plain})"
    assert d_tau < 0.01 * d_plain and d_all < 3e-5, (f"hooked AdamW differs from the wd=0-group reference: tau {d_tau}, "
                                                     f"any {d_all}, decay effect {d_plain}")
    return f"hooked = reference to {d_all:.1e} (tau {d_tau:.1e}); with decay tau differs by {d_plain:.1e}"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    torch.set_num_threads(4)
    ok = 0
    for t in TESTS:
        try:
            msg = t()
            print(f"  ok   {t.__name__}: {msg}"); ok += 1
        except Exception:
            print(f"  FAIL {t.__name__}"); traceback.print_exc()
    print(f"{ok}/{len(TESTS)} passed")
    sys.exit(0 if ok == len(TESTS) else 1)


if __name__ == "__main__":
    main()

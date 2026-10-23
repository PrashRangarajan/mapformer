"""Unit checks for G-3 (model_ssm_sign.py, construct_ssm_sign.py). CPU, ~1 min.

  cd /home/prashr && python3 -m mapformer.test_ssm_sign
"""
import math
import sys

import numpy as np
import torch

from mapformer.construct_ssm_sign import CLASSES, construct, sensitivity, wrap_err
from mapformer.environment_steps import GridWorldSteps
from mapformer.model_signroute import ARMS as ATTN_ARMS
from mapformer.model_ssm_sign import MODES, STEPS, build

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


def _tokens(size=32, T=48, n=3, seed=0):
    env = GridWorldSteps(size=size, seed=seed, p_first=0.5, deltas=(1, -1))
    np.random.seed(seed)
    return env, torch.stack([env.generate_trajectory(T)[0][:-1] for _ in range(n)])


def _perturb(m, scale=8.0, seed=1):
    """Non-trivial steps (init steps are ~0.1 rad), so the checks are not passed by near-identity rotations."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.action_to_lie.w_out.weight.copy_(torch.randn(m.action_to_lie.w_out.weight.shape, generator=g) * scale / 4)
        m.action_to_lie.w_in.weight.mul_(4.0)
    return m


def test_parallel_equals_recurrence():
    """Rule 19: the cumsum-of-angles (RoPE-trick) form equals the sequential complex recurrence."""
    env, tok = _tokens()
    worst64, worst32 = 0.0, 0.0
    for mode in MODES:
        for steps in STEPS:
            for decay in ("none", "learned"):
                torch.manual_seed(3)
                # exp of a perturbed latent reaches steps ~e^20 and angles where cos() itself is ill-conditioned in
                # float64; a milder perturbation keeps exp's steps O(1-10) like every other arm's
                m = _perturb(build(mode, steps, env.unified_vocab_size, 32, decay=decay),
                             scale=1.0 if mode == "exp" else 8.0).eval()
                if decay == "learned":
                    with torch.no_grad():
                        m.layers[0].a_proj.weight.normal_(0, 0.3)
                ref32 = m.sequential_reference(tok)
                with torch.no_grad():
                    out32 = m(tok)
                md = build(mode, steps, env.unified_vocab_size, 32, decay=decay).double().eval()
                md.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
                with torch.no_grad():
                    out64 = md(tok)
                ref64 = md.sequential_reference(tok)
                s = float(ref64.abs().max())
                worst64 = max(worst64, float((out64 - ref64).abs().max()) / s)
                worst32 = max(worst32, float((out32.double() - ref64).abs().max()) / s)
    check("parallel form == sequential recurrence, float64 (relative)", worst64 < 1e-9, f"max rel {worst64:.1e}")
    check("float32 parallel vs float64 recurrence (relative)", worst32 < 1e-3, f"max rel {worst32:.1e}")


def test_long_sequence_tolerance():
    """At the 4T evaluation length (1023 tokens) in float32, against the float64 recurrence (stated tolerance)."""
    env, tok = _tokens(T=512, n=1)
    torch.manual_seed(4)
    m = build("signed", "moves", env.unified_vocab_size, 32).eval()
    _perturb(m, scale=2.0)
    with torch.no_grad():
        out32 = m(tok)
    md = build("signed", "moves", env.unified_vocab_size, 32).double().eval()
    md.load_state_dict({k: v.double() for k, v in m.state_dict().items()})
    ref = md.sequential_reference(tok)
    rel = float((out32.double() - ref).abs().max() / ref.abs().max())
    phi = float((m.step_table()[tok] * m.path_integrator.omega).sum(1).abs().max())
    check("T=512 float32 vs float64 recurrence", rel < 5e-3, f"max rel {rel:.1e} (max |Phi| {phi:.0f} rad)")


def test_shared_init_matches_attention_arm():
    """Same construction order as MapFormerWM: at a seed, every shared module starts from the attention arm's weights."""
    for mode in ("signed", "abs", "softplus", "exp"):
        torch.manual_seed(2100)
        a = ATTN_ARMS[mode](vocab_size=19, d_model=128, n_heads=2, n_layers=1, grid_size=32, action_ids=(0, 1))
        torch.manual_seed(2100)
        s = build(mode, "moves", 19, 32)
        sa, ss = a.state_dict(), s.state_dict()
        shared = [k for k in ss if k in sa]
        diff = [k for k in shared if not torch.equal(sa[k], ss[k])]
        only = sorted(set(ss) - set(sa))
        check(f"init identical to attention {mode} on {len(shared)} shared tensors", not diff and only == [
            "layers.0.y_norm.weight"], f"diff {diff[:3]} ssm-only {only}")


def test_parameter_parity():
    n = {(mo, st): sum(p.numel() for p in build(mo, st, 19, 32).parameters()) for mo in MODES for st in STEPS}
    check("identical parameter count in every arm", len(set(n.values())) == 1, f"{set(n.values())}")


def test_step_classes():
    env, tok = _tokens()
    for steps in STEPS:
        for mode in MODES:
            torch.manual_seed(5)
            m = _perturb(build(mode, steps, env.unified_vocab_size, 32))
            st = m.step_table().detach()
            obs = [i for i in range(st.shape[0]) if i not in m.act.tolist()]
            if mode == "index":
                ok = bool((st == 1).all())
            elif mode == "signed":
                ok = bool((st[m.act] < 0).any() and (st[m.act] > 0).any())
            elif mode == "abs":
                ok = bool((st >= 0).all())
            else:
                ok = bool((st[m.act] > 0).all())
            if steps == "moves" and mode != "index":
                ok &= bool((st[obs] == 0).all())
            if steps == "all" and mode != "index":
                ok &= bool((st[obs] != 0).any())
            check(f"step class {mode}/{steps}", ok)


def test_causal():
    env, tok = _tokens()
    torch.manual_seed(6)
    m = _perturb(build("signed", "moves", env.unified_vocab_size, 32)).eval()
    t2 = tok.clone()
    t2[:, 40:] = (t2[:, 40:] + 3) % env.unified_vocab_size
    with torch.no_grad():
        d = float((m(tok)[:, :40] - m(t2)[:, :40]).abs().max())
    check("causal (future tokens do not change past logits)", d == 0.0, f"{d:.1e}")


def test_memoryless():
    env, tok = _tokens()
    torch.manual_seed(7)
    m = build("signed", "moves", env.unified_vocab_size, 32, mixer="none").eval()
    t2 = tok.clone()
    t2[:, :20] = (t2[:, :20] + 5) % env.unified_vocab_size
    with torch.no_grad():
        d = float((m(tok)[:, 20:] - m(t2)[:, 20:]).abs().max())
    check("memoryless arm reads only the current token", d == 0.0, f"{d:.1e}")


def test_index_angles():
    env, tok = _tokens()
    m = build("index", "moves", env.unified_vocab_size, 32)
    cos, sin = m.path_integrator(m.step_table()[tok])
    T = tok.shape[1]
    ref = torch.arange(1, T + 1).float().view(1, 1, T, 1) * m.path_integrator.omega.detach().view(1, 2, 1, -1)
    check("index arm: angle = omega * token index", torch.allclose(torch.atan2(sin, cos), torch.atan2(
        torch.sin(ref), torch.cos(ref)), atol=1e-4))


def test_constructions_exact():
    """Every class's construction reproduces the signed angle table mod 2 pi; the step constraint holds; logits equal
    the source (steps = moves) or the signed control (steps = all; observations at 0 mod 2 pi in both)."""
    env, tok = _tokens()
    for steps in STEPS:
        torch.manual_seed(8)
        src = _perturb(build("signed", steps, env.unified_vocab_size, 32), scale=2.0).eval()
        ctl = construct(src, "signed")[0]
        for k in CLASSES:
            m, info = construct(src, k)
            ref = src if steps == "moves" else ctl
            ea = wrap_err(m.angle_table(), ref.angle_table())
            st = m.step_table().detach()
            rows = st if steps == "all" else st[m.act]       # steps = moves: non-moves are 0 by construction
            cons = {"abs": bool((st >= 0).all()), "softplus": bool((rows > 0).all()), "exp": bool((rows > 0).all()),
                    "signed": True}[k]
            with torch.no_grad():
                dl = float((m(tok) - ref(tok)).abs().max())
            check(f"construction {k}/{steps}", ea < 2e-5 and cons and dl < 2e-3 and info["w_in_solve_resid"] < 1e-8,
                  f"angle err {ea:.1e}, logits {dl:.1e}, min step {float(rows.min()):+.3f}, W_in resid "
                  f"{info['w_in_solve_resid']:.1e}")
            if steps == "all" and k in ("softplus", "exp"):
                obs = [i for i in range(st.shape[0]) if i not in m.act.tolist()]
                full = wrap_err(m.angle_table()[obs], torch.zeros_like(m.angle_table()[obs]))
                stp = float(st[obs].mean())
                check(f"{k}/all: annihilated tokens step {stp:.4f} = a full turn", full < 1e-5,
                      f"angle mod 2 pi {full:.1e}")


def test_construction_moves_to_all():
    """D4's full-arm construction: a moves-only source rebuilt in the every-token-steps classes, exact against the source."""
    env, tok = _tokens()
    torch.manual_seed(9)
    src = _perturb(build("signed", "moves", env.unified_vocab_size, 32), scale=2.0).eval()
    for k in CLASSES:
        m, info = construct(src, k, steps="all")
        st = m.step_table().detach()
        obs = [i for i in range(st.shape[0]) if i not in m.act.tolist()]
        ea = wrap_err(m.angle_table(), src.angle_table())
        with torch.no_grad():
            dl = float((m(tok) - src(tok)).abs().max())
        ok = m.steps == "all" and ea < 2e-5 and dl < 2e-3 and (k == "signed" or float(st.min()) >= 0)
        check(f"construction moves -> all {k}", ok, f"angle err {ea:.1e} (obs step {float(st[obs].mean()):.4f}), "
              f"logits {dl:.1e}, lambda {info['balance_lambda']:.3f}")


def test_balance_invariant():
    """The W_in / W_out rescaling leaves every step unchanged and equalises the two maxima."""
    env, tok = _tokens()
    torch.manual_seed(10)
    src = _perturb(build("signed", "moves", env.unified_vocab_size, 32), scale=2.0).eval()
    for k in CLASSES:
        a, _ = construct(src, k, balance=False)
        b, ib = construct(src, k, balance=True)
        d = float((a.step_table() - b.step_table()).abs().max())
        eq = abs(ib["max_abs_w_in"] - ib["max_abs_w_out"]) < 1e-6 * ib["max_abs_w_in"]
        check(f"balance leaves {k} steps unchanged", d < 1e-6 and eq, f"{d:.1e}")


def test_sensitivity():
    """Amendment 1.2: the move-angle sensitivity recorded in report.json is finite and positive for every construction."""
    env, tok = _tokens()
    torch.manual_seed(11)
    src = _perturb(build("signed", "moves", env.unified_vocab_size, 32), scale=2.0).eval()
    vals = {k: sensitivity(construct(src, k)[0]) for k in CLASSES}
    ok = all(np.isfinite(m) and m > 0 and mx >= m for m, mx in vals.values())
    check("sensitivity finite and positive", ok, " ".join(f"{k} {v[0]:.1f}" for k, v in vals.items()))


def main():
    for f in (test_parallel_equals_recurrence, test_long_sequence_tolerance, test_shared_init_matches_attention_arm,
              test_parameter_parity, test_step_classes, test_causal, test_memoryless, test_index_angles,
              test_constructions_exact, test_construction_moves_to_all, test_balance_invariant,
              test_sensitivity):
        print(f.__name__, flush=True)
        f()
    n_ok = sum(ok for _, ok in RESULTS)
    print(f"test_ssm_sign: {n_ok}/{len(RESULTS)} PASS")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()

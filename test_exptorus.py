"""Unit checks for EXPTORUS (EXPTORUS_PREREG.md), before any GPU time. CPU, ~1 min.

  cd /home/prashr && python3 -m mapformer.test_exptorus

What is measured, not asserted in prose: the exp / softplus torus arms are wired exactly like the signed / |.| anchors (the
signroute `signed` mode with the torus's four moves is bitwise SignedAO4_r4); at one seed all four arms start from bitwise
identical weights; only the four move tokens step; the step signs are what each class says; the analysis's init
reconstruction equals what train_variant builds (a checkpoint trained at lr 0); the runtime registration cannot shadow a
different class. Each check is written so that it can fail (rule 29: unfailable tests were found before).
"""
import os
import subprocess
import sys
import tempfile
import time
import traceback

import torch

from mapformer import model_exptorus as MX
from mapformer import model_wrap_exist as MWE
from mapformer.model_signroute import _signroute_variant

V, D, H = 21, 128, 2                 # torus vocabulary: 4 moves + 17 observation tokens (blank + 16 types)
ARMS = {"SO": MWE.MapFormerWM_SignedAO4_r4, "AO": MWE.MapFormerWM_AbsAO4_r4, "EXP": MX.MapFormerWM_ExpAO4_r4,
        "SPL": MX.MapFormerWM_SoftplusAO4_r4}


def _mk(cls, seed=0, **kw):
    torch.manual_seed(seed)
    return cls(vocab_size=V, d_model=D, n_heads=H, n_layers=1, grid_size=64, **kw).eval()


def test_register_idempotent_and_refuses_shadowing():
    from mapformer.train_variant import VARIANT_MAP
    MX.register(); MX.register()
    assert VARIANT_MAP["ExpAO4_r4"] is MX.MapFormerWM_ExpAO4_r4
    assert VARIANT_MAP["SoftplusAO4_r4"] is MX.MapFormerWM_SoftplusAO4_r4
    assert VARIANT_MAP["SignedAO4_r4"] is MWE.MapFormerWM_SignedAO4_r4      # anchors untouched
    VARIANT_MAP["ExpAO4_r4"] = MWE.MapFormerWM_AbsAO4_r4
    try:
        MX.register()
    except RuntimeError:
        pass
    else:
        raise AssertionError("register() overrode a different class registered under ExpAO4_r4")
    finally:
        VARIANT_MAP["ExpAO4_r4"] = MX.MapFormerWM_ExpAO4_r4
    return "idempotent; refuses to shadow"


def test_signroute_signed_with_four_moves_is_SignedAO4():
    """The exp / softplus classes come from model_signroute; its `signed` mode with the torus's four moves must be bitwise
    the anchor SignedAO4_r4 (state dict, step table and forward), so the arms differ ONLY in the readout function."""
    S4 = _signroute_variant("signed")
    new = _mk(S4, 7, action_ids=MX.TORUS_MOVES); old = _mk(MWE.MapFormerWM_SignedAO4_r4, 7)
    assert set(new.state_dict()) == set(old.state_dict())
    for k in new.state_dict():
        assert torch.equal(new.state_dict()[k], old.state_dict()[k]), k
    tok = torch.randint(0, V, (3, 64))
    with torch.no_grad():
        assert torch.equal(new.step_table(), old.step_table())
        assert torch.equal(new(tok), old(tok))
    A4 = _signroute_variant("abs")
    new = _mk(A4, 7, action_ids=MX.TORUS_MOVES); old = _mk(MWE.MapFormerWM_AbsAO4_r4, 7)
    with torch.no_grad():
        assert torch.equal(new.step_table(), old.step_table()) and torch.equal(new(tok), old(tok))
    return "signroute signed / abs with moves 0-3 == SignedAO4_r4 / AbsAO4_r4 bitwise (state dict, steps, forward)"


def test_same_initial_weights_across_arms():
    for seed in (0, 3400):
        sd = {a: _mk(c, seed).state_dict() for a, c in ARMS.items()}
        for a in ARMS:
            assert set(sd[a]) == set(sd["SO"]), a
            for k in sd["SO"]:
                assert torch.equal(sd[a][k], sd["SO"][k]), (seed, a, k)
    # and the readout really differs: the four step tables are not equal
    st = {a: _mk(c, 1).step_table() for a, c in ARMS.items()}
    assert not torch.equal(st["EXP"], st["SPL"]) and not torch.equal(st["SO"], st["AO"])
    return "bitwise-identical state dicts at seeds 0 and 3400; step tables differ by the readout only"


def test_only_moves_step():
    rows = []
    for a, c in ARMS.items():
        m = _mk(c, 3)
        with torch.no_grad():
            st = m.step_table()
        other = [i for i in range(V) if i not in MX.TORUS_MOVES]
        assert torch.count_nonzero(st[other]) == 0, a
        assert torch.count_nonzero(st[list(MX.TORUS_MOVES)]) > 0, a
        rows.append(a)
    return "observation / blank steps exactly 0 for " + ", ".join(rows)


def test_step_signs():
    out = []
    for a, c in ARMS.items():
        st = torch.cat([_mk(c, s).step_table()[list(MX.TORUS_MOVES)].detach().reshape(-1) for s in range(4)])
        neg = float((st < 0).double().mean())
        if a == "SO":
            assert 0.3 < neg < 0.7, neg
        elif a == "AO":
            assert neg == 0.0 and float(st.min()) >= 0.0
        else:
            assert float(st.min()) > 0.0, (a, float(st.min()))           # strictly positive
        out.append(f"{a} neg {100 * neg:.1f}% mean {float(st.mean()):+.3f}")
    return "; ".join(out)


def test_param_parity_and_gradients():
    n = {a: sum(p.numel() for p in _mk(c).parameters()) for a, c in ARMS.items()}
    assert len(set(n.values())) == 1, n
    for a in ("EXP", "SPL"):
        m = _mk(ARMS[a]).train()
        tok = torch.randint(0, V, (2, 32))
        m(tok).sum().backward()
        for nm in ("action_to_lie.w_in.weight", "action_to_lie.w_out.weight", "path_integrator.omega"):
            g = dict(m.named_parameters())[nm].grad
            assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, (a, nm)
    return f"{n['SO']:,} params in every arm; exp / softplus gradients reach w_in, w_out, omega"


def test_init_reconstruction_equals_trainer():
    """analyze_exptorus.init_model must equal what train_variant (via train_exptorus) builds at the seed. A run at lr 0
    saves the initial weights unchanged (AdamW: decay lr x wd = 0, step 0); compared bitwise for the exp and signed arms."""
    from mapformer.analyze_exptorus import init_model, VAR
    out = []
    with tempfile.TemporaryDirectory() as d:
        for arm in ("EXP", "SO"):
            od = os.path.join(d, arm)
            cmd = [sys.executable, "-m", "mapformer.train_exptorus", "--variant", VAR[arm], "--seed", "3497", "--lr", "0",
                   "--weight-decay", "0.05", "--n-batches", "1", "--batch-size", "2", "--n-steps", "16", "--n-layers", "1",
                   "--n-heads", "2", "--d-model", "128", "--n-landmarks", "0", "--schedule", "cosine", "--epochs", "1",
                   "--fast-attn", "--device", "cpu", "--output-dir", od]
            r = subprocess.run(cmd, cwd="/home/prashr", capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            assert r.returncode == 0, r.stderr[-2000:]
            b = torch.load(os.path.join(od, f"{VAR[arm]}.pt"), map_location="cpu", weights_only=False)
            m0 = init_model(arm, 3497).state_dict()
            for k, v in b["model_state_dict"].items():
                assert torch.equal(v, m0[k]), (arm, k)
            out.append(arm)
    return "init_model == train_exptorus's lr-0 checkpoint, bitwise, for " + ", ".join(out)


TESTS = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]


def main():
    fails = 0; t00 = time.time()
    for t in TESTS:
        t0 = time.time()
        try:
            info = t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)" + (f"\n      {info}" if info else ""))
        except Exception:
            fails += 1
            print(f"FAIL {t.__name__} ({time.time() - t0:.1f}s)"); traceback.print_exc()
    print(f"\n{len(TESTS) - fails}/{len(TESTS)} passed in {time.time() - t00:.1f}s")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

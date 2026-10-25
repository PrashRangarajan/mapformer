"""CPU unit checks for the image-observation torus (IMAGE_TORUS_PREREG.md), before any GPU time. Each check that a
property holds is paired with a check that the arm or condition it should NOT hold for fails it (rule 29: no
unfailable tests).

  cd /home/prashr && python3 -m mapformer.test_image_torus
"""
import sys
import traceback

import numpy as np
import torch

from mapformer import image_bank
from mapformer import model_image as MI
from mapformer import eval_image_torus as EV
from mapformer.environment import GridWorld

SYN = image_bank.synthetic_banks(n_per_class=(5, 3), fill=(10, 200))


def _walk_tokens(seed=0, T=48, B=3, env_seed=5):
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=env_seed)
    np.random.seed(seed)
    tok, _o, rev, _ = env.generate_batch(B, T)
    return tok, rev


class _EncIn:
    """Records the encoder's input (pixel values, x255) on each forward."""

    def __init__(self, m):
        self.x = None
        MI.image_emb(m).encoder.register_forward_pre_hook(self._h)

    def _h(self, mod, inp):
        self.x = (inp[0] * 255).round().long()


def test_bank_real():
    b = image_bank.load_banks()
    assert tuple(b["train"]["images"].shape) == (96000, 28, 28) and tuple(b["test"]["images"].shape) == (16000, 28, 28)
    for s in ("train", "test"):
        L = b[s]["labels"]; off, cnt = b[s]["off"], b[s]["cnt"]
        for k in range(16):
            assert bool((L[off[k]:off[k] + cnt[k]] == k).all()), (s, k)
    # the two splits share no image (exact byte match), on the full sets (Amendment 1, n7)
    h = lambda X: set(map(bytes, X.reshape(len(X), -1).numpy()))
    inter = h(b["train"]["images"]) & h(b["test"]["images"])
    assert len(inter) == 0, len(inter)
    assert image_bank.bank_md5() == "72cdafaf92a9a42b1b6266ff72423e09"


def test_train_split_only_in_training_and_test_at_eval():
    m = MI.build("MapWM", "image", 0, banks=SYN); rec = _EncIn(m)
    tok, _ = _walk_tokens()
    inp = tok[:, :-1]
    obs = inp[:, 1::2]; is_obj = (obs >= 4) & (obs < 20)
    m.train(); m(inp)
    v = rec.x.view(obs.shape[0], obs.shape[1], 28, 28)[..., 0, 0]
    assert bool((v[is_obj] == 10 + (obs[is_obj] - 4)).all()), "training must draw the TRAIN split, right class"
    assert bool((v[~is_obj] == 0).all()), "blank slots must get the zero image"
    MI.set_split(m, "test")
    try:
        m(inp); raise AssertionError("test split accepted in training mode")
    except RuntimeError:
        pass
    m.eval(); m(inp)
    v = rec.x.view(obs.shape[0], obs.shape[1], 28, 28)[..., 0, 0]
    assert bool((v[is_obj] == 200 + (obs[is_obj] - 4)).all()), "eval with split test must draw the TEST split"
    MI.set_split(m, None); m(inp)                                    # auto: eval mode -> test
    v = rec.x.view(obs.shape[0], obs.shape[1], 28, 28)[..., 0, 0]
    assert bool((v[is_obj] == 200 + (obs[is_obj] - 4)).all()), "auto split in eval mode must be TEST"
    MI.set_split(m, "train"); m(inp)                                 # an explicit train-images readout
    v = rec.x.view(obs.shape[0], obs.shape[1], 28, 28)[..., 0, 0]
    assert bool((v[is_obj] == 10 + (obs[is_obj] - 4)).all())
    MI.set_split(m, None); m.train(); m(inp)                        # auto in training -> train
    v = rec.x.view(obs.shape[0], obs.shape[1], 28, 28)[..., 0, 0]
    assert bool((v[is_obj] == 10 + (obs[is_obj] - 4)).all())
    e = MI.image_emb(m)
    assert e.draws["train"] == 3 * obs.numel() and e.draws["test"] == 2 * obs.numel(), e.draws


def test_fresh_instance_every_visit_and_paired_reseed():
    b = image_bank.load_banks()
    m = MI.build("MapWM", "image", 0, banks=b).eval(); e = MI.image_emb(m)
    tok, _ = _walk_tokens(T=256)
    inp = tok[:, :-1]
    e.reseed(5); m(inp); i1 = e.last_idx.clone()
    m(inp); i2 = e.last_idx.clone()
    obj = (inp[:, 1::2] >= 4) & (inp[:, 1::2] < 20)
    assert float((i1[obj] != i2[obj]).float().mean()) > 0.99, "a second forward must draw fresh instances"
    # within one forward, repeated visits to the same class get different instances
    cls = inp[:, 1::2][obj]; idx = i1[obj]
    same = [(idx[cls == c].unique().numel(), int((cls == c).sum())) for c in range(4, 20) if int((cls == c).sum()) > 5]
    U, N = sum(u for u, _ in same), sum(n for _, n in same)
    assert U >= 0.95 * N, same            # ~1000 instances per class: birthday collisions only (expected ~1%)
    assert len(same) >= 10
    e.reseed(5); m(inp); assert torch.equal(e.last_idx, i1), "reseed must reproduce the draw (arm pairing)"
    # drawn instances have the right class in the real bank
    lab = b["test"]["labels"][i1[obj]]
    assert torch.equal(lab, cls - 4)
    # pairing across arms: same seed, same walk -> same images
    m2 = MI.build("RoPE", "image", 0, banks=b).eval(); e2 = MI.image_emb(m2)
    e2.reseed(5); m2(inp); assert torch.equal(e2.last_idx, i1)


def test_revisit_mask_and_stream_unchanged():
    """train_image builds GridWorld(64, 16, 0.5, seed); its batches equal a fresh GridWorld's under the same np seed,
    and the image wrap never touches the tokens / targets train() uses."""
    import mapformer.train_image as TI
    for s in (3100, 3107):
        a = TI.make_env(s)                                   # the trainer's own construction (Amendment 1, m5)
        st = np.random.get_state(); TI.check_interleaving(a, 1024)
        assert all((x == y).all() if hasattr(x, "all") else x == y for x, y in zip(st, np.random.get_state()))
        b = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=s)
        np.random.seed(s); ta, oa, ra, _ = a.generate_batch(4, 512)
        np.random.seed(s); tb, ob, rb, _ = b.generate_batch(4, 512)
        assert torch.equal(ta, tb) and torch.equal(ra, rb) and torch.equal(oa, ob)
        assert int(ta.max()) < MI.VOCAB
    m = MI.build("MapWM", "image", 0, banks=SYN)
    tok, _ = _walk_tokens(); before = tok.clone(); m.train(); out = m(tok[:, :-1])
    assert torch.equal(tok, before) and out.shape[-1] == MI.VOCAB
    # negative control: a different seed's map differs
    assert not torch.equal(GridWorld(64, 16, 0.5, seed=3100).obs_map, GridWorld(64, 16, 0.5, seed=3101).obs_map)


def test_step_map_input_per_arm():
    b = image_bank.load_banks()
    tok, _ = _walk_tokens(T=64)
    inp = tok[:, :-1]
    obj = torch.zeros_like(inp, dtype=torch.bool); obj[:, 1::2] = (inp[:, 1::2] >= 4) & (inp[:, 1::2] < 20)
    act = inp < 4
    res = {}
    for arm in ("MapWM", "ActOnly", "NormStep"):
        m = MI.build(arm, "image", 0, banks=b).eval(); e = MI.image_emb(m)
        with torch.no_grad():
            e.reseed(1); d1 = MI.step_of(m, inp); e.reseed(2); d2 = MI.step_of(m, inp)
        res[arm] = (d1, d2)
        moved = (d1 - d2).abs().flatten(2).amax(-1)                  # per position: did the image instance change it?
        assert float(moved[act].max()) == 0.0, f"{arm}: action steps must not depend on images"
        if arm == "ActOnly":
            assert float(d1[obj].abs().max()) == 0.0 and float(d1[act].abs().max()) > 0, "ActOnly: obs steps 0"
        else:
            assert float(moved[obj].min()) > 0, f"{arm}: object steps must read the image (per-instance)"
    # NormStep reads LayerNorm(emb): invariant to a scale of the embedding; MapWM is not (negative control)
    for arm, inv in (("NormStep", True), ("MapWM", False)):
        m = MI.build(arm, "image", 0, banks=b).eval()
        with torch.no_grad():
            x = m.token_emb.tok(inp)
            f = (lambda z: m.step(inp, z)) if arm == "NormStep" else m.action_to_lie
            same = torch.allclose(f(x), f(3 * x), atol=1e-5)
        assert same == inv, (arm, same)
    m = MI.build("RoPE", "image", 0, banks=b)
    assert not hasattr(m, "action_to_lie") and MI.step_of(m, inp) is None


def test_token_twin_is_train_variant_vanilla_r4_and_pairing():
    from mapformer.train_variant import VARIANT_MAP
    s = 3100
    torch.manual_seed(s); ref = VARIANT_MAP["Vanilla_r4"](vocab_size=21, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    rng_ref = torch.get_rng_state()
    tw = MI.build("MapWM", "token", s); rng_tw = torch.get_rng_state()
    for k, v in ref.state_dict().items():
        assert torch.equal(v, tw.state_dict()[k]), k
    assert torch.equal(rng_ref, rng_tw)
    im = MI.build("MapWM", "image", s, banks=SYN); rng_im = torch.get_rng_state()
    assert torch.equal(rng_im, rng_ref), "the encoder init must not consume the global RNG (dropout stream)"
    sd_im = im.state_dict()
    for k, v in ref.state_dict().items():
        k2 = k.replace("token_emb.", "token_emb.tok.")
        assert torch.equal(v, sd_im[k2]), k
    enc = {a: MI.build(a, "image", s, banks=SYN).token_emb.encoder.state_dict() for a in MI.ARMS}
    for a in MI.ARMS:
        for k, v in enc["MapWM"].items():
            assert torch.equal(v, enc[a][k]), (a, k)
    other = MI.build("MapWM", "image", s + 1, banks=SYN).token_emb.encoder.state_dict()
    assert not torch.equal(other["net.0.weight"], enc["MapWM"]["net.0.weight"]), "seed must change the encoder"
    # base weights shared by the LEAK arms at a seed (model_codes builds them on the r=4 base)
    for a in ("ActOnly", "NormStep"):
        sd = MI.build(a, "image", s, banks=SYN).state_dict()
        for k, v in sd_im.items():
            assert torch.equal(v, sd[k]), (a, k)


def test_checkpoint_roundtrip_excludes_banks():
    m = MI.build("ActOnly", "image", 7, banks=SYN)
    sd = m.state_dict()
    assert not any("imgs_" in k or "off_" in k or "cnt_" in k for k in sd), [k for k in sd if "imgs" in k]
    m2 = MI.build("ActOnly", "image", 8, banks=SYN, seed_rng=False)
    m2.load_state_dict(sd)                                            # strict
    tok, _ = _walk_tokens()
    m.eval(); m2.eval(); MI.image_emb(m).reseed(3); MI.image_emb(m2).reseed(3)
    with torch.no_grad():
        assert torch.equal(m(tok[:, :-1]), m2(tok[:, :-1]))


def test_strict_interleaving():
    m = MI.build("MapWM", "image", 0, banks=SYN).eval()
    tok, _ = _walk_tokens()
    bad = tok[:, :-1].clone(); bad[0, 0] = 7                         # an observation token at an action slot
    try:
        m(bad); raise AssertionError("strict check missed an observation at an even position")
    except ValueError:
        pass
    m.train(); m(bad)                                                # training skips the check (no host sync)


def test_eval_readouts_split_and_leak_wiring():
    W = EV.walks(10000, 1234, 6, 48)
    for arm in ("MapWM", "ActOnly"):
        m = MI.build(arm, "image", 0, banks=SYN).eval(); e = MI.image_emb(m)
        d0 = dict(e.draws); r = EV.score(m, W, 0, "cpu", "test"); d1 = dict(e.draws)
        assert d1["train"] == d0["train"] and d1["test"] > d0["test"], (d0, d1)
        EV.score(m, W, 0, "cpu", "train"); d2 = dict(e.draws)
        assert d2["train"] > d1["train"] and e.split is None and e.active_split() == "test"
        # zeroing object steps: a no-op for ActOnly (exact), not for MapWM at init
        with torch.no_grad():
            tok = torch.stack([w[0] for w in W])[:, :-1]
            e.reseed(9); a = m(tok)
            z = EV._ZeroObj(m); z.keep = (tok < 4) | (tok == 20); e.reseed(9); b = m(tok); z.close()
        assert torch.equal(a, b) == (arm == "ActOnly"), arm
        assert 0 <= r["acc"] <= 1 and r["n"] > 0


def test_shuffle_keeps_actions_and_multiset():
    tok, _ = _walk_tokens(T=64)
    inp = tok[:, :-1]
    sh = EV.shuffle_obs(inp, np.random.default_rng(0))
    assert torch.equal(sh[:, 0::2], inp[:, 0::2])
    for r in range(inp.shape[0]):
        assert torch.equal(sh[r, 1::2].sort().values, inp[r, 1::2].sort().values)
    assert not torch.equal(sh, inp)


class _Rec(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.seen = []

    def forward(self, tok):
        self.seen.append(tok.cpu().clone())
        return torch.zeros(tok.shape[0], tok.shape[1], MI.VOCAB)


def test_eval_stream_equals_eval_noise_refine():
    """eval_image_torus.walks is eval_noise_refine's stream, checked by CALLING eval_noise_refine.evaluate with a stub
    model that records the inputs it is given (Amendment 1, m5)."""
    from mapformer import eval_noise_refine as ENR
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
    rec = _Rec()
    ENR.evaluate(rec, env, 128, 3, 0.0, torch.device("cpu"), seed=1234 + 3100)
    W = EV.walks(10000, 1234 + 3100, 3, 128)
    assert len(rec.seen) == 3
    for x, (t, _r) in zip(rec.seen, W):
        assert torch.equal(x[0], t[:-1])
    W2 = EV.walks(10000, 1234 + 3101, 3, 128)                     # negative control: another seed's walks differ
    assert not torch.equal(rec.seen[0][0], W2[0][0][:-1])


def test_encoder_gain_fixed_and_calibrated():
    g, norms = MI.calibrate_gain()
    assert abs(g - MI.ENC_GAIN) / MI.ENC_GAIN < 1e-4, (g, MI.ENC_GAIN)
    for arm in MI.ARMS:
        m = MI.build(arm, "image", 3100, banks=SYN)
        e = MI.image_emb(m)
        assert "enc_gain" in dict(e.named_buffers()) and "enc_gain" not in dict(e.named_parameters())
        assert abs(float(e.enc_gain) - MI.ENC_GAIN) < 1e-5 and "token_emb.enc_gain" in m.state_dict()
    # init: an image observation embedding has about a token row's norm (sqrt(128) = 11.3); without the gain ~0.4
    b = image_bank.load_banks()
    m = MI.build("MapWM", "image", 3100, banks=b).eval()
    tok, _ = _walk_tokens(T=256); inp = tok[:, :-1]
    with torch.no_grad():
        x = m.token_emb(inp)
    obj = (inp >= 4) & (inp < 20)
    n_obj, n_act = float(x[obj].norm(dim=1).mean()), float(x[inp < 4].norm(dim=1).mean())
    assert 0.7 < n_obj / n_act < 1.5, (n_obj, n_act)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    torch.set_num_threads(4)
    bad = 0
    for t in TESTS:
        try:
            t(); print(f"[ok]   {t.__name__}")
        except Exception:
            bad += 1; print(f"[FAIL] {t.__name__}"); traceback.print_exc()
    print(f"{len(TESTS) - bad}/{len(TESTS)} passed")
    sys.exit(1 if bad else 0)

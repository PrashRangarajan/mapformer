"""CPU unit checks for RNN_TORUS_PREREG.md, before any GPU time. Each property check is paired with a case that must fail it
(rule 29: no unfailable tests).

  cd /home/prashr && python3 -m mapformer.test_rnn_torus            (all checks incl. the CPU smoke of every arm)
  cd /home/prashr && python3 -m mapformer.test_rnn_torus --fast     (skip the CPU smoke)
"""
import json, os, re, subprocess, sys, tempfile, traceback

import numpy as np
import torch
import torch.nn as nn

from mapformer import model_rnn as MR
from mapformer import analyze_rnn_torus as AR
from mapformer.environment import GridWorld

REPO = "/home/prashr/mapformer"


def test_param_counts():
    for k, (c, l, h) in MR.ARM_SPECS.items():
        m = MR.RNN_VARIANTS[k](MR.VOCAB, n_layers=l)
        n = sum(p.numel() for p in m.parameters())
        assert n == MR.n_params(c, h, l), (k, n, MR.n_params(c, h, l))
        tgt = MR.TARGET_PATH if n < 500_000 else MR.TARGET_ROPE4
        assert MR.match_hidden(c, l, tgt) == h, (k, MR.match_hidden(c, l, tgt))
        assert abs(n - tgt) / tgt < 0.01, (k, n, tgt)
    # negative: the repo's fixed-width LSTM (h = 128) is NOT matched to the path model
    from mapformer.model_baselines_extra import LSTMBaseline
    n0 = sum(p.numel() for p in LSTMBaseline(MR.VOCAB).parameters())
    assert abs(n0 - MR.TARGET_PATH) / MR.TARGET_PATH > 0.3, n0


def test_targets_are_the_anchor_counts():
    from mapformer.train_variant import VARIANT_MAP
    c = lambda v, nl: sum(p.numel() for p in VARIANT_MAP[v](vocab_size=MR.VOCAB, d_model=128, n_heads=2, n_layers=nl,
                                                            grid_size=64).parameters())
    assert c("Vanilla_r4", 1) == MR.TARGET_PATH and c("RoPE", 4) == MR.TARGET_ROPE4
    assert GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=0).unified_vocab_size == MR.VOCAB


def test_causal():
    torch.manual_seed(0)
    tok = torch.randint(0, MR.VOCAB, (2, 40))
    for k, (c, l, h) in MR.ARM_SPECS.items():
        m = MR.RNN_VARIANTS[k](MR.VOCAB, n_layers=l).eval()
        a = m(tok); t2 = tok.clone(); t2[:, 25] = (t2[:, 25] + 1) % MR.VOCAB; b = m(t2)
        assert a.shape == (2, 40, MR.VOCAB)
        assert torch.equal(a[:, :25], b[:, :25]) and not torch.allclose(a[:, 25:], b[:, 25:]), k
    # negative control: a bidirectional LSTM fails the same check
    bi = nn.LSTM(8, 8, batch_first=True, bidirectional=True)
    x = torch.randn(1, 40, 8); y = x.clone(); y[:, 25] += 1
    assert not torch.allclose(bi(x)[0][:, :25], bi(y)[0][:, :25])


def test_forget_bias_and_shapes():
    m = MR.RNN_VARIANTS["LSTM_L2_h111"](MR.VOCAB, n_layers=2)
    for l in range(2):
        bi, bh = getattr(m.rnn, f"bias_ih_l{l}"), getattr(m.rnn, f"bias_hh_l{l}")
        assert torch.all(bi[111:222] == 1) and torch.all(bh[111:222] == 0)
        assert not torch.all(bi[:111] == 1)                       # only the forget slice
    g = MR.RNN_VARIANTS["GRU_L1_h181"](MR.VOCAB, n_layers=1)
    assert not torch.all(g.rnn.bias_ih_l0[181:362] == 1)          # GRU keeps PyTorch's default init
    assert m.rnn.dropout == 0.1 and MR.RNN_VARIANTS["LSTM_L1_h157"](MR.VOCAB, n_layers=1).rnn.dropout == 0.0
    try:
        MR.RNN_VARIANTS["LSTM_L1_h157"](MR.VOCAB, n_layers=2); raise RuntimeError("layer mismatch not caught")
    except AssertionError:
        pass


def test_register_reaches_evaluators():
    from mapformer import eval_noise_refine, eval_rank_strata
    assert "GRU_L2_h128" not in eval_noise_refine.VARIANT_MAP or eval_noise_refine.VARIANT_MAP["GRU_L2_h128"] is MR.GRU_L2_h128
    MR.register()
    assert eval_noise_refine.VARIANT_MAP["GRU_L2_h128"] is MR.RNN_VARIANTS["GRU_L2_h128"]
    assert eval_rank_strata.VARIANT_MAP["LSTM_L1_h313"] is MR.RNN_VARIANTS["LSTM_L1_h313"]
    MR.register()                                                 # idempotent
    from mapformer import train_variant as TV
    old = TV.VARIANT_MAP["GRU_L2_h128"]; TV.VARIANT_MAP["GRU_L2_h128"] = nn.Linear
    try:
        MR.register(); raise RuntimeError("shadowing not refused")
    except RuntimeError as e:
        assert "already exists" in str(e)
    finally:
        TV.VARIANT_MAP["GRU_L2_h128"] = old


def test_train_rnn_refusals():
    from mapformer import train_rnn
    for argv, frag in ((["--variant", "Vanilla_r4", "--output-dir", "x"], "must be one of"),
                       (["--variant", "GRU_L2_h128", "--fast-attn", "--output-dir", "x"], "--fast-attn"),
                       (["--variant", "GRU_L2_h128", "--deterministic", "--output-dir", "x"], "--deterministic")):
        try:
            train_rnn.main(argv); raise RuntimeError(f"not refused: {argv}")
        except SystemExit as e:
            assert frag in str(e), (argv, str(e))


def _driver_cells():
    txt = open(f"{REPO}/run_rnn_torus.sh").read()
    block = re.search(r"^CELLS=\((.*?)\)\s*$", txt, re.S | re.M).group(1)
    return re.findall(r'"([^"]+)"', block), txt


def test_driver_matches_analysis():
    cells, txt = _driver_cells()
    got = {}
    for c in cells:
        root, v, nl, lr, kind = c.split(":")
        got[(root, v)] = (int(nl), float(lr), kind == "fa")
        if kind == "rnn":
            assert root == AR.lr_root(float(lr)), c
    want = {(r, v): (nl, lr, fa) for (r, v, nl, lr, fa) in AR.all_cells().values()}
    assert got == want, (got, want)
    seeds = [int(x) for x in re.search(r"^SEEDS=\(([^)]*)\)", txt, re.M).group(1).split()]
    assert seeds == AR.SEEDS and AR.PILOT_SEED not in seeds
    # negative: a perturbed table is detected
    k0 = next(k for k in got if k[1] == "GRU_L2_h128")
    bad = dict(got); bad[k0] = (2, got[k0][1] * 3, False)
    assert bad != want


def test_seeds_fresh():
    """No run dir anywhere in runs/ carries a batch seed (checked at build time; the batch's own dir is excluded)."""
    hits = []
    for root, dirs, _ in os.walk(f"{REPO}/runs"):
        if "/runs/rnn_torus" in root:
            dirs[:] = []; continue
        hits += [d for d in dirs if re.search(r"_s33[0-7]\d?$", d) and int(re.search(r"_s(\d+)$", d).group(1)) in AR.SEEDS]
        if root.count("/") - REPO.count("/") > 5:
            dirs[:] = []
    assert not hits, hits[:5]


class _Oracle(nn.Module):
    """In-context perfect memory: tracks position from the action tokens and predicts the last observation seen at the new
    cell IN THE INPUT (blank if none). Uses context; knows no map."""
    def forward(self, tok):
        D = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        t = tok[0].tolist(); out = torch.full((1, len(t), MR.VOCAB), -10.0)
        pos = np.zeros(2, int); mem = {}; cur = None
        for i, x in enumerate(t):
            if i % 2 == 0:
                pos = (pos + D[x]) % 64; cur = tuple(pos)
                out[0, i, mem.get(cur, 20)] = 10.0
            else:
                mem[cur] = x
        return out


class _Blank(nn.Module):
    def forward(self, tok):
        o = torch.full((1, tok.shape[1], MR.VOCAB), -10.0); o[..., 20] = 10.0; return o


def test_context_destruction_controls():
    assert GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=0).unified_vocab_size - 1 == 20   # blank id
    r_or = AR.cd_eval(_Oracle(), 3300, torch.device("cpu"), T=256, n_trials=4)
    r_bl = AR.cd_eval(_Blank(), 3300, torch.device("cpu"), T=256, n_trials=4)
    assert r_or["clean"] == 1.0 and r_or["destroyed"] < 0.4, r_or          # context user: collapses
    assert r_bl["clean"] == r_bl["destroyed"] and 0.4 < r_bl["clean"] < 0.6, r_bl   # context-free: unchanged


def test_cd_walks_are_eval_walks():
    """cd_eval's intact walks are eval_noise_refine's walks (same np seed, same draw order)."""
    from mapformer import eval_noise_refine as E
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
    acc_e, _ = E.evaluate(_Oracle(), env, 256, 4, 0.0, torch.device("cpu"), seed=1234 + 3301)
    r = AR.cd_eval(_Oracle(), 3301, torch.device("cpu"), T=256, n_trials=4)
    assert acc_e == r["clean"] == 1.0
    acc_b, _ = E.evaluate(_Blank(), env, 256, 4, 0.0, torch.device("cpu"), seed=1234 + 3301)
    rb = AR.cd_eval(_Blank(), 3301, torch.device("cpu"), T=256, n_trials=4)
    assert abs(acc_b - rb["clean"]) < 1e-12, (acc_b, rb)
    rb2 = AR.cd_eval(_Blank(), 3302, torch.device("cpu"), T=256, n_trials=4)
    assert rb2["n"] != rb["n"] or rb2["clean"] != rb["clean"]          # another seed draws other walks


def smoke_every_arm():
    """CPU smoke: each RNN arm through train_rnn (2 batches x 2 epochs, T = 64, cosine, 1 data worker), then both
    evaluators and the context-destruction eval through eval_rnn_torus / analyze_rnn_torus on its checkpoint."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    with tempfile.TemporaryDirectory() as tmp:
        for k, (c, l, h) in MR.ARM_SPECS.items():
            out = f"{tmp}/r/p0/{k}_s3390"
            cmd = [sys.executable, "-m", "mapformer.train_rnn", "--variant", k, "--seed", "3390", "--epochs", "2", "--lr", "1e-3",
                   "--n-batches", "2", "--batch-size", "4", "--n-steps", "64", "--n-layers", str(l), "--n-heads", "2",
                   "--d-model", "128", "--n-landmarks", "0", "--schedule", "cosine", "--data-workers", "1", "--device", "cpu",
                   "--output-dir", out]
            p = subprocess.run(cmd, cwd="/home/prashr", env=env, capture_output=True, text=True)
            assert p.returncode == 0, p.stderr[-2000:]
            assert f"params={MR.n_params(c, h, l):,}" in p.stdout and "cudnn.allow_tf32 False" in p.stdout, p.stdout
            b = torch.load(f"{out}/{k}.pt", weights_only=False)
            assert len(b["losses"]) == 2 and np.isfinite(b["losses"]).all() and b["config"]["fast_attn"] is False
            a = b["config"]["args"]
            assert a["lr"] == 1e-3 and a["schedule"] == "cosine" and a["weight_decay"] == 0.05 and a["variant"] == k
            print(f"    smoke {k}: losses {[round(x, 3) for x in b['losses']]}", flush=True)
        vs = list(MR.ARM_SPECS)
        p = subprocess.run([sys.executable, "-m", "mapformer.eval_rnn_torus", "noise_refine", "--runs-dir", f"{tmp}/r",
                            "--variants", *vs, "--noises", "0.0", "--seeds", "3390", "--lengths", "64", "128",
                            "--n-trials", "3", "--device", "cpu", "--title", "smoke", "--out", f"{tmp}/E.md"],
                           cwd="/home/prashr", env=env, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-2000:]
        J = json.load(open(f"{tmp}/E.json"))
        assert all(f"0.0|{v}|128" in J and J[f"0.0|{v}|128"][0][1] is not None for v in vs), J.keys()
        p = subprocess.run([sys.executable, "-m", "mapformer.eval_rnn_torus", "strata", "--runs-dir", f"{tmp}/r",
                            "--variants", *vs, "--seeds", "3390", "--lengths", "128", "--n-trials", "3", "--device", "cpu",
                            "--check-json", f"{tmp}/E.json", "--out", f"{tmp}/S.json"],
                           cwd="/home/prashr", env=env, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-2000:]
        S = json.load(open(f"{tmp}/S.json"))
        assert all(abs(S[f"{v}|3390|128"]["all"]["acc"] - J[f"0.0|{v}|128"][0][1]) < 1e-9 for v in vs)
        cells = {v: ("r", v, MR.ARM_SPECS[v][1], 1e-3, False) for v in vs}
        AR.run_cd(torch.device("cpu"), cells=cells, runs=tmp, seeds=[3390], out=f"{tmp}/CD.json")
        C = json.load(open(f"{tmp}/CD.json"))
        assert len(C) == len(vs) and all(0 <= x["destroyed"] <= 1 for x in C.values())


def main():
    fast = "--fast" in sys.argv
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    if not fast:
        tests.append(smoke_every_arm)
    bad = 0
    for t in tests:
        try:
            t(); print(f"  ok   {t.__name__}", flush=True)
        except Exception:
            bad += 1; print(f"  FAIL {t.__name__}"); traceback.print_exc()
    print(f"{len(tests) - bad}/{len(tests)} passed"); sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

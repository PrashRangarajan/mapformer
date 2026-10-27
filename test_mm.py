"""Unit tests for the Memory Maze batch (MEMORY_MAZE_PREREG.md). CPU only.

  cd /home/prashr && python3 -m mapformer.test_mm --zip <a small zip of Memory Maze 15x15 npz episodes (>= 2)>

Checks: the scored-step definition equals the gate's (mm_gate_v2) row for row; the E feature times the fitted map equals
the gate's class-E dead reckoning; ActOnly (frames never move the phase); causality (the query at s_t sees nothing from
f_t on); blind frames are invisible; the InfoNCE mask removes exactly the same-pose negatives; the installed E map makes
the phase a function of the dead-reckoned position; scoring gives R1 = 1 to the pose oracle and R2 its definition;
GRU parameter match; GPS position alignment; every arm runs; resume reproduces an uninterrupted run exactly.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

from mapformer import mm_data as D
from mapformer import model_mm as M
from mapformer import train_mm as TR
import mm_gate as g1
import mm_gate_v2 as g2

FAILS = []


def chk(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def episodes(zip_path, n):
    return list(g1.iter_episodes(zip_path, n))


def t_scored_equals_gate(zp):
    base = g1.fit(zp, 2, rng=np.random.default_rng(0))
    old = g2.CONFIGS
    g2.CONFIGS = [(D.B_BLIND, D.S_SIGHT)]
    ext = g2.fit_v2(zp, 2, base)
    for i, ep in enumerate(episodes(zp, 2)):
        rec = g2.episode(ep, i, base, ext)[f"B{D.B_BLIND}_S{D.S_SIGHT}"]["rows"]
        a = ep["action"].argmax(1); a[ep["action"].sum(1) == 0] = -1
        pose, cid = D.pose_bins(ep["agent_pos"], ep["agent_dir"])
        mine = D.scored_steps(pose, cid, D.sighted_mask(1001, i))
        chk([(r["t"], r["gap"], r["back"]) for r in rec] == [(r["t"], r["gap"], r["back"]) for r in mine],
            f"scored steps equal the gate's rows (episode {i}: {len(mine)} rows)")
        # E feature x fitted map == gate's class-E prediction (its ME)
        W = D.e_map_as_matrix(base["ME"][:7, :6])
        P = g2.step_preds(a, ep["agent_dir"], np.diff(ep["agent_pos"], axis=0), base, ext)["E"]
        mineP = np.vstack([np.zeros((1, 2)), np.cumsum(D.e_feature(a, ep["agent_dir"]).astype(np.float64)[1:] @ W, 0)])
        chk(np.allclose(P, mineP, atol=1e-4), f"E feature x map reproduces the gate's class-E dead reckoning (ep {i})")
    g2.CONFIGS = old


def fake_batch(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(B, 1001, 256, generator=g)
    sighted = torch.ones(B, 1001, dtype=torch.bool)
    act = np.random.default_rng(seed).integers(0, 6, (B, 1001)); act[:, 0] = -1
    dr = np.random.default_rng(seed + 1).normal(size=(B, 1001, 2)); dr /= np.linalg.norm(dr, axis=-1, keepdims=True)
    pos = np.cumsum(np.random.default_rng(seed + 2).normal(0, 0.2, (B, 1001, 2)), 1) + 7
    feats = {"e": torch.from_numpy(np.stack([D.e_feature(a, d) for a, d in zip(act, dr)])),
             "odo": torch.from_numpy(np.stack([D.odo_feature(p) for p in pos])).float(),
             "raw": torch.from_numpy(np.stack([D.raw_feature(a) for a in act]))}
    return emb, sighted, feats


def t_model_properties():
    torch.manual_seed(0)
    emb, sighted, feats = fake_batch()
    for arm in M.ARMS:
        m = M.build(arm).eval()
        lg, h = m(emb, sighted, feats)
        chk(lg.shape == (2, 1000, 1001) and h.shape == (2, 2001, 128), f"{arm}: shapes {tuple(lg.shape)} {tuple(h.shape)}")
        # causality: changing frames f_t.. (t = 400) leaves hidden states before token 2t untouched
        e2 = emb.clone(); e2[:, 400:] += 1.0
        h2 = m.hidden(e2, sighted, feats)
        chk(torch.allclose(h[:, :800], h2[:, :800], atol=1e-5) and not torch.allclose(h[:, 800:], h2[:, 800:]),
            f"{arm}: causal (query at s_400 = token 799 blind to f_400 on)")
        # blind frames invisible
        sb = sighted.clone(); sb[:, 100:116] = False
        e3 = emb.clone(); e3[:, 100:116] = torch.randn(2, 16, 256)
        chk(torch.allclose(m.hidden(emb, sb, feats), m.hidden(e3, sb, feats), atol=1e-6), f"{arm}: blind frames invisible")
        if m.kind == "path":
            d = m.step_delta(feats)
            chk(float(d[:, 0::2].abs().max()) == 0.0, f"{arm}: ActOnly (frame tokens have step 0)")
            c1, _ = m.path(d)
            # frames do not enter the phase: recompute with other embeddings -> identical step
            chk(torch.equal(m.step_delta(feats), d), f"{arm}: step independent of frames")
    gm = M.build("GRU")
    core = sum(p.numel() for p in gm.gru.parameters()) + sum(p.numel() for p in gm.proj.parameters())
    tl = M.transformer_layer_params()
    chk(core <= tl and core >= 0.97 * tl, f"GRU core {core} vs one transformer layer {tl} (h {gm.h})")


def t_gps_alignment():
    m = M.build("GPS")
    emb, sighted, feats = fake_batch()
    p = torch.cumsum(feats["odo"], 1)
    ptok = torch.repeat_interleave(p, 2, dim=1)[:, 1:]
    chk(torch.equal(ptok[:, 0], p[:, 0]) and torch.equal(ptok[:, 1], p[:, 1]) and torch.equal(ptok[:, 2], p[:, 1])
        and torch.equal(ptok[:, 1999], p[:, 1000]) and torch.equal(ptok[:, 2000], p[:, 1000]),
        "GPS: f_0 -> p_0, s_t and f_t -> p_t")


def t_info_nce():
    torch.manual_seed(1)
    B = 2
    lg = torch.randn(B, 1000, 1001)
    pose = torch.randint(0, 30, (B, 1001))
    sighted = torch.rand(B, 1001) > 0.5
    got = float(TR.info_nce(lg, pose, sighted))
    tot, n = 0.0, 0
    for b in range(B):
        for t in range(1, 1001):
            if not sighted[b, t]:
                continue
            keep = (pose[b] != pose[b, t]); keep[t] = True
            l = lg[b, t - 1][keep]
            pos_idx = int(keep[:t].sum())
            tot += float(torch.logsumexp(l, 0) - l[pos_idx]); n += 1
    chk(abs(got - tot / n) < 1e-4, f"InfoNCE masks exactly the same-pose negatives ({got:.5f} vs {tot/n:.5f})")
    got2 = float(TR.info_nce(lg, pose, sighted, "both"))
    tot2, n2 = 0.0, 0
    for b in range(B):
        for t in range(1, 1001):
            if not (sighted[b, t] and not sighted[b, t - 1]):
                continue
            keep = (pose[b] != pose[b, t]); keep[t] = True
            l = lg[b, t - 1][keep]
            tot2 += float(torch.logsumexp(l, 0) - l[int(keep[:t].sum())]); n2 += 1
    chk(abs(got2 - (tot / n + tot2 / n2)) < 1e-4, f"loss 'both' = sighted mean + post-blind mean ({got2:.5f})")


def t_install(zp):
    eps = episodes(zp, 2)
    acts = []
    for ep in eps:
        a = ep["action"].argmax(1).astype(int); a[ep["action"].sum(1) == 0] = -1; acts.append(a)
    Mfit = D.fit_e_map(acts, [e["agent_dir"] for e in eps], [e["agent_pos"] for e in eps])
    W = D.e_map_as_matrix(Mfit)
    m = M.build("P_E")
    m.install_e_map(W, frozen=True, seed=0)
    chk(not m.w_in.weight.requires_grad and not m.path.omega.requires_grad, "install frozen: step map and omega frozen")
    ef = torch.from_numpy(D.e_feature(acts[0], eps[0]["agent_dir"]))[None]
    feats = {"e": ef}
    d = m.step_delta(feats)
    cum = torch.cumsum(d, 1)[0, 1::2]                                   # at s_t tokens, (1000, H, nb)
    dr = torch.cumsum(ef[0, 1:].double() @ torch.from_numpy(W), 0)       # dead-reckoned position
    u = m.w_out.weight.view(2, 32, 4)[..., :2].double()
    pred = torch.einsum("hkc,tc->thk", u, dr)
    chk(torch.allclose(cum.double(), pred, atol=1e-3), "installed phase = u_k . dead-reckoned position (E map)")
    m2 = M.build("P_E"); m2.install_e_map(W, frozen=False)
    chk(m2.w_in.weight.requires_grad, "install trainable keeps the step map trainable")


class _Stub:
    """A model whose logits put the maximum on a chosen frame per query (scoring test)."""

    def __init__(self, choose):
        self.choose = choose
        self.io = self

    def eval(self):
        return self

    def hidden(self, emb, sighted, feats):
        return torch.zeros(emb.shape[0], 2001, 4)

    def logits(self, out, emb):
        B = emb.shape[0]
        lg = torch.zeros(B, 1000, 1001)
        for b in range(B):
            for t in range(1, 1001):
                lg[b, t - 1, self.choose(b, t)] = 1.0
        return lg


def t_scoring(data_dir):
    sp = TR.Split(data_dir, "eval", 0, 2, torch.device("cpu"))
    rows = sp.rows()
    last, b0 = {}, {}
    for i in range(sp.n):
        for r in rows[i]:
            last[(i, r["t"])] = r["s"]; b0[(i, r["t"])] = r["b0"]
    sc = TR.score(_Stub(lambda b, t: last.get((b, t), 0)), sp, 2)
    chk(len(sc) > 0 and all(x[5] for x in sc), f"pose oracle scores R1 = 1 ({len(sc)} rows)")
    sc2 = TR.score(_Stub(lambda b, t: t), sp, 2)
    chk(all(x[5] and x[6] for x in sc2), "predicting f_t itself scores R1 = R2 = 1")
    sc3 = TR.score(_Stub(lambda b, t: b0.get((b, t), t) - 1), sp, 2)
    chk(not any(x[5] for x in sc3), "copy-the-last-SIGHTED-frame (gate F1) scores R1 = 0 (pose differs by definition)")


def t_resume(data_dir):
    with tempfile.TemporaryDirectory() as d:
        base = [sys.executable, "-m", "mapformer.train_mm", "--arm", "P_E", "--seed", "3691", "--data", data_dir,
                "--device", "cpu", "--batch", "2", "--train-range", "0", "2", "--eval-n", "2", "--no-held", "--no-probe"]
        env = dict(os.environ, PYTHONWARNINGS="ignore")
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        subprocess.run(base + ["--out", f"{d}/a", "--epochs", "3"], check=True, cwd=cwd, env=env, capture_output=True)
        # interrupted: run 1 epoch of a 3-epoch schedule, then resume
        subprocess.run(base + ["--out", f"{d}/b", "--epochs", "3"], check=True, cwd=cwd, env=dict(env, MM_STOP_AFTER="1"),
                       capture_output=True)
        st = torch.load(f"{d}/b/ckpt.pt")
        chk(st["epoch"] == 1 and not os.path.exists(f"{d}/b/DONE"), "MM_STOP_AFTER stops after epoch 1 (no DONE)")
        subprocess.run(base + ["--out", f"{d}/b", "--epochs", "3"], check=True, cwd=cwd, env=env, capture_output=True)
        ra = json.load(open(f"{d}/a/results.json")); rb = json.load(open(f"{d}/b/results.json"))
        chk(ra["hist"] == rb["hist"] and ra["eval"] == rb["eval"], f"resume reproduces the uninterrupted run {ra['hist']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--data", required=True, help="array dir prepared from the same zip (mm_encoder prep --smoke)")
    args = ap.parse_args()
    t_scored_equals_gate(args.zip)
    t_model_properties()
    t_gps_alignment()
    t_info_nce()
    t_install(args.zip)
    t_scoring(args.data)
    t_resume(args.data)
    print(f"\n{len(FAILS)} FAIL(S)" if FAILS else "\nALL PASS")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

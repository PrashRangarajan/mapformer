"""SEARCH_PREREG.md S1 -- retrieval anatomy of MapFormer-EM on recency, eval-only.

Per (checkpoint, k), on held-out episodes:
  acc      the model's own prediction
  sel_h    per head: is the A_P argmax over PRIOR SYMBOL keys the answer?
  sel0_h   the same with Delta(q_k) set to 0 at the query position -- the counterfactual
           without the query token's own step. max_h sel_h - max_h sel0_h is the
           per-token rewind's contribution, WRAPPED OR NOT (theta enters via cos/sin, so a
           block's rewind is defined only modulo 2 pi / omega; the linear slope in
           probe_rewind cannot see a wrapped one)
  att      attention weight on the answer key, max over heads (the real softmax(A_X (*) A_P))
  gain     sign of A_X at the answer in the head with the largest attention on it
  profile  mean A_P with Delta(q) = 0 against symbol distance n (n = 0 is the query's own
           key, n >= 1 the n-th most recent symbol) -- the kernel-peak readout

`em_forward` replicates MapFormerEM / MapFormerEM_SingleP0 (1 layer, eval mode) and is
CHECKED against model(x) on the first batch of every checkpoint.

    python3 -m mapformer.probe_anatomy            # compute (if needed) + analyse
    python3 -m mapformer.probe_anatomy --analyse  # analyse the saved JSON only
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.ckpt_guard import REPO, require_checkpoints
from mapformer.model import _apply_rope

ARMS = ["VanillaEM_P0_r4", "EMDoF_alignlock", "EMDoF_magonly", "EMDoF_alignfree", "VanillaEM_r4"]
RHO1 = ["VanillaEM_P0_r4", "EMDoF_alignlock", "EMDoF_magonly"]
NMAX = 80
OUT_JSON = REPO / "_ANATOMY.json"


# ----------------------------------------------------------------------------- internals
def origins(m, e=None):
    """(q0, k0) WITH graph; k0 through a property where that is what forward() reads.

    EMPair (`model_em_pairorigin`) has PER-TOKEN origins, so it exposes `_origins(x)` returning
    (B, H, L, d_head). Callers pass the embeddings `e`; the probe then rotates those instead of
    broadcasting one vector. Added 2026-09-12: without it `em_forward` reproduced the wrong
    model, which its own assert caught (max diff 13.8) rather than reporting a wrong number.
    """
    if hasattr(m, "_origins") and e is not None:
        return m._origins(e)
    if "p0_pos" in dict(m.named_parameters()):
        return m.p0_pos, m.p0_pos
    return m.q0_pos, m.k0_pos


def ap_from_delta(m, delta, e=None):
    cos_a, sin_a = m.path_integrator(delta)
    q0, k0 = origins(m, e)
    B, L = delta.shape[:2]
    if q0.dim() == 2:                                   # one vector, shared by every pair
        q0 = q0[None, :, None, :].expand(B, -1, L, -1)
        k0 = k0[None, :, None, :].expand(B, -1, L, -1)
    qp, kp = _apply_rope(q0, cos_a, sin_a), _apply_rope(k0, cos_a, sin_a)
    return qp @ kp.transpose(-1, -2) / math.sqrt(q0.shape[-1])


def em_forward(m, x, delta=None):
    assert len(m.layers) == 1, "one-layer EM only"
    B, L = x.shape
    e = m.token_emb(x)
    if delta is None:
        delta = m.action_to_lie(e)
    AP = ap_from_delta(m, delta, e)
    lay = m.layers[0]
    H, dh = lay.n_heads, lay.d_head
    h = lay.norm1(e)
    Qc = lay.q_content(h).view(B, L, H, dh).transpose(1, 2)
    Kc = lay.k_content(h).view(B, L, H, dh).transpose(1, 2)
    V = lay.v_proj(h).view(B, L, H, dh).transpose(1, 2)
    AX = Qc @ Kc.transpose(-1, -2) / math.sqrt(dh)
    mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
    attn = F.softmax((AX * AP).masked_fill(mask, float("-inf")), dim=-1)
    y = e + lay.o_proj((attn @ V).transpose(1, 2).reshape(B, L, -1))
    y = y + lay.ffn(lay.norm2(y))
    return dict(logits=m.out_proj(m.out_norm(y)), AX=AX, AP=AP, attn=attn, delta=delta)


def query_mask(x, env):
    return (x >= env.query_offset) & (x < env.query_offset + env.k_max)


# ----------------------------------------------------------------------------- one checkpoint
@torch.no_grad()
def anatomy(m, env, seed, episodes=256, T=1024, device="cpu"):
    m = m.to(device).eval()
    K = env.k_max
    H = m.n_heads
    z = lambda *s: np.zeros(s)
    n, acc, att, gain = z(K + 1), z(K + 1), z(K + 1), z(K + 1)
    sel, sel0 = z(K + 1, H), z(K + 1, H)
    selmin, selmin0, gain_h, att_h = z(K + 1, H), z(K + 1, H), z(K + 1, H), z(K + 1, H)
    prof, prof_n = z(H, NMAX + 1), z(NMAX + 1)
    rng = np.random.RandomState(7000 + seed)
    checked = False
    for _ in range(episodes // 16):
        toks, sps, ans, infos = env.generate_batch(16, T, rng)
        x = toks[:, :-1].to(device)
        o = em_forward(m, x)
        if not checked:
            diff = (o["logits"] - m(x)).abs().max().item()
            assert diff < 1e-3, f"em_forward does not reproduce the model: {diff}"
            checked = True
        d0 = o["delta"].clone()
        d0[query_mask(x, env)] = 0.0
        AP0 = ap_from_delta(m, d0, m.token_emb(x))
        lo = env.sym_offset
        for b in range(x.shape[0]):
            sympos = np.asarray(infos[b]["sym_positions"])
            qs = [(p, a, k) for p, a, k in zip(sps[b], ans[b], infos[b]["offsets"]) if p < x.shape[1]]
            if not qs:
                continue
            P = torch.tensor([q[0] for q in qs], device=device)
            ap = o["AP"][b][:, P].cpu().numpy()          # (H, nq, L)
            ap0 = AP0[b][:, P].cpu().numpy()
            at = o["attn"][b][:, P].cpu().numpy()
            ax = o["AX"][b][:, P].cpu().numpy()
            pred = o["logits"][b, P, lo:lo + env.n_symbols].argmax(-1).cpu().numpy()
            for i, (p, a, k) in enumerate(qs):
                cand = sympos[sympos < p]
                tgt = cand[-k]
                n[k] += 1
                acc[k] += int(pred[i] == a - lo)
                sel[k] += (cand[ap[:, i, cand].argmax(-1)] == tgt)
                sel0[k] += (cand[ap0[:, i, cand].argmax(-1)] == tgt)
                selmin[k] += (cand[ap[:, i, cand].argmin(-1)] == tgt)
                selmin0[k] += (cand[ap0[:, i, cand].argmin(-1)] == tgt)
                gain_h[k] += (ax[:, i, tgt] > 0)
                att_h[k] += at[:, i, tgt]
                hs = int(at[:, i, tgt].argmax())
                att[k] += at[hs, i, tgt]
                gain[k] += int(ax[hs, i, tgt] > 0)
                prof[:, 0] += ap0[:, i, p]; prof_n[0] += 1
                for d in range(1, min(NMAX, len(cand)) + 1):
                    prof[:, d] += ap0[:, i, cand[-d]]; prof_n[d] += 1
    nn_ = np.maximum(n, 1)
    return dict(n=n[1:].tolist(), acc=(acc / nn_)[1:].tolist(), att=(att / nn_)[1:].tolist(),
                gain_pos=(gain / nn_)[1:].tolist(),
                sel=(sel / nn_[:, None])[1:].tolist(), sel0=(sel0 / nn_[:, None])[1:].tolist(),
                selmin=(selmin / nn_[:, None])[1:].tolist(),
                selmin0=(selmin0 / nn_[:, None])[1:].tolist(),
                gain_h=(gain_h / nn_[:, None])[1:].tolist(), att_h=(att_h / nn_[:, None])[1:].tolist(),
                profile=(prof / np.maximum(prof_n, 1)).tolist())


def compute(device, out=OUT_JSON, runs_dir="runs/dof/recency", arms=ARMS, seeds=range(24),
            **env_overrides):
    from mapformer.probe_rewind import load_model
    f = require_checkpoints(runs_dir, "recency", arms, seeds)
    rows = []
    for (arm, s), pt in sorted(f.found.items()):
        m, ck, env = load_model(pt, ck_variant(pt, arm), **env_overrides)
        r = anatomy(m, env, s, device=device)
        r.update(arm=arm, seed=s)
        rows.append(r)
        print(f"{arm:18s} s{s:<2} acc {np.average(r['acc'], weights=r['n']):.3f}", flush=True)
    json.dump(rows, open(out, "w"))
    return rows


def ck_variant(pt, arm):
    """The variant stored in the checkpoint (run dirs in runs/search are named by ARM)."""
    return torch.load(pt, map_location="cpu", weights_only=False).get("variant", arm)


# ----------------------------------------------------------------------------- exploratory
def explore(rows, arms=ARMS, kmin=8):
    """NOT pre-registered. Route of every solved cell, read in the head that retrieves it,
    and whether distance from that arm's kernel peak to k predicts success."""
    L = ["## Exploratory (not registered): route of each solved cell (acc >= 0.9, k >= 8)\n",
         "Read in the head with the most attention on the answer. Categories in priority order: "
         "peak-rewind (sel - sel0 >= 0.5), trough-rewind (selmin - selmin0 >= 0.5), peak-static "
         "(sel0 >= 0.5), trough-static (selmin0 >= 0.5), other.\n",
         "| arm | solved | peak-rewind | trough-rewind | peak-static | trough-static | other | A_X>0 in peak-* | A_X>0 in trough-* |",
         "|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        cat = []; gp = {"peak": [], "trough": []}
        for r in rows:
            if r["arm"] != arm:
                continue
            S, S0 = np.array(r["sel"]), np.array(r["sel0"])
            M, M0 = np.array(r["selmin"]), np.array(r["selmin0"])
            G, A = np.array(r["gain_h"]), np.array(r["att_h"])
            for k in range(kmin, len(r["acc"]) + 1):
                i = k - 1
                if r["n"][i] == 0 or r["acc"][i] < 0.9:
                    continue
                h = int(A[i].argmax())
                c = ("peak-rewind" if S[i, h] - S0[i, h] >= 0.5 else
                     "trough-rewind" if M[i, h] - M0[i, h] >= 0.5 else
                     "peak-static" if S0[i, h] >= 0.5 else
                     "trough-static" if M0[i, h] >= 0.5 else "other")
                cat.append(c)
                if c != "other":
                    gp[c.split("-")[0]].append(G[i, h])
        cat = np.array(cat)
        fr = lambda c: (cat == c).mean() if len(cat) else float("nan")
        mp = lambda v: f"{np.mean(v):.3f}" if v else "-"
        L.append(f"| {arm} | {len(cat)} | {fr('peak-rewind'):.3f} | {fr('trough-rewind'):.3f} | "
                 f"{fr('peak-static'):.3f} | {fr('trough-static'):.3f} | {fr('other'):.3f} | "
                 f"{mp(gp['peak'])} | {mp(gp['trough'])} |")
    L += ["", "## Exploratory: P(solved) against distance from the nearest head's kernel peak to k\n",
          "n*_h = argmax of head h's Delta(q)=0 profile; d = min_h |k - n*_h|. If the difficulty "
          "of search is the size of the shift the query token must make, the arms fall on one curve.\n",
          "| arm | d 0-3 | d 4-7 | d 8-15 | d 16-31 | d 32-64 |", "|---|---|---|---|---|---|"]
    bins = [(0, 3), (4, 7), (8, 15), (16, 31), (32, 64)]
    for arm in arms:
        acc_b = {b: [] for b in bins}
        for r in rows:
            if r["arm"] != arm:
                continue
            ns = [int(np.argmax(p)) for p in r["profile"]]
            for k in range(1, len(r["acc"]) + 1):
                if r["n"][k - 1] == 0:
                    continue
                d = min(abs(k - n_) for n_ in ns)
                for b in bins:
                    if b[0] <= d <= b[1]:
                        acc_b[b].append(r["acc"][k - 1] >= 0.9)
        L.append(f"| {arm} | " + " | ".join(
            f"{np.mean(acc_b[b]):.3f} (n={len(acc_b[b])})" if acc_b[b] else "-" for b in bins) + " |")
    txt = "\n".join(L)
    print(txt)
    return txt


# ----------------------------------------------------------------------------- analysis
def analyse(rows):
    from mapformer.stats_guard import paired, table
    L = []
    by = {(r["arm"], r["seed"]): r for r in rows}
    def cells(arm, kmin=8):
        out = []
        for r in rows:
            if r["arm"] != arm:
                continue
            S, S0 = np.array(r["sel"]), np.array(r["sel0"])
            for k in range(kmin, len(r["acc"]) + 1):
                i = k - 1
                if r["n"][i] == 0:
                    continue
                out.append((r["acc"][i], S[i].max() - S0[i].max(), (S[i] - S0[i]).max(),
                            r["gain_pos"][i], r["att"][i]))
        return np.array(out)

    L.append("## S1-a  H-wrap: do solved large-k cells carry a per-token (wrapped) rewind?\n")
    L.append("diff = max_h sel_h - max_h sel0_h (registered primary); alt = max_h (sel_h - sel0_h)\n")
    L.append("| arm | cells k>=8 | solved (acc>=0.9) | solved with diff>=0.5 | failed (acc<=0.3) | failed with diff>=0.5 | r(acc, diff) | solved: mean att | solved: gain>0 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    route = {}
    for arm in ARMS:
        c = cells(arm)
        sv, fl = c[c[:, 0] >= 0.9], c[c[:, 0] <= 0.3]
        fs = (sv[:, 1] >= 0.5).mean() if len(sv) else float("nan")
        ff = (fl[:, 1] >= 0.5).mean() if len(fl) else float("nan")
        route[arm] = fs
        r = np.corrcoef(c[:, 0], c[:, 1])[0, 1]
        L.append(f"| {arm} | {len(c)} | {len(sv)} | {fs:.3f} | {len(fl)} | {ff:.3f} | {r:+.3f} | "
                 f"{sv[:, 4].mean():.3f} | {sv[:, 3].mean():.3f} |")
    c = cells("VanillaEM_P0_r4")
    sv, fl = c[c[:, 0] >= 0.9], c[c[:, 0] <= 0.3]
    a, b = (sv[:, 1] >= 0.5).mean(), (fl[:, 1] >= 0.5).mean()
    v = ("CONFIRMED" if a >= 0.7 and b <= 0.2 else "REFUTED" if a < 0.3 else "PARTIAL")
    L.append(f"\nH-wrap (P0): solved {a:.3f} (>= 0.70 needed), failed {b:.3f} (<= 0.20 needed) -> **{v}**")
    L.append(f"alt statistic, P0: solved {(sv[:, 2] >= 0.5).mean():.3f}, failed {(fl[:, 2] >= 0.5).mean():.3f}\n")

    L.append("## S1-b  H-phase: AlignFree - MagOnly by k bin (paired by seed, n=24)\n")
    cs = []
    for lo_, hi_ in [(1, 16), (17, 32), (33, 48), (49, 64)]:
        def binacc(arm):
            out = {}
            for s in range(24):
                r = by[(arm, s)]
                a_ = np.array(r["acc"][lo_ - 1:hi_]); w = np.array(r["n"][lo_ - 1:hi_])
                out[s] = float((a_ * w).sum() / w.sum())
            return out
        cs.append(paired(binacc("EMDoF_alignfree"), binacc("EMDoF_magonly"), f"k {lo_}-{hi_}"))
    L.append(table(cs))
    d = route["EMDoF_alignfree"] - route["EMDoF_magonly"]
    L.append(f"\nRewind-route fraction of solved cells: AlignFree {route['EMDoF_alignfree']:.3f}, "
             f"MagOnly {route['EMDoF_magonly']:.3f}, difference {d:+.3f} (<= -0.15 predicted) -> "
             f"**{'CONFIRMED' if d <= -0.15 else 'REFUTED'}**\n")

    L.append("## Kernel-peak readout (Delta(q) = 0): argmax over n in 0..80 of the mean A_P profile, per head\n")
    L.append("| arm | heads | peak at n in {0,1} | peak at n >= 2 | peaks (n) |")
    L.append("|---|---|---|---|---|")
    ok = True
    for arm in ARMS:
        pk = [int(np.argmax(np.array(r["profile"])[h])) for r in rows if r["arm"] == arm
              for h in range(len(r["profile"]))]
        pk = np.array(pk)
        L.append(f"| {arm} | {len(pk)} | {(pk <= 1).mean():.3f} | {(pk >= 2).mean():.3f} | "
                 f"{sorted(pk.tolist())} |")
        if arm in RHO1 and (pk > 1).any():
            ok = False
    L.append(f"\nManipulation check (rho=1 arms all peak at n <= 1): **{'PASS' if ok else 'FAIL -- H-phase route reading not interpretable'}**")
    txt = "\n".join(L)
    print(txt)
    (REPO / "runs/search").mkdir(parents=True, exist_ok=True)
    (REPO / "runs/search/S1_report.md").write_text(txt + "\n")
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--explore", action="store_true", help="exploratory tables (not registered)")
    a = ap.parse_args()
    if a.analyse:
        rows = json.load(open(a.out))
    else:
        rows = compute(a.device, a.out)
    if a.explore:
        txt = explore(rows)
        (REPO / "runs/search/S1_explore.md").write_text(txt + "\n")
    else:
        analyse(rows)


if __name__ == "__main__":
    main()

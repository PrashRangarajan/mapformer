"""T = 1024 version (2026-10-20) of docs/audits/2026-10-03/causal_whatwhere.py, run on the SIGN_MATCHED checkpoints
(runs/sign_matched/p0/{Signed_r4,Abs_r4,Pos_r4,RoPE}_s0-7: torus, 1 layer, 2 heads, rank 4, trained AND tested at
T = 1024, 900 epochs) -- the regime where a map is needed (retrace floor 0.843 vs 0.934 at T = 128). Identical
conditions and code path, with three changes for memory/CPU: the evaluation batch (eval_noise_refine's 100
trajectories, np seed 1234 + s) is processed in chunks of 10 and accuracy pooled over all revisit targets; the
calibration set is 20 trajectories (np seed 50000 + s) instead of 100; the graded phase-offset lesion and the lam_add
grid are dropped (lam_pos kept). Floors are the committed own-stream ones (docs/audits/2026-10-16/sign_floors_out.txt).
Original docstring follows.

Causal test of the what/where interaction in the layer-1 attention score (WHAT_WHERE_ANALYSIS.md
sec. 6; LIT_WHAT_WHERE.md P1). Evaluation only; no training. Not pre-registered: descriptive, post hoc.

Checkpoints: runs/paper2x2/p0/{arm}_s{0..7} (1 layer, 2 heads, d_head 64, torus paper task, T=128).
Evaluation: exactly eval_noise_refine.py's -- GridWorld(64, 16 objects, p_empty 0.5, seed=10000)
(held-out map), np.random.seed(1234 + seed), 100 trajectories of 128 steps, input tok[:-1], accuracy
pooled over revisited-observation targets. Here the 100 trajectories are run as one batch.

The score is bilinear in a per-token content representation given the phases:
  RoPE-form (RoPE, MapWM):  S_ts = (R(theta_t) q_t)^T (R(theta_s) k_s) / 8,   q = W_q LN(emb) + b_q
  PoPE-form (PoPE, MapPoPE): S_ts = sum_c mu^q_tc mu^k_sc cos(phi_s - phi_t + delta_c) / 8,  mu = softplus(q)
In a 1-layer model q_t, k_t (RoPE) or mu^q_t, mu^k_t (PoPE) are functions of the token id alone, so the
content of a position is its token id v in 0..20 (4 actions, 16 objects, blank). Phases are left as the
model computes them in every condition: the "where" variable is never touched.

Conditions (per head; the softmax, values, FFN and readout are the model's own):
  intact      the rebuilt forward with no change. Verified against model(tokens) (max |dlogit|).
  pos_class   POSITION-ONLY (TEM-t form, L5). q_t, k_s replaced by the frequency-weighted mean of the
              content representation over the token's CLASS (action tokens / observation tokens). Because
              S is bilinear in (q, k) given the phases, mean(q)^T R mean(k) = E_{q,k}[q^T R k] exactly when
              query and key content are drawn independently from their class marginals: this IS the score
              averaged over the content of query and key at the same phase difference. The class (action vs
              observation) is kept because it is a token TYPE, not object identity; TEM-t also feeds
              actions and observations through different paths. For PoPE the mean is of mu = softplus(q),
              not softplus of the mean, for the same reason (the score is linear in mu).
  pos_global  stricter: one mean over all tokens (type removed as well).
  pos_tmatch  pos_class with the score multiplied by a per-head scalar beta that restores the intact
              score's mean within-row sd over action-query rows (calibration set): separates "the kernel's
              shape" from "its sharpness", which content averaging also reduces.
  additive    L4: P + C. P = pos_class score; C(v_t, v_s) = mean of (S - P) over all causal (t, s) pairs
              with that token pair (21 x 21 table per head, calibration set). The content-position
              interaction I = S - P - C is removed; content keeps a per-pair constant.
  gain        L3-like: alpha(v_t, v_s) + gamma(v_t, v_s) P, both fitted per token pair and head by least
              squares over causal pairs (calibration set). Content may scale and offset the shared
              (content-averaged) kernel but cannot change its shape or move its peak.
  content     the reverse lesion: phases set to 0 (no rotation; PoPE keeps delta). Shows the readout can fail.
  lam_pos[l]  graded: P + l (S - P)       (l = 0 is pos_class, 1 is intact, > 1 amplifies)
  lam_add[l]  graded: P + C + l I         (l = 0 is additive, 1 is intact)
  phase[s]    graded lesion: each (query token, key token) pair gets a fixed random phase offset per head
              and per rotary channel, psi ~ N(0, s^2) rad, added to the key's phase (RoPE-style
              psi_k - psi_q per pair): content now MOVES the kernel, by an amount set by s.
The calibration set (token frequencies, class means, C, beta) is 100 different trajectories on the same
held-out map (np seed 50000 + seed), so nothing fitted touches the evaluation trajectories.

Floor: (i) always-blank constant on the evaluation targets; (ii) the best token n-gram (orders 1-5, the
n tokens before the target, backoff, fitted on 1000 trajectories from each of the 8 training maps,
revisit targets only) evaluated on the same targets. Reported per seed set, best of the two.

Output: causal_whatwhere.json, causal_whatwhere.txt (next to this file). Run from anywhere:
  python3 docs/audits/2026-10-03/causal_whatwhere.py --device cuda:0
"""
import argparse, collections, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

REPO = "/home/prashr/mapformer"
sys.path.insert(0, "/home/prashr")
from mapformer import model as M                       # noqa: E402
from mapformer.model import _apply_rope, _is_pow2       # noqa: E402
from mapformer.model_pope import DELTA_MIN, DELTA_MAX   # noqa: E402
from mapformer.environment import GridWorld             # noqa: E402
from mapformer.train_variant import VARIANT_MAP         # noqa: E402
from mapformer import stats_core as SC                  # noqa: E402

ARMS = ["Signed_r4", "Abs_r4", "Pos_r4", "RoPE"]
SEEDS = list(range(8))
T_STEPS, N_TRIALS, ENV_SEED = 1024, 100, 10000
N_CAL, CHUNK = 20, 10
N_ACT, VOCAB = 4, 21
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
SIGMAS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
OUT = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- data
def trajectories(env, seed, n):
    np.random.seed(seed)
    toks, revs = [], []
    for _ in range(n):
        t, _om, r = env.generate_trajectory(T_STEPS)
        toks.append(t); revs.append(r)
    return torch.stack(toks), torch.stack(revs)


def floors(eval_sets):
    """Always-blank constant and the best backoff token n-gram (fit on the 8 training maps)."""
    blank = N_ACT + 16
    counts = [collections.defaultdict(collections.Counter) for _ in range(6)]
    for k in SEEDS:
        env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=k)
        tok, rev = trajectories(env, 70000 + k, 1000)
        tok = tok.numpy(); rev = rev.numpy()
        for b in range(tok.shape[0]):
            for t in np.nonzero(rev[b])[0]:
                for n in range(6):
                    if t - n >= 0:
                        counts[n][tuple(tok[b, t - n:t])][tok[b, t]] += 1
    best = {n: {c: cnt.most_common(1)[0][0] for c, cnt in counts[n].items()} for n in range(6)}
    res = {}
    for s, (tok, rev) in eval_sets.items():
        tok = tok.numpy(); rev = rev.numpy()
        ok = np.zeros(6); tot = 0; okb = 0
        for b in range(tok.shape[0]):
            for t in np.nonzero(rev[b])[0]:
                tot += 1; okb += tok[b, t] == blank
                for n in range(6):
                    pred = None
                    for m in range(n, -1, -1):            # backoff
                        pred = best[m].get(tuple(tok[b, t - m:t]))
                        if pred is not None:
                            break
                    ok[n] += pred == tok[b, t]
        res[s] = {"blank": okb / tot, "ngram": (ok / tot).tolist(), "n_targets": tot}
    return res


# ----------------------------------------------------------------------------- model pieces
def load(arm, s, dev):
    ck = torch.load(f"{REPO}/runs/sign_matched/p0/{arm}_s{s}/{arm}.pt", map_location="cpu", weights_only=False)
    c = ck["config"]
    m = VARIANT_MAP[arm](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                         n_layers=c["n_layers"], grid_size=c["grid_size"])
    m.load_state_dict(ck["model_state_dict"])
    assert c["n_layers"] == 1 and len(m.layers) == 1
    return m.to(dev).eval()


def is_pope(m):
    return hasattr(m.layers[0], "pope_delta")


def phases(m, tokens):
    B, L = tokens.shape
    x = m.token_emb(tokens)
    if hasattr(m, "action_to_lie"):
        c, s = m.path_integrator(m.action_to_lie(x))
    elif is_pope(m):
        ang = torch.outer(torch.arange(L, device=tokens.device, dtype=x.dtype), m.theta_c)
        c = ang.cos()[None, None].expand(B, m.n_heads, L, -1); s = ang.sin()[None, None].expand(B, m.n_heads, L, -1)
    else:
        c, s = m._rope_cos_sin(L, tokens.device, x.dtype)
        c, s = c.expand(B, -1, -1, -1), s.expand(B, -1, -1, -1)
    return x, c, s


def heads(layer, z, B, T):
    return z.view(B, T, layer.n_heads, layer.d_head).transpose(1, 2)


def score(m, qr, kr, c, s, koff=None):
    """Pre-softmax score (B,H,T,T), same op sequence as the layer code. qr, kr: content reps (B,H,T,dh)
    -- Q, K for RoPE-form, softplus(Q), softplus(K) for PoPE-form. koff: extra key phase (B,H,T,nb)."""
    layer = m.layers[0]; dh = layer.d_head
    if is_pope(m):
        d = layer.pope_delta.clamp(DELTA_MIN, DELTA_MAX).view(1, layer.n_heads, 1, -1)
        cd, sd = torch.cos(d), torch.sin(d)
        cK, sK = c * cd - s * sd, s * cd + c * sd
        if koff is not None:
            co, so = torch.cos(koff), torch.sin(koff)
            cK, sK = cK * co - sK * so, sK * co + cK * so
        return (torch.matmul(qr * c, (kr * cK).transpose(-1, -2))
                + torch.matmul(qr * s, (kr * sK).transpose(-1, -2))) / math.sqrt(dh)
    Q, K = _apply_rope(qr, c, s), _apply_rope(kr, c, s)
    if koff is not None:
        K = _apply_rope(K, torch.cos(koff), torch.sin(koff))
    scale = math.sqrt(dh)
    if M._POW2_SCALE_FOLD and _is_pow2(scale):
        return torch.matmul(Q / scale, K.transpose(-1, -2))
    return torch.matmul(Q, K.transpose(-1, -2)) / scale


def finish(m, x, S, V):
    """Everything after the score, as in WMTransformerLayer(_PoPE).forward + readout (eval mode)."""
    layer = m.layers[0]; B, T, D = x.shape
    mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
    S = S.masked_fill(mask[None, None], float("-inf"))
    out = torch.matmul(F.softmax(S, dim=-1), V).transpose(1, 2).reshape(B, T, D)
    x = x + layer.o_proj(out)
    x = x + layer.ffn(layer.norm2(x))
    return m.out_proj(m.out_norm(x))


def seq_reps(m, x):
    """Content reps computed from the sequence exactly as the layer does (for the intact rebuild)."""
    layer = m.layers[0]; B, T, _ = x.shape
    h = layer.norm1(x)
    Q, K, V = heads(layer, layer.q_proj(h), B, T), heads(layer, layer.k_proj(h), B, T), heads(layer, layer.v_proj(h), B, T)
    if is_pope(m):
        Q, K = F.softplus(Q), F.softplus(K)
    return Q, K, V


def token_tables(m):
    """Per-token content reps (VOCAB, H, dh): in a 1-layer model these are exact functions of the id."""
    layer = m.layers[0]
    h = layer.norm1(m.token_emb.weight)
    q = layer.q_proj(h).view(VOCAB, layer.n_heads, layer.d_head)
    k = layer.k_proj(h).view(VOCAB, layer.n_heads, layer.d_head)
    if is_pope(m):
        q, k = F.softplus(q), F.softplus(k)
    return q, k


def gather(tab, tokens):
    """(VOCAB,H,dh) table -> (B,H,T,dh) by token id."""
    return tab[tokens].permute(0, 2, 1, 3)


def class_tables(tab, freq, mode):
    w = freq.clone().float()
    out = tab.clone()
    groups = [list(range(N_ACT)), list(range(N_ACT, VOCAB))] if mode == "class" else [list(range(VOCAB))]
    for g in groups:
        wg = w[g] / w[g].sum()
        out[g] = (tab[g] * wg[:, None, None]).sum(0, keepdim=True)
    return out


def accuracy(logits, tok, rev):
    pred = logits.argmax(-1); tgt = tok[:, 1:]; msk = rev[:, 1:]
    return float((pred[msk] == tgt[msk]).float().mean())


# ----------------------------------------------------------------------------- per checkpoint (chunked)
@torch.no_grad()
def calibrate(m, cal, dev):
    layer = m.layers[0]; H = layer.n_heads
    qt, kt = token_tables(m)
    tokC = cal[0].to(dev)[:, :-1]
    freq = torch.bincount(tokC.flatten(), minlength=VOCAB).float()
    qc, kc = class_tables(qt, freq, "class"), class_tables(kt, freq, "class")
    qg, kg = class_tables(qt, freq, "global"), class_tables(kt, freq, "global")
    xC, cC, sC = phases(m, tokC)
    SC_ = score(m, gather(qt, tokC), gather(kt, tokC), cC, sC)
    PC_ = score(m, gather(qc, tokC), gather(kc, tokC), cC, sC)
    B, Hh, T, _ = SC_.shape
    causal = torch.tril(torch.ones(T, T, device=dev, dtype=torch.bool))
    pair = (tokC[:, :, None] * VOCAB + tokC[:, None, :])
    idx = pair[causal.expand(B, T, T)]
    cnt = torch.zeros(VOCAB * VOCAB, device=dev, dtype=torch.float64)
    cnt.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float64))
    Ctab = torch.zeros(H, VOCAB * VOCAB, device=dev, dtype=torch.float64)
    Atab = torch.zeros(H, VOCAB * VOCAB, device=dev); Gtab = torch.ones(H, VOCAB * VOCAB, device=dev)
    for h in range(H):
        Pv = PC_[:, h][causal.expand(B, T, T)].double(); Sv = SC_[:, h][causal.expand(B, T, T)].double()
        Ctab[h].index_add_(0, idx, Sv - Pv)
        z = lambda v: torch.zeros(VOCAB * VOCAB, device=dev, dtype=torch.float64).index_add_(0, idx, v)
        sP, sS, sPP, sPS = z(Pv), z(Sv), z(Pv * Pv), z(Pv * Sv)
        n = cnt.clamp(min=1); vP = sPP - sP * sP / n
        ok = (cnt >= 50) & (vP > 1e-9 * n)
        gam = torch.where(ok, (sPS - sP * sS / n) / vP.clamp(min=1e-30), torch.ones_like(vP))
        Atab[h] = ((sS - gam * sP) / n).float(); Gtab[h] = gam.float()
    Ctab = (Ctab / cnt.clamp(min=1)).float().view(H, VOCAB, VOCAB)
    Atab, Gtab = Atab.view(H, VOCAB, VOCAB), Gtab.view(H, VOCAB, VOCAB)
    actrow = (tokC < N_ACT)

    def row_sd(Z):
        Zm = Z.masked_fill(~causal[None, None], float("nan"))
        mu = torch.nanmean(Zm, -1, keepdim=True)
        sd = torch.sqrt(torch.nanmean((Zm - mu) ** 2, -1))
        return torch.stack([sd[:, h][actrow & (torch.arange(T, device=dev) > 0)].mean() for h in range(H)])
    beta = row_sd(SC_) / row_sd(PC_)
    CpC = Ctab[:, tokC[:, :, None], tokC[:, None, :]].permute(1, 0, 2, 3)

    def row_ss(Z):
        Zm = Z.masked_fill(~causal[None, None], float("nan"))
        Zc = Zm - torch.nanmean(Zm, -1, keepdim=True)
        ss = torch.nansum(Zc ** 2, -1)
        return torch.stack([ss[:, h][actrow].sum() for h in range(H)])
    st = {"beta": beta.tolist(), "row_inter_share": (row_ss(SC_ - PC_ - CpC) / row_ss(SC_)).tolist(),
          "row_pos_share": (row_ss(PC_) / row_ss(SC_)).tolist(), "row_add_share": (row_ss(PC_ + CpC) / row_ss(SC_)).tolist()}
    GpC = (Atab[:, tokC[:, :, None], tokC[:, None, :]] + Gtab[:, tokC[:, :, None], tokC[:, None, :]]
           * PC_.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
    st["row_gain_resid_share"] = (row_ss(SC_ - GpC) / row_ss(SC_)).tolist()
    st["gain_gamma_sd"] = [float(Gtab[h][cnt.view(VOCAB, VOCAB) >= 50].std()) for h in range(H)]
    return dict(qt=qt, kt=kt, qc=qc, kc=kc, qg=qg, kg=kg, Ctab=Ctab, Atab=Atab, Gtab=Gtab, beta=beta), st


@torch.no_grad()
def run_one(m, ev, cal, dev):
    H = m.layers[0].n_heads
    K, st = calibrate(m, cal, dev)
    keys = ["intact", "pos_class", "pos_global", "pos_tmatch", "additive", "gain", "content"] + [f"lam{l:g}" for l in LAMS]
    hit = {k: 0 for k in keys}; tot = 0; vmax = 0.0; vtab = 0.0; hit_model = 0
    for i in range(0, ev[0].shape[0], CHUNK):
        tokE, revE = ev[0][i:i + CHUNK].to(dev), ev[1][i:i + CHUNK].to(dev)
        inpE = tokE[:, :-1]; msk = revE[:, 1:]; tgt = tokE[:, 1:]; tot += int(msk.sum())
        x, c, s = phases(m, inpE)
        Qs, Ks, V = seq_reps(m, x)
        S = score(m, Qs, Ks, c, s)
        lg_model = m(inpE); lg = finish(m, x, S, V)
        vmax = max(vmax, float((lg - lg_model).abs().max()))
        hit_model += int((lg_model.argmax(-1)[msk] == tgt[msk]).sum())
        S_tab = score(m, gather(K["qt"], inpE), gather(K["kt"], inpE), c, s)
        vtab = max(vtab, float((finish(m, x, S_tab, V) - lg_model).abs().max()))
        h = lambda Z: int((finish(m, x, Z, V).argmax(-1)[msk] == tgt[msk]).sum())
        P = score(m, gather(K["qc"], inpE), gather(K["kc"], inpE), c, s)
        Pg = score(m, gather(K["qg"], inpE), gather(K["kg"], inpE), c, s)
        pr = lambda tab: tab[:, inpE[:, :, None], inpE[:, None, :]].permute(1, 0, 2, 3)
        hit["intact"] += h(S); hit["pos_class"] += h(P); hit["pos_global"] += h(Pg)
        hit["pos_tmatch"] += h(P * K["beta"].view(1, H, 1, 1))
        hit["additive"] += h(P + pr(K["Ctab"])); hit["gain"] += h(pr(K["Atab"]) + pr(K["Gtab"]) * P)
        hit["content"] += h(score(m, Qs, Ks, torch.ones_like(c), torch.zeros_like(s)))
        for l in LAMS:
            hit[f"lam{l:g}"] += h(P + l * (S - P))
        del S, P, Pg, S_tab
    out = {k: v / tot for k, v in hit.items()}
    out.update(st, verify_maxabs=vmax, verify_table_maxabs_logit=vtab, acc_model=hit_model / tot, n_targets=tot)
    return out


def summarize(vals):
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); a = ap.parse_args()
    dev = torch.device(a.device); torch.set_num_threads(16)
    t0 = time.time()
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=ENV_SEED)
    stored = json.load(open(f"{REPO}/SIGN_MATCHED.json"))
    res = {"runs": {}}
    if os.path.exists(f"{OUT}/causal_whatwhere_T1024.json"):
        res = json.load(open(f"{OUT}/causal_whatwhere_T1024.json"))
    for arm in ARMS:
        res["runs"].setdefault(arm, {})
        for s in SEEDS:
            if str(s) in res["runs"][arm]:
                continue
            ev = trajectories(env, 1234 + s, N_TRIALS); cal = trajectories(env, 50000 + s, N_CAL)
            m = load(arm, s, dev)
            r = run_one(m, ev, cal, dev)
            r["acc_stored_eval"] = [x[1] for x in stored[f"0.0|{arm}|1024"] if x[0] == s][0]
            res["runs"][arm][str(s)] = r
            json.dump(res, open(f"{OUT}/causal_whatwhere_T1024.json", "w"), indent=1)
            print(f"{arm:10s} s{s} verify {r['verify_maxabs']:.1e} tab {r['verify_table_maxabs_logit']:.1e} intact {r['intact']:.4f} "
                  f"(model {r['acc_model']:.4f}, stored {r['acc_stored_eval']:.4f}) pos {r['pos_class']:.4f} posT {r['pos_tmatch']:.4f} "
                  f"gain {r['gain']:.4f} add {r['additive']:.4f} posG {r['pos_global']:.4f} cont {r['content']:.4f}  [{time.time() - t0:.0f}s]", flush=True)
            del m
    report(res)


def report(res):
    R = {a: {int(k): v for k, v in r.items()} for a, r in res["runs"].items()}
    L = []; p = L.append
    p("# What/where causal test at T = 1024 (causal_whatwhere_T1024.py; post hoc, eval-only)\n")
    p("Checkpoints: SIGN_MATCHED (torus, trained and tested at T = 1024, 1 layer, rank 4, n = 8). Floors on these walks "
      "(docs/audits/2026-10-16/sign_floors_out.txt): blank 0.506, n-gram 0.575, retrace 0.843.\n")
    p("| arm | max dlogit rebuilt vs model | via tables | intact | stored eval | max abs diff |"); p("|---|---|---|---|---|---|")
    for arm in ARMS:
        rs = [R[arm][s] for s in sorted(R[arm])]
        p(f"| {arm} | {max(r['verify_maxabs'] for r in rs):.1e} | {max(r['verify_table_maxabs_logit'] for r in rs):.1e} | "
          f"{np.mean([r['intact'] for r in rs]):.4f} | {np.mean([r['acc_stored_eval'] for r in rs]):.4f} | "
          f"{max(abs(r['intact'] - r['acc_stored_eval']) for r in rs):.4f} |")
    conds = [("intact", "intact"), ("pos_class", "position-only (object identity removed, type kept)"),
             ("pos_tmatch", "position-only, sharpness-matched"), ("gain", "shared kernel x content gain"),
             ("additive", "additive position + content"), ("pos_global", "one global mean (type removed too)"),
             ("content", "content-only (no rotation)")]
    p("\n## Accuracy, held-out map, T = 1024, mean +/- sd over seeds\n")
    p("| condition | " + " | ".join(ARMS) + " |"); p("|---" * (len(ARMS) + 1) + "|")
    get = lambda arm, key: [R[arm][s][key] for s in sorted(R[arm])]
    for key, name in conds:
        p(f"| {name} | " + " | ".join("%.3f +/- %.3f" % summarize(get(arm, key)) for arm in ARMS) + " |")
    p("\n## Paired change vs intact (per seed): mean, sd, exact-t MDE, t-test p, seeds lower\n")
    p("| condition | arm | delta | sd | MDE | p | lower |"); p("|---|---|---|---|---|---|---|")
    for key, name in conds[1:]:
        for arm in ARMS:
            d = np.array(get(arm, key)) - np.array(get(arm, "intact")); sd = d.std(ddof=1)
            p(f"| {name} | {arm} | {d.mean():+.3f} | {sd:.3f} | {SC.mde(sd, len(d)):.3f} | {SC.paired_p(d):.3g} | {(d < 0).sum()}/{len(d)} |")
    p("\n## graded: P + l (S - P)  [l = 0 position-only, 1 intact]\n")
    p("| l | " + " | ".join(ARMS) + " |"); p("|---" * (len(ARMS) + 1) + "|")
    for l in LAMS:
        p(f"| {l:g} | " + " | ".join("%.3f +/- %.3f" % summarize(get(arm, f"lam{l:g}")) for arm in ARMS) + " |")
    p("\n## Score variance within action-query rows (calibration set), mean over heads and seeds\n")
    p("| arm | SS(P)/SS(S) | SS(P+C)/SS(S) | SS(I)/SS(S) | SS(S - gain fit)/SS(S) | beta |"); p("|---|---|---|---|---|---|")
    for arm in ARMS:
        f = lambda k: np.mean([np.mean(v) for v in get(arm, k)])
        p(f"| {arm} | {f('row_pos_share'):.3f} | {f('row_add_share'):.3f} | {f('row_inter_share'):.3f} | {f('row_gain_resid_share'):.3f} | {f('beta'):.2f} |")
    txt = "\n".join(L) + "\n"
    open(f"{OUT}/causal_whatwhere_T1024.txt", "w").write(txt); print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report(json.load(open(f"{OUT}/causal_whatwhere_T1024.json")))
    else:
        main()

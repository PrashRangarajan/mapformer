"""Causal test of the what/where interaction in the layer-1 attention score (WHAT_WHERE_ANALYSIS.md
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

ARMS = ["RoPE", "PoPE-Flat", "Vanilla", "Vanilla_r4", "MapPoPE-Flat", "MapPoPE_r4"]
SEEDS = list(range(8))
T_STEPS, N_TRIALS, ENV_SEED = 128, 100, 10000
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
    ck = torch.load(f"{REPO}/runs/paper2x2/p0/{arm}_s{s}/{arm}.pt", map_location="cpu", weights_only=False)
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


# ----------------------------------------------------------------------------- per checkpoint
@torch.no_grad()
def run_one(m, ev, cal, dev, psi_seed):
    layer = m.layers[0]; H = layer.n_heads
    out = {}
    tokE, revE = ev[0].to(dev), ev[1].to(dev)
    inpE = tokE[:, :-1]
    # --- verification (rule 9): rebuilt forward vs the model's own logits, same batch
    x, c, s = phases(m, inpE)
    Qs, Ks, V = seq_reps(m, x)
    S = score(m, Qs, Ks, c, s)
    lg_mine = finish(m, x, S, V)
    lg_model = m(inpE)
    out["verify_maxabs"] = float((lg_mine - lg_model).abs().max())
    out["acc_model_batched"] = accuracy(lg_model, tokE, revE)
    # rebuild from the per-token tables (the form every intervention uses)
    qt, kt = token_tables(m)
    S_tab = score(m, gather(qt, inpE), gather(kt, inpE), c, s)
    out["verify_table_maxabs_logit"] = float((finish(m, x, S_tab, V) - lg_model).abs().max())
    out["intact"] = accuracy(lg_mine, tokE, revE)

    # --- calibration on separate trajectories
    tokC = cal[0].to(dev)[:, :-1]
    freq = torch.bincount(tokC.flatten(), minlength=VOCAB).float()
    qc, kc = class_tables(qt, freq, "class"), class_tables(kt, freq, "class")
    qg, kg = class_tables(qt, freq, "global"), class_tables(kt, freq, "global")
    xC, cC, sC = phases(m, tokC)
    SC_ = score(m, gather(qt, tokC), gather(kt, tokC), cC, sC)
    PC_ = score(m, gather(qc, tokC), gather(kc, tokC), cC, sC)
    B, Hh, T, _ = SC_.shape
    causal = torch.tril(torch.ones(T, T, device=dev, dtype=torch.bool))
    pair = (tokC[:, :, None] * VOCAB + tokC[:, None, :])          # (B,T,T)
    Ctab = torch.zeros(H, VOCAB * VOCAB, device=dev, dtype=torch.float64)
    cnt = torch.zeros(VOCAB * VOCAB, device=dev, dtype=torch.float64)
    idx = pair[causal.expand(B, T, T)]
    cnt.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float64))
    for h in range(H):
        Ctab[h].index_add_(0, idx, (SC_[:, h] - PC_[:, h])[causal.expand(B, T, T)].double())
    Ctab = (Ctab / cnt.clamp(min=1)).float().view(H, VOCAB, VOCAB)
    # gain surrogate (L3-like): per token pair, S ~ alpha + gamma * P by OLS over causal pairs
    Atab = torch.zeros(H, VOCAB * VOCAB, device=dev); Gtab = torch.ones(H, VOCAB * VOCAB, device=dev)
    for h in range(H):
        Pv = PC_[:, h][causal.expand(B, T, T)].double(); Sv = SC_[:, h][causal.expand(B, T, T)].double()
        z = lambda v: torch.zeros(VOCAB * VOCAB, device=dev, dtype=torch.float64).index_add_(0, idx, v)
        sP, sS, sPP, sPS = z(Pv), z(Sv), z(Pv * Pv), z(Pv * Sv)
        n = cnt.clamp(min=1)
        vP = sPP - sP * sP / n
        ok = (cnt >= 50) & (vP > 1e-9 * n)
        gam = torch.where(ok, (sPS - sP * sS / n) / vP.clamp(min=1e-30), torch.ones_like(vP))
        Atab[h] = ((sS - gam * sP) / n).float(); Gtab[h] = gam.float()
    Atab, Gtab = Atab.view(H, VOCAB, VOCAB), Gtab.view(H, VOCAB, VOCAB)
    out["gain_gamma_sd"] = [float(Gtab[h][cnt.view(VOCAB, VOCAB) >= 50].std()) for h in range(H)]
    # beta: restore the intact score's mean within-row sd on action-query rows
    actrow = (tokC < N_ACT)                                           # (B,T)

    def row_sd(Z):
        Zm = Z.masked_fill(~causal[None, None], float("nan"))
        mu = torch.nanmean(Zm, -1, keepdim=True)
        sd = torch.sqrt(torch.nanmean((Zm - mu) ** 2, -1))          # (B,H,T)
        return torch.stack([sd[:, h][actrow & (torch.arange(T, device=dev) > 0)].mean() for h in range(H)])
    beta = row_sd(SC_) / row_sd(PC_)
    out["beta"] = beta.tolist()
    # within-row interaction share of the calibration scores (action-query rows, causal keys)
    CpC = Ctab[:, tokC[:, :, None], tokC[:, None, :]].permute(1, 0, 2, 3)
    I = SC_ - PC_ - CpC

    def row_ss(Z):
        Zm = Z.masked_fill(~causal[None, None], float("nan"))
        Zc = Zm - torch.nanmean(Zm, -1, keepdim=True)
        ss = torch.nansum(Zc ** 2, -1)                                # (B,H,T)
        return torch.stack([ss[:, h][actrow].sum() for h in range(H)])
    out["row_inter_share"] = (row_ss(I) / row_ss(SC_)).tolist()
    out["row_pos_share"] = (row_ss(PC_) / row_ss(SC_)).tolist()
    out["row_add_share"] = (row_ss(PC_ + CpC) / row_ss(SC_)).tolist()
    GpC = (Atab[:, tokC[:, :, None], tokC[:, None, :]] + Gtab[:, tokC[:, :, None], tokC[:, None, :]]
           * PC_.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
    out["row_gain_resid_share"] = (row_ss(SC_ - GpC) / row_ss(SC_)).tolist()
    del SC_, PC_, CpC, I, GpC

    # --- interventions on the evaluation batch
    P = score(m, gather(qc, inpE), gather(kc, inpE), c, s)
    Pg = score(m, gather(qg, inpE), gather(kg, inpE), c, s)
    Cp = Ctab[:, inpE[:, :, None], inpE[:, None, :]].permute(1, 0, 2, 3)
    Iev = S - P - Cp
    acc = lambda Z: accuracy(finish(m, x, Z, V), tokE, revE)
    out["pos_class"] = acc(P)
    out["pos_global"] = acc(Pg)
    out["pos_tmatch"] = acc(P * beta.view(1, H, 1, 1))
    out["additive"] = acc(P + Cp)
    pr = lambda tab: tab[:, inpE[:, :, None], inpE[:, None, :]].permute(1, 0, 2, 3)
    out["gain"] = acc(pr(Atab) + pr(Gtab) * P)
    out["content"] = acc(score(m, Qs, Ks, torch.ones_like(c), torch.zeros_like(s)))
    out["lam_pos"] = {f"{l:g}": acc(P + l * (S - P)) for l in LAMS}
    out["lam_add"] = {f"{l:g}": acc(P + Cp + l * Iev) for l in LAMS}
    # per-pair random phase offsets (fixed draw per checkpoint)
    g = torch.Generator().manual_seed(psi_seed)
    nb = c.shape[-1]
    psi = torch.randn(VOCAB, VOCAB, H, nb, generator=g).to(dev)        # (query tok, key tok, H, nb)
    ph = {}
    for sg in SIGMAS:
        Z = torch.empty_like(S)
        for a in range(VOCAB):
            rows = (inpE == a)                                             # (B,T) queries with token a
            if not rows.any():
                continue
            koff = (sg * psi[a][inpE]).permute(0, 2, 1, 3)                # (B,H,T,nb) key offsets
            Sa = score(m, Qs, Ks, c, s, koff=koff)
            Z = torch.where(rows[:, None, :, None], Sa, Z)
        ph[f"{sg:g}"] = acc(Z)
    out["phase"] = ph
    return out


def summarize(vals):
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=ENV_SEED)
    ev = {s: trajectories(env, 1234 + s, N_TRIALS) for s in SEEDS}
    cal = {s: trajectories(env, 50000 + s, N_TRIALS) for s in SEEDS}
    fl = floors(ev)
    stored = json.load(open(f"{REPO}/_PAPER2X2_RAW.json"))
    res = {"floors": fl, "runs": {}}
    for arm in ARMS:
        res["runs"][arm] = {}
        for s in SEEDS:
            m = load(arm, s, dev)
            r = run_one(m, ev[s], cal[s], dev, psi_seed=9000 + s)
            r["acc_stored_eval"] = [x[1] for x in stored[f"0.0|{arm}|128"] if x[0] == s][0]
            res["runs"][arm][s] = r
            print(f"{arm:13s} s{s} verify {r['verify_maxabs']:.1e} tab {r['verify_table_maxabs_logit']:.1e} "
                  f"intact {r['intact']:.4f} (stored {r['acc_stored_eval']:.4f}) pos {r['pos_class']:.4f} "
                  f"posG {r['pos_global']:.4f} posT {r['pos_tmatch']:.4f} add {r['additive']:.4f} "
                  f"cont {r['content']:.4f}", flush=True)
            del m; torch.cuda.empty_cache()
    json.dump(res, open(f"{OUT}/causal_whatwhere.json", "w"), indent=1)
    print(f"elapsed {time.time() - t0:.0f} s")
    report(res)


def report(res):
    L = []
    p = L.append
    fl = {int(k): v for k, v in res["floors"].items()}
    res = dict(res, runs={a: {int(k): v for k, v in r.items()} for a, r in res["runs"].items()})
    blank = [fl[s]["blank"] for s in SEEDS]
    ng = np.array([fl[s]["ngram"] for s in SEEDS])
    p("# Causal what/where test (causal_whatwhere.py)\n")
    p(f"Floor (same 8 evaluation sets): always-blank {np.mean(blank):.3f} +/- {np.std(blank, ddof=1):.3f}; "
      f"token n-gram by order 0..5: " + " ".join(f"{v:.3f}" for v in ng.mean(0))
      + f"; best trivial predictor {max(np.mean(blank), ng.mean(0).max()):.3f}.\n")
    R = res["runs"]
    p("## Verification (rule 9)\n")
    p("| arm | max abs dlogit, rebuilt vs model(tokens) | via per-token tables | intact acc (batched) | stored eval acc (batch 1) | max abs acc diff |")
    p("|---|---|---|---|---|---|")
    for arm in ARMS:
        rs = [R[arm][s] for s in SEEDS]
        p(f"| {arm} | {max(r['verify_maxabs'] for r in rs):.2e} | {max(r['verify_table_maxabs_logit'] for r in rs):.2e} | "
          f"{np.mean([r['intact'] for r in rs]):.4f} | {np.mean([r['acc_stored_eval'] for r in rs]):.4f} | "
          f"{max(abs(r['intact'] - r['acc_stored_eval']) for r in rs):.4f} |")
    conds = [("intact", "intact"), ("pos_class", "position-only (class means)"),
             ("pos_tmatch", "position-only, sharpness-matched"), ("pos_global", "position-only (global mean)"),
             ("additive", "additive P + C"), ("gain", "gain alpha(c) + gamma(c) P"),
             ("content", "content-only (no rotation)")]
    p("\n## Accuracy, held-out map, T=128, mean +/- sd over 8 seeds\n")
    p("| condition | " + " | ".join(ARMS) + " |")
    p("|---" * (len(ARMS) + 1) + "|")
    get = lambda arm, key: [R[arm][s][key] for s in SEEDS]
    for key, name in conds:
        p(f"| {name} | " + " | ".join("%.3f +/- %.3f" % summarize(get(arm, key)) for arm in ARMS) + " |")
    p("\n## Paired change vs intact (per seed), exact-t MDE (stats_core), t-test p, seeds where the intervention is lower\n")
    p("| condition | arm | delta | sd | MDE | p | lower |")
    p("|---|---|---|---|---|---|---|")
    for key, name in conds[1:]:
        for arm in ARMS:
            d = np.array(get(arm, key)) - np.array(get(arm, "intact"))
            sd = d.std(ddof=1)
            p(f"| {name} | {arm} | {d.mean():+.3f} | {sd:.3f} | {SC.mde(sd, len(d)):.3f} | "
              f"{SC.paired_p(d):.3g} | {(d < 0).sum()}/8 |")
    for key, name, grid in (("lam_pos", "graded: P + l (S - P)  [l=0 position-only, 1 intact]", LAMS),
                            ("lam_add", "graded: P + C + l I  [l=0 additive, 1 intact]", LAMS),
                            ("phase", "graded lesion: per-pair random phase offset, sd (rad)", SIGMAS)):
        p(f"\n## {name}\n")
        p("| level | " + " | ".join(ARMS) + " |")
        p("|---" * (len(ARMS) + 1) + "|")
        for l in grid:
            p(f"| {l:g} | " + " | ".join("%.3f +/- %.3f" % summarize([v[f'{l:g}'] for v in get(arm, key)]) for arm in ARMS) + " |")
    p("\n## Score variance on the calibration set, within action-query rows (softmax-relevant), mean over heads and seeds\n")
    p("| arm | SS(P)/SS(S) | SS(P + C)/SS(S) | SS(I)/SS(S), I = S - P - C | SS(S - gain fit)/SS(S) | sd of gamma over pairs | beta (sharpness factor) |")
    p("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        f = lambda k: np.mean([np.mean(v) for v in get(arm, k)])
        p(f"| {arm} | {f('row_pos_share'):.3f} | {f('row_add_share'):.3f} | {f('row_inter_share'):.3f} | {f('row_gain_resid_share'):.3f} | {f('gain_gamma_sd'):.2f} | {f('beta'):.2f} |")
    txt = "\n".join(L) + "\n"
    open(f"{OUT}/causal_whatwhere.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report(json.load(open(f"{OUT}/causal_whatwhere.json")))
    else:
        main()

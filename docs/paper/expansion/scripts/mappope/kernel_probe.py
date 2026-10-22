"""MapPoPE expansion, ANALYSIS (post hoc, eval-only, CPU). Why does PoPE's polar score help a path phase and hurt an
index phase, and why does it not repair wrap-only revisits?

Two batches of stored checkpoints, evaluated on the stream of eval_noise_refine / eval_rank_strata exactly
(held-out map GridWorld(64, 16 obs, p_empty 0.5, seed 10000), np.random.seed(1234 + s), env.generate_trajectory(T)):
  paper2x2   runs/paper2x2/p0, T=128, arms RoPE, PoPE-Flat, Vanilla (MapWM r2), MapPoPE-Flat (r2), seeds 0-7
  score_rank runs/score_rank/p0, T=1024, arms Vanilla (A2), MapPoPE-Pair (P2), Vanilla_r4mi (A4), MapPoPE-Pair_r4mi (P4)

Layer 0 is rebuilt by hand (scores, softmax, values, FFN, readout) and checked against the model's own logits
(`verify` = max |dlogit|). For every scored revisit target o_t (query = the action token a_t at index 2t):
  strata      retrace class: R1 / R2 / R3+ = step j of a run that reverses the previous run (sign_floors.py's
              retrace rule: the target is the observation 2j steps back, token offset 4j-1); 'none' otherwise.
              wrap-only (unwrapped position new) / plain gap < 128 / plain gap >= 128 (eval_rank_strata.kinds_vec).
  mass_cell   attention mass on EARLIER OBSERVATION keys in the same (wrapped) cell, best head (max over heads
              of the run-mean), and per head.
  mass_rt     attention mass on the retrace target key (offset 4j-1), retrace queries only.
  mass_off1   attention mass on the observation key at offset 1 (the observation just before the query).
  prof        mean attention by token offset 0..24 (index arms' kernel in use).
  coherence   for the MOST RECENT same-cell observation key: C  = sum_c A_c cos(kernel phase_c) / sum_c A_c,
              C0 = the same with the content phase (RoPE form: psi_c = arg q_c conj k_c) or delta (PoPE form) removed.
              1 = the kernel peaks exactly on this key; C0 is the code alone, C - C0 what the score adds.
Content-pair kernel checks (no walks; 1-layer Q/K depend only on the token): for index arms the kernel
K_ao(D) over token offset D for action query a and observation key o; fraction of (head, a, o) with K(3) > K(1)
(needed to prefer the 1-step retrace target over the newest observation); PoPE delta dead-zone share (>= 0).

Run from /home/prashr:  CUDA_VISIBLE_DEVICES= python3 mapformer/docs/paper/expansion/scripts/mappope/kernel_probe.py
Outputs: kernel_probe_<batch>_out.txt, kernel_probe_<batch>.json next to this file.
HEAD_RULE=stratum (added 2026-10-21, docs/paper/review_draft5/ANALYSIS_SCRIPT_AUDIT.md B2) reads C, C0, mAct and mObsX
of each stratum on the head with the most same-cell mass in that stratum (default: the head with the most same-cell
mass over all revisits, used for every stratum); outputs then carry the suffix _hstratum.
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(8)
sys.path.insert(0, "/home/prashr")
from mapformer import train_variant  # noqa: E402
from mapformer.model_pope_pair import MapFormerWM_PoPEPair  # noqa: E402
from mapformer.model_pope_pair_mi import MapFormerWM_PoPEPair_r4mi  # noqa: E402
from mapformer.model_pope import DELTA_MIN, DELTA_MAX  # noqa: E402
from mapformer.model import _apply_rope  # noqa: E402
from mapformer.environment import GridWorld  # noqa: E402
from mapformer.eval_rank_strata import kinds_vec  # noqa: E402

VM = train_variant.VARIANT_MAP
VM["MapPoPE-Pair"] = MapFormerWM_PoPEPair
VM["MapPoPE-Pair_r4mi"] = MapFormerWM_PoPEPair_r4mi
REPO = "/home/prashr/mapformer"
HERE = os.path.dirname(os.path.abspath(__file__))
NA, NO = 4, 17
OFFMAX = 24
HEAD_RULE = os.environ.get("HEAD_RULE", "all")

BATCHES = {
    "paper2x2": dict(dir=f"{REPO}/runs/paper2x2/p0", T=128, trials=100, seeds=range(8),
                     arms=["RoPE", "PoPE-Flat", "Vanilla", "MapPoPE-Flat"]),
    "score_rank": dict(dir=f"{REPO}/runs/score_rank/p0", T=1024, trials=int(os.environ.get("SR_TRIALS", 40)),
                       seeds=None, arms=["Vanilla", "MapPoPE-Pair", "Vanilla_r4mi", "MapPoPE-Pair_r4mi"]),
}


def load(d, v, s):
    b = torch.load(f"{d}/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)
    c = b["config"]
    m = VM[v](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"], n_layers=c["n_layers"],
              grid_size=c["grid_size"])
    m.load_state_dict(b["model_state_dict"])
    return m.eval(), c


def is_pope(m):
    return hasattr(m.layers[0], "pope_delta")


def phases(m, tok):
    """Angles (H, L, n) exactly as the model builds cos/sin (per element for PoPE layers)."""
    L = tok.shape[1]; H = m.n_heads
    name = type(m).__name__
    if hasattr(m, "inv_freq"):                                   # RoPE index
        a = torch.outer(torch.arange(L, dtype=torch.float32), m.inv_freq)
        return a[None].expand(H, -1, -1)
    if hasattr(m, "theta_c"):                                    # PoPE index
        a = torch.outer(torch.arange(L, dtype=torch.float32), m.theta_c)
        return a[None].expand(H, -1, -1)
    x = m.token_emb(tok)
    a = (torch.cumsum(m.action_to_lie(x), 1) * m.path_integrator.omega)[0].transpose(0, 1)   # (H, L, nb)
    if name.startswith("MapFormerWM_PoPEPair"):
        a = a.repeat_interleave(2, -1)
    return a


@torch.no_grad()
def forward_manual(m, tok):
    """Layer-0 rebuild. Returns logits, attn (H, L, L), Q, K (H, L, dh) pre-rotation, phases (H, L, n)."""
    lay = m.layers[0]; B, L = tok.shape; H, dh = m.n_heads, lay.d_head
    x = m.token_emb(tok); h = lay.norm1(x)
    Q = lay.q_proj(h).view(B, L, H, dh).transpose(1, 2)
    K = lay.k_proj(h).view(B, L, H, dh).transpose(1, 2)
    V = lay.v_proj(h).view(B, L, H, dh).transpose(1, 2)
    ang = phases(m, tok)
    ca, sa = ang.cos()[None], ang.sin()[None]
    if is_pope(m):
        mq, mk = F.softplus(Q), F.softplus(K)
        d = lay.pope_delta.clamp(DELTA_MIN, DELTA_MAX).view(1, H, 1, -1)
        cK = ca * d.cos() - sa * d.sin(); sK = sa * d.cos() + ca * d.sin()
        S = (torch.matmul(mq * ca, (mk * cK).transpose(-1, -2)) + torch.matmul(mq * sa, (mk * sK).transpose(-1, -2))) / math.sqrt(dh)
    else:
        S = torch.matmul(_apply_rope(Q, ca, sa), _apply_rope(K, ca, sa).transpose(-1, -2)) / math.sqrt(dh)
    S = S.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), 1), float("-inf"))
    A = F.softmax(S, -1)
    out = lay.o_proj(torch.matmul(A, V).transpose(1, 2).reshape(B, L, -1))
    x = x + out; x = x + lay.ffn(lay.norm2(x))
    logits = m.out_proj(m.out_norm(x))
    return logits, A[0], Q[0], K[0], ang


def retrace_class(a):
    """Per step t: j (step within a run that reverses the previous run, j <= previous run length) or 0."""
    T = len(a); out = np.zeros(T, dtype=np.int64); prev_run = cur = 0
    for t in range(T):
        if t > 0 and a[t] == a[t - 1]:
            cur += 1
        else:
            prev_run = cur if (t > 0 and a[t] == (a[t - 1] ^ 1)) else 0
            cur = 1
        if prev_run > 0 and cur <= prev_run and t - 2 * cur >= 0:
            out[t] = cur
    return out


@torch.no_grad()
def coherence(m, Q, K, ang, tq, ts):
    """C, C0 per head for query tokens tq and key tokens ts (arrays)."""
    lay = m.layers[0]; H = m.n_heads
    dphi = ang[:, tq] - ang[:, ts]                                  # (H, P, n)
    if is_pope(m):
        A = F.softplus(Q[:, tq]) * F.softplus(K[:, ts])              # (H, P, dh)
        d = lay.pope_delta.clamp(DELTA_MIN, DELTA_MAX)[:, None, :]
        C = (A * torch.cos(dphi - d)).sum(-1) / A.sum(-1)
        C0 = (A * torch.cos(dphi)).sum(-1) / A.sum(-1)
    else:
        qc = torch.complex(Q[:, tq, 0::2], Q[:, tq, 1::2]); kc = torch.complex(K[:, ts, 0::2], K[:, ts, 1::2])
        Z = qc * kc.conj(); A = Z.abs(); psi = Z.angle()
        C = (A * torch.cos(dphi + psi)).sum(-1) / A.sum(-1)
        C0 = (A * torch.cos(dphi)).sum(-1) / A.sum(-1)
    return C.numpy(), C0.numpy()


@torch.no_grad()
def content_kernel_checks(m):
    """Index arms: fraction of (h, a, o) with K(3) > K(1), and argmax over D in 1..16 of K_ao(D)."""
    lay = m.layers[0]; H, dh = m.n_heads, lay.d_head; E = m.token_emb.weight
    hq = lay.norm1(E[:NA]); hk = lay.norm1(E[NA:NA + NO])
    q = lay.q_proj(hq).view(NA, H, dh); k = lay.k_proj(hk).view(NO, H, dh)
    D = torch.arange(0, 17, dtype=torch.float32)
    out = {}
    if hasattr(m, "theta_c"):
        th = m.theta_c; mq, mk = F.softplus(q), F.softplus(k)
        d = lay.pope_delta.clamp(DELTA_MIN, DELTA_MAX)                                   # (H, dh)
        Kd = torch.einsum("ahc,ohc,hcd->hao d".replace(" ", ""), mq, mk, torch.cos(th[None, :, None] * D[None, None] - d[..., None]))
        out["delta_deadzone_share"] = float((lay.pope_delta >= 0).float().mean())
        out["delta_abs_mean"] = float(lay.pope_delta.clamp(DELTA_MIN, DELTA_MAX).abs().mean())
    elif hasattr(m, "inv_freq"):
        th = m.inv_freq
        qc = torch.complex(q[..., 0::2], q[..., 1::2]); kc = torch.complex(k[..., 0::2], k[..., 1::2])
        Z = qc[:, None] * kc[None].conj()                                                # (a, o, H, nb)
        Kd = (Z[..., None] * torch.exp(1j * th[None, None, None, :, None] * D)).real.sum(-2)  # (a, o, H, D)
        Kd = Kd.permute(2, 0, 1, 3)
    else:                                                       # path arms: zero-displacement score, action vs observation keys
        ka = lay.k_proj(lay.norm1(E[:NA])).view(NA, H, dh)
        if hasattr(lay, "pope_delta"):
            d = lay.pope_delta.clamp(DELTA_MIN, DELTA_MAX)
            mq = F.softplus(q)
            g = lambda kk: torch.einsum("ahc,khc,hc->hak", mq, F.softplus(kk), torch.cos(d))
            out["K0_obs"] = float(g(k).median()); out["K0_act"] = float(g(ka).median())
            out["amp_obs"] = out["K0_obs"]; out["amp_act"] = out["K0_act"]
            out["kmag_obs"] = float(F.softplus(k).mean()); out["kmag_act"] = float(F.softplus(ka).mean())
        else:
            qc = torch.complex(q[..., 0::2], q[..., 1::2])
            def z(kk):
                kc = torch.complex(kk[..., 0::2], kk[..., 1::2]); return qc[:, None] * kc[None].conj()   # (a, k, H, nb)
            Zo, Za = z(k), z(ka)
            out["K0_obs"] = float(Zo.real.sum(-1).median()); out["K0_act"] = float(Za.real.sum(-1).median())
            out["amp_obs"] = float(Zo.abs().sum(-1).median()); out["amp_act"] = float(Za.abs().sum(-1).median())
            out["cospsi_obs"] = float((Zo.real.sum(-1) / Zo.abs().sum(-1)).median())
            out["cospsi_act"] = float((Za.real.sum(-1) / Za.abs().sum(-1)).median())
        return out
    Kd = Kd.reshape(H, NA, NO, -1)
    out["frac_K3_gt_K1"] = float((Kd[..., 3] > Kd[..., 1]).float().mean())
    out["frac_K3_gt_K1_per_head"] = [float((Kd[h, ..., 3] > Kd[h, ..., 1]).float().mean()) for h in range(H)]
    am = Kd[..., 1:].argmax(-1) + 1
    out["argmax_offset_hist"] = np.bincount(am.flatten().numpy(), minlength=17).tolist()
    return out


def run_batch(name, cfg):
    res = {}
    for v in cfg["arms"]:
        seeds = cfg["seeds"]
        if seeds is None:
            seeds = sorted(int(x.rsplit("_s", 1)[1]) for x in os.listdir(cfg["dir"]) if x.rsplit("_s", 1)[0] == v)
        for s in seeds:
            m, c = load(cfg["dir"], v, s)
            env = GridWorld(size=c["grid_size"], n_obs_types=c.get("n_obs_types", 16), p_empty=c.get("p_empty", 0.5),
                            seed=10000)
            np.random.seed(1234 + s)
            H = m.n_heads
            rec = defaultdict(list); prof = np.zeros((H, OFFMAX + 1)); nprof = 0; verify = 0.0
            for _ in range(cfg["trials"]):
                tok, _o, rev = env.generate_trajectory(cfg["T"])
                locs = env.visited_locations
                inp = tok[None, :-1]
                logits, A, Q, K, ang = forward_manual(m, inp)
                if _ < 2:
                    verify = max(verify, float((logits - m(inp)).abs().max().detach()))
                pred = logits[0].argmax(-1).numpy(); tgt = tok[1:].numpy(); msk = rev[1:].numpy()
                a = tok[0::2].numpy(); cell = np.array([x * 1000 + y for x, y in locs])
                rc = retrace_class(a); new_u, lag = kinds_vec(tok, env.size)
                An = A.numpy()
                tq_l, ts_l = [], []
                for i in np.nonzero(msk)[0]:
                    t = i // 2; qi = 2 * t
                    same = np.nonzero(cell[:t] == cell[t])[0]
                    keys = 2 * same + 1
                    rec["ok"].append(pred[qi] == tgt[qi])
                    rec["rc"].append(min(rc[t], 3)); rec["wrap"].append(bool(new_u[t])); rec["lag"].append(int(lag[t]))
                    rec["mass_cell"].append(An[:, qi, keys].sum(-1))
                    rec["mass_off1"].append(An[:, qi, qi - 1])
                    rec["mass_act"].append(An[:, qi, 0:qi + 1:2].sum(-1))          # all action keys (incl. self)
                    rec["mass_obs_other"].append(An[:, qi, 1:qi:2].sum(-1) - An[:, qi, keys].sum(-1))
                    rec["mass_rt"].append(An[:, qi, qi - (4 * rc[t] - 1)] if rc[t] > 0 else np.full(H, np.nan))
                    tq_l.append(qi); ts_l.append(keys[-1])
                    if qi >= OFFMAX:
                        prof += An[:, qi, qi - np.arange(OFFMAX + 1)]; nprof += 1
                if tq_l:
                    C, C0 = coherence(m, Q, K, ang, np.array(tq_l), np.array(ts_l))
                    rec["C"].extend(list(C.T)); rec["C0"].extend(list(C0.T))
            R = {k: np.array(x) for k, x in rec.items()}
            res[(v, s)] = dict(R=R, prof=prof / max(nprof, 1), verify=verify, kern=content_kernel_checks(m))
            print(f"{name} {v:18s} s{s:<3d} acc {R['ok'].mean():.4f} n {len(R['ok'])} verify {verify:.1e}", flush=True)
    return res


def summarise(name, res, arms, lines):
    L = lines.append
    L(f"\n=== {name} ===")
    strata = {
        "all": lambda R: np.ones(len(R["ok"]), bool),
        "R1 (1-step reversal)": lambda R: R["rc"] == 1, "R2": lambda R: R["rc"] == 2, "R3+": lambda R: R["rc"] == 3,
        "not retrace": lambda R: R["rc"] == 0,
        "plain gap<128": lambda R: (~R["wrap"]) & (R["lag"] < 128), "plain gap>=128": lambda R: (~R["wrap"]) & (R["lag"] >= 128),
        "wrap-only": lambda R: R["wrap"],
    }
    J = {}
    for v in arms:
        runs = [res[k] for k in res if k[0] == v]
        if not runs:
            continue
        L(f"\n-- {v} (n={len(runs)} seeds; verify max {max(r['verify'] for r in runs):.1e}) --")
        L(f"{'stratum':22s} {'share':>6s} {'acc':>7s} {'massCell':>9s} {'massRT':>7s} {'massOff1':>8s} {'C':>6s} {'C0':>6s} {'mAct':>6s} {'mObsX':>6s}   (medians over seeds; masses = best head)")
        for sn, f in strata.items():
            row = []
            for r in runs:
                R = r["R"]; sel = f(R)
                if sel.sum() < 5:
                    continue
                mc = R["mass_cell"][sel].mean(0)
                # HEAD_RULE=all (default; the committed outputs): best head over ALL revisits, used for every stratum.
                # HEAD_RULE=stratum (ANALYSIS_SCRIPT_AUDIT.md B2): the head with most same-cell mass IN THIS STRATUM.
                h = int(np.argmax(mc)) if HEAD_RULE == "stratum" else int(np.argmax(R["mass_cell"].mean(0)))
                mrt = np.nanmean(R["mass_rt"][sel], 0) if np.isfinite(R["mass_rt"][sel]).any() else np.full(2, np.nan)
                row.append((sel.mean(), R["ok"][sel].mean(), mc[h], np.nanmax(mrt) if np.isfinite(mrt).any() else np.nan,
                            R["mass_off1"][sel].mean(0).max(), R["C"][sel][:, h].mean(), R["C0"][sel][:, h].mean(),
                            R["mass_act"][sel][:, h].mean(), R["mass_obs_other"][sel][:, h].mean()))
            if not row:
                continue
            a = np.array(row, dtype=float); md = np.nanmedian(a, 0)
            J[(v, sn)] = a.tolist()
            L(f"{sn:22s} {md[0]:6.3f} {md[1]:7.4f} {md[2]:9.3f} {md[3]:7.3f} {md[4]:8.3f} {md[5]:6.3f} {md[6]:6.3f} {md[7]:6.3f} {md[8]:6.3f}")
        kk = [r["kern"] for r in runs if r["kern"]]
        if kk and "K0_obs" in kk[0]:
            L("zero-displacement score (median over h, a, key): obs keys " + " ".join(f"{x['K0_obs']:.1f}" for x in kk)
              + " | action keys " + " ".join(f"{x['K0_act']:.1f}" for x in kk))
            L("  amplitude sum_c|.| obs " + " ".join(f"{x['amp_obs']:.1f}" for x in kk) + " | act " + " ".join(f"{x['amp_act']:.1f}" for x in kk))
            if "cospsi_obs" in kk[0]:
                L("  amplitude-weighted cos(psi): obs keys " + " ".join(f"{x['cospsi_obs']:+.2f}" for x in kk)
                  + " | action keys " + " ".join(f"{x['cospsi_act']:+.2f}" for x in kk))
            else:
                L("  mean key magnitude softplus(k): obs " + " ".join(f"{x['kmag_obs']:.2f}" for x in kk) + " | act " + " ".join(f"{x['kmag_act']:.2f}" for x in kk))
        elif kk:
            L("content-pair kernel: frac(K(3) > K(1)) per run: " + " ".join(f"{x['frac_K3_gt_K1']:.2f}" for x in kk))
            if "delta_deadzone_share" in kk[0]:
                L("PoPE delta: share >= 0 (dead zone, gradient 0) per run: " + " ".join(f"{x['delta_deadzone_share']:.2f}" for x in kk)
                  + "; mean |delta| " + " ".join(f"{x['delta_abs_mean']:.2f}" for x in kk))
            hist = np.sum([x["argmax_offset_hist"] for x in kk], 0)
            L("argmax_D in 1..16 of K_ao(D), pooled (h, a, o) counts: " + " ".join(f"{d}:{int(c)}" for d, c in enumerate(hist) if c))
        if name == "paper2x2":
            P = np.median([r["prof"].max(0) for r in runs], 0)
            L("attention by token offset 0..12 (max over heads, median over seeds): " + " ".join(f"{x:.3f}" for x in P[:13]))
            Pb = np.median([r["prof"][int(np.argmax(r["prof"][:, 3]))] for r in runs], 0)
            L("  head with most offset-3 mass, offsets 0..12:                        " + " ".join(f"{x:.3f}" for x in Pb[:13]))
    return J


def main():
    which = sys.argv[1:] or list(BATCHES)
    lines = []; js = {}
    for name in which:
        res = run_batch(name, BATCHES[name])
        js[name] = summarise(name, res, BATCHES[name]["arms"], lines)
        # per-seed accuracies and per-seed stratum rows for downstream tests
        js[name + "_perseed"] = {f"{v}|{s}": {"acc": float(r["R"]["ok"].mean()),
                                              "R1_acc": float(r["R"]["ok"][r["R"]["rc"] == 1].mean()),
                                              "wrap_acc": float(r["R"]["ok"][r["R"]["wrap"]].mean()) if r["R"]["wrap"].any() else None,
                                              "kern": r["kern"]} for (v, s), r in res.items()}
        print("\n".join(lines), flush=True)
    tag = "_".join(which) + ("_hstratum" if HEAD_RULE == "stratum" else "")
    open(f"{HERE}/kernel_probe_{tag}_out.txt", "w").write("\n".join(lines) + "\n")
    json.dump({k: {str(kk): vv for kk, vv in v.items()} for k, v in js.items()}, open(f"{HERE}/kernel_probe_{tag}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

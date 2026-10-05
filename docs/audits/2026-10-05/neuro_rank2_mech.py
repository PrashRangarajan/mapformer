"""Why does PoPE's score rescue rank 2 at T=128? Post hoc, eval-only, CPU (docs/theory/2026-10-05/neuro_design.md sec. 2).

Checkpoints: runs/mappope_pair/p0 (the registered MAPPOPE_PAIR batch; paper torus, T=128, 1 layer, 2 heads, 300 ep cosine):
  MapWM r2 (Vanilla) s10-25, MapPoPE-Pair r2 s10-25, MapPoPE r2 (64 angles) s10-25, and the r4 arms s10-17.
Registered accuracy per seed is read from MAPPOPE_PAIR_R2.json / _R4.json (T=128 column), not recomputed.

Readouts per checkpoint:
 1. Phase-code geometry per head (03_formal.md Theorem 1; probe_winding2.py logic, generalised to PoPE channels):
    K = per-channel phase per unit move on each axis, tau = per-move common phase (mean action step + expected
    observation step), w = channel weight (RoPE form: mean_a |q_b(a)| * E_o |k_b(o)| per 2-D block; PoPE: the same with
    softplus magnitudes per element, summed over the two elements that share an angle in the pairwise model).
    cond = s_min/s_max of the w-weighted frame (0 = collinear axes, COLLAPSE), clk = |w^.5 tau| / |w^.5 K|_F (DRIFT).
    Class without the lattice criterion (T=128 has essentially no wraps): A if clk >= 0.05, else B if cond <= 0.25, else ok.
 2. Retrieval on real held-out sequences (GridWorld 64, 16 objects, p_empty 0.5, env seed 10000, 60 walks of 128
    steps, np seed 4321): for every revisit target, the query is the preceding action token; keys are earlier
    observation tokens. top1 = fraction of queries whose highest-scoring observation key lies in the query's cell
    (best head). The rebuilt forward is verified against model(tokens).
 3. Content-phase intervention (RoPE-form arms only): replace each 2-D block's content q_b, k_b by (|q_b|, 0), (|k_b|, 0)
    -- content keeps every amplitude but loses its phase, so the score becomes PoPE-like sum_b |q_b||k_b| cos(dtheta_b)
    with no content phase. Accuracy on the same walks (the network after the score is untouched).
    And the PoPE-form converse: delta forced to 0 (exact peak invariance).
 4. psi_disp (RoPE-form): amplitude-weighted circular dispersion of the content phase psi_b(a,o) = arg k_b(o) - arg q_b(a)
    across the 68 action x observation pairs (0 = one shared offset, content cannot move peaks; 1 = uniform).
Run: cd /home/prashr && PYTHONPATH=/home/prashr python3 mapformer/docs/audits/2026-10-05/neuro_rank2_mech.py
"""
import json, math, sys
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(4)
sys.path.insert(0, "/home/prashr")
from mapformer import train_variant
from mapformer.model import _apply_rope
from mapformer.model_pope import DELTA_MIN, DELTA_MAX
from mapformer.model_pope_pair import MapFormerWM_PoPEPair, MapFormerWM_PoPEPair_r4
from mapformer.environment import GridWorld
VM = train_variant.VARIANT_MAP
VM["MapPoPE-Pair"] = MapFormerWM_PoPEPair; VM["MapPoPE-Pair_r4"] = MapFormerWM_PoPEPair_r4
REPO = "/home/prashr/mapformer"; RUN = f"{REPO}/runs/mappope_pair/p0"
NA, V = 4, 21; PLUS, MINUS = [1, 3], [0, 2]          # GridWorld: N=-x S=+x W=-y E=+y
MOVE = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def load(arm, s):
    ck = torch.load(f"{RUN}/{arm}_s{s}/{arm}.pt", map_location="cpu", weights_only=False); c = ck["config"]
    m = VM[arm](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"], n_layers=1, grid_size=c["grid_size"])
    m.load_state_dict(ck["model_state_dict"]); return m.eval()


def kind(m):
    n = type(m).__name__
    return "pair" if n.startswith("MapFormerWM_PoPEPair") else ("pope" if hasattr(m.layers[0], "pope_delta") else "rope")


@torch.no_grad()
def tables(m):
    lay = m.layers[0]; h = lay.norm1(m.token_emb.weight); H, dh = m.n_heads, lay.d_head
    q = lay.q_proj(h).view(V, H, dh); k = lay.k_proj(h).view(V, H, dh)
    return q, k


@torch.no_grad()
def geometry(m):
    k_ = kind(m); q, k = tables(m)
    st = (m.action_to_lie(m.token_emb.weight[None])[0] * m.path_integrator.omega).numpy()   # (V, H, nb)
    if k_ == "rope":
        qa = torch.hypot(q[..., 0::2], q[..., 1::2]).numpy(); ka = torch.hypot(k[..., 0::2], k[..., 1::2]).numpy()
    else:
        qa, ka = F.softplus(q).numpy(), F.softplus(k).numpy()
        if k_ == "pair":
            pass                                         # per element; summed over each pair below
    wobs = np.array([0.5 / 16] * 16 + [0.5])
    out = []
    for hh in range(m.n_heads):
        P = st[:, hh]
        K = np.stack([(P[PLUS[j]] - P[MINUS[j]]) / 2 for j in range(2)], 1)
        tau = P[:NA].mean(0) + (wobs[:, None] * P[NA:]).sum(0)
        A = qa[:NA, hh].mean(0) * (wobs[:, None] * ka[NA:, hh]).sum(0)
        if k_ == "pair":
            A = A.reshape(-1, 2).sum(1)
        w = A / A.sum(); sw = np.sqrt(w)
        Kw, tw = K * sw[:, None], tau * sw
        sv = np.linalg.svd(Kw, compute_uv=False)
        cond, clk = sv[-1] / sv[0], np.linalg.norm(tw) / np.linalg.norm(Kw)
        cls = "A" if clk >= 0.05 else ("B" if cond <= 0.25 else "ok")
        r = dict(cond=float(cond), clk=float(clk), cls=cls)
        if k_ == "rope":                                  # content-phase dispersion over the 68 pairs
            qc = torch.complex(q[:NA, hh, 0::2], q[:NA, hh, 1::2]); kc = torch.complex(k[NA:, hh, 0::2], k[NA:, hh, 1::2])
            z = kc[None] * qc[:, None].conj()             # (4, 17, nb): amplitude * e^{i psi}
            amp = z.abs(); u = z / amp.clamp_min(1e-12)
            mean_dir = (amp * u).sum((0, 1)); mean_dir = mean_dir / mean_dir.abs().clamp_min(1e-12)
            R = (amp * (u * mean_dir.conj()).real).sum((0, 1)) / amp.sum((0, 1))   # per-channel resultant length
            wch = amp.sum((0, 1)); r["psi_disp"] = float(1 - (wch * R).sum() / wch.sum())
        else:
            d = m.layers[0].pope_delta.clamp(DELTA_MIN, DELTA_MAX)[hh]
            r["delta_at0"] = float((d.abs() < 1e-6).float().mean())
        out.append(r)
    return out


def walks(n=60, T=128):
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000); np.random.seed(4321)
    toks, revs = [], []
    for _ in range(n):
        t, _o, r = env.generate_trajectory(T); toks.append(t); revs.append(r)
    toks, revs = torch.stack(toks), torch.stack(revs)
    acts = toks[:, 0::2].numpy()
    pos = np.zeros((n, T, 2), int)
    for b in range(n):
        p = np.zeros(2, int)
        for i in range(T):
            p = (p + MOVE[int(acts[b, i])]) % 64; pos[b, i] = p
    return toks, revs, pos


@torch.no_grad()
def forward(m, toks, mode="intact"):
    lay = m.layers[0]; k_ = kind(m); x = m.token_emb(toks); B, L, D = x.shape; H, dh = m.n_heads, lay.d_head
    c, s = m.path_integrator(m.action_to_lie(x))
    if k_ == "pair":
        c, s = c.repeat_interleave(2, -1), s.repeat_interleave(2, -1)
    h = lay.norm1(x); sh = lambda z: z.view(B, L, H, dh).transpose(1, 2)
    Q, K, Vv = sh(lay.q_proj(h)), sh(lay.k_proj(h)), sh(lay.v_proj(h))
    if k_ == "rope":
        if mode == "strip_psi":
            Q = torch.stack([torch.hypot(Q[..., 0::2], Q[..., 1::2]), torch.zeros_like(Q[..., 0::2])], -1).flatten(-2)
            K = torch.stack([torch.hypot(K[..., 0::2], K[..., 1::2]), torch.zeros_like(K[..., 0::2])], -1).flatten(-2)
        S = _apply_rope(Q, c, s) @ _apply_rope(K, c, s).transpose(-1, -2) / math.sqrt(dh)
        if mode == "strip_obs":                       # strip psi only where the key is an observation token
            Q2 = torch.stack([torch.hypot(Q[..., 0::2], Q[..., 1::2]), torch.zeros_like(Q[..., 0::2])], -1).flatten(-2)
            K2 = torch.stack([torch.hypot(K[..., 0::2], K[..., 1::2]), torch.zeros_like(K[..., 0::2])], -1).flatten(-2)
            S2 = _apply_rope(Q2, c, s) @ _apply_rope(K2, c, s).transpose(-1, -2) / math.sqrt(dh)
            obs_key = (toks >= NA)[:, None, None, :]
            S = torch.where(obs_key, S2, S)
    else:
        mq, mk = F.softplus(Q), F.softplus(K)
        d = lay.pope_delta.clamp(DELTA_MIN, DELTA_MAX).view(1, H, 1, -1)
        if mode == "delta0":
            d = torch.zeros_like(d)
        cd, sd = torch.cos(d), torch.sin(d); cK, sK = c * cd - s * sd, s * cd + c * sd
        S = ((mq * c) @ (mk * cK).transpose(-1, -2) + (mq * s) @ (mk * sK).transpose(-1, -2)) / math.sqrt(dh)
    S = S.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), 1), float("-inf"))
    out = (F.softmax(S, -1) @ Vv).transpose(1, 2).reshape(B, L, D)
    x2 = x + lay.o_proj(out); x2 = x2 + lay.ffn(lay.norm2(x2))
    return m.out_proj(m.out_norm(x2)), S


def readouts(m, toks, revs, pos):
    inp, tgt, rv = toks[:, :-1], toks[:, 1:], revs[:, 1:]
    logits, S = forward(m, inp)
    err = (logits - m(inp)).abs().max().item()
    acc = (logits.argmax(-1) == tgt)[rv].float().mean().item()
    # top-1 key in the right cell: query = action at index 2i (predicts obs at 2i+1, a revisit); keys = obs at 2j+1, j < i
    hits = np.zeros(m.n_heads); nq = 0
    Sn = S.numpy()
    for b in range(inp.shape[0]):
        for i in range(1, 128):
            if not bool(revs[b, 2 * i + 1]):
                continue
            q = 2 * i; keys = np.arange(1, q, 2); same = (pos[b, (keys - 1) // 2] == pos[b, i]).all(1)
            if not same.any():
                continue
            nq += 1
            for hh in range(m.n_heads):
                hits[hh] += same[np.argmax(Sn[b, hh, q, keys])]
    r = dict(acc=acc, verify=err, top1_best=float(hits.max() / nq), top1_heads=(hits / nq).tolist())
    if kind(m) == "rope":
        lg, _ = forward(m, inp, "strip_psi"); r["acc_strip_psi"] = (lg.argmax(-1) == tgt)[rv].float().mean().item()
    else:
        lg, _ = forward(m, inp, "delta0"); r["acc_delta0"] = (lg.argmax(-1) == tgt)[rv].float().mean().item()
    return r


def main():
    toks, revs, pos = walks()
    reg = {}
    for f in ("MAPPOPE_PAIR_R2.json", "MAPPOPE_PAIR_R4.json"):
        for key, rows in json.load(open(f"{REPO}/{f}")).items():
            _, arm, T = key.split("|")
            if T == "128":
                for s, a, _l in rows:
                    reg[(arm, int(s))] = a
    sets = [("Vanilla", range(10, 26)), ("MapPoPE-Pair", range(10, 26)), ("MapPoPE-Flat", range(10, 26)),
            ("Vanilla_r4", range(10, 18)), ("MapPoPE-Pair_r4", range(10, 18)), ("MapPoPE_r4", range(10, 18))]
    allr = {}
    for arm, seeds in sets:
        rows = []
        for s in seeds:
            m = load(arm, s); g = geometry(m); r = readouts(m, toks, revs, pos)
            ck = torch.load(f"{RUN}/{arm}_s{s}/{arm}.pt", map_location="cpu", weights_only=False)
            L = np.array(ck["losses"]); fl = float(L[-max(1, len(L) // 20):].mean())
            r.update(seed=s, heads=g, final_loss=fl, reg_acc=reg.get((arm, s)))
            rows.append(r)
            fmt = lambda key, f: " ".join(format(h[key], f) for h in g)
            ex = ("strip %.4f psi_disp %s" % (r["acc_strip_psi"], fmt("psi_disp", ".2f")) if "acc_strip_psi" in r
                  else "delta0 %.4f d@0 %s" % (r["acc_delta0"], fmt("delta_at0", ".2f")))
            print("%-16s s%2d reg %.4f acc %.4f loss %.4f %s cls %s cond %s clk %s top1 %.3f %s verify %.1e" % (
                arm, s, r["reg_acc"], r["acc"], fl, "SOLVED" if fl < 0.05 else "      ", "".join(h["cls"][0] for h in g),
                fmt("cond", ".2f"), fmt("clk", ".3f"), r["top1_best"], ex, r["verify"]), flush=True)
        allr[arm] = rows
        cls = [h["cls"] for r in rows for h in r["heads"]]
        print(f"== {arm}: heads ok {cls.count('ok')} A {cls.count('A')} B {cls.count('B')}; SOLVED {sum(r['final_loss'] < 0.05 for r in rows)}/{len(rows)}; "
              f"mean acc {np.mean([r['acc'] for r in rows]):.4f}; seeds with an ok head {sum(any(h['cls'] == 'ok' for h in r['heads']) for r in rows)}", flush=True)
    for arm in ("Vanilla",):
        rows = allr[arm]; a = np.array([r["acc"] for r in rows]); st = np.array([r["acc_strip_psi"] for r in rows])
        pd_ = np.array([max(h["psi_disp"] for h in r["heads"]) for r in rows])
        print(f"\n{arm}: strip_psi - intact mean {np.mean(st - a):+.4f} (seeds up {(st > a + 1e-4).sum()}, down {(st < a - 1e-4).sum()}); "
              f"Spearman(acc, max psi_disp) {spearman(a, pd_):+.2f}")
    json.dump(allr, open(f"{REPO}/docs/audits/2026-10-05/neuro_rank2_mech.json", "w"), indent=1)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    main()

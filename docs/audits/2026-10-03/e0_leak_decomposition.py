"""E0 (docs/lit/LIT_NEW_OBJECTS.md): leak decomposition on the new-object pilot checkpoints. Eval only.

For every arm in runs/newobj_pilot/<arm>_s100/<arm>.pt and every code condition, on the TEST pool and on the
TRAIN pool (reference), with the same 200 sequences per pool for every arm and condition (env seed 10000,
np seed 10**6 before each pool, generated in train_newobj.evaluate's order, so ID/intact all-target accuracy
must reproduce eval.json):

  readouts at revisit targets
    obj_acc   object-identity accuracy: targets whose answer is an object; argmax over the ACTIVE pool's object
              logits only (no blank competitor)
    blank_auc AUC of (blank logit - logsumexp(active-pool object logits)) for "target is blank", all revisits
    all_acc   legacy all-target accuracy (argmax over the full masked vocabulary)
  interventions on the path-integration step (forward hook on model.action_to_lie, output multiplied by 0/1)
    intact    none
    zobj      object-token steps zeroed
    zob       object AND blank steps zeroed: only action steps remain
  C = obj_acc(zob)   (content floor: the "what" channel with an exact action-only "where")
  L = C - obj_acc(intact)   (leak cost)

Code conditions. c = the fixed codebook (model_codes.codebook, N(0, I_64), E||c||^2 = 64). Every condition
transforms all 2P rows; only the active pool's rows ever appear, so this equals transforming that pool.
"emb" = the codes the encoder A reads; "read" = the codes the readout scores against.
    ID                 emb = read = c
    ROT_d              emb = read = Q c, Q Haar-orthogonal (draw d). In distribution (Q c ~ N(0, I)): a control.
    NORMEMB_xs         emb = s c, read = c          (s in 0.5, 2, 4, 8)
    NORMBOTH_xs        emb = read = s c
    RANDMAP_sS_d       emb = read = M c, M = S G / 8, G_ij ~ N(0,1) (draw d) => E||M c||^2 = S^2 ||c||^2.
                       Covariance becomes S^2 G G^T / 64 (Marchenko-Pastur spread, eigenvalues ~0..4 S^2).
    RANDMAPnm_d        emb = read = (G c) * ||c|| / ||G c||: per-code norm matched; S drops out. Direction /
                       covariance shift only.
    SPARSE_d           fresh codes, 8 random coordinates = +/-1, scaled by sqrt(8): ||c||^2 = 64 exactly.
    PROJIN / PROJOUT   emb = read = P c * ||c|| / ||P c||, P the projector onto the top-r right-singular
                       subspace of the trained object-step map Ms = W_out W_in A (H*nb x 64), r = rank(W_in) = 4
                       (PROJIN) or onto its orthogonal complement (PROJOUT). Per arm. Norm matched.

Verification (rule 9) runs first and is written to the output: the hooked model with no intervention
reproduces a freshly loaded model's logits bit-for-bit; masking leaves every non-masked step bit-identical and
the masked ones exactly 0; the path integrator receives the masked steps; zeroing all steps changes logits;
NORMEMB scales object embeddings only; train_newobj.evaluate on the fresh model reproduces eval.json.

Run from anywhere: python3 /home/prashr/mapformer/docs/audits/2026-10-03/e0_leak_decomposition.py
n=1 seed per arm: descriptive only.
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/prashr")
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata

from mapformer.train_newobj import ARMS, evaluate
from mapformer.environment_newobj import NewObjectWorld, N_SPECIAL, BLANK
from mapformer.model_codes import use_object_codes, set_pool, D_CODE

REPO = Path("/home/prashr/mapformer")
OUT = REPO / "docs/audits/2026-10-03"
RUNS = REPO / "runs/newobj_pilot"
ARM_NAMES = ["MapWM", "MapPoPE", "MapEM", "PosOnly", "RoPE", "PoPE"]
PATH_ARMS = {"MapWM", "MapPoPE", "MapEM", "PosOnly"}
P, SIZE, T_STEPS, N_SEQ, BATCH = 1000, 32, 1024, 200, 20
EVAL_SEED, ENV_SEED, COND_SEED = 10 ** 6, 10000, 20261003
DEV = "cuda:0"
INTERVENTIONS = ["intact", "zobj", "zob"]

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load(arm):
    ck = torch.load(RUNS / f"{arm}_s100" / f"{arm}.pt", map_location="cpu", weights_only=False)
    a = ck["config"]["args"]
    assert a["pool_size"] == P and a["size"] == SIZE and a["n_steps"] == T_STEPS
    m = ARMS[arm](vocab_size=ck["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=a["n_layers"],
                  grid_size=a["size"])
    use_object_codes(m, N_SPECIAL, P)
    m.load_state_dict(ck["model_state_dict"])
    return m.to(DEV).eval()


class Probe:
    """Hooks a loaded model: separate embedding / readout codebooks, a 0/1 step mask, step capture."""

    def __init__(self, model):
        self.m = model
        self.orig = model.token_emb.codes.clone()
        self.read = self.orig
        self.keep = None          # (B, L) bool; None = no intervention
        self.cap = {}
        ro = model.out_proj.ro
        probe = self

        class _Out(nn.Module):
            def __init__(self):
                super().__init__(); self.ro = ro

            def forward(self, h):
                return self.ro(h, probe.read)
        model.out_proj = _Out()
        self.has_step = hasattr(model, "action_to_lie")
        if self.has_step:
            model.action_to_lie.register_forward_hook(self._hook)
            model.path_integrator.register_forward_pre_hook(self._pi_hook)

    def _hook(self, mod, inp, out):
        self.cap["raw"] = out
        if self.keep is None:
            return out
        return torch.where(self.keep[:, :, None, None], out, torch.zeros_like(out))

    def _pi_hook(self, mod, inp):
        self.cap["pi_in"] = inp[0]

    def set_codes(self, emb, read):
        self.m.token_emb.codes = emb.to(DEV)
        self.read = read.to(DEV)


def keep_mask(tok, kind):
    if kind == "intact":
        return None
    if kind == "zobj":
        return tok < N_SPECIAL
    if kind == "zob":
        return tok < BLANK
    if kind == "zall":
        return torch.zeros_like(tok, dtype=torch.bool)
    raise ValueError(kind)


def sequences(pool):
    env = NewObjectWorld(size=SIZE, seed=ENV_SEED, pool_size=P, pool=pool)
    np.random.seed(EVAL_SEED)
    toks, revs = [], []
    for _ in range(N_SEQ):
        t, _om, r = env.generate_trajectory(T_STEPS)
        toks.append(t); revs.append(r)
    return torch.stack(toks), torch.stack(revs)


def auc(score, label):
    pos, n1 = label.sum(), label.size
    n0 = n1 - pos
    if pos == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * n0))


@torch.no_grad()
def run(probe, toks, revs, pool, kind):
    m = probe.m
    lo = N_SPECIAL + (0 if pool == "train" else P)
    ok_obj, n_obj, ok_all, n_all, sc, lab = [], [], [], [], [], []
    st = {"act": [0.0, 0], "obj": [0.0, 0], "blank": [0.0, 0]}
    for i in range(0, len(toks), BATCH):
        tok = toks[i:i + BATCH].to(DEV); rev = revs[i:i + BATCH].to(DEV)
        inp = tok[:, :-1]
        probe.keep = keep_mask(inp, kind) if probe.has_step else None
        lg = m(inp).float()
        probe.keep = None
        tgt, msk = tok[:, 1:], rev[:, 1:]
        all_ok = (lg.argmax(-1) == tgt) & msk
        ok_all.append(all_ok.sum(1).cpu()); n_all.append(msk.sum(1).cpu())
        ol = lg[..., lo:lo + P]
        om = msk & (tgt >= N_SPECIAL)
        ook = ((ol.argmax(-1) + lo) == tgt) & om
        ok_obj.append(ook.sum(1).cpu()); n_obj.append(om.sum(1).cpu())
        s = lg[..., BLANK] - torch.logsumexp(ol, -1)
        sc.append(s[msk].cpu()); lab.append((tgt[msk] == BLANK).cpu())
        if probe.has_step and kind == "intact":
            d = probe.cap["raw"].float().flatten(2).pow(2).sum(-1)       # (B, L) squared step norm
            for name, sel in (("act", inp < BLANK), ("blank", inp == BLANK), ("obj", inp >= N_SPECIAL)):
                st[name][0] += float(d[sel].sum()); st[name][1] += int(sel.sum())
    ok_obj, n_obj = torch.cat(ok_obj).numpy(), torch.cat(n_obj).numpy()
    ok_all, n_all = torch.cat(ok_all).numpy(), torch.cat(n_all).numpy()
    res = {"obj_acc": ok_obj.sum() / n_obj.sum(), "all_acc": ok_all.sum() / n_all.sum(),
           "blank_auc": auc(torch.cat(sc).numpy(), torch.cat(lab).numpy()),
           "n_obj_targets": int(n_obj.sum()), "n_targets": int(n_all.sum()),
           "_ok_obj": ok_obj, "_n_obj": n_obj}
    if probe.has_step and kind == "intact":
        rms = {k: math.sqrt(v[0] / max(v[1], 1)) for k, v in st.items()}
        res["step_rms"] = rms
        res["gain_obj"] = rms["obj"] / rms["act"]
        res["gain_blank"] = rms["blank"] / rms["act"]
    return res


def cluster_se(ok_a, ok_b, n):
    """Sequence-clustered SE of (sum ok_a - sum ok_b) / sum n (eval-sample noise only, not seed noise)."""
    d = ok_a - ok_b
    L = d.sum() / n.sum()
    k = len(n)
    return float(math.sqrt(k / (k - 1) * ((d - L * n) ** 2).sum()) / n.sum())


def haar(g, n):
    q, r = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    return (q * torch.sign(torch.diagonal(r))).float()


def conditions(c, step_map=None):
    """Return {name: (emb_codes, read_codes)}; step_map = Ms (H*nb x 64) for PROJ conditions."""
    g = torch.Generator().manual_seed(COND_SEED)
    out = {"ID": (c, c)}
    for d in range(3):
        Q = haar(g, D_CODE)
        out[f"ROT_{d}"] = (c @ Q.T,) * 2
    for s in (0.5, 2, 4, 8):
        out[f"NORMEMB_x{s:g}"] = (c * s, c)
        out[f"NORMBOTH_x{s:g}"] = (c * s, c * s)
    nrm = c.norm(dim=1, keepdim=True)
    for d in range(3):
        G = torch.randn(D_CODE, D_CODE, generator=g)
        for S in (0.5, 1, 2):
            out[f"RANDMAP_s{S:g}_{d}"] = ((c @ (S * G / 8).T),) * 2
        gc = c @ G.T
        out[f"RANDMAPnm_{d}"] = (gc * nrm / gc.norm(dim=1, keepdim=True),) * 2
    for d in range(3):
        n = c.shape[0]
        idx = torch.stack([torch.randperm(D_CODE, generator=g)[:8] for _ in range(n)])
        sgn = torch.randint(0, 2, (n, 8), generator=g).float() * 2 - 1
        sp = torch.zeros(n, D_CODE).scatter_(1, idx, sgn) * math.sqrt(8)
        out[f"SPARSE_{d}"] = (sp, sp)
    if step_map is not None:
        U, S, Vh = torch.linalg.svd(step_map.double(), full_matrices=True)
        r = int((S > S[0] * 1e-6).sum())
        V = Vh[:r].float()                               # (r, 64) row space
        pin = (c @ V.T) @ V
        pout = c - pin
        out["PROJIN"] = (pin * nrm / pin.norm(dim=1, keepdim=True),) * 2
        out["PROJOUT"] = (pout * nrm / pout.norm(dim=1, keepdim=True),) * 2
    return out


def step_map(m):
    W = (m.action_to_lie.w_out.weight @ m.action_to_lie.w_in.weight @ m.token_emb.encoder.weight)
    return W.detach().cpu()


@torch.no_grad()
def verify(arm, seqs, eval_json):
    v = {}
    fresh = load(arm)
    hooked = load(arm)
    probe = Probe(hooked)
    tok = seqs["test"][0][:2].to(DEV)[:, :-1]
    set_pool(fresh, "test"); set_pool(hooked, "test")
    ref = fresh(tok)
    v["identity_no_intervention_bitexact"] = bool(torch.equal(ref, hooked(tok)))
    if probe.has_step:
        probe.keep = torch.ones_like(tok, dtype=torch.bool)
        v["identity_allones_mask_bitexact"] = bool(torch.equal(ref, hooked(tok)))
        probe.keep = None
        hooked(tok); raw = probe.cap["raw"].clone()
        for kind in ("zobj", "zob"):
            km = keep_mask(tok, kind)
            probe.keep = km
            lg = hooked(tok)
            probe.keep = None
            pi_in = probe.cap["pi_in"]
            kept = km[:, :, None, None].expand_as(raw)
            v[f"{kind}_kept_steps_bitexact"] = bool(torch.equal(pi_in[kept], raw[kept]))
            v[f"{kind}_masked_steps_exact_zero"] = bool((pi_in[~kept] == 0).all())
            v[f"{kind}_n_masked_tokens"] = int((~km).sum())
            v[f"{kind}_masked_token_ids_ok"] = bool(
                (tok[~km] >= (N_SPECIAL if kind == "zobj" else BLANK)).all())
            v[f"{kind}_logits_changed"] = bool(not torch.equal(lg, ref))
            # embedding stream untouched: token_emb output identical with the hook active
        probe.keep = keep_mask(tok, "zall")
        fin = torch.isfinite(ref)                       # the inactive pool's logits are -inf by design
        v["zall_logits_maxabs_change"] = float((hooked(tok) - ref)[fin].abs().max())
        probe.keep = None
        # the masked step must be the one the attention uses: the integrator input equals hook output
        v["integrator_receives_hook_output"] = v["zob_kept_steps_bitexact"] and v["zob_masked_steps_exact_zero"]
        # NORMEMB: special embeddings unchanged, object embeddings scaled, readout codes unchanged
        e0 = hooked.token_emb(tok)
        probe.set_codes(probe.orig * 4, probe.orig)
        e4 = hooked.token_emb(tok)
        isob = tok >= N_SPECIAL
        v["normemb_special_emb_unchanged"] = bool(torch.equal(e0[~isob], e4[~isob]))
        v["normemb_obj_emb_scaled_x4_maxrel"] = float(((e4[isob] - 4 * e0[isob]).norm(dim=-1)
                                                       / (4 * e0[isob]).norm(dim=-1)).max())
        v["normemb_readout_codes_unchanged"] = bool(torch.equal(probe.read, probe.orig.to(DEV)))
        probe.set_codes(probe.orig, probe.orig)
        v["restore_bitexact"] = bool(torch.equal(ref, hooked(tok)))
    # eval.json reproduction through the repo's own evaluate() on the fresh model (batch 1)
    for pool in ("train", "test"):
        env = NewObjectWorld(size=SIZE, seed=ENV_SEED, pool_size=P, pool=pool); set_pool(fresh, pool)
        acc, nll = evaluate(fresh, env, T_STEPS, N_SEQ, DEV)
        v[f"evaluate_{pool}_acc"] = acc
        v[f"evaluate_{pool}_minus_eval_json"] = acc - eval_json[f"{pool}|{T_STEPS}"]["acc"]
    del fresh, hooked
    return v


def main():
    t0 = time.time()
    seqs = {p: sequences(p) for p in ("train", "test")}
    for p, (tk, rv) in seqs.items():
        tgt, m = tk[:, 1:], rv[:, 1:]
        frac_blank = float((tgt[m] == BLANK).float().mean())
        print(f"{p}: {int(m.sum())} revisit targets, always-blank floor {frac_blank:.4f}", flush=True)
    floors = {p: {"always_blank_all_acc": float((seqs[p][0][:, 1:][seqs[p][1][:, 1:]] == BLANK).float().mean()),
                  "obj_chance_pool": 1 / P, "obj_chance_in_seq": 1 / 16} for p in seqs}
    out = {"meta": {"n_seq": N_SEQ, "T_steps": T_STEPS, "env_seed": ENV_SEED, "eval_seed": EVAL_SEED,
                    "cond_seed": COND_SEED, "batch": BATCH, "tf32": False}, "floors": floors,
           "verify": {}, "results": {}, "step_map": {}}
    for arm in ARM_NAMES:
        ej = json.load(open(RUNS / f"{arm}_s100" / "eval.json"))["eval"]
        out["verify"][arm] = verify(arm, seqs, ej)
        print(arm, "verify", out["verify"][arm], flush=True)
        m = load(arm)
        probe = Probe(m)
        sm = step_map(m) if probe.has_step else None
        if sm is not None:
            sv = torch.linalg.svdvals(sm.double())
            out["step_map"][arm] = {"singular_values_top8": sv[:8].tolist(),
                                    "action_step_rms_note": "see step_rms in results"}
        conds = conditions(probe.orig.cpu(), sm)
        res = {}
        for pool in ("test", "train"):
            set_pool(m, pool)
            tk, rv = seqs[pool]
            for cname, (emb, read) in conds.items():
                probe.set_codes(emb, read)
                r = {k: run(probe, tk, rv, pool, k) for k in (INTERVENTIONS if probe.has_step else ["intact"])}
                row = {k: {kk: vv for kk, vv in r[k].items() if not kk.startswith("_")} for k in r}
                if probe.has_step:
                    C = r["zob"]["obj_acc"]
                    row["C"] = float(C)
                    row["L"] = float(C - r["intact"]["obj_acc"])
                    row["L_se_eval"] = cluster_se(r["zob"]["_ok_obj"], r["intact"]["_ok_obj"], r["intact"]["_n_obj"])
                    row["L_obj_only"] = float(r["zobj"]["obj_acc"] - r["intact"]["obj_acc"])
                res[f"{pool}|{cname}"] = row
            probe.set_codes(probe.orig, probe.orig)
            print(f"{arm} {pool} done {time.time() - t0:.0f}s", flush=True)
        out["results"][arm] = res
        del m, probe
        torch.cuda.empty_cache()

    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    json.dump(conv(out), open(OUT / "e0_results.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s", flush=True)
    (OUT / "e0.done").write_text("ok\n")


if __name__ == "__main__":
    main()

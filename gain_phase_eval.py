"""Per-run readouts for GAIN_PHASE_PREREG.md (new-object task, test pool, LEAK's eval stream).

For a checkpoint of any arm in model_gain_phase.ARMS (and LEAK's ActOnly, for the readout validation):
  acc            object-identity accuracy on the TEST pool at x1 (argmax over the active pool's objects, at revisit
                 targets whose answer is an object), 200 sequences, eval seed 10**6, env seed 10000: leak_eval's
                 sequences and scoring rule exactly.
  L_ms           MEAN-SUBSTITUTION leak (registered leak readout): every object token's step replaced by the mean step
                 over all 1000 test-pool codes (computed per model, x1); L_ms = acc(substituted) - acc. It removes the
                 object-IDENTITY-dependent part of the step only and keeps the part shared by all objects, so it is
                 invariant to the per-move gauge that made LEAK's zeroing readout wrong for NormStep
                 (docs/NORMSTEP_NOTES.md 2). Blank and action steps untouched.
  L_zero         LEAK's readout (object steps zeroed), for continuity; valid for raw steps, not for NormStep.
  S_id           geometric identity spread (gauge-invariant, score-independent): in phase units (omega * Delta),
                 rms over the 1000 test codes of ||Delta(o) - mean_o Delta|| divided by the mean over the two axes of
                 ||(Delta(+a) - Delta(-a)) / 2|| (half the difference of opposite actions: a per-move displacement that a
                 common offset cannot change).
  shift_cells    least-squares projection of omega * (Delta(o) - mean) onto the two axis displacements: the field shift
                 (in cells) object o's own step gives its key; rms over objects; resid = share of ||.||^2 not explained
                 by a displacement (field distortion).
  shared         ||omega * mean_o Delta(o)|| / axis (the shared observation step, the gauge part), blank the same.
  x2 / x4        accuracy with the object codes scaled on the embedding side (readout codes unchanged): a real shift for
                 raw steps, a construction check for NormStep steps (LN(s e) = LN(e)) -- no verdict.
  train_x1       train-pool accuracy at x1.
  gains          gain arms only: mu_k over test objects (mean, cv), blank, actions; mu_q over actions; spectrum share of
                 sum A_c per 8-channel band (fine -> coarse). Per head.
`python3 -m mapformer.gain_phase_eval CKPT ARM [device]` prints the dict.
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn

from mapformer.environment_newobj import N_SPECIAL
from mapformer.leak_eval import sequences, P, SIZE, T_STEPS, BATCH
from mapformer.model_codes import use_object_codes, set_pool, MapWM_ActOnly
from mapformer.model_em_pope import GainKernelLayer
from mapformer.model_gain_phase import ARMS as GP_ARMS

ALL_ARMS = dict(GP_ARMS, ActOnly=MapWM_ActOnly)
TEST_IDS = torch.arange(N_SPECIAL + P, N_SPECIAL + 2 * P)


def load(ckpt, arm, dev):
    b = torch.load(ckpt, map_location="cpu", weights_only=False); a = b["config"]["args"]
    assert a["pool_size"] == P and a["size"] == SIZE and a["n_steps"] == T_STEPS and a["variant"] == arm, (a, arm)
    m = ALL_ARMS[arm](vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=a["n_layers"],
                      grid_size=SIZE)
    use_object_codes(m, N_SPECIAL, P); m.load_state_dict(b["model_state_dict"])
    return m.to(dev).eval(), b


def step_of(m, tok):
    x = m.token_emb(tok)
    return m.step(tok, x) if hasattr(m, "step") else m.action_to_lie(x)


class Probe:
    """Hooks on action_to_lie's output (the per-token step, after NormStep's LayerNorm where present) and on the
    embedding-side codes. mode None: untouched; 'zero': object steps 0; 'mean': object steps = mean test-pool step;
    'add': object steps + self.extra[code index] (positive control). Readout codes are never scaled."""

    def __init__(self, m):
        self.m, self.mode, self.obj, self.tok = m, None, None, None
        self.codes = m.token_emb.codes.clone()
        m.action_to_lie.register_forward_hook(self._hook)
        read = self.codes; ro = m.out_proj.ro

        class _Out(nn.Module):
            def __init__(self):
                super().__init__(); self.ro = ro

            def forward(self, h):
                return self.ro(h, read)
        m.out_proj = _Out()
        self.mean = self.mean_step()
        self.extra = None

    def _hook(self, mod, inp, out):
        if self.mode is None:
            return out
        obj = self.obj[..., None, None]
        if self.mode == "zero":
            return torch.where(obj, torch.zeros_like(out), out)
        if self.mode == "mean":
            return torch.where(obj, self.mean.to(out.dtype).expand_as(out), out)
        if self.mode == "add":
            idx = (self.tok - N_SPECIAL).clamp(min=0)
            return torch.where(obj, out + self.extra[idx].to(out.dtype), out)
        raise ValueError(self.mode)

    @torch.no_grad()
    def mean_step(self):
        mode, self.mode = self.mode, None
        d = step_of(self.m, TEST_IDS[None].to(self.codes.device))[0]          # (1000, H, nb)
        self.mode = mode
        return d.mean(0)

    def set_scale(self, s):
        self.m.token_emb.codes = self.codes * s


@torch.no_grad()
def obj_acc(probe, toks, revs, pool, mode, dev):
    m = probe.m; lo = N_SPECIAL + (0 if pool == "train" else P); ok = n = 0
    for i in range(0, len(toks), BATCH):
        tok = toks[i:i + BATCH].to(dev); rev = revs[i:i + BATCH].to(dev); inp = tok[:, :-1]
        probe.mode, probe.obj, probe.tok = mode, inp >= N_SPECIAL, inp
        lg = m(inp).float(); probe.mode = None
        tgt, msk = tok[:, 1:], rev[:, 1:] & (tok[:, 1:] >= N_SPECIAL)
        ok += int((((lg[..., lo:lo + P].argmax(-1) + lo) == tgt) & msk).sum()); n += int(msk.sum())
    return ok / n


@torch.no_grad()
def step_geometry(m, extra=None):
    """S_id, shift_cells, resid, shared, blank (module docstring). extra: (2P, H, nb) added to object steps (control)."""
    dev = next(m.parameters()).device
    ids = torch.cat([torch.arange(N_SPECIAL), TEST_IDS])[None].to(dev)
    d = step_of(m, ids)[0]
    if extra is not None:
        d = d.clone(); d[N_SPECIAL:] += extra[P:2 * P]
    d = (d * m.path_integrator.omega).flatten(1).double()                     # phase units, (5 + P, H*nb)
    ux, uy = (d[0] - d[1]) / 2, (d[2] - d[3]) / 2
    axis = float((ux.norm() + uy.norm()) / 2)
    obj = d[N_SPECIAL:]; mu = obj.mean(0); dev_ = obj - mu
    U = torch.stack([ux, uy], 1)                                               # (C, 2)
    coef = torch.linalg.lstsq(U, dev_.T).solution.T                            # (P, 2) cells
    resid = dev_ - coef @ U.T
    return {"S_id": float(dev_.norm(dim=1).pow(2).mean().sqrt()) / axis,
            "shift_cells": float(coef.norm(dim=1).pow(2).mean().sqrt()),
            "resid": float(resid.pow(2).sum() / dev_.pow(2).sum().clamp_min(1e-30)),
            "shared": float(mu.norm()) / axis, "blank": float(d[4].norm()) / axis, "axis": axis}


@torch.no_grad()
def gain_stats(m):
    layer = m.layers[0]
    if not isinstance(layer, GainKernelLayer):
        return None
    dev = next(m.parameters()).device
    ids = torch.cat([torch.arange(N_SPECIAL), TEST_IDS])[None].to(dev)
    gq, gk = layer.gains(layer.norm1(m.token_emb(ids)))                      # (1, H, 5+P, nb), equal over nb
    gq, gk = gq[0, :, :, 0].double(), gk[0, :, :, 0].double()
    A = layer.amplitude().double(); band = A.view(A.shape[0], 4, -1).sum(-1) / A.sum(-1, keepdim=True)
    ob = gk[:, N_SPECIAL:]
    return {"muk_obj_mean": ob.mean(1).tolist(), "muk_obj_cv": (ob.std(1) / ob.mean(1)).tolist(),
            "muk_blank": gk[:, 4].tolist(), "muk_act": gk[:, :4].mean(1).tolist(), "muq_act": gq[:, :4].mean(1).tolist(),
            "band_share": band.tolist()}


def evaluate_run(ckpt, arm, dev="cuda:0", data=None):
    m, b = load(ckpt, arm, dev); probe = Probe(m); out = {}
    data = data or {pool: sequences(pool) for pool in ("test", "train")}
    set_pool(m, "test"); toks, revs = data["test"]
    a1 = obj_acc(probe, toks, revs, "test", None, dev)
    out["acc"] = a1
    out["L_ms"] = obj_acc(probe, toks, revs, "test", "mean", dev) - a1
    out["L_zero"] = obj_acc(probe, toks, revs, "test", "zero", dev) - a1
    for s in (2.0, 4.0):
        probe.set_scale(s); out[f"x{s:g}"] = obj_acc(probe, toks, revs, "test", None, dev)
    probe.set_scale(1.0)
    set_pool(m, "train"); toks, revs = data["train"]
    out["train_x1"] = obj_acc(probe, toks, revs, "train", None, dev)
    set_pool(m, "test")
    out.update(step_geometry(m)); out["gains"] = gain_stats(m)
    return out


if __name__ == "__main__":
    print(json.dumps(evaluate_run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "cuda:0"), indent=1))

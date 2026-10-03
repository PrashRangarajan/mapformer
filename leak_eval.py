"""Leak readouts for the new-object task (LEAK_PREREG.md), adapted from docs/audits/2026-10-03/e0_leak_decomposition.py.

For a checkpoint: object-identity accuracy on the TEST pool (argmax over the active pool's objects, at revisit
targets whose answer is an object), with the object codes scaled on the EMBEDDING side by s in {1, 2, 4} (the
readout codes unchanged), each with the model intact and with object-token steps zeroed. Leak cost L = zeroed -
intact. 200 eval sequences (eval seed 10**6, held-out env seed 10000), the stream train_newobj's eval uses.
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn

from mapformer.environment_newobj import NewObjectWorld, N_SPECIAL, BLANK
from mapformer.model_codes import use_object_codes, set_pool
from mapformer.train_newobj import ARMS

P, SIZE, T_STEPS, N_SEQ, BATCH, EVAL_SEED, ENV_SEED = 1000, 32, 1024, 200, 20, 10**6, 10000


def load(ckpt, arm, dev):
    b = torch.load(ckpt, map_location="cpu", weights_only=False); a = b["config"]["args"]
    assert a["pool_size"] == P and a["size"] == SIZE and a["n_steps"] == T_STEPS, a
    m = ARMS[arm](vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=a["n_layers"], grid_size=SIZE)
    use_object_codes(m, N_SPECIAL, P); m.load_state_dict(b["model_state_dict"])
    return m.to(dev).eval(), b


def sequences(pool):
    env = NewObjectWorld(size=SIZE, seed=ENV_SEED, pool_size=P, pool=pool)
    np.random.seed(EVAL_SEED)
    T, R = [], []
    for _ in range(N_SEQ):
        t, _o, r = env.generate_trajectory(T_STEPS); T.append(t); R.append(r)
    return torch.stack(T), torch.stack(R)


class Probe:
    """Embedding-side code scaling and a step mask, by hooks; with no intervention the model is unchanged."""

    def __init__(self, m):
        self.m, self.keep, self.scale = m, None, 1.0
        self.codes = m.token_emb.codes.clone()
        self.has_step = hasattr(m, "action_to_lie")
        if self.has_step:
            m.action_to_lie.register_forward_hook(self._hook)
        emb = m.token_emb; read = self.codes; probe = self
        ro = m.out_proj.ro

        class _Out(nn.Module):
            def __init__(self):
                super().__init__(); self.ro = ro

            def forward(self, h):
                return self.ro(h, read)                       # readout codes never scaled
        m.out_proj = _Out()

    def _hook(self, mod, inp, out):
        if self.keep is None:
            return out
        return torch.where(self.keep[:, :, None, None], out, torch.zeros_like(out))

    def set_scale(self, s):
        self.scale = s; self.m.token_emb.codes = self.codes * s


@torch.no_grad()
def obj_acc(probe, toks, revs, pool, zero_obj, dev):
    m = probe.m; lo = N_SPECIAL + (0 if pool == "train" else P); ok = n = 0
    for i in range(0, len(toks), BATCH):
        tok = toks[i:i + BATCH].to(dev); rev = revs[i:i + BATCH].to(dev); inp = tok[:, :-1]
        probe.keep = (inp < N_SPECIAL) if (zero_obj and probe.has_step) else None
        lg = m(inp).float(); probe.keep = None
        tgt, msk = tok[:, 1:], rev[:, 1:] & (tok[:, 1:] >= N_SPECIAL)
        ok += int((((lg[..., lo:lo + P].argmax(-1) + lo) == tgt) & msk).sum()); n += int(msk.sum())
    return ok / n


def evaluate_run(ckpt, arm, dev="cuda:0", scales=(1.0, 2.0, 4.0)):
    m, b = load(ckpt, arm, dev); probe = Probe(m); out = {}
    for pool in ("test", "train"):
        set_pool(m, pool); toks, revs = sequences(pool)
        for s in (scales if pool == "test" else (1.0,)):
            probe.set_scale(s)
            a = obj_acc(probe, toks, revs, pool, False, dev); z = obj_acc(probe, toks, revs, pool, True, dev)
            out[f"{pool}|x{s:g}"] = {"intact": a, "zobj": z, "leak": z - a}
        probe.set_scale(1.0)
    return out


if __name__ == "__main__":
    print(json.dumps(evaluate_run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "cuda:0"), indent=1))

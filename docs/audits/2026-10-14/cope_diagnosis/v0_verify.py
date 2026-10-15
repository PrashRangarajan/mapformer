"""Rule-9 checks of the probes before any learned quantity is read.
(1) construction (generality audit c3): increments() reads alpha 17/16, beta 15/16, kappa 1; func_eval gives 1.000 on
    every stratum. (2) learned model, one run: tables() gates equal the gates inside CoPE.forward on real sequences;
    the snap hook with the model's own gates is a no-op (bitwise accuracy); score() reproduces eval.json."""
import json, sys
sys.path.insert(0, "/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis")
import numpy as np, torch
torch.set_num_threads(4)
import lib_cope as lc
from func_cope import func_eval, make_table, gate_table

trajs = lc.trajectories(100)
# (1) construction
G = gate_table(0.9625, 0.8375, 0.1)
inc = lc.increments(G.numpy()[None])
print("construction increments (h0, a0):", {k: round(v, 5) for k, v in inc[(0, 0)].items() if k != "g_obs"})
acc, loss, rows = func_eval(G, make_table(200.0), trajs)
print(f"construction func_eval acc {acc:.4f} loss {loss:.4f}; strata", {k: (round(v[0], 4), v[1]) for k, v in lc.strat_table(rows).items()})

# (2) learned run
path = lc.REPO / "runs/wrap_cope/p0/CoPEF_s900/CoPEF.pt"
m, ck = lc.load(path)
L, Gt, Z = lc.tables(m)
cap = {}
orig = lc.mmb.CoPE.forward
def hook(self, query, attn_logits):
    cap["g"] = torch.sigmoid(attn_logits); cap["q"] = query
    return orig(self, query, attn_logits)
lc.mmb.CoPE.forward = hook
tok, rev = trajs[0]; x = tok[:-1].unsqueeze(0)
with torch.no_grad():
    out0 = m(x)
g = cap["g"][0]                                          # (H, L, L)
pred = torch.as_tensor(Gt)[:, x[0][:, None], x[0][None, :]] * torch.ones(x.shape[1], x.shape[1]).tril()
print("max |gate(forward) - gate(table)|:", float((g - pred).abs().max()))
zq = torch.einsum("hld,dp->hlp", cap["q"][0], m.layers[0].attn.cope.pos_emb[0])
zt = torch.as_tensor(Z)[:, x[0]]
print("max |q.e (forward) - Z table|:", float((zq - zt).abs().max()))
lc.mmb.CoPE.forward = orig

ev = json.load(open(path.parent / "eval.json"))["eval"]["128"]["acc"]
full = lc.trajectories(400)
a0, n0, r0 = lc.score(m, full)
a1, n1, r1 = lc.score(m, full, Gsnap=Gt)
print(f"score() acc {a0:.6f} vs eval.json {ev:.6f}; snap with own gates acc {a1:.6f} nll {n1:.6f} vs {n0:.6f}")

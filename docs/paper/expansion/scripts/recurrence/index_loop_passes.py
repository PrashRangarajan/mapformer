"""Per-pass revisit accuracy by stratum for INDEX attention (FACTORIAL batch, torus T=1024, seeds 500-503):
RoPELooped run with n_loops = 1..6 (exact) and 4-layer RoPE read by a logit lens after each layer.
Tests T2's P3 on single models: with an index code, extra passes should add accuracy on short-gap revisits only.
Same evaluation stream as the registered one (env seed 10000, np seed 1234 + s, 100 trajectories); the 'all'
column at the trained depth is checked against FACTORIAL_{LOOP,L4}.json. The FACTORIAL models trained with
--fast-attn (SDPA); the manual attention here agrees to ~1e-6 in logits (rule 19)."""
import json, math, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/recurrence")
from common import OUT, REPO, env_for, step_info, stratum, trajectories  # noqa
from mapformer.train_variant import VARIANT_MAP  # noqa: E402
from mapformer.model import _apply_rope  # noqa: E402
torch.set_num_threads(4)
STR = ("all", "short", "long", "wrap")
N_TR = int(sys.argv[1]) if len(sys.argv) > 1 else 100


@torch.no_grad()
def passes(m, blocks, tok):
    x = m.token_emb(tok[None]); L = x.shape[1]
    cos, sin = m._rope_cos_sin(L, x.device, x.dtype)
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    out = []
    for blk in blocks:
        h = blk.norm1(x); B, T, D = h.shape
        Q = blk.q_proj(h).view(B, T, blk.n_heads, blk.d_head).transpose(1, 2)
        K = blk.k_proj(h).view(B, T, blk.n_heads, blk.d_head).transpose(1, 2)
        V = blk.v_proj(h).view(B, T, blk.n_heads, blk.d_head).transpose(1, 2)
        Q = _apply_rope(Q, cos, sin); K = _apply_rope(K, cos, sin)
        a = F.softmax((Q @ K.transpose(-1, -2) / math.sqrt(blk.d_head)).masked_fill(mask, float("-inf")), -1)
        x = x + blk.o_proj((a @ V).transpose(1, 2).reshape(B, T, D))
        x = x + blk.ffn(blk.norm2(x))
        out.append(m.out_proj(m.out_norm(x))[0])
    return out


res = {}
for tag, v, sub in (("LOOP", "RoPELooped", "LOOP"), ("L4", "RoPE", "L4")):
    stored = json.load(open(f"{REPO}/FACTORIAL_{tag}.json"))
    for s in (500, 501, 502, 503):
        b = torch.load(f"{REPO}/runs/factorial/{sub}/p0/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)
        c = b["config"]
        m = VARIANT_MAP[v](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                           n_layers=c["n_layers"], grid_size=c["grid_size"])
        m.load_state_dict(b["model_state_dict"]); m.eval()
        blocks = [m.layers[0]] * 6 if tag == "LOOP" else list(m.layers)
        acc = None
        for tok, rev in trajectories(env_for(c), 1024, N_TR, 1234 + s):
            lg = passes(m, blocks, tok[:-1])
            if acc is None:
                acc = {k: np.zeros((len(lg), 2)) for k in STR}
            idx = torch.nonzero(rev[1:]).flatten().numpy()
            cell, new_u, lag = step_info(tok, c["grid_size"]); st = idx // 2
            strat = stratum(new_u[st], lag[st])
            for p, l in enumerate(lg):
                hit = (l.argmax(-1)[idx] == tok[1:][idx]).numpy()
                for k in STR:
                    sel = np.ones(len(idx), bool) if k == "all" else strat == k
                    acc[k][p] += [hit[sel].sum(), sel.sum()]
        r = {k: (acc[k][:, 0] / acc[k][:, 1]).tolist() for k in STR}
        res[f"{tag}|{s}"] = r
        key = [k for k in stored if v in k and "1024" in k]
        ref = {x[0]: x[1] for x in stored[key[0]]}.get(s) if key else None
        print(f"{tag} s{s} trained-depth all {r['all'][3]:.4f} stored {ref}  | " +
              " ".join(f"{k} " + "/".join(f"{x:.3f}" for x in r[k]) for k in STR), flush=True)
json.dump(res, open(f"{OUT}/index_loop_passes.json", "w"), indent=1)
for tag, P in (("LOOP", 6), ("L4", 4)):
    print(f"== {tag} mean over 4 seeds, per pass")
    for k in STR:
        x = np.array([res[f"{tag}|{s}"][k] for s in (500, 501, 502, 503)]).mean(0)
        print(f"  {k:5s} " + " ".join(f"{v:.3f}" for v in x))

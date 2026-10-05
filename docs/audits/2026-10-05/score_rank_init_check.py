"""Construction check for SCORE_RANK_PREREG.md (CPU, no training). For each seed, build the four arms exactly as
train_variant does (torch.manual_seed(s); np.random.seed(s); the GridWorld env is built first, as in main(), but
GridWorld's constructor uses numpy only) and check:

  1. shared components equal across arms: token_emb, path_integrator, out_norm, out_proj (all four);
     action_to_lie: Vanilla == MapPoPE-Pair (r2), Vanilla_r4mi == MapPoPE-Pair_r4mi (r4);
     layers: Vanilla == Vanilla_r4mi (MapWM), MapPoPE-Pair == MapPoPE-Pair_r4mi (PoPE, incl. pope_delta);
  2. structure: 32 angles per head in every arm, omega identical, rank(W_out W_in) = 2 / 2 / 4 / 4;
  3. function: MapPoPE-Pair_r4mi given MapPoPE-Pair's weights embedded in its rank-4 bottleneck (extra latent dims
     zero) gives MapPoPE-Pair's logits to float32 rounding (< 1e-5; the zero dims change the reduction length) (so the r4mi class differs from the r2 class only in the bottleneck);
  4. causality: changing tokens after position t leaves logits at <= t unchanged (every arm);
  5. positive controls -- the check must FAIL on: (a) the r4mi class without the RNG rewind; (b) the older
     MapPoPE-Pair_r4 (model_pope_pair), whose r4 bottleneck is drawn inside the base constructor.
Output: score_rank_init_check_out.txt next to this file.  Run: cd /home/prashr && python3 mapformer/docs/audits/2026-10-05/score_rank_init_check.py
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, "/home/prashr")
from mapformer import train_score_rank  # noqa: F401  (registers the arms)
from mapformer.train_variant import VARIANT_MAP
from mapformer.environment import GridWorld
from mapformer.model import MapFormerWM, ActionToLieAlgebra
from mapformer.model_pope import _swap
from mapformer.model_pope_pair import MapFormerWM_PoPEPair, MapFormerWM_PoPEPair_r4
from mapformer.model_pope_pair_mi import MapFormerWM_PoPEPair_r4mi

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_rank_init_check_out.txt")
ARMS = ["Vanilla", "MapPoPE-Pair", "Vanilla_r4mi", "MapPoPE-Pair_r4mi"]
lines = []


def log(s):
    print(s, flush=True); lines.append(s)


class NoRewind(MapFormerWM_PoPEPair):            # positive control (a): r4 drawn, layers drawn after it
    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64, **kw):
        MapFormerWM.__init__(self, vocab_size, d_model, n_heads, n_layers, dropout, grid_size, 2)
        lie4 = ActionToLieAlgebra(d_model, n_heads, self.n_blocks, 4)
        self.layers = _swap(self.layers, d_model, n_heads)
        self.action_to_lie = lie4


def build(cls, s):
    torch.manual_seed(s); np.random.seed(s)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=s)
    return cls(vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64).eval(), env


def sd(m, prefix):
    return {k: v for k, v in m.state_dict().items() if k.startswith(prefix)}


def same(a, b, prefix):
    x, y = sd(a, prefix), sd(b, prefix)
    return x.keys() == y.keys() and len(x) > 0 and all(torch.equal(x[k], y[k]) for k in x)


def lie_rank(m):
    """Per-head rank of the content-to-angle map (float64 product; rows h*nb:(h+1)*nb belong to head h)."""
    W = m.action_to_lie.w_out.weight.double() @ m.action_to_lie.w_in.weight.double()
    nb = m.n_blocks
    r = {int(torch.linalg.matrix_rank(W[h * nb:(h + 1) * nb])) for h in range(m.n_heads)}
    assert len(r) == 1, r
    return r.pop()


def causal_ok(m, V, s):
    g = torch.Generator().manual_seed(s)
    x = torch.randint(0, V, (2, 300), generator=g)
    y = x.clone(); y[:, 150:] = torch.randint(0, V, (2, 150), generator=g)
    with torch.no_grad():
        return float((m(x)[:, :150] - m(y)[:, :150]).abs().max()) == 0.0


ok_all = True
SHARED = ["token_emb.", "path_integrator.", "out_norm.", "out_proj."]
for s in list(range(10, 18)) + [100]:
    M = {a: build(VARIANT_MAP[a], s)[0] for a in ARMS}
    V = M["Vanilla"].vocab_size
    checks = {
        "shared (emb, omega, readout) equal across 4 arms": all(same(M["Vanilla"], M[a], p) for a in ARMS[1:] for p in SHARED),
        "r2 bottleneck Vanilla == MapPoPE-Pair": same(M["Vanilla"], M["MapPoPE-Pair"], "action_to_lie."),
        "r4 bottleneck Vanilla_r4mi == MapPoPE-Pair_r4mi": same(M["Vanilla_r4mi"], M["MapPoPE-Pair_r4mi"], "action_to_lie."),
        "MapWM layers Vanilla == Vanilla_r4mi": same(M["Vanilla"], M["Vanilla_r4mi"], "layers."),
        "PoPE layers MapPoPE-Pair == MapPoPE-Pair_r4mi": same(M["MapPoPE-Pair"], M["MapPoPE-Pair_r4mi"], "layers."),
        "32 angles per head everywhere": all(M[a].n_blocks == 32 and M[a].path_integrator.omega.shape[-1] == 32 for a in ARMS),
        "per-head ranks 2/2/4/4": [lie_rank(M[a]) for a in ARMS] == [2, 2, 4, 4],
        "causal (all arms)": all(causal_ok(M[a], V, s) for a in ARMS),
    }
    # 3. function: embed MapPoPE-Pair's r2 bottleneck in the r4mi model -> identical logits
    P, Q = M["MapPoPE-Pair"], build(VARIANT_MAP["MapPoPE-Pair_r4mi"], s)[0]
    st = Q.state_dict()
    for k, v in P.state_dict().items():
        if k.startswith("action_to_lie."):
            continue
        st[k] = v.clone()
    wi = torch.zeros_like(st["action_to_lie.w_in.weight"]); wi[:2] = P.action_to_lie.w_in.weight
    wo = torch.zeros_like(st["action_to_lie.w_out.weight"]); wo[:, :2] = P.action_to_lie.w_out.weight
    st["action_to_lie.w_in.weight"], st["action_to_lie.w_out.weight"] = wi, wo
    Q.load_state_dict(st)
    x = torch.randint(0, V, (2, 400), generator=torch.Generator().manual_seed(s))
    with torch.no_grad():
        d = float((P(x) - Q(x)).abs().max())
    # float32: the zero latent dims change the matmul's reduction length, so equality is to rounding, not bitwise
    checks["r4mi with r2 weights embedded == MapPoPE-Pair (max logit diff < 1e-5)"] = d < 1e-5
    # 5. positive controls (must be False)
    nr = build(NoRewind, s)[0]
    old = build(MapFormerWM_PoPEPair_r4, s)[0]
    pc = {"control (a) no RNG rewind: PoPE layers equal?": same(M["MapPoPE-Pair"], nr, "layers."),
          "control (b) old MapPoPE-Pair_r4: r4 bottleneck equal to Vanilla_r4mi?": same(M["Vanilla_r4mi"], old, "action_to_lie."),
          "control (b) old MapPoPE-Pair_r4: PoPE layers equal?": same(M["MapPoPE-Pair"], old, "layers.")}
    params = {a: sum(p.numel() for p in M[a].parameters()) for a in ARMS}
    good = all(checks.values()) and not any(pc.values())
    ok_all &= good
    log(f"seed {s}: {'PASS' if good else 'FAIL'}  params {params}  logit diff {d:.1e}")
    for k, v in checks.items():
        log(f"    {'ok ' if v else 'BAD'} {k}")
    for k, v in pc.items():
        log(f"    {'ok ' if not v else 'BAD'} {k} -> {v} (must be False)")
log(f"\nOVERALL: {'PASS' if ok_all else 'FAIL'}")
open(OUT, "w").write("\n".join(lines) + "\n")

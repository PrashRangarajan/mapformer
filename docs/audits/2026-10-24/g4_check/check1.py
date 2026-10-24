import torch, numpy as np, math
import mapformer.model as M
from mapformer.environment import GridWorld
from mapformer.model_wrap_exist import MapFormerWM_SignedAO4_r4 as SO, MapFormerWM_AbsAO4_r4 as AO
from mapformer.analyze_torus_lrsweep import decay_exponent
# 1. init equality at batch seeds
for s in (2400, 2403, 2407):
    sds = []
    for cls in (SO, AO):
        torch.manual_seed(s); np.random.seed(s)
        env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=s)
        m = cls(vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)
        sds.append(m.state_dict())
    assert sds[0].keys() == sds[1].keys()
    print(s, "init identical:", all(torch.equal(sds[0][k], sds[1][k]) for k in sds[0]), "vocab", env.unified_vocab_size)
# 2. SDPA equivalence, eval mode, forward and grads
env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=2470)
toks = torch.stack([env.generate_trajectory(256)[0] for _ in range(3)])
for cls in (SO, AO):
    torch.manual_seed(2470); m = cls(vocab_size=21, d_model=128, n_heads=2, n_layers=1, grid_size=64).double(); m.eval()
    outs, grads = [], []
    for sdpa in (False, True):
        M.USE_SDPA = sdpa; m.zero_grad()
        y = m(toks[:, :-1]); y.logsumexp(-1).sum().backward()
        outs.append(y.detach().clone()); grads.append(torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]))
    M.USE_SDPA = False
    print(cls.__name__, "fwd maxdiff %.2e" % (outs[0]-outs[1]).abs().max().item(), "grad rel %.2e" % ((grads[0]-grads[1]).norm()/grads[0].norm()).item())
    # float32 too
    m = m.float(); o = []
    for sdpa in (False, True):
        M.USE_SDPA = sdpa
        with torch.no_grad(): o.append(m(toks[:, :-1]))
    M.USE_SDPA = False
    print("  fp32 fwd maxdiff %.2e" % (o[0]-o[1]).abs().max().item())
# 3. memoryless entropy per training map
H0 = -0.5*math.log(0.5) + 0.5*(math.log(2) + math.log(16))
print("analytic marginal entropy %.4f" % H0)
for s in range(2400, 2408):
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=s)
    om = env.obs_map.numpy().ravel(); p = np.bincount(om, minlength=17)/om.size
    print(s, "map entropy %.4f" % -(p[p>0]*np.log(p[p>0])).sum(), "blank frac %.3f" % p[16])
print("decay exponents", {lr: round(decay_exponent(lr), 2) for lr in (1e-3, 3e-3, 1e-2, 3e-2)}, "x lr", round(decay_exponent(1.0),1))

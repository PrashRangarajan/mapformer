import numpy as np, torch
from mapformer.environment_steps import GridWorldSteps
from mapformer.model_ssm_sign import build
from mapformer.model_signroute import ARMS
for size in (32, 8):
    for arm in ("signed", "abs", "softplus", "exp"):
        for s in (2177, 2178):
            torch.manual_seed(s); np.random.seed(s)
            env = GridWorldSteps(size=size, seed=s, p_first=0.5, deltas=(1, -1))
            act = tuple(env.action_offset + i for i in range(2))
            m = build(arm, "moves", env.unified_vocab_size, size, action_ids=act, bottleneck_r=4)
            torch.manual_seed(s); np.random.seed(s)
            env2 = GridWorldSteps(size=size, seed=s, p_first=0.5, deltas=(1, -1))
            a = ARMS[arm](vocab_size=env2.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=size,
                          action_ids=(env2.action_offset, env2.action_offset + 1))
            sa, ss = a.state_dict(), m.state_dict()
            sh = [k for k in ss if k in sa]
            diff = [k for k in sh if not torch.equal(sa[k], ss[k])]
            # also: the step tables at init agree (same readout code)
            dst = float((a.step_table() - m.step_table()).abs().max())
            print(size, arm, s, len(sh), "shared; diff", diff, "only-ssm", sorted(set(ss) - set(sa)), "only-attn",
                  sorted(set(sa) - set(ss)), "step-table diff", dst)
# index arm: same shared init as the signed attention arm?
torch.manual_seed(2177); m = build("index", "moves", 19, 32)
torch.manual_seed(2177); a = ARMS["signed"](vocab_size=19, d_model=128, n_heads=2, n_layers=1, grid_size=32, action_ids=(0, 1))
print("index vs attention signed shared diff", [k for k in m.state_dict() if k in a.state_dict() and not torch.equal(m.state_dict()[k], a.state_dict()[k])])
torch.manual_seed(2177); m0 = build("signed", "moves", 19, 32, mixer="none")
print("none vs attention signed shared diff", [k for k in m0.state_dict() if k in a.state_dict() and not torch.equal(m0.state_dict()[k], a.state_dict()[k])])
# init statistics: share of move steps negative; init move angle sums > pi (the past-pi start claim for exp)
for arm in ("signed", "abs", "softplus", "exp"):
    for size in (32, 8):
        fr, past = [], []
        for s in range(2100, 2108):
            torch.manual_seed(s); m = build(arm, "moves", 19, size)
            A = (m.step_table()[m.act] * m.path_integrator.omega).detach()      # (2, H, N)
            fr.append(float((m.step_table()[m.act] < 0).double().mean()))
            past.append(float((A.sum(0).abs() > np.pi).double().sum(-1).mean()))
        print(f"init {arm:8s} m={size:2d}: frac move steps negative {np.mean(fr):.3f}; channels per head with |move-angle sum| > pi {np.mean(past):.1f} of 32")

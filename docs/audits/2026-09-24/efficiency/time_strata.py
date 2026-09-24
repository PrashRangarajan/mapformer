import sys, time, importlib
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, "/tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/audit_efficiency")
OS = importlib.import_module("mapformer.eval_rank_strata"); NS = importlib.import_module("mfprop.eval_rank_strata")
from mapformer.environment import GridWorld
from mfprop.environment import GridWorld as NewGW
torch.manual_seed(0)
class Stub(torch.nn.Module):          # constant-cost stand-in for the GPU forward
    def forward(self, x): return torch.randn(x.shape[0], x.shape[1], 21)
m = Stub()
for T in (1024, 2048):
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
    np.random.seed(0); t = time.perf_counter()
    for _ in range(5): env.generate_trajectory(T)
    g_old = (time.perf_counter() - t) / 5
    t = time.perf_counter(); OS.evaluate(m, env, T, 5, 1, "cpu"); a = (time.perf_counter() - t) / 5
    t = time.perf_counter(); NS.evaluate(m, env, T, 5, 1, "cpu"); b = (time.perf_counter() - t) / 5
    env2 = NewGW(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
    t = time.perf_counter(); NS.evaluate(m, env2, T, 5, 1, "cpu"); c = (time.perf_counter() - t) / 5
    print(f"T={T}: per trial  old eval {a*1e3:.1f} ms (of which walk {g_old*1e3:.1f})   vectorised scoring {b*1e3:.1f} ms   + fast walk {c*1e3:.1f} ms")

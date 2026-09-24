import time, torch, numpy as np, cProfile, pstats, io
torch.set_num_threads(1)
from mapformer.environment import GridWorld
env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=0)
np.random.seed(1)
t=time.perf_counter(); env.generate_batch(4, 1024); dt=time.perf_counter()-t
print(f"generate_batch(4,1024): {dt*1e3:.1f} ms -> per traj {dt/4*1e3:.2f} ms, per step {dt/4/1024*1e6:.2f} us")
print(f"extrapolated per epoch (98 x 16 traj) single core: {dt/4*16*98:.2f} s")
pr=cProfile.Profile(); np.random.seed(1); pr.enable(); env.generate_batch(2,1024); pr.disable()
s=io.StringIO(); pstats.Stats(pr,stream=s).sort_stats('tottime').print_stats(8); print(s.getvalue()[:2500])
# micro-costs
n=20000
t=time.perf_counter()
for _ in range(n): np.random.randint(0,4)
print("scalar randint us", (time.perf_counter()-t)/n*1e6)
om=env.obs_map
t=time.perf_counter()
for i in range(n): om[i%64, 3].item()
print("torch obs_map[i,j].item() us", (time.perf_counter()-t)/n*1e6)
rm=torch.zeros(2048,dtype=torch.bool)
t=time.perf_counter()
for i in range(n): rm[(2*i+1)%2048]=True
print("torch bool setitem us", (time.perf_counter()-t)/n*1e6)

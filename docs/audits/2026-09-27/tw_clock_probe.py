"""Text-world clock probe (audit 2026-09-30 M1): per seed, how many of the 64 (head, block) phase channels
drift between first visit and revisit of the same cell (mean |wrapped dtheta| > 1.0 rad; random ~1.57).
A per-step clock moves those channels with time, so they cannot serve as a map."""
import numpy as np, torch
from mapformer.environment_textworld import TextWorld, VERBS, SEE
from mapformer.probe_textworld import table
from mapformer.analyze_textworld_secondary import load
R="/home/prashr/mapformer/runs/textworld/p0"
te=TextWorld(size=64,seed=10000)
for s in range(8):
    ck=f"{R}/Vanilla_r4_L1_s{s}/Vanilla_r4.pt"
    m=load(ck,"cpu")
    om=m.path_integrator.omega.detach().numpy()  # H,nb
    with torch.no_grad():
        D=m.action_to_lie(m.token_emb.weight[None])[0].numpy()  # V,H,nb
    env=te
    A={a:D[env.dir_ids[a]].mean(0) for a in range(4)}
    c=np.mean([A[a] for a in range(4)],0); v=D[[env.idx[w] for w in VERBS]].mean(0)
    per=(c+v)*om   # clock part per step, radians
    wrap=lambda x:(x+np.pi)%(2*np.pi)-np.pi
    # empirical: phases at revisit object slots vs first visit
    np.random.seed(0); dif=[]; lag=[]
    for _ in range(40):
        tok,obs,rev=te.generate_trajectory(1024)
        locs=te.visited_locations
        with torch.no_grad():
            d=m.action_to_lie(m.token_emb(tok[None]))[0].numpy()
        th=np.cumsum(d,0)*om[None]
        slots=np.nonzero(obs.numpy())[0][:len(locs)]
        first={}
        for k,i in enumerate(slots):
            L=tuple(locs[k])
            if L in first:
                j,kk=first[L]; dif.append(wrap(th[i]-th[j])); lag.append(k-kk)
            else: first[L]=(i,k)
    dif=np.array(dif); lag=np.array(lag)
    md=np.abs(dif).mean(0)   # H,nb ; random ~ pi/2
    print(f"s{s}: blocks with revisit mean|dtheta|>1.0: {(md>1.0).sum()}/{md.size}; clock|>0.3 rad/step: {(np.abs(wrap(per))>0.3).sum()}")

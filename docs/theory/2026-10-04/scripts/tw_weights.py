"""Text world: do the clock channels (revisit drift > 1 rad) carry attention weight? Weight per (head, block) =
mean over query words |q_pair| x mean over object-word keys |k_pair| (layer-1 Q,K are token functions). Post hoc, CPU."""
import numpy as np, torch
torch.set_num_threads(4)
from mapformer.environment_textworld import TextWorld, VERBS, SEE, OBJECTS
from mapformer.analyze_textworld_secondary import load
R="/home/prashr/mapformer/runs/textworld/p0"
te=TextWorld(size=64,seed=10000)
wrap=lambda x:(x+np.pi)%(2*np.pi)-np.pi
for s in range(8):
    m=load(f"{R}/Vanilla_r4_L1_s{s}/Vanilla_r4.pt","cpu"); om=m.path_integrator.omega.detach().numpy()
    np.random.seed(0); dif=[]
    for _ in range(40):
        tok,obs,rev=te.generate_trajectory(1024); locs=te.visited_locations
        with torch.no_grad(): d=m.action_to_lie(m.token_emb(tok[None]))[0].numpy()
        th=np.cumsum(d,0)*om[None]; slots=np.nonzero(obs.numpy())[0][:len(locs)]; first={}
        for k,i in enumerate(slots):
            L=tuple(locs[k])
            if L in first: dif.append(wrap(th[i]-th[first[L]]))
            else: first[L]=i
    md=np.abs(np.array(dif)).mean(0)                     # H,nb
    with torch.no_grad():
        L1=m.layers[0]; e=m.token_emb.weight; h=L1.norm1(e); H=m.n_heads; dh=m.d_model//H; V=e.shape[0]
        Q=L1.q_proj(h).view(V,H,dh); K=L1.k_proj(h).view(V,H,dh)
        qa=torch.sqrt(Q[...,0::2]**2+Q[...,1::2]**2).mean(0); 
        oid=[te.idx[w] for w in OBJECTS]; ka=torch.sqrt(K[...,0::2]**2+K[...,1::2]**2)[oid].mean(0)
        w=(qa*ka).numpy(); w=w/w.sum()
    drift=md>1.0
    print(f"s{s}: drifting {drift.sum()}/64; weight on drifting channels {w[drift].sum():.3f}; per head drift "
          f"{drift.sum(1)} weight-share per head {np.round(w.sum(1),2)}; weight on drifting within head {[round(float(w[h][drift[h]].sum()/w[h].sum()),3) for h in range(H)]}")

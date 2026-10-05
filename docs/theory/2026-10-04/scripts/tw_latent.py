"""Text world, r=4 shared latent, D=2. Lemma check: a per-move clock can coexist with a full 2-D drift-free map only if
the clock's latent has a component outside span(u,v). Per seed: share of the per-move clock latent outside span(u,v);
and in the drift-free channels (per-move clock |wrapped| < 0.05 rad), the independence s_min/s_max of the w-weighted
2-column matrix of position covectors (omega_i * W_out_i . u, omega_i * W_out_i . v). Post hoc, CPU."""
import numpy as np, torch
torch.set_num_threads(4)
from mapformer.environment_textworld import TextWorld, VERBS, OBJECTS
from mapformer.analyze_textworld_secondary import load
R="/home/prashr/mapformer/runs/textworld/p0"; te=TextWorld(size=64,seed=10000)
wrap=lambda x:(x+np.pi)%(2*np.pi)-np.pi
for s in range(8):
    m=load(f"{R}/Vanilla_r4_L1_s{s}/Vanilla_r4.pt","cpu")
    with torch.no_grad():
        E=m.token_emb.weight; Z=m.action_to_lie.w_in(E).numpy()            # V,r
        Wo=m.action_to_lie.w_out.weight.numpy()                               # H*nb, r
        om=m.path_integrator.omega.detach().numpy().reshape(-1)
        L1=m.layers[0]; h=L1.norm1(E); H=m.n_heads; dh=m.d_model//H; V=E.shape[0]
        Q=L1.q_proj(h).view(V,H,dh); K=L1.k_proj(h).view(V,H,dh)
        qa=torch.sqrt(Q[...,0::2]**2+Q[...,1::2]**2).mean(0)
        ka=torch.sqrt(K[...,0::2]**2+K[...,1::2]**2)[[te.idx[w] for w in OBJECTS]].mean(0)
        w=(qa*ka).numpy().reshape(-1); w=w/w.sum()
    Dz={a:Z[te.dir_ids[a]].mean(0) for a in range(4)}
    # pair opposite directions using the env's deltas
    dl=te.ACTION_DELTAS if hasattr(te,'ACTION_DELTAS') else {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}
    u=(Dz[0]-Dz[1])/2; v=(Dz[2]-Dz[3])/2
    mlat=np.mean([Dz[a] for a in range(4)],0)+Z[[te.idx[x] for x in VERBS]].mean(0)
    B,_=np.linalg.qr(np.stack([u,v],1)); perp=np.linalg.norm(mlat-B@(B.T@mlat))/(np.linalg.norm(mlat)+1e-12)
    clock=wrap(om*(Wo@mlat)); P=np.stack([om*(Wo@u), om*(Wo@v)],1)          # per channel
    clean=np.abs(clock)<0.05
    Pw=P[clean]*np.sqrt(w[clean])[:,None]; sv=np.linalg.svd(Pw,compute_uv=False)
    print(f"s{s}: |m_lat|/|u_lat| {np.linalg.norm(mlat)/np.linalg.norm(u):.3f}; share outside span(u,v) {perp:.3f}; "
          f"clean channels {clean.sum()}/64 carrying {w[clean].sum():.2f} of weight; clean-map independence {sv[-1]/sv[0]:.3f}")

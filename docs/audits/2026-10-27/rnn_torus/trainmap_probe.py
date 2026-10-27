# Run: cd /home/prashr && OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/prashr python3 mapformer/docs/audits/2026-10-27/rnn_torus/trainmap_probe.py 10 OUT.json
"""Post hoc (unregistered), CPU, eval-only: on held-out-map walks (env seed 10000, walks np seed 1234+s, the first
N of the eval's walks), how often does each model's prediction equal the TRAINING map's content at that cell?
Also vs an unrelated map (GridWorld seed 20000, the context-destruction map) on cells where the three maps differ."""
import os, sys, json, numpy as np, torch
torch.set_num_threads(8)
from mapformer import model_rnn, train_variant
from mapformer.environment import GridWorld, _fast_walk_batch
model_rnn.register()
R='/home/prashr/mapformer/runs/rnn_torus'
cells={'Path_L1':('anchor','Vanilla_r4'),'RoPE_L1':('anchor','RoPE'),
 'LSTM_L1_h313@0.001':('lr1e-03','LSTM_L1_h313'),'LSTM_L1_h313@0.003':('lr3e-03','LSTM_L1_h313'),
 'LSTM_L1_h157@0.003':('lr3e-03','LSTM_L1_h157'),'GRU_L1_h181@0.001':('lr1e-03','GRU_L1_h181'),
 'LSTM_L2_h111@0.003':('lr3e-03','LSTM_L2_h111'),'GRU_L2_h128@0.003':('lr3e-03','GRU_L2_h128')}
N=int(sys.argv[1]) if len(sys.argv)>1 else 10; T=1024
held=GridWorld(size=64,n_obs_types=16,p_empty=0.5,seed=10000)
out={}
for c,(root,v) in cells.items():
    rows=[]
    for s in range(3300,3308):
        blob=torch.load(f'{R}/{root}/p0/{v}_s{s}/{v}.pt',map_location='cpu',weights_only=False); cfg=blob['config']
        m=train_variant.VARIANT_MAP[v](vocab_size=cfg['vocab_size'],d_model=cfg['d_model'],n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers'],grid_size=cfg['grid_size']); m.load_state_dict(blob['model_state_dict']); m.eval()
        trm=GridWorld(size=64,n_obs_types=16,p_empty=0.5,seed=s).obs_map.numpy()+held.obs_offset
        ctl=GridWorld(size=64,n_obs_types=16,p_empty=0.5,seed=20000).obs_map.numpy()+held.obs_offset
        np.random.seed(1234+s)
        dd_tr=[];dd_ct=[];lg_dd_tr=[];lg_dd_ct=[];acc=[];eqtr=[];agree=[];dis_tr=[];dis_held=[];lg_acc=[];lg_tr=[];lg_dis_tr=[]
        for _ in range(N):
            tok,om,rev,locs=_fast_walk_batch(held,1,T,None)
            tok=torch.as_tensor(tok); rev=np.asarray(rev)[0]
            with torch.no_grad(): pred=m(tok[:,:-1]).argmax(-1)[0].numpy()
            tgt=tok[0,1:].numpy(); msk=rev[1:]
            L=locs[0]; last={}; gap=np.zeros(2*T-1,dtype=int)
            tcell=np.zeros(2*T-1,dtype=np.int64); ccell=np.zeros(2*T-1,dtype=np.int64)
            for j,(x,y) in enumerate(L):
                p=2*j  # obs of step j sits at token 2j+1 -> target index 2j
                tcell[p]=trm[x,y]; ccell[p]=ctl[x,y]; gap[p]=j-last.get((x,y),-10**6); last[(x,y)]=j
            P=pred[msk];G=tgt[msk];TR=tcell[msk];CT=ccell[msk];g=gap[msk]
            dd=(G!=TR)&(G!=CT)&(TR!=CT); dd_tr.append((P==TR)[dd]); dd_ct.append((P==CT)[dd]); lgd=dd&(g>=128); lg_dd_tr.append((P==TR)[lgd]); lg_dd_ct.append((P==CT)[lgd])
            acc.append(P==G); eqtr.append(P==TR); agree.append(G==TR)
            d=G!=TR; dis_tr.append((P==TR)[d]); dis_held.append((P==G)[d])
            lg=g>=128; lg_acc.append((P==G)[lg]); lg_tr.append((P==TR)[lg]); lg_dis_tr.append((P==TR)[lg&d])
        f=lambda z: float(np.concatenate(z).mean())
        rows.append(dict(seed=s,acc=f(acc),pred_eq_train=f(eqtr),maps_agree=f(agree),disagree_pred_train=f(dis_tr),
            disagree_pred_held=f(dis_held),longgap_acc=f(lg_acc),longgap_pred_train=f(lg_tr),longgap_dis_pred_train=f(lg_dis_tr),three_way_pred_train=f(dd_tr),three_way_pred_ctrl=f(dd_ct),lg_three_way_pred_train=f(lg_dd_tr),lg_three_way_pred_ctrl=f(lg_dd_ct),n_three_way=int(sum(len(z) for z in dd_tr))))
    out[c]=rows
    mean={k:np.mean([r[k] for r in rows]) for k in rows[0] if k!='seed'}
    print(c,' '.join(f'{k} {v:.3f}' for k,v in mean.items()),flush=True)
json.dump(out,open(sys.argv[2],'w'),indent=1)

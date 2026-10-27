# Independent re-derivation of every RNN_TORUS_ANALYSIS.txt number from the eval / strata / own-map / CD JSONs and the checkpoints (2026-10-27).
import json, itertools, numpy as np, torch, glob
from scipy.stats import fisher_exact, pearsonr
R='/home/prashr/mapformer'
cells={'Path_L1':('anchor','Vanilla_r4'),'RoPE_L1':('anchor','RoPE'),
 'LSTM_L1_h313@0.001':('lr1e-03','LSTM_L1_h313'),'LSTM_L1_h313@0.003':('lr3e-03','LSTM_L1_h313'),
 'LSTM_L1_h157@0.003':('lr3e-03','LSTM_L1_h157'),'GRU_L1_h181@0.001':('lr1e-03','GRU_L1_h181'),
 'LSTM_L2_h111@0.003':('lr3e-03','LSTM_L2_h111'),'GRU_L2_h128@0.003':('lr3e-03','GRU_L2_h128')}
S=list(range(3300,3308))
cd=json.load(open(f'{R}/RNN_TORUS_CD.json'))
D={}
for c,(root,v) in cells.items():
    E=json.load(open(f'{R}/RNN_TORUS_{root}.json')); ST=json.load(open(f'{R}/RNN_TORUS_{root}_STRATA.json'))
    row={}
    for T in (512,1024,2048):
        m={s:a for s,a,_ in E[f'0.0|{v}|{T}']}; row[T]=[m[s] for s in S]
    own=[]
    for s in S:
        o=json.load(open(f'{R}/RNN_TORUS_{root}_OWN_s{s}.json')); own.append(o[f'0.0|{v}|1024'][0][1])
    row['own']=own
    tail=[];nl=[];cfgs=[]
    for s in S:
        b=torch.load(glob.glob(f'{R}/runs/rnn_torus/{root}/p0/{v}_s{s}/*.pt')[0],map_location='cpu',weights_only=False)
        L=b['losses']; nl.append(len(L)); tail.append(float(np.mean(L[-45:]))); cfgs.append(b.get('config',{}))
    row['tail']=tail; row['nl']=set(nl)
    row['np']=sum(1 for _ in [0])
    st={k:np.mean([ST[f'{v}|{s}|1024'][k]['acc'] for s in S]) for k in ('all','plain_lag<128','plain_lag>=128','wrap')}
    stall=[ST[f'{v}|{s}|1024']['all']['acc'] for s in S]
    assert np.allclose(stall,row[1024]),c
    row['strata']=st
    row['cd']=np.mean([cd[f'{root}|{v}|{s}']['destroyed'] for s in S]); row['cdc']=np.mean([cd[f'{root}|{v}|{s}']['clean'] for s in S])
    D[c]=row
def perm(a,b):
    a=np.array(a);b=np.array(b);x=np.r_[a,b];obs=abs(a.mean()-b.mean());k=0;n=0
    for idx in itertools.combinations(range(16),8):
        m=np.zeros(16,bool);m[list(idx)]=True;d=abs(x[m].mean()-x[~m].mean());n+=1;k+=d>=obs-1e-12
    return k/n
for c,r in D.items():
    print(f"{c:20s} nl{r['nl']} 512 {np.mean(r[512]):.4f} 1024 {np.mean(r[1024]):.4f} sd {np.std(r[1024],ddof=1):.4f} 2048 {np.mean(r[2048]):.4f} own {np.mean(r['own']):.4f} tail {np.mean(r['tail']):.4f} solved {sum(t<0.05 for t in r['tail'])} strata {' '.join(f'{v:.3f}' for v in r['strata'].values())} cd {r['cd']:.4f} cdclean {r['cdc']:.4f} ownmin {min(r['own']):.3f} ownmax {max(r['own']):.3f}")
P=D['Path_L1']
for c in list(D)[1:]:
    print(c,'d',round(np.mean(P[1024])-np.mean(D[c][1024]),4),'perm',round(perm(P[1024],D[c][1024]),5),'vsRoPE',round(np.mean(D[c][1024])-np.mean(D['RoPE_L1'][1024]),4))
print('fisher 8v0',fisher_exact([[8,0],[0,8]])[1])
x=sum((D[c]['tail'] for c in D),[]);y=sum((D[c][1024] for c in D),[])
print('r(loss,acc) all64',round(pearsonr(x,y)[0],3))
xr=sum((D[c]['tail'] for c in list(D)[2:]),[]);yr=sum((D[c][1024] for c in list(D)[2:]),[])
print('r within RNN 48', round(pearsonr(xr,yr)[0],3))
print('S9',round(np.mean(D['LSTM_L1_h313@0.001'][1024])-np.mean(D['LSTM_L1_h157@0.003'][1024]),4))
print('--- own vs held per seed')
for c,r in D.items():
    g=np.array(r['own'])-np.array(r[1024]); print(c, 'gap mean %.4f min %.4f max %.4f k>0 %d'%(g.mean(),g.min(),g.max(),(g>0).sum()), 'T2048-T1024 %.4f'%(np.mean(r[2048])-np.mean(r[1024])))
for c in list(D)[2:]:
    x=np.array(D[c]['tail']);y=np.array(D[c][1024]); print(c,'within-cell r %.2f'%np.corrcoef(x,y)[0,1])

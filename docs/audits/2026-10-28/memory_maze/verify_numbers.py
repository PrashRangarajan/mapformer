"""Independent re-derivation of every Memory Maze batch number from the 64 results.json (MEMORY_MAZE_RESULTS.md).
Does not import analyze_mm or stats_core: own exact two-sample permutation (all 12,870 splits), own window slope.
  python3 -I verify_numbers.py <run root>      (<root>/<ARM>_s<seed>/{results.json, DONE})
"""
import json, glob, os, itertools, numpy as np, sys
root=sys.argv[1]
R={}
md5s=set(); md5start=set(); torch=set(); tf=set(); gpus={}; changed=[]
for d in sorted(glob.glob(root+'/*_s*')):
    b=os.path.basename(d); arm,s=b.rsplit('_s',1); s=int(s)
    assert os.path.exists(d+'/DONE')
    r=json.load(open(d+'/results.json')); R.setdefault(arm,{})[s]=r
    md5s.add(json.dumps(r['md5'],sort_keys=True)); md5start.add(json.dumps(r['config']['md5_start'],sort_keys=True))
    torch.add(r['env']['torch']); tf.add(json.dumps(r['config']['tf32'],sort_keys=True)); gpus[r['env']['gpu']]=gpus.get(r['env']['gpu'],0)+1
    if r['md5_changed']: changed.append(b)
print('n runs', sum(len(v) for v in R.values()), {a:sorted(v) for a,v in R.items()})
print('md5 end sets',len(md5s), list(md5s)); print('md5 start sets',len(md5start), md5start==md5s)
print('torch',torch,'tf32',tf,'gpus',gpus,'changed',changed)
for a in R:
    c=[R[a][s]['config'] for s in R[a]]
    print(a,'n_params',{x['n_params'] for x in c},'n_trainable',{x['n_trainable'] for x in c},'install',{x['install'] for x in c},
          'epochs',{x['epochs'] for x in c},'loss',{x['loss_steps'] for x in c},'hist len',{len(R[a][s]['hist']) for s in R[a]},
          'sec/ep %.2f'%np.mean([R[a][s]['sec_per_epoch'] for s in R[a]]), 'probe_sec %.0f'%np.mean([R[a][s].get('probe_sec',0) for s in R[a]]))
def v(a,split='eval',mode='normal',st='nonlocal',k='R1'):
    return np.array([R[a][s][split][mode][st][k] for s in sorted(R[a])])
def perm(x,y):
    z=np.concatenate([x,y]); n=len(x); obs=abs(x.mean()-y.mean()); cnt=0; tot=0
    for idx in itertools.combinations(range(len(z)),n):
        m=np.zeros(len(z),bool); m[list(idx)]=True
        dd=abs(z[m].mean()-z[~m].mean()); tot+=1; cnt+= dd>=obs-1e-12
    return cnt/tot
print('\nper arm eval nonlocal R1 / held / R2 / all / swap / shuffle / n scored mean')
for a in R:
    e=v(a); h=v(a,'held')
    print(f"{a:10s} {e.mean():.4f} sd {e.std(ddof=1):.4f} min {e.min():.3f} max {e.max():.3f} | held {h.mean():.4f} sd {h.std(ddof=1):.4f} | R2 {v(a,k='R2').mean():.4f} heldR2 {v(a,'held',k='R2').mean():.4f} | all {v(a,st='all').mean():.4f} | swap {v(a,mode='frames_swapped').mean():.4f} shuf {v(a,mode='steps_shuffled').mean():.4f} | n {v(a,k='n').mean():.0f} heldn {v(a,'held',k='n').mean():.0f}")
for a,b in [('P_E','R1'),('P_E','R4'),('P_E','GRU'),('P_odo','P_E'),('P_odo','GPS'),('P_E_frozen','R1'),('P_raw','P_E'),('R4','R1'),('GPS','P_E'),('P_E_frozen','P_E')]:
    x,y=v(a),v(b); print(f"{a}-{b}: d {x.mean()-y.mean():+.4f} p {perm(x,y):.5f}")
x,y=v('P_E','held'),v('R1','held'); print('held P_E-R1', x.mean()-y.mean(), perm(x,y))
print('\nper seed P_E',np.round(v('P_E'),3)); print('per seed R4',np.round(v('R4'),3)); print('per seed R1',np.round(v('R1'),3))
# descending
def ws(h,a=.6,b=.85):
    h=np.asarray(h); i0,i1=int(round(a*len(h))),int(round(b*len(h))); y=h[i0:i1]
    return np.polyfit(np.arange(len(y)),y,1)[0]*len(y)/abs(y.mean())
for a in R:
    w=[ws(R[a][s]['hist']) for s in sorted(R[a])]
    fl=[R[a][s]['hist'][-1] for s in sorted(R[a])]
    print(a,'window slopes %',np.round(np.array(w)*100,1),'DESC',sum(x<-0.03 for x in w),'final loss %.3f'%np.mean(fl))
print('\nprobe per arm blinded walls/obj, unblinded, sd')
for a in R:
    bw=np.array([R[a][s]['probe']['blinded']['walls_acc'] for s in R[a]]); bo=np.array([R[a][s]['probe']['blinded']['objects_mse_sqdist'] for s in R[a]])
    uw=np.array([R[a][s]['probe']['unblinded']['walls_acc'] for s in R[a]]); uo=np.array([R[a][s]['probe']['unblinded']['objects_mse_sqdist'] for s in R[a]])
    print(f"{a:10s} B walls {bw.mean():.2f}+-{bw.std(ddof=1):.2f} obj {bo.mean():.2f}+-{bo.std(ddof=1):.2f} | U walls {uw.mean():.2f} obj {uo.mean():.2f}")
print('probe keys', list(R['P_E'][3600]['probe'].keys()), 'constant set', {json.dumps(R[a][s]['probe']['constant']) for a in R for s in R[a]}.__len__())

# ---- per-seed tables, loss vs accuracy, P_E loss trajectories
import json, glob, os, sys, numpy as np
root=sys.argv[1]
R={}
for d in sorted(glob.glob(root+'/*_s*')):
    arm,s=os.path.basename(d).rsplit('_s',1); R.setdefault(arm,{})[int(s)]=json.load(open(d+'/results.json'))
order=['P_odo','R4','P_E','GPS','P_E_frozen','GRU','P_raw','R1']
print('| arm | '+' | '.join(str(s) for s in range(3600,3608))+' | mean |')
for a in order:
    v=[R[a][s]['eval']['normal']['nonlocal']['R1'] for s in range(3600,3608)]
    print(f'| {a} | '+' | '.join(f'{x:.3f}' for x in v)+f' | {np.mean(v):.3f} |')
print('held')
for a in order:
    v=[R[a][s]['held']['normal']['nonlocal']['R1'] for s in range(3600,3608)]
    print(f'| {a} | '+' | '.join(f'{x:.3f}' for x in v)+f' | {np.mean(v):.3f} |')
allx=[];ally=[]
for a in order:
    fl=np.array([np.mean(R[a][s]['hist'][-30:]) for s in range(3600,3608)]); ac=np.array([R[a][s]['eval']['normal']['nonlocal']['R1'] for s in range(3600,3608)])
    allx+=list(fl); ally+=list(ac)
    print(a,'final-5%% loss %.3f (%.3f-%.3f)'%(fl.mean(),fl.min(),fl.max()),'r within %.2f'%np.corrcoef(fl,ac)[0,1], 'longgap %.4f'%np.mean([R[a][s]['eval']['normal']['longgap']['R1'] for s in R[a]]), 'ep-mean %.4f'%np.mean([R[a][s]['eval']['normal']['nonlocal_episode_mean_R1'] for s in R[a]]))
print('pooled r(final loss, acc) over 64: %.3f'%np.corrcoef(allx,ally)[0,1])
# P_E jump: epoch where loss first falls below midpoint between epoch 100 and final? print loss at epochs 100,200,300,400,600
for s in range(3600,3608):
    h=R['P_E'][s]['hist']; print('P_E',s,[round(h[i],2) for i in (99,149,199,249,299,399,499,599)], round(R['P_E'][s]['eval']['normal']['nonlocal']['R1'],3))
for s in range(3600,3602):
    h=R['R4'][s]['hist']; print('R4',s,[round(h[i],2) for i in (99,199,299,399,499,599)])
print('wall per run h: ', {a: round(np.mean([R[a][s]['sec_per_epoch']*600/3600 for s in R[a]]),2) for a in R})
print('total A40-ish h train', sum(R[a][s]['sec_per_epoch']*600/3600 + (R[a][s].get('probe_sec',0)+R[a][s].get('score_sec',0))/3600 for a in R for s in R[a]))

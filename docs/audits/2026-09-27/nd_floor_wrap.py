"""Floors and wrap-only revisit share for the rank-ND task (audit 2026-10-01): best constant, reversal-copy
(lag 2), retrace (copy o[t-2j] while a run reverses the previous one), and the fraction of revisits whose
earlier visit is only at a different UNWRAPPED position (needs the phase to be periodic in N)."""
import numpy as np
from collections import Counter
from mapformer.environment_nd import GridWorldND
def opp(a): return a^1
def run(D,N,T,ntraj=100,seed=10000):
    env=GridWorldND(D,N,seed=seed); np.random.seed(seed)
    blank=env.unified_blank
    tot=0; c_const=0; c_rev=0; c_ret=0; c_ret_or=0
    wrap_only=0; lag=[]; c_local=Counter()
    for _ in range(ntraj):
        tok,_,rev=env.generate_trajectory(T)
        a=tok[0::2].numpy(); o=tok[1::2].numpy(); r=rev[1::2].numpy()
        # unwrapped positions
        U=np.cumsum(env.action_deltas[a],axis=0)
        P=[tuple(x) for x in env.visited_locations]
        last_cell={}; last_unw={}
        # retrace state
        ret_j=0; prev_run_len=0; cur_len=0
        for t in range(T):
            # run bookkeeping
            if t>0 and a[t]==a[t-1]:
                cur_len+=1
            else:
                if t>0 and a[t]==opp(a[t-1]):
                    prev_run_len=cur_len; ret_j=0; retr=True
                else:
                    retr=False; prev_run_len=0
                cur_len=1
            if t>0 and a[t]!=a[t-1]:
                reversing = a[t]==opp(a[t-1])
            # retrace predictor: in a run that reversed the previous run, step j (1-based) copies o[t-2j] if j<=prev_run_len
            j=cur_len
            if r[t]:
                tot+=1
                c_const+= o[t]==blank
                # reversal-copy (lag 2 only, at the reversal step)
                if t>=2 and a[t]==opp(a[t-1]): pred=o[t-2]
                else: pred=blank
                c_rev+= pred==o[t]
                # retrace within reversed run
                pr=blank
                if prev_run_len>0 and j<=prev_run_len and t-2*j>=0 and a[t-j]==opp(a[t]) if t-j>=0 else False:
                    pr=o[t-2*j]
                c_ret+= pr==o[t]
                key=P[t]; uk=tuple(U[t])
                L=t-last_cell[key]
                lag.append(L)
                if uk not in last_unw: wrap_only+=1
                c_local[min(L,10**9)]+=0
            last_cell[P[t]]=t; last_unw[tuple(U[t])]=t
        # note: start cell not 'seen' by env; our last_cell includes only post-step cells, consistent
    lag=np.array(lag)
    print(f"D={D} N={N} T={T}: scored {tot}  const(blank) {c_const/tot:.3f}  reversal-copy {c_rev/tot:.3f}  retrace {c_ret/tot:.3f}  "
          f"wrap-only revisits {wrap_only/tot:.3f}  lag to last visit: median {np.median(lag):.0f}, <=2 {np.mean(lag<=2):.3f}, <=20 {np.mean(lag<=20):.3f}, <=100 {np.mean(lag<=100):.3f}")
for D,N in [(2,64),(2,32),(3,10)]:
    for T in [1024,2048]:
        run(D,N,T)

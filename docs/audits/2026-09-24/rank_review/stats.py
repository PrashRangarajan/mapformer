import json, numpy as np, torch
from scipy import stats
REPO="/home/prashr/mapformer"
M=json.load(open(f"{REPO}/RANK_MATCHED.json"))
def acc(v,T=1024): d={x[0]:x[1] for x in M[f"0.0|{v}|{T}"]}; return np.array([d[s] for s in range(8)])
L={}
for v in ("Vanilla","Vanilla_r4"):
    L[v]=np.array([torch.load(f"{REPO}/runs/rank_matched/p0/{v}_s{s}/{v}.pt",map_location="cpu",weights_only=False)["losses"][-1] for s in range(8)])
a2,a4=acc("Vanilla"),acc("Vanilla_r4"); l2,l4=L["Vanilla"],L["Vanilla_r4"]
d=a4-a2
print("paired d",np.round(d,3),"mean",d.mean().round(4),"sd",d.std(ddof=1).round(4))
print("corr of acc across arms by seed (pairing benefit):",np.corrcoef(a2,a4)[0,1].round(3))
n=8; se=d.std(ddof=1)/np.sqrt(n)
print(f"t={d.mean()/se:.2f}, p(t7, two-sided)={2*stats.t.sf(abs(d.mean()/se),7):.4f}")
print(f"MDE normal 2.8*se={2.8*se:.3f}; t-based (t.975,7 + t.80,7)*se={(stats.t.ppf(.975,7)+stats.t.ppf(.8,7))*se:.3f}")
print("threshold |mean|>2.8 se corresponds to two-sided p(t7) =", round(2*stats.t.sf(2.8,7),4))
print("Wilcoxon signed-rank p:", stats.wilcoxon(d).pvalue.round(4), " sign test 7/8 p two-sided:", round(stats.binomtest(7,8).pvalue,4))
# exact paired permutation (sign-flip) test
import itertools
obs=d.mean(); cnt=0
for signs in itertools.product([1,-1],repeat=8):
    if abs((np.array(signs)*d).mean())>=abs(obs)-1e-12: cnt+=1
print("exact sign-flip permutation p:",cnt/256)
# success-rate view
for thr in (0.5,0.2):
    s2=(l2<thr).sum(); s4=(l4<thr).sum()
    print(f"train loss<{thr}: r2 {s2}/8, r4 {s4}/8, Fisher p={stats.fisher_exact([[s2,8-s2],[s4,8-s4]])[1]:.3f}")
# loss-matched variants
x=np.log(np.r_[l2,l4]); y=np.r_[a2,a4]; g=np.r_[np.zeros(8),np.ones(8)]
b=np.polyfit(x,y,1); res=y-np.polyval(b,x); print("pooled OLS (as in analyze): slope",b[0].round(4),"resid diff",(res[8:]-res[:8]).mean().round(4))
X=np.c_[np.ones(16),g,x]; beta,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@beta; s2e=r@r/(16-3); cov=s2e*np.linalg.inv(X.T@X)
print(f"ANCOVA acc~arm+logloss: arm {beta[1]:+.4f} (se {np.sqrt(cov[1,1]):.4f}), slope {beta[2]:+.4f}")
for v,xx,yy in (("r2",np.log(l2),a2),("r4",np.log(l4),a4)):
    print(v,"within slope",np.polyfit(xx,yy,1)[0].round(4),"spearman",stats.spearmanr(xx,yy)[0].round(3))
# in the overlap region only
lo,hi=max(l2.min(),l4.min()),min(l2.max(),l4.max())
print(f"overlap region {lo:.3f}-{hi:.3f}:")
for v,ll,aa in (("r2",l2,a2),("r4",l4,a4)):
    k=(ll>=lo)&(ll<=hi); print("  ",v,[(round(float(p),3),round(float(q),3)) for p,q in sorted(zip(ll[k],aa[k]))])
# old batch, training loss vs acc at T=128 and T=1024
O=json.load(open(f"{REPO}/RANK_SWEEP.json"))
print("RANK_SWEEP.json keys sample:",list(O.keys())[:8])

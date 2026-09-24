import numpy as np
from scipy import stats
np.random.seed(0)
print("P(|t_{n-1}| > 2.8) under H0 -- the actual alpha of the 'DETECTABLE' verdict")
for n in (3,4,5,6,8,12,16,24):
    a = 2*stats.t.sf(2.8, n-1)
    k = stats.t.ppf(0.975,n-1)+stats.t.ppf(0.80,n-1)
    print(f"  n={n:2d} alpha={a:.3f}   exact 80%-power/alpha.05 multiplier {k:.2f} (repo uses 2.8; ratio {k/2.8:.2f})")
# Bimodal, uncorrelated arms: each seed solves w.p. p (acc ~0.99) or not (acc ~ U-ish around 0.62)
def draw(n,p,rng):
    s = rng.random(n)<p
    return np.where(s, 0.99+0.005*rng.standard_normal(n), 0.62+0.04*rng.standard_normal(n))
def sf_exact(d):
    n=len(d); import itertools
    obs=abs(d.mean()); cnt=0
    for signs in itertools.product((1,-1),repeat=n):
        cnt += abs((d*np.array(signs)).mean()) >= obs-1e-12
    return cnt/2**n
rng=np.random.default_rng(1)
print("\nBimodal null (both arms solve w.p. 0.5, independent), n=8, 4000 sims")
fp_mde=fp_signflip=fp_t=0; N=4000
for _ in range(N):
    a=draw(8,.5,rng); b=draw(8,.5,rng); d=b-a
    sd=d.std(ddof=1); 
    fp_mde += abs(d.mean()) > 2.8*sd/np.sqrt(8) if sd>0 else d.mean()!=0
    fp_t += stats.ttest_1samp(d,0).pvalue<0.05
print(f"  'DETECTABLE' rate {fp_mde/N:.3f}; paired t p<.05 rate {fp_t/N:.3f}")
# Same with a few ceiling seeds: both near 1.0 on most seeds -> sd tiny
print("\nCeiling null: both arms 0.999+-0.0005 on 8 seeds; diff tiny but sd tinier?")
hits=0
for _ in range(N):
    a=0.999+0.0005*rng.standard_normal(8); b=0.999+0.0005*rng.standard_normal(8)
    d=b-a; hits += abs(d.mean())>2.8*d.std(ddof=1)/np.sqrt(8)
print(f"  'DETECTABLE' rate {hits/N:.3f} (scale-free, same as normal case)")
# Power: r4 solves w.p. 1.0, r2 w.p. 0.5 -> how often does paired MDE call DETECTABLE vs Fisher/perm
print("\nAlternative: arm B solves 8/8, arm A solves w.p. 0.5; n=8, 2000 sims")
det=perm=fish=0; M=2000
import itertools
for _ in range(M):
    a=draw(8,.5,rng); b=draw(8,1.0,rng); d=b-a
    det += abs(d.mean())>2.8*d.std(ddof=1)/np.sqrt(8)
    sa=(a>0.9).sum(); sb=(b>0.9).sum()
    fish += stats.fisher_exact([[sb,8-sb],[sa,8-sa]])[1]<0.05
print(f"  paired-MDE DETECTABLE {det/M:.3f}   Fisher p<.05 {fish/M:.3f}")

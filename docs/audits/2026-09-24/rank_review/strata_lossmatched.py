import json, numpy as np, torch
REPO="/home/prashr/mapformer"
S=json.load(open(f"{REPO}/RANK_MATCHED_STRATA.json"))
L={v:np.array([torch.load(f"{REPO}/runs/rank_matched/p0/{v}_s{s}/{v}.pt",map_location="cpu",weights_only=False)["losses"][-1] for s in range(8)]) for v in ("Vanilla","Vanilla_r4")}
x=np.log(np.r_[L["Vanilla"],L["Vanilla_r4"]]); g=np.r_[np.zeros(8),np.ones(8)]
alla=np.array([S[f"{v}|{s}|1024"]["all"]["acc"] for v in ("Vanilla","Vanilla_r4") for s in range(8)])
for k in ("plain_lag<128","plain_lag>=128","wrap"):
    y=np.array([S[f"{v}|{s}|1024"][k]["acc"] for v in ("Vanilla","Vanilla_r4") for s in range(8)])
    for cov_name,cov in (("log loss",x),("overall acc",alla)):
        X=np.c_[np.ones(16),g,cov]; b,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@b
        se=np.sqrt(r@r/13*np.linalg.inv(X.T@X)[1,1])
        print(f"{k:15s} ~ arm + {cov_name:11s}: arm {b[1]:+.3f} (se {se:.3f}, t {b[1]/se:+.2f})")
# stratum gap = acc(stratum) - acc(plain_lag<128), per seed: a composition measure that does not need loss matching
for k in ("plain_lag>=128","wrap"):
    for v in ("Vanilla","Vanilla_r4"):
        d=[S[f"{v}|{s}|1024"][k]["acc"]-S[f"{v}|{s}|1024"]["plain_lag<128"]["acc"] for s in range(8)]
        print(f"{v:11s} {k} - lag<128 per seed:", " ".join(f"{z:+.3f}" for z in d))

"""Action vs observation tokens on the PoPE paper's tasks (post hoc, CPU). Step table of the path-integrated models,
read from the weights: step(t) = omega * W_out W_in emb(t). Each token's step is split into its projection on the
reference direction u (the mean step of the task's 'observation' tokens) -- coefficient a_t in units of u -- and the
residual (|residual| / |u|).
Indirect Indexing (`<string>, <char>, <shift>, <target>`, runs/indirect_200k MapPoPE r2 at 200k iters -- the solved
setting -- and runs/indirect MapWM r2 / MapPoPE r2 at 100k, mostly unsolved): u = mean LETTER step. If letters act as a
position counter they share u (cos to u near 1, a near 1); the 'actions' are the shift tokens: do digits step in
proportion to their value, do '+' and '-' differ in sign?
Bach chorales (runs/jsb, MapWM r2 / MapPoPE r2, 5 seeds; raster scan S A T B per 16th note; token = pitch + 2, 1 =
silence): u = frequency-weighted mean pitched step. Is every note a clock tick (cos near 1), or do steps depend on pitch?"""
import glob, numpy as np, torch
from mapformer.environment_indirect import STOI
from mapformer.environment_jsb import splits

def steps(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    E, Wi, Wo, om = sd["token_emb.weight"], sd["action_to_lie.w_in.weight"], sd["action_to_lie.w_out.weight"], sd["path_integrator.omega"]
    return ((E @ Wi.T @ Wo.T).view(E.shape[0], *om.shape) * om).reshape(E.shape[0], -1).numpy()

def proj(st, ids, u):
    a = st[ids] @ u / (u @ u); res = np.linalg.norm(st[ids] - np.outer(a, u), axis=1) / np.linalg.norm(u)
    cos = st[ids] @ u / (np.linalg.norm(st[ids], axis=1) * np.linalg.norm(u))
    return a, res, cos

print("== Indirect Indexing ==")
letters = [STOI[c] for c in [chr(x) for x in range(65, 91)] + [chr(x) for x in range(97, 123)]]
digits = [STOI[str(i)] for i in range(10)]
for tag, pat in (("MapPoPE r2, 200k (7/8 solved)", "runs/indirect_200k/MapPoPE_r2_s*/MapPoPE_r2.pt"),
                 ("MapPoPE r2, 100k", "runs/indirect/MapPoPE_r2_s*/MapPoPE_r2.pt"), ("MapWM r2, 100k", "runs/indirect/MapWM_r2_s*/MapWM_r2.pt")):
    R = []
    for p in sorted(glob.glob(f"/home/prashr/mapformer/{pat}")):
        st = steps(p); u = st[letters].mean(0)
        aL, rL, cL = proj(st, letters, u); aD, rD, _ = proj(st, digits, u)
        a = lambda c: proj(st, [STOI[c]], u)[0][0]; r = lambda c: proj(st, [STOI[c]], u)[1][0]
        pm = st[STOI["+"]] - st[STOI["-"]]
        R.append([cL.mean(), aL.std(), np.corrcoef(aD, np.arange(10))[0, 1], np.abs(aD).mean(), rD.mean(), a("+"), a("-"), r("+"), r("-"),
                  np.linalg.norm(pm) / np.linalg.norm(u), a(","), a(" ")])
    R = np.array(R)
    print(f"  {tag} (n={len(R)}), medians:")
    print(f"    letters: cos to their mean step {np.median(R[:,0]):.3f}; spread of a {np.median(R[:,1]):.3f}  (1, 0 = one shared counter step)")
    print(f"    digits:  corr(a, digit value) {np.median(R[:,2]):+.2f} [{R[:,2].min():+.2f}, {R[:,2].max():+.2f}]; |a| {np.median(R[:,3]):.2f}; residual {np.median(R[:,4]):.2f}")
    print(f"    '+': a {np.median(R[:,5]):+.2f} res {np.median(R[:,7]):.2f};  '-': a {np.median(R[:,6]):+.2f} res {np.median(R[:,8]):.2f};  |s(+) - s(-)| / |u| {np.median(R[:,9]):.2f}")
    print(f"    ',': a {np.median(R[:,10]):+.2f};  ' ': a {np.median(R[:,11]):+.2f}")

print("\n== Bach chorales ==")
X, M = splits()["train"]
toks = X[M].numpy()
freq = np.bincount(toks[toks > 0], minlength=90).astype(float); freq /= freq.sum()
pitched = [t for t in range(2, 90) if freq[t] > 1e-4]
for arm in ("MapWM_r2", "MapPoPE_r2"):
    R = []
    for p in sorted(glob.glob(f"/home/prashr/mapformer/runs/jsb/{arm}_s*/{arm}.pt")):
        st = steps(p); w = freq[pitched]; u = (w[:, None] * st[pitched]).sum(0) / w.sum()
        a, res, cos = proj(st, pitched, u); aS, rS, cS = proj(st, [1], u)
        R.append([(w * cos).sum() / w.sum(), np.corrcoef(a, pitched)[0, 1], (w * res).sum() / w.sum(), aS[0], cS[0]])
    R = np.array(R)
    print(f"  {arm} (n={len(R)}), medians: notes' cos to the mean note step (freq-weighted) {np.median(R[:,0]):.3f}; "
          f"corr(a, pitch) {np.median(R[:,1]):+.2f} [{R[:,1].min():+.2f}, {R[:,1].max():+.2f}]; residual {np.median(R[:,2]):.2f}; "
          f"silence: a {np.median(R[:,3]):+.2f}, cos {np.median(R[:,4]):+.2f}")

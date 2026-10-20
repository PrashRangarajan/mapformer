"""ANALYSIS of train_snap.py / train_nd_cpu.py outputs (post hoc, CPU pilots) for RANK_THEORY.md checks C5-C7.
Usage: python3 analyze_snap.py <dir> [<dir> ...]   -> prints, and writes <dir>/analyze_snap_out.txt
Per run, per head, over training:
  kappa (visible drift), lam (latent clock, read or not), off = lam_perp / lam (share of the latent clock outside the
  map plane), indep (frame conditioning), r_beta (resolution along the drift / mean).
Derived:
  t_k  = first snapshot at which kappa <= 0.1 * kappa(0);  t_l = first snapshot at which lam <= 0.1 * lam(0)
         'parking first' = t_k < t_l (the visible drift goes before the latent clock does)
  drift-direction persistence (D >= 2): |cos| between beta-hat at the snapshot nearest step 200 and at the end (random
         expectation in 2D: 2/pi = 0.64), with r_beta at both: the feedback account predicts r_beta near 1 early and
         falling along a persistent drift direction; a 'prune after collapse' account predicts the drift direction
         turning onto an already weak axis."""
import glob, json, os, sys
import numpy as np


def summarise(path):
    d = json.load(open(path))
    S = d["snaps"]
    out = []
    if not S:
        return d, out
    for h in range(len(S[0]["heads"])):
        k = np.array([s["heads"][h]["kappa"] for s in S]); lam = np.array([s["heads"][h]["lam"] for s in S])
        lp = np.array([s["heads"][h]["lam_perp"] for s in S]); ind = np.array([s["heads"][h]["indep"] for s in S])
        rb = np.array([s["heads"][h]["r_beta"] for s in S]); ep = np.array([s["epoch"] for s in S])
        st = np.array([s["step"] for s in S])

        def first(cond):
            i = np.where(cond)[0]
            return int(st[i[0]]) if len(i) else None
        t_k = first(k <= 0.1 * k[0]); t_l = first(lam <= 0.1 * lam[0])
        off = np.where(lam > 0, lp / np.maximum(lam, 1e-12), np.nan)
        B = [np.array(s["heads"][h]["beta"]) for s in S]
        i200 = int(np.argmin(np.abs(st - 200)))
        b0, b1 = B[i200], B[-1]
        cosb = float(abs(b0 @ b1) / max(np.linalg.norm(b0) * np.linalg.norm(b1), 1e-12)) if len(b0) > 1 else float("nan")
        out.append(dict(cosb=cosb, rb200=float(rb[i200]), h=h, k0=k[0], kf=k[-1], lam0=lam[0], lamf=lam[-1], off0=off[0], offf=off[-1],
                        off_max=np.nanmax(off), ind0=ind[0], indf=ind[-1], ind_min=ind.min(), rbf=rb[-1],
                        t_k=t_k, t_l=t_l, steps=int(st[-1]),
                        traj=[(int(a), round(float(b), 3), round(float(c), 3), round(float(e), 2), round(float(f), 2))
                              for a, b, c, e, f in zip(st, k, lam, off, ind)]))
    return d, out


def main():
    for D in sys.argv[1:]:
        L = []
        files = sorted(glob.glob(f"{D}/*.json"))
        arms = sorted({json.load(open(f))["arm"] for f in files})
        for arm in arms:
            L.append(f"===== {arm} =====")
            pf = []
            for f in files:
                d, hs = summarise(f) if json.load(open(f))["arm"] == arm else (None, None)
                if d is None:
                    continue
                fl = np.mean(d["losses"][-max(1, len(d["losses"]) // 20):])
                L.append(f" s{d['seed']} final-5% loss {fl:.4f} acc {d['acc']} basin {d.get('final_basin')} class {d.get('cls')}")
                for x in hs:
                    pf.append((x["t_k"], x["t_l"]))
                    L.append(f"   h{x['h']} kappa {x['k0']:.3f}->{x['kf']:.4f} | lam {x['lam0']:.3f}->{x['lamf']:.4f} | "
                             f"off {x['off0']:.2f}->{x['offf']:.2f} (max {x['off_max']:.2f}) | indep {x['ind0']:.2f}->"
                             f"{x['indf']:.2f} (min {x['ind_min']:.2f}) | r_beta_f {x['rbf']:.2f} | t_kappa/10 {x['t_k']} "
                             f"t_lam/10 {x['t_l']} of {x['steps']} | |cos(beta@200, beta_end)| {x['cosb']:.2f} r_beta@200 {x['rb200']:.2f}")
                    tr = x["traj"]; idx = sorted(set([0, 1, 2, 4, 8, 16, len(tr) // 4, len(tr) // 2, len(tr) - 1]))
                    L.append("      (step, kappa, lam, off, indep): " + " ".join(str(tr[i]) for i in idx if i < len(tr)))
            first_k = sum(1 for a, b in pf if a is not None and (b is None or a < b))
            L.append(f" heads where the visible drift fell 10x before the latent clock did: {first_k}/{len(pf)}")
        txt = "\n".join(L)
        open(f"{D}/analyze_snap_out.txt", "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()

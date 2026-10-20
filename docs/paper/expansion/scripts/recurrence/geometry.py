"""Omega-weighted, basis-free geometry of each model's action phase steps (the step map as the path
integrator sees it). For actions W/E (x) and S/N (y): the x-step and y-step phase vectors are the
antisymmetric parts phi_x = (phi_E - phi_W)/2, phi_y = (phi_N - phi_S)/2 over all head x channel angles.
  dim2   = sigma_2 / sigma_1 of [phi_x; phi_y]: 0 = x and y move the phase along ONE direction (a 1D code
           of the 2D torus), ~1 = two independent directions (a 2D code)
  cancel = mean over axes of |phi_+ + phi_-| / |phi_+ - phi_-|: 0 = opposite actions cancel through zero
           (a map); large = a common forward drift (a clock). Not wrap-aware (rule 8): a code can cancel
           mod 2 pi with cancel > 0; read it with the phase-reader columns.
  kappa  = (phi_y . phi_x) / |phi_x|^2: for a 1D map, the code reads x + kappa y
  wrapch = channels (of those that move) whose phase after 64 E steps and after 64 N steps is within 0.3 rad of
           0 mod 2 pi (unweighted; a rough count of wrap-consistent channels)
Per seed, beside the stored tail loss, the model accuracy and the soft reader on long-gap revisits."""
import json, sys
import numpy as np, torch
sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/recurrence")
from common import OUT, load  # noqa
ps = {a: json.load(open(f"{OUT}/pass_strata_{a}.json")) for a in ("A", "L", "L4", "C")}
sr = {a: json.load(open(f"{OUT}/phase_reader_soft_{a}.json")) for a in ("A", "L", "L4", "C")}
fin = {"A": 0, "C": 0, "L": 3, "L4": 3}
res = {}
print("| arm | seed | dim2 | cancel | code class | kappa | wrapch | tail loss | model long / wrap | soft reader long / wrap |")
print("|---|---|---|---|---|---|---|---|---|---|")
for a in ("A", "L", "L4", "C"):
    res[a] = {}
    for s in range(8):
        m, c, b = load(a, s)
        with torch.no_grad():
            e = m.token_emb.weight[:4]
            ph = m.action_to_lie(e[None])[0].reshape(4, -1) * m.path_integrator.omega.reshape(-1)
        X = torch.stack([ph[1] - ph[0], ph[3] - ph[2]]) / 2
        sv = torch.linalg.svdvals(X).numpy()
        canc = float(((ph[0] + ph[1]).norm() / (ph[1] - ph[0]).norm() + (ph[2] + ph[3]).norm() / (ph[3] - ph[2]).norm()) / 2)
        d2 = float(sv[1] / sv[0])
        cls = ("2D map" if d2 > 0.3 else "1D map") if canc < 0.1 else "drifting"
        tail = float(np.mean(b["losses"][-45:]))
        px, py = X[0], X[1]
        kap = float((py @ px) / (px @ px))
        wr = lambda v: (torch.remainder(64 * v + np.pi, 2 * np.pi) - np.pi).abs()
        mv = (ph[1].abs() > 1e-3) | (ph[3].abs() > 1e-3)
        wch = f"{int(((wr(ph[1]) < 0.3) & (wr(ph[3]) < 0.3) & mv).sum())}/{int(mv.sum())}"
        res[a][s] = dict(dim2=d2, cancel=canc, cls=cls, tail=tail, kappa=kap, wrapch=wch)
        P = ps[a][str(s)]
        print(f"| {a} | {s} | {d2:.3f} | {canc:.2f} | {cls} | {kap:+.3f} | {wch} | {tail:.3f} | {P['long']['acc'][fin[a]]:.3f} / "
              f"{P['wrap']['acc'][fin[a]]:.3f} | {sr[a][str(s)]['long']:.3f} / {sr[a][str(s)]['wrap']:.3f} |")
json.dump(res, open(f"{OUT}/geometry.json", "w"), indent=1)
for a in res:
    cl = [res[a][s]["cls"] for s in range(8)]
    print(f"{a}: " + ", ".join(f"{k} {cl.count(k)}/8" for k in ("2D map", "1D map", "drifting")))
from scipy.stats import fisher_exact
n1 = lambda a: sum(res[a][s]["cls"] == "1D map" for s in range(8))
print("1D map, aids pooled vs A: %d/16 vs %d/8, Fisher p %.4f (post hoc)" % (n1("L") + n1("L4"), n1("A"),
      fisher_exact([[n1("L") + n1("L4"), 16 - n1("L") - n1("L4")], [n1("A"), 8 - n1("A")]])[1]))
n2 = lambda a: sum(res[a][s]["cls"] == "2D map" for s in range(8))
print("2D map, aids pooled vs C: %d/16 vs %d/8, Fisher p %.2e" % (n2("L") + n2("L4"), n2("C"),
      fisher_exact([[n2("L") + n2("L4"), 16 - n2("L") - n2("L4")], [n2("C"), 8 - n2("C")]])[1]))

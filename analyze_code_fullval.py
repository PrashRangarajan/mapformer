"""Readouts for CODE_FULLVAL_PREREG.md: full-val bpc of the .final.pt code checkpoints against the
best_val_bpc (min over 36 evals of 40 windows) that every quoted code contrast was read on."""
import json
import numpy as np

from mapformer.stats_core import paired_p, mde

REPO = "/home/prashr/mapformer"; S = (0, 1, 2)
W = {"0-511": 512, "512-1023": 512, "1024-2047": 1024}


def full(path, L):
    J = json.load(open(path)); out = {}
    for k, v in J.items():
        if k.startswith("_"):
            continue
        b = v["bpc_by_position"]; ks = [x for x in W if int(x.split("-")[0]) < L]
        out[k] = sum(b[x] * W[x] for x in ks) / sum(W[x] for x in ks)
    return out


def best(d, name):
    return json.load(open(f"{REPO}/runs/{d}/{name}.json")).get("best_val_bpc")


def row(name, f, o, lines):
    d = np.array([f(s) for s in S]); dold = np.array([o(s) for s in S])
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else np.inf
    p = paired_p(d)
    flip = np.sign(d.mean()) != np.sign(dold.mean())
    v = "SIGN FLIP -> withdraw" if flip else ("keep (p<0.05)" if p < 0.05 else "UNMEASURED")
    lines.append(f"| {name} | {dold.mean():+.4f} | **{d.mean():+.4f}** | {(d > 0).sum()}/3 | {t:+.2f} | "
                 f"{'DETECTABLE' if abs(t) > 2.8 else 'unmeasured'} | {p:.3f} | {mde(d.std(ddof=1), 3):.4f} | {v} |")


def main():
    F = {"c2048": full(f"{REPO}/runs/code2048/FULLVAL_2048.json", 2048),
         "dec": full(f"{REPO}/runs/code_decay/FULLVAL_512.json", 512),
         "c512": full(f"{REPO}/runs/code/FULLVAL_512.json", 512)}
    D = {"c2048": "code2048", "dec": "code_decay", "c512": "code"}
    L = ["# Full-val bpc per run (.final.pt) vs best_val_bpc", "",
         "| batch | run | full val | best_val_bpc | diff |", "|---|---|---|---|---|"]
    for b, m in F.items():
        for k in sorted(m):
            o = best(D[b], k); L.append(f"| {D[b]} | {k} | {m[k]:.4f} | {o:.4f} | {m[k] - o:+.4f} |")
    g = lambda b, a, s: F[b][f"{a}_s{s}"]
    go = lambda b, a, s: best(D[b], f"{a}_s{s}")
    L += ["", "# Contrasts (paired over seeds 0-2; t-test p and exact-t MDE from stats_core)", "",
          "| contrast | best_val_bpc (old) | full val (new) | >0 | t | house | t-test p | t-MDE | registered |",
          "|---|---|---|---|---|---|---|---|---|"]
    for get, tag in ((g, "new"),):
        pass
    C = [("C1 encoding main", "c2048", lambda G, s: ((G("c2048", "PoPE-Flat", s) - G("c2048", "RoPE", s)) +
                                                      (G("c2048", "MapPoPE-Flat", s) - G("c2048", "Vanilla", s))) / 2),
         ("C1 position main", "c2048", lambda G, s: ((G("c2048", "Vanilla", s) - G("c2048", "RoPE", s)) +
                                                      (G("c2048", "MapPoPE-Flat", s) - G("c2048", "PoPE-Flat", s))) / 2),
         ("C1 MapPoPE - PoPE", "c2048", lambda G, s: G("c2048", "MapPoPE-Flat", s) - G("c2048", "PoPE-Flat", s)),
         ("C1 MapPoPE - MapWM", "c2048", lambda G, s: G("c2048", "MapPoPE-Flat", s) - G("c2048", "Vanilla", s)),
         ("C2 PoPE-Decay - RoPE-Decay", "dec", lambda G, s: G("dec", "PoPE-Decay", s) - G("dec", "RoPE-Decay", s)),
         ("C2 MapPoPE-Decay - PoPE-Decay", "dec", lambda G, s: G("dec", "MapPoPE-Decay", s) - G("dec", "PoPE-Decay", s)),
         ("C2 MapWM-Decay - RoPE-Decay", "dec", lambda G, s: G("dec", "MapWM-Decay", s) - G("dec", "RoPE-Decay", s))]
    for x, base in (("RoPE-Decay", "RoPE"), ("PoPE-Decay", "PoPE-Flat"), ("MapPoPE-Decay", "MapPoPE-Flat"),
                    ("MapWM-Decay", "Vanilla")):
        C.append((f"envelope {base} (cross-batch)", "dec",
                  lambda G, s, x=x, base=base: G("dec", x, s) - G("c512", base, s)))
    for name, _, f in C:
        row(name, lambda s: f(g, s), lambda s: f(go, s), L)
    print("\n".join(L))


if __name__ == "__main__":
    main()

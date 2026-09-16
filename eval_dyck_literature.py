"""Every standard Dyck metric in the literature, on one set of checkpoints.

A. close accuracy, Hewitt et al. 2020 (arXiv 2010.07515) and Yao et al. 2021 (ACL, sec. D):
   "the accuracy of correct close bracket predictions: p(>_j | >) = p(>_j) / sum_i p(>_i)", i.e.
   the probability of the legal closer renormalised over the closing brackets. Opening brackets
   are never scored -- which bracket gets opened is nondeterministic, so it is not predictable.
     A1 mean over POSITIONS (Yao: "average accuracy of generating close brackets")
     A2 mean over DISTANCES j (Hewitt: "let p_j be the probability ... given that j tokens
        separate it from its open bracket. We report mean_j p_j")
     A3 the 0/1 form: is the legal closer ranked above the illegal one. Chance 0.500.
B. set prediction, Gers & Schmidhuber 2001 / Suzgun et al. 2019 / Bhattamishra et al. 2020
   (COLING, MapFormer's ref [28]) / Ebrahimi et al. 2020: the predicted SET must equal the set of
   valid next symbols. "The model's prediction is considered to be correct if and only if its
   output at every step is correct" -- per-sequence accuracy. Those models emit per-symbol
   sigmoids at threshold 0.5; a softmax cannot put 0.5 on three symbols, so the threshold-free
   analogue is used: a step is correct iff min_{valid} p > max_{invalid} p.
     B1 per step   B2 per sequence (the published criterion)
C. Goodale et al. 2025 (ACL) F1, the metric MapFormer reports, plus the strict variant with a min
   over Val(s) in place of BT's mean.
D. language-model view: cross-entropy against the sampler's own entropy (the achievable floor).

Run from /home/prashr: python3 -m mapformer.eval_dyck_literature
"""
import glob, json
import numpy as np, torch
import torch.nn.functional as F

from mapformer.environment_dyck import DyckWorld, f1_valid, CLOSE_P, CLOSE_B
from mapformer.validate_dyck import ngram_fit, ngram_probs
from mapformer.probe_dyck_stack import top_distance, strict_f1
from mapformer.train_dyck import build

RUNS = "/home/prashr/mapformer/runs/dyck_bs128"
OUT_MD = "/home/prashr/mapformer/DYCK_LITERATURE_METRICS.md"
CELLS = [(32, 4), (128, 4), (32, 12), (128, 12)]
ARMS = [("MapPoPE-1L", "MapPoPE-1L_r2", "MapPoPE", 1, 1), ("MapWM-1L", "MapWM-1L_r2", "MapWM", 1, 1),
        ("MapEM-1L", "MapEM-1L_r2", "MapEM", 1, 1), ("PoPE-1L", "PoPE-1L", "PoPE", 1, 1),
        ("PoPE-2L", "PoPE-2L", "PoPE", 2, 2), ("RoPE-1L", "RoPE-1L", "RoPE", 1, 1),
        ("RoPE-2L", "RoPE-2L", "RoPE", 2, 2), ("CoPE-1L", "CoPE-1L", "CoPE", 1, 1),
        ("CoPE-2L", "CoPE-2L", "CoPE", 2, 2)]
N = 512
BUCK = [("1-2", 1, 2), ("3-8", 3, 8), ("9-32", 9, 32), ("33+", 33, 10 ** 6)]


def cell_data(w, L, D):
    inp, tgt, valid, ent = w.batch(N, L, D, np.random.default_rng(424242 + 1000 * L + D))
    d = dict(inp=inp, tgt=tgt, valid=valid, ent=float(ent.mean()), dist=top_distance(inp, tgt, L))
    d["dp"] = valid[..., CLOSE_P] | valid[..., CLOSE_B]
    d["corr"] = torch.where(valid[..., CLOSE_P], CLOSE_P, CLOSE_B)
    d["wrng"] = torch.where(valid[..., CLOSE_P], CLOSE_B, CLOSE_P)
    return d


def metrics(P, d, logits=None):
    valid, dp = d["valid"], d["dp"]
    pc = P.gather(-1, d["corr"].unsqueeze(-1)).squeeze(-1)
    pw = P.gather(-1, d["wrng"].unsqueeze(-1)).squeeze(-1)
    ratio = pc / (pc + pw).clamp_min(1e-12)
    js = sorted({int(x) for x in d["dist"][dp].unique()})
    pj = [float(ratio[dp & (d["dist"] == j)].mean()) for j in js if bool((dp & (d["dist"] == j)).any())]
    ok = (torch.where(valid, P, torch.full_like(P, 1.0)).min(-1).values >
          torch.where(~valid, P, torch.zeros_like(P)).max(-1).values)
    m = dict(A1_close_acc_pos=float(ratio[dp].mean()), A2_close_acc_dist=float(np.mean(pj)),
             A3_closer_top1=float((pc > pw)[dp].double().mean()),
             B1_setmatch_step=float(ok.double().mean()), B2_setmatch_seq=float(ok.all(-1).double().mean()),
             C1_paper_f1=float(f1_valid(P, valid)[0].mean()), C2_strict_f1=float(strict_f1(P, valid).mean()),
             invalid_mass=float((P * ~valid).sum(-1).mean()))
    for nm, lo, hi in BUCK:
        sel = dp & (d["dist"] >= lo) & (d["dist"] <= hi)
        m[f"A3_d{nm}"] = float((pc > pw)[sel].double().mean()) if bool(sel.any()) else float("nan")
    if logits is not None:
        m["D_ce"] = float(F.cross_entropy(logits.transpose(1, 2), d["tgt"]))
    return m


def main():
    w = DyckWorld()
    data = {c: cell_data(w, *c) for c in CELLS}
    R = {}
    for k in (1, 3):
        fit = ngram_fit(w, k, np.random.default_rng(100 + k))
        for c in CELLS:
            R[(f"n-gram k={k}", c)] = [metrics(ngram_probs(fit, k, data[c]["inp"]), data[c])]
        print(f"n-gram k={k} done", flush=True)
    for lab, name, arch, nl, nh in ARMS:
        for pt in sorted(glob.glob(f"{RUNS}/{name}_s*/{name}.pt")):
            m = build(arch, 5, nl, nh, 2, 32).cuda().eval()
            m.load_state_dict(torch.load(pt, map_location="cuda"))
            for c in CELLS:
                inp = data[c]["inp"]
                with torch.no_grad():
                    lg = torch.cat([m(inp[i:i + 128].cuda()).float().cpu() for i in range(0, N, 128)])
                R.setdefault((lab, c), []).append(metrics(lg.softmax(-1), data[c], lg))
        print(f"{lab}: {len(R[(lab, CELLS[0])])} seeds", flush=True)
    agg = {f"{lab}|L{c[0]}D{c[1]}": {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
           for (lab, c), rs in R.items()}
    for (lab, c), rs in R.items():
        agg[f"{lab}|L{c[0]}D{c[1]}"]["n_seeds"] = len(rs)
        for k in rs[0]:
            agg[f"{lab}|L{c[0]}D{c[1]}"][k + "_sd"] = float(np.std([r[k] for r in rs], ddof=1)) if len(rs) > 1 else 0.0
    json.dump(agg, open(OUT_MD.replace(".md", ".json"), "w"), indent=1)

    rows = [f"n-gram k={k}" for k in (1, 3)] + [a[0] for a in ARMS]
    L = ["# Dyck-2 under every standard metric in the literature", "",
         "Same checkpoints and same 512 evaluation sequences per cell throughout; cells are seed means "
         f"(n=8 per model, 1 for the n-grams). Trained on L=32 D=4 only. Definitions and citations in "
         "`eval_dyck_literature.py`.", ""]
    blocks = [("A1 close accuracy, mean over positions -- Yao et al. 2021 (ACL), following Hewitt et al. 2020. "
               "p(legal closer) renormalised over closing brackets. Chance 0.500.", "A1_close_acc_pos"),
              ("A2 close accuracy, mean over DISTANCES j -- Hewitt et al. 2020's 'bracket-closing memory'. "
               "Chance 0.500.", "A2_close_acc_dist"),
              ("A3 the same comparison as 0/1 -- is the legal closer ranked above the illegal one. Chance 0.500.",
               "A3_closer_top1"),
              ("B1 valid-set prediction, per step -- Gers & Schmidhuber 2001 / Suzgun et al. 2019 / "
               "Bhattamishra et al. 2020 / Ebrahimi et al. 2020, threshold-free form.", "B1_setmatch_step"),
              ("B2 valid-set prediction, PER SEQUENCE -- the published criterion: correct only if every step "
               "is correct.", "B2_setmatch_seq"),
              ("C1 F1 valid continuation -- Goodale et al. 2025, the metric MapFormer reports.", "C1_paper_f1"),
              ("C2 strict F1 -- C1 with a min over Val(s) in place of BT's mean.", "C2_strict_f1"),
              ("invalid mass -- probability on ungrammatical brackets. Uniform guesser ~0.25.", "invalid_mass")]
    for title, key in blocks:
        L += [f"## {title}", "", "| model | " + " | ".join(f"L{c[0]} D{c[1]}" for c in CELLS) + " |",
              "|---|" + "---|" * len(CELLS)]
        for r in rows:
            L.append(f"| {r} | " + " | ".join(f"{agg[f'{r}|L{c[0]}D{c[1]}'][key]:.3f}" for c in CELLS) + " |")
        L.append("")
    L += ["## A3 by distance to the bracket that must be closed (L=128, D=12)", "",
          "| model | " + " | ".join(f"d {b[0]}" for b in BUCK) + " |", "|---|" + "---|" * len(BUCK)]
    for r in rows:
        L.append(f"| {r} | " + " | ".join(f"{agg[f'{r}|L128D12']['A3_d' + b[0]]:.3f}" for b in BUCK) + " |")
    L += ["", "## D cross-entropy against the sampler's own entropy (the achievable floor)", "",
          "| model | " + " | ".join(f"L{c[0]} D{c[1]}" for c in CELLS) + " |", "|---|" + "---|" * len(CELLS)]
    for r in [a[0] for a in ARMS]:
        L.append(f"| {r} | " + " | ".join(f"{agg[f'{r}|L{c[0]}D{c[1]}']['D_ce']:.3f}" for c in CELLS) + " |")
    L.append("| *floor (sampler entropy)* | " + " | ".join(f"*{data[c]['ent']:.3f}*" for c in CELLS) + " |")
    open(OUT_MD, "w").write("\n".join(L) + "\n")
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()

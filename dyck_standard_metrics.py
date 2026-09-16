"""The Dyck metrics the literature actually uses, computed on our checkpoints.

(1) Hewitt et al. 2020 (arXiv 2010.07515), "bracket-closing memory": "let p_j be the probability
    that the model predicts the correct closing bracket given that j tokens separate it from its
    open bracket. We report mean_j p_j". Probability is renormalised over closing brackets.
    Note the average is over DISTANCES j, not over positions -- short distances do not dominate.
(2) Suzgun et al. 2019 / Bhattamishra et al. 2020 (COLING, the paper MapFormer cites as [28]),
    Next Character Prediction: the predicted SET must equal the valid set at every step, and a
    sequence counts correct only if every step is correct. Those models emit per-character
    sigmoids thresholded at 0.5; ours emit a softmax, so the threshold-free analogue is used:
    a step is correct iff min_{valid} p > max_{invalid} p.
Run from /home/prashr: python3 -m mapformer.dyck_standard_metrics
"""
import glob, json
import numpy as np, torch

from mapformer.environment_dyck import DyckWorld, CLOSE_P, CLOSE_B
from mapformer.validate_dyck import ngram_fit, ngram_probs
from mapformer.probe_dyck_stack import top_distance
from mapformer.train_dyck import build

RUNS = "/home/prashr/mapformer/runs/dyck_bs128"
CELLS = [(32, 4), (128, 4), (128, 12)]
ARMS = [("MapPoPE", "MapPoPE-1L_r2", "MapPoPE", 1, 1), ("MapFormer (MapWM)", "MapWM-1L_r2", "MapWM", 1, 1),
        ("PoPE", "PoPE-1L", "PoPE", 1, 1), ("RoPE", "RoPE-1L", "RoPE", 1, 1)]
N = 512


def main():
    w = DyckWorld(); out = {}
    for L, D in CELLS:
        inp, tgt, valid, _ = w.batch(N, L, D, np.random.default_rng(424242 + 1000 * L + D))
        dist = top_distance(inp, tgt, L)
        dp = valid[..., CLOSE_P] | valid[..., CLOSE_B]
        corr = torch.where(valid[..., CLOSE_P], CLOSE_P, CLOSE_B)
        wrg = torch.where(valid[..., CLOSE_P], CLOSE_B, CLOSE_P)
        js = sorted({int(x) for x in dist[dp].unique()})
        rows = [("n-gram baseline", [ngram_probs(ngram_fit(w, 1, np.random.default_rng(101)), 1, inp)])]
        for lab, name, arch, nl, nh in ARMS:
            Ps = []
            for pt in sorted(glob.glob(f"{RUNS}/{name}_s*/{name}.pt")):
                m = build(arch, 5, nl, nh, 2, 32).cuda().eval()
                m.load_state_dict(torch.load(pt, map_location="cuda"))
                with torch.no_grad():
                    Ps.append(torch.cat([m(inp[i:i + 128].cuda()).float().softmax(-1).cpu()
                                         for i in range(0, N, 128)]))
            rows.append((lab, Ps))
        for lab, Ps in rows:
            hew, sm_pos, sm_seq = [], [], []
            for P in Ps:
                pc = P.gather(-1, corr.unsqueeze(-1)).squeeze(-1)
                pw = P.gather(-1, wrg.unsqueeze(-1)).squeeze(-1)
                ratio = pc / (pc + pw).clamp_min(1e-12)              # renormalised over closers
                pj = [float(ratio[dp & (dist == j)].mean()) for j in js
                      if bool((dp & (dist == j)).any())]
                hew.append(float(np.mean(pj)))                        # mean over DISTANCES
                big = torch.where(valid, P, torch.zeros_like(P)).masked_fill(~valid, 1.0)
                ok = (torch.where(valid, P, torch.full_like(P, 1.0)).min(-1).values >
                      torch.where(~valid, P, torch.zeros_like(P)).max(-1).values)
                sm_pos.append(float(ok.double().mean())); sm_seq.append(float(ok.all(-1).double().mean()))
            out[f"{lab}|L{L}D{D}"] = dict(hewitt=float(np.mean(hew)), hewitt_sd=float(np.std(hew, ddof=1)) if len(hew) > 1 else 0.0,
                                          setmatch_pos=float(np.mean(sm_pos)), setmatch_seq=float(np.mean(sm_seq)), n=len(Ps))
            print(f"L{L} D{D} {lab:<20} Hewitt mean_j p_j {np.mean(hew):.3f}   set-match per step "
                  f"{np.mean(sm_pos):.3f}   per sequence {np.mean(sm_seq):.3f}", flush=True)
    json.dump(out, open("/home/prashr/mapformer/DYCK_STANDARD_METRICS.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""What metric does the decay envelope actually decay in? (DYCK_DECAY_RESULTS.md)

Committed after an audit noted the 0.862 / 0.238 correlation had no script behind it. Correlates the
envelope's distance |S_t - S_s| against token distance and against stack-depth difference, on the
trained MapPoPE_decay Dyck models. NOTE this is an association on one task, not the isolating test:
the crossed arms (path integration decayed over |t-s|, index decayed over a state distance) are not run.
Run from /home/prashr: python3 -m mapformer.probe_dyck_metric
"""
import glob
import numpy as np, torch

from mapformer.environment_dyck import DyckWorld, OPEN_P, OPEN_B
from mapformer.train_dyck import build


def main(n=16, L=128, D=12, dev="cuda:0", runs_dir="/home/prashr/mapformer/runs/dyck_decay"):
    w = DyckWorld(); inp, tgt, valid, _ = w.batch(n, L, D, np.random.default_rng(5))
    dep = np.zeros(inp.shape, dtype=int)
    for i in range(n):
        d = 0
        for t in range(L):
            dep[i, t] = d; d += 1 if tgt[i, t].item() in (OPEN_P, OPEN_B) else -1
    rows = []
    for pt in sorted(glob.glob(runs_dir + "/MapPoPE_decay-1L_r2_s*/*.pt")):
        m = build("MapPoPE_decay", 5, 1, 1, 2, 32).to(dev).eval()
        m.load_state_dict(torch.load(pt, map_location=dev, weights_only=False))
        with torch.no_grad():
            S = m.action_to_lie(m.token_emb(inp.to(dev))).mean(-1).cumsum(1).transpose(1, 2)[:, 0].cpu().numpy()
        ct, cd = [], []
        for i in range(n):
            t = np.arange(L); k = np.tril_indices(L, -1)
            Dm = np.abs(S[i][:, None] - S[i][None, :])
            ct.append(np.corrcoef(Dm[k], np.abs(t[:, None] - t[None, :])[k])[0, 1])
            cd.append(np.corrcoef(Dm[k], np.abs(dep[i][:, None] - dep[i][None, :])[k])[0, 1])
        rows.append((np.mean(ct), np.mean(cd)))
    A = np.array(rows)
    print(f"seeds {len(A)}: r(|S_t-S_s|, token distance) = {A[:,0].mean():.3f} +/- {A[:,0].std(ddof=1):.3f}")
    print(f"           r(|S_t-S_s|, depth difference) = {A[:,1].mean():.3f} +/- {A[:,1].std(ddof=1):.3f}")


if __name__ == "__main__":
    main()

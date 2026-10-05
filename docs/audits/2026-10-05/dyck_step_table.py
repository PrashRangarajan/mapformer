"""Dyck-2 step table (post hoc, CPU): what does the path integrator's step do for each bracket? Trained 1-layer
MapWM / MapPoPE at matched depth (runs/dyck_mdepth/T12, L32 D12, rank 2, 8 seeds). step(t) = omega * W_out W_in emb(t).
Readouts per seed: |step| of each token relative to the mean opener step; cos('(' , '[') and cos(')' , ']') (same move,
different 'what'); opposition || s(open) + s(close) || / mean norm for matching pairs (0 = push and pop cancel); BOS."""
import numpy as np, torch
from mapformer.train_dyck import build
names = ["(", ")", "[", "]", "BOS"]
for arch in ("MapWM", "MapPoPE"):
    rows = []
    for s in range(8):
        nm = f"{arch}-1L_r2_tL32D12"
        m = build(arch, 5, 1, 2, 2, 32); m.load_state_dict(torch.load(f"/home/prashr/mapformer/runs/dyck_mdepth/T12/{nm}_s{s}/{nm}.pt", map_location="cpu")); m.eval()
        with torch.no_grad():
            st = (m.action_to_lie(m.token_emb.weight[None])[0] * m.path_integrator.omega).reshape(5, -1).numpy()
        n = np.linalg.norm(st, axis=1); cos = lambda a, b: st[a] @ st[b] / (n[a] * n[b])
        opp = lambda a, b: np.linalg.norm(st[a] + st[b]) / ((n[a] + n[b]) / 2)
        rows.append([n[1] / n[[0, 2]].mean(), n[3] / n[[0, 2]].mean(), cos(0, 2), cos(1, 3), opp(0, 1), opp(2, 3), opp(0, 3), n[4] / n[[0, 2]].mean()])
    r = np.array(rows)
    print(f"{arch:8s} (8 seeds, median [min, max]):")
    for j, lab in enumerate(["|step ')'| / opener", "|step ']'| / opener", "cos( '(' , '[' )  same push", "cos( ')' , ']' )  same pop",
                             "opposition ( vs )", "opposition [ vs ]", "opposition ( vs ]  (mismatched pair)", "|step BOS| / opener"]):
        print(f"    {lab:40s} {np.median(r[:, j]):+.3f}  [{r[:, j].min():+.3f}, {r[:, j].max():+.3f}]")

"""Analyse a Dyck-2 batch against DYCK_PREREG.md. Run from /home/prashr:
python3 -m mapformer.analyze_dyck --runs-dir <dir> --out <md>"""
import argparse, glob, json, math
import numpy as np
import torch

REPO = "/home/prashr/mapformer"
LS, DS = [32, 64, 96, 128], [4, 6, 8, 12]
PAPER = {  # Fig 6, rows D4..D12, columns L32..L128
    "RoPE-1L": [[.91, .84, .80, .78], [.89, .84, .80, .78], [.88, .83, .79, .78], [.86, .81, .78, .77]],
    "RoPE-2L": [[.97, .70, .63, .59], [.89, .66, .60, .56], [.79, .61, .56, .53], [.66, .54, .52, .50]],
    "MapWM-1L_r2": [[1.0, .99, .98, .97], [.99, .97, .96, .96], [.98, .96, .95, .95], [.98, .94, .94, .94]],
    "MapEM-1L_r2": [[1.0, 1.0, .99, .99], [1.0, .98, .98, .97], [.99, .97, .96, .96], [.98, .95, .95, .95]],
}
FIG3A = {"CoPE-1L": [.92, .86, .85, .84], "CoPE-2L": [.96, .82, .79, .78]}   # D4 row
FIG3B = {"CoPE-1L": [.92, .93, .93, .93], "CoPE-2L": [.96, .90, .85, .82]}   # L32 column
ORDER = ["MapWM-1L_r2", "MapEM-1L_r2", "RoPE-1L", "RoPE-2L", "CoPE-1L", "CoPE-2L",
         "MapWM-1L_r4", "MapEM-1L_r4"]


def load(runs):
    R = {}
    for f in sorted(glob.glob(f"{runs}/*_s*/*.json")):
        j = json.load(open(f)); R.setdefault(j["name"], {})[j["seed"]] = j
    return R


def cell(R, arm, c):
    return np.array([R[arm][s]["grid"][c]["F1"] for s in sorted(R[arm])])


def contrast(R, a, b, c):
    seeds = sorted(set(R[a]) & set(R[b]))
    d = np.array([R[a][s]["grid"][c]["F1"] - R[b][s]["grid"][c]["F1"] for s in seeds])
    sd = d.std(ddof=1) if len(d) > 1 else float("nan")
    mde = 2.8 * sd / math.sqrt(len(d))
    return d.mean(), sd, mde, int((d > 0).sum()), len(d)


def mechanism(runs, arm):
    """Fig 3f: cosines of W_in applied to bracket embeddings (MapWM/MapEM)."""
    rows = []
    for pt in sorted(glob.glob(f"{runs}/{arm}_s*/{arm}.pt")):
        sd = torch.load(pt, map_location="cpu")
        E = sd["token_emb.weight"]; W = sd["action_to_lie.w_in.weight"]
        v = E @ W.T                                    # (vocab, r)
        cs = lambda i, j: float(torch.nn.functional.cosine_similarity(v[i], v[j], dim=0))
        rows.append((cs(0, 1), cs(2, 3), abs(cs(0, 2)), float(v[:4].norm(dim=1).mean()),
                     float(v[4].norm())))
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    R = load(a.runs_dir)
    G = json.load(open(f"{REPO}/DYCK_GATES.json"))["grid"]
    floor = {c: max(G[c][f"G4_ngram{k}"] for k in range(1, 7)) for c in G}
    L = []
    w = L.append
    w(f"# Dyck-2 replication -- `{a.runs_dir}`\n")
    w("Pre-registration: `DYCK_PREREG.md`. F1 = mean per-prefix valid-continuation F1 "
      "(Goodale et al.), 1024 sequences per cell. Floor = best n-gram (orders 1-6) from `DYCK_GATES.md`.\n")
    arms = [x for x in ORDER if x in R]
    w("## Seeds, convergence (rule 10)\n")
    w("| arm | params | seeds | final CE (floor 0.800) | slope /1k steps, median seed | max slope |")
    w("|---|---|---|---|---|---|")
    for arm in arms:
        js = list(R[arm].values())
        sl = np.array([j["final_slope_per_1k"] for j in js])
        w(f"| {arm} | {js[0]['params']:,} | {len(js)} | {np.mean([j['final_loss'] for j in js]):.4f} "
          f"| {np.median(sl):+.4f} | {sl.min():+.4f} |")
    w("")
    key = ["L32_D4", "L128_D4", "L32_D12", "L128_D12"]
    w("## Headline cells (mean +/- sd over seeds; paper value in brackets)\n")
    w("| arm | " + " | ".join(key) + " |"); w("|---|" + "---|" * len(key))
    for arm in arms:
        cells = []
        for c in key:
            x = cell(R, arm, c); Li, Di = LS.index(int(c[1:c.index('_')])), DS.index(int(c.split('D')[1]))
            p = PAPER[arm][Di][Li] if arm in PAPER else None
            if p is None and arm in FIG3A and Di == 0: p = FIG3A[arm][Li]
            if p is None and arm in FIG3B and Li == 0: p = FIG3B[arm][Di]
            cells.append(f"{x.mean():.3f} +/- {x.std(ddof=1):.3f}" + (f" [{p:.2f}]" if p is not None else ""))
        w(f"| {arm} | " + " | ".join(cells) + " |")
    w("| n-gram floor | " + " | ".join(f"{floor[c]:.3f}" for c in key) + " |")
    w("| CE-optimal predictor | " + " | ".join(f"{G[c]['G3_sampler_exact']:.3f}" for c in key) + " |\n")
    w("## Full grids (ours / paper, rows D, columns L)\n")
    for arm in arms:
        w(f"**{arm}**\n"); w("| D \\ L | " + " | ".join(map(str, LS)) + " |"); w("|---|" + "---|" * 4)
        dev = []
        for Di, D in enumerate(DS):
            row = []
            for Li, Lx in enumerate(LS):
                m = cell(R, arm, f"L{Lx}_D{D}").mean()
                if arm in PAPER:
                    p = PAPER[arm][Di][Li]; dev.append(m - p); row.append(f"{m:.3f} / {p:.2f}")
                else:
                    row.append(f"{m:.3f}")
            w(f"| {D} | " + " | ".join(row) + " |")
        if dev:
            dev = np.array(dev)
            w(f"\nmax |ours - paper| = {np.abs(dev).max():.3f}; mean (ours - paper) = {dev.mean():+.3f}; "
              f"cells within 0.03: {(np.abs(dev) <= 0.03).sum()}/16\n")
        else:
            w("")
    V = []
    def m(arm, c): return cell(R, arm, c).mean() if arm in R else float("nan")
    w("## Registered verdicts\n")
    for arm, thr1, thr2 in [("MapWM-1L_r2", 0.97, 0.91), ("MapEM-1L_r2", 0.97, 0.92)]:
        if arm in R:
            w(f"- **R1 {arm}** L32 D4 = {m(arm,'L32_D4'):.3f} (needs >= {thr1}) -> "
              f"**{'REPLICATES' if m(arm,'L32_D4') >= thr1 else 'DOES NOT REPLICATE'}**")
            w(f"- **R2 {arm}** L128 D12 = {m(arm,'L128_D12'):.3f} (needs >= {thr2}) -> "
              f"**{'REPLICATES' if m(arm,'L128_D12') >= thr2 else 'DOES NOT REPLICATE'}**")
    if "RoPE-2L" in R:
        ok = m("RoPE-2L", "L32_D4") >= 0.94 and m("RoPE-2L", "L128_D4") <= 0.70 and m("RoPE-2L", "L128_D12") <= 0.70
        w(f"- **R3 RoPE-2L** L32 D4 {m('RoPE-2L','L32_D4'):.3f} (>= 0.94), L128 D4 {m('RoPE-2L','L128_D4'):.3f} "
          f"and L128 D12 {m('RoPE-2L','L128_D12'):.3f} (both <= 0.70) -> **{'REPLICATES' if ok else 'DOES NOT REPLICATE'}**")
    if "RoPE-1L" in R:
        w(f"- **R4 RoPE-1L** L32 D4 {m('RoPE-1L','L32_D4'):.3f} (< 0.94) -> "
          f"**{'REPLICATES' if m('RoPE-1L','L32_D4') < 0.94 else 'DOES NOT REPLICATE'}**")
    for arm in ["CoPE-1L", "CoPE-2L"]:
        if arm in R:
            a1, a2 = m(arm, "L32_D4"), m(arm, "L128_D4")
            ok = abs(a1 - FIG3A[arm][0]) <= 0.03 and abs(a2 - FIG3A[arm][3]) <= 0.03
            w(f"- **R5 {arm}** (exploratory) L32 D4 {a1:.3f} [{FIG3A[arm][0]}], L128 D4 {a2:.3f} "
              f"[{FIG3A[arm][3]}] -> **{'WITHIN 0.03' if ok else 'OUTSIDE 0.03'}**")
    w("\n**R6 contrasts at L128 D12 (paired by seed)**\n")
    w("| contrast | paper | delta | sd | MDE | seeds + | verdict |"); w("|---|---|---|---|---|---|---|")
    for x, y, p in [("MapWM-1L_r2", "RoPE-2L", 0.44), ("MapWM-1L_r2", "RoPE-1L", 0.17),
                    ("MapEM-1L_r2", "RoPE-2L", 0.45), ("MapEM-1L_r2", "MapWM-1L_r2", 0.01)]:
        if x in R and y in R:
            d, sd, mde, pos, n = contrast(R, x, y, "L128_D12")
            w(f"| {x} - {y} | {p:+.2f} | {d:+.3f} | {sd:.3f} | {mde:.3f} | {pos}/{n} | "
              f"{'DETECTABLE' if abs(d) > mde else 'unmeasured'} |")
    w("\n**R7 floor reading at L128 D12** (best n-gram %.3f)\n" % floor["L128_D12"])
    for arm in arms:
        x = cell(R, arm, "L128_D12") - floor["L128_D12"]
        mde = 2.8 * x.std(ddof=1) / math.sqrt(len(x))
        w(f"- {arm}: {x.mean():+.3f} over the floor (MDE {mde:.3f}, {(x > 0).sum()}/{len(x)} seeds above) -> "
          f"{'ABOVE' if x.mean() > mde else ('BELOW' if x.mean() < -mde else 'at the floor')}")
    w("\n## Rule 9: r(final loss, F1) across all runs\n")
    fl = np.array([j["final_loss"] for arm in arms for j in R[arm].values()])
    for c in key:
        f1 = np.array([j["grid"][c]["F1"] for arm in arms for j in R[arm].values()])
        w(f"- {c}: r = {np.corrcoef(fl, f1)[0, 1]:+.3f}")
    w("\n## Mechanism (Fig 3f): cosines of W_in on bracket embeddings\n")
    w("| arm | cos('(' , ')') | cos('[' , ']') | abs cos('(' , '[') | mean norm brackets | norm BOS |")
    w("|---|---|---|---|---|---|")
    for arm in [x for x in arms if x.startswith("Map")]:
        M = mechanism(a.runs_dir, arm)
        if len(M):
            w(f"| {arm} | " + " | ".join(f"{M[:, i].mean():+.3f} +/- {M[:, i].std(ddof=1):.3f}" for i in range(5)) + " |")
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

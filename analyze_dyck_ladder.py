"""Fixed-width depth ladder, DYCK_LADDER_PREREG.md. n_heads=2 / d=128 throughout,
depth 1-4, 4 arms x 8 seeds, one batch. Primary: A2 at L32 D4 and L32 D12 (training
length -- no extrapolation confound). F1 and invalid mass alongside."""
import glob, json
import numpy as np, torch
from mapformer.eval_dyck_literature import cell_data, metrics, N
from mapformer.environment_dyck import DyckWorld
from mapformer.train_dyck import build

RUNS = "/home/prashr/mapformer/runs/dyck_ladder"
OUT = "/home/prashr/mapformer/DYCK_LADDER_RESULTS"
CELLS = [(32, 4), (32, 12), (128, 12)]
ARCH = {"RoPE": "RoPE", "PoPE": "PoPE", "MapWM": "MapWM_r2", "MapPoPE": "MapPoPE_r2"}


def main():
    w = DyckWorld()
    data = {c: cell_data(w, *c) for c in CELLS}
    R = {}
    for L in (1, 2, 3, 4):
        for arch in ARCH:
            nm = f"{arch}-{L}L" + ("_r2" if arch.startswith("Map") else "")
            pts = sorted(glob.glob(f"{RUNS}/{nm}_s*/{nm}.pt"))
            for pt in pts:
                m = build(arch, 5, L, 2, 2, 32).cuda().eval()
                m.load_state_dict(torch.load(pt, map_location="cuda"))
                for c in CELLS:
                    inp = data[c]["inp"]
                    with torch.no_grad():
                        lg = torch.cat([m(inp[i:i+128].cuda()).float().cpu() for i in range(0, N, 128)])
                    R.setdefault((arch, L, c), []).append(metrics(lg.softmax(-1), data[c], lg))
                del m; torch.cuda.empty_cache()
            print(f"  {nm}: {len(pts)} seeds", flush=True)
    json.dump({f"{a}|{L}L|L{c[0]}D{c[1]}": v for (a, L, c), v in R.items()},
              open(OUT + ".json", "w"), indent=1, default=float)

    def v(a, L, c, k="A2_close_acc_dist"):
        return np.array([r[k] for r in R[(a, L, c)]])
    mde = lambda d: 2.8 * d.std(ddof=1) / np.sqrt(len(d))
    lines = []
    def say(s=""): print(s); lines.append(s)
    say("# Dyck depth ladder at FIXED width\n")
    say("n_heads=2, d_model=128 throughout; only depth varies. 4 arms x 8 seeds x 4 depths, one batch.")
    say("Primary A2 (chance 0.500) at the TRAINING length.\n")
    for c in CELLS:
        say(f"## {'L%dD%d' % c}" + ("  (co-primary)" if c[0] == 32 else "  (extrapolation, reference only)"))
        say("\n| depth | RoPE | PoPE | MapWM | MapPoPE | POSITION main | ENCODING main | interaction |")
        say("|---|---|---|---|---|---|---|---|")
        for L in (1, 2, 3, 4):
            r, p, mw, mp = (v(a, L, c) for a in ("RoPE", "PoPE", "MapWM", "MapPoPE"))
            pos = ((mw - r) + (mp - p)) / 2; enc = ((p - r) + (mp - mw)) / 2; it = (mp - p) - (mw - r)
            fmt = lambda d: f"{d.mean():+.3f} ({mde(d):.3f}, {int((d>0).sum())}/8)" + ("**" if abs(d.mean()) > mde(d) else "")
            say(f"| {L}L | {r.mean():.3f} | {p.mean():.3f} | {mw.mean():.3f} | {mp.mean():.3f} | {fmt(pos)} | {fmt(enc)} | {fmt(it)} |")
        say("\n`**` = clears MDE. Cells show mean (MDE, seeds positive).\n")
    say("## Convergence\n")
    for L in (1, 2, 3, 4):
        sl = []
        for f in glob.glob(f"{RUNS}/*-{L}L*_s*/*.json"):
            sl.append(json.load(open(f))["final_slope_per_1k"])
        say(f"- {L}L: median final slope {np.median(sl):+.5f}/1k, worst {min(sl):+.5f} (void if median < -0.005)")
    open(OUT + ".md", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

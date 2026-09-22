"""E1 (depth-matched 2x2) and E2 (frequency ladder). DYCK_DEPTH_PREREG.md.

Primary readout is A2 (Hewitt distance-averaged closing accuracy), NOT F1: F1
inflates the position effect 2.3x and this project's rules forbid it as a
headline. F1 and invalid mass are reported alongside, never selectively.
"""
import glob, json, os
import numpy as np, torch

from mapformer.eval_dyck_literature import cell_data, metrics, CELLS, N
from mapformer.environment_dyck import DyckWorld
from mapformer.train_dyck import build

RUNS = "/home/prashr/mapformer/runs/dyck_depth"
OLD = "/home/prashr/mapformer/runs/dyck_bs128"
OUT = "/home/prashr/mapformer/DYCK_DEPTH_RESULTS.md"
KEYS = ["A2_close_acc_dist", "A3_d33+", "A3_d9-32", "C1_paper_f1", "invalid_mass"]

# label, dir-name, arch, n_layers, n_heads, rope_base, runs-dir
E1 = [("MapWM-2L", "MapWM-2L_r2", "MapWM", 2, 2, None, RUNS),
      ("MapPoPE-2L", "MapPoPE-2L_r2", "MapPoPE", 2, 2, None, RUNS),
      ("RoPE-2L", "RoPE-2L", "RoPE", 2, 2, None, RUNS),
      ("PoPE-2L", "PoPE-2L", "PoPE", 2, 2, None, RUNS)]
E1_OLD = [("MapWM-1L", "MapWM-1L_r2", "MapWM", 1, 1, None, OLD),
          ("MapPoPE-1L", "MapPoPE-1L_r2", "MapPoPE", 1, 1, None, OLD),
          ("RoPE-1L", "RoPE-1L", "RoPE", 1, 1, None, OLD),
          ("PoPE-1L", "PoPE-1L", "PoPE", 1, 1, None, OLD)]
E2 = [(f"{a}-1L_b{b}", f"{a}-1L_b{b}", a, 1, 1, float(b), RUNS)
      for b in (32, 128, 10000) for a in ("RoPE", "PoPE")]


def score(arms, data):
    R = {}
    for lab, name, arch, nl, nh, rb, rd in arms:
        pts = sorted(glob.glob(f"{rd}/{name}_s*/{name}.pt"))
        for pt in pts:
            m = build(arch, 5, nl, nh, 2, 32, rope_base=rb).cuda().eval()
            m.load_state_dict(torch.load(pt, map_location="cuda"))
            for c in CELLS:
                inp = data[c]["inp"]
                with torch.no_grad():
                    lg = torch.cat([m(inp[i:i+128].cuda()).float().cpu() for i in range(0, N, 128)])
                R.setdefault((lab, c), []).append(metrics(lg.softmax(-1), data[c], lg))
            del m; torch.cuda.empty_cache()
        print(f"  {lab}: {len(pts)} seeds", flush=True)
    return R


def mde(d):
    d = np.asarray(d, float)
    return 2.8 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")


def main():
    w = DyckWorld()
    data = {c: cell_data(w, *c) for c in CELLS}
    print("scoring E1 (new 2L batch)"); R = score(E1, data)
    print("scoring E1 reference (stored 1L)"); R.update(score(E1_OLD, data))
    print("scoring E2 (ladder)"); R.update(score(E2, data))
    json.dump({f"{l}|L{c[0]}D{c[1]}": v for (l, c), v in R.items()},
              open(OUT.replace(".md", ".json"), "w"), indent=1, default=float)

    L, cell = [], (128, 12)
    def say(s=""): print(s); L.append(s)
    def arr(lab, key): return np.array([r[key] for r in R[(lab, cell)]])

    say("# Dyck: depth-matched 2x2 (E1) and the frequency ladder (E2)\n")
    say("Pre-registration `DYCK_DEPTH_PREREG.md`. Cell L128 D12. **Primary readout "
        "A2 = Hewitt distance-averaged closing accuracy, chance 0.500.** F1 and "
        "invalid mass alongside, never selectively.\n")

    say("## Levels\n")
    say("| arm | n | A2 (primary) | A3 d33+ | A3 d9-32 | F1 | invalid mass |")
    say("|" + "---|" * 7)
    for lab in [a[0] for a in E1_OLD] + [a[0] for a in E1] + [a[0] for a in E2]:
        if (lab, cell) not in R: continue
        n = len(R[(lab, cell)])
        say(f"| {lab} | {n} | " + " | ".join(
            f"{arr(lab,k).mean():.3f}" for k in KEYS) + " |")
    say("")

    def cross(tag, idx, path, key):
        ri, pi, rp, pp = idx
        d_pos = ((arr(rp,key)-arr(ri,key)) + (arr(pp,key)-arr(pi,key)))/2
        d_enc = ((arr(pi,key)-arr(ri,key)) + (arr(pp,key)-arr(rp,key)))/2
        d_int = (arr(pp,key)-arr(pi,key)) - (arr(rp,key)-arr(ri,key))
        say(f"**{tag}** ({key})")
        for nm, d in (("POSITION main", d_pos), ("ENCODING main", d_enc), ("INTERACTION", d_int)):
            m, M = d.mean(), mde(d)
            say(f"- {nm}: {m:+.4f} (MDE {M:.4f}, {int((d>0).sum())}/{len(d)} positive) "
                + ("**DETECTABLE**" if abs(m) > M else "unmeasured"))
        say("")

    say("## E1 -- does the position effect survive at 2 layers?\n")
    for key in ("A2_close_acc_dist", "A3_d33+", "C1_paper_f1"):
        cross("1 LAYER (stored)", ("RoPE-1L","PoPE-1L","MapWM-1L","MapPoPE-1L"), None, key)
        cross("2 LAYERS (new)",   ("RoPE-2L","PoPE-2L","MapWM-2L","MapPoPE-2L"), None, key)

    say("## E2 -- does giving the index arms the path arms' frequency ladder close the gap?\n")
    key = "A2_close_acc_dist"
    base_gap_r = arr("MapWM-1L",key).mean() - arr("RoPE-1L_b10000",key).mean()
    base_gap_p = arr("MapPoPE-1L",key).mean() - arr("PoPE-1L_b10000",key).mean()
    say(f"Fresh base-10000 controls: RoPE {arr('RoPE-1L_b10000',key).mean():.3f} "
        f"(stored {arr('RoPE-1L',key).mean():.3f}), PoPE "
        f"{arr('PoPE-1L_b10000',key).mean():.3f} (stored {arr('PoPE-1L',key).mean():.3f})\n")
    say("| ladder | RoPE-1L | closes | PoPE-1L | closes |")
    say("|" + "---|" * 5)
    for b in (10000, 32, 128):
        r, p = arr(f"RoPE-1L_b{b}",key).mean(), arr(f"PoPE-1L_b{b}",key).mean()
        cr = (r - arr("RoPE-1L_b10000",key).mean())/base_gap_r*100 if base_gap_r else float('nan')
        cp = (p - arr("PoPE-1L_b10000",key).mean())/base_gap_p*100 if base_gap_p else float('nan')
        say(f"| base {b} | {r:.3f} | {cr:+.0f}% | {p:.3f} | {cp:+.0f}% |")
    say(f"\nGap to close: MapWM-1L - RoPE-1L(b10000) = {base_gap_r:+.3f}; "
        f"MapPoPE-1L - PoPE-1L(b10000) = {base_gap_p:+.3f}")
    say("\n**F3 fires if either ladder closes >= 50% of its gap.**\n")
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

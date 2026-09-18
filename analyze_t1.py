"""T1: does a centred (random-walk) accumulator rescue MapPoPE? See T1_PREREG.md."""
import argparse, glob, json, math
import numpy as np, torch

from mapformer.environment_jsb import splits, VOCAB_SIZE
from mapformer.train_jsb import build
from mapformer.model_centered import center_model, token_frequencies

REPO = "/home/prashr/mapformer"
BUCK = ["0-512", "512-1024", "1024-2048"]
PAIRS = [("MapPoPE_r2", "runs/jsb_len512", "MapPoPE_r2_centered", "runs/jsb_centered"),
         ("MapWM_r2", "runs/jsb_len512", "MapWM_r2_centered", "runs/jsb_centered")]


def load(name, d):
    out = {}
    for f in sorted(glob.glob(f"{REPO}/{d}/{name}_s*/{name}.json")):
        j = json.load(open(f))
        if j.get("train_len") == 512 and j.get("base", 2048) == 2048 and j.get("rank", 2) == 2:
            out[j["seed"]] = j
    return out


def alpha_and_range(name, arch, d, centered, dev):
    """Manipulation check: growth exponent of the accumulator on long test pieces."""
    S = splits(); Xte, Mte = S["test"]; Xtr, Mtr = S["train"]
    X = Xte[Mte.sum(1) >= 2000][:8].to(dev)
    out = []
    for pt in sorted(glob.glob(f"{REPO}/{d}/{name}_s*/{name}.pt"))[:3]:
        m = build(arch, 256, 8, 6, 2, 2048, 0.2, "uniform", torch.Generator())
        if centered:
            m = center_model(m, token_frequencies(Xtr, Mtr, VOCAB_SIZE))
        m = m.to(dev).eval(); m.load_state_dict(torch.load(pt, map_location=dev))
        with torch.no_grad():
            Sc = m.action_to_lie(m.token_emb(X)).cumsum(1)
        rng = lambda T: float((Sc[:, :T].max(1).values - Sc[:, :T].min(1).values).mean())
        Ts = [64, 128, 256, 512, 1024, 2048]
        out.append((float(np.polyfit(np.log(Ts), np.log([max(rng(T), 1e-9) for T in Ts]), 1)[0]),
                    rng(512), rng(2048)))
    A = np.array(out)
    return A[:, 0].mean(), A[:, 1].mean(), A[:, 2].mean()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    dev = torch.device("cuda:0")
    L = ["# T1: does bounding the accumulator rescue MapPoPE?", "",
         "Pre-registration: `T1_PREREG.md`. Centring makes E[Delta] = 0, so the accumulator is a "
         "mean-zero random walk instead of a clock. Test NLL by position bucket, **lower is "
         "better**, 5 seeds; baselines are the uncentred runs in `runs/jsb_len512`.", "",
         "## M1 manipulation check (3 seeds per arm)", "",
         "| arm | alpha | range(S) at 512 | at 2048 | growth |", "|---|---|---|---|---|"]
    for base_name, bd, cen_name, cd in PAIRS:
        arch = "MapPoPE" if "PoPE" in base_name else "MapWM"
        for nm, d, c in ((base_name, bd, False), (cen_name, cd, True)):
            al, r5, r20 = alpha_and_range(nm, arch, d, c, dev)
            L.append(f"| {nm} | {al:.3f} | {r5:.1f} | {r20:.1f} | {r20/max(r5,1e-9):.2f}x |")
    L += ["", "## Test NLL by bucket", "", "| arm | " + " | ".join(BUCK) + " | seeds |",
          "|---|---|---|---|---|"]
    R = {}
    for base_name, bd, cen_name, cd in PAIRS:
        for nm, d in ((base_name, bd), (cen_name, cd)):
            R[nm] = load(nm, d)
            v = R[nm]
            if not v:
                continue
            L.append(f"| {nm} | " + " | ".join(
                f"{np.mean([v[s]['test_buckets'][k] for s in v]):.4f} +/- "
                f"{np.std([v[s]['test_buckets'][k] for s in v], ddof=1):.4f}" for k in BUCK) +
                f" | {len(v)} |")
    L += ["", "## Registered contrasts (centred - uncentred, negative = better)", "",
          "| contrast | " + " | ".join(BUCK) + " |", "|---|---|---|---|"]
    for base_name, _, cen_name, _ in PAIRS:
        if not (R.get(base_name) and R.get(cen_name)):
            continue
        ss = sorted(set(R[base_name]) & set(R[cen_name]))
        cells = []
        for k in BUCK:
            d = np.array([R[cen_name][s]["test_buckets"][k] - R[base_name][s]["test_buckets"][k] for s in ss])
            mde = 2.8 * d.std(ddof=1) / math.sqrt(len(d))
            cells.append(f"{d.mean():+.4f} (MDE {mde:.4f}, {int((d<0).sum())}/{len(d)})" +
                         (" **DET**" if abs(d.mean()) > mde else ""))
        L.append(f"| {cen_name} - {base_name} | " + " | ".join(cells) + " |")
    if R.get("MapPoPE_r2_centered") and R.get("MapWM_r2"):
        c = np.mean([v["test_buckets"]["1024-2048"] for v in R["MapPoPE_r2_centered"].values()])
        w = np.mean([v["test_buckets"]["1024-2048"] for v in R["MapWM_r2"].values()])
        L += ["", f"**P4**: centred MapPoPE at 1024-2048 = {c:.4f} against uncentred MapWM's "
              f"{w:.4f} -> {'BEATS it' if c < w else 'still behind'}"]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

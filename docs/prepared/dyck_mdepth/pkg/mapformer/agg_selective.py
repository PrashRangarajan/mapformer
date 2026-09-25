"""Selective RoPE's angle generator vs MapFormer's, attributed knob by knob.

Both mechanisms drive the rotation phase from a content-dependent cumsum. This
reports where each of Selective RoPE's three additions -- no rank bottleneck, a
causal conv over positions, a sigmoid gate -- helps or hurts, on an iterative task
(parity) and a navigation task (torus).

Read every contrast against the parameter column: SRoPEGen carries +8.2% over
MapFormer and that is not removable, since the missing bottleneck IS the design.
"""
import argparse, json, statistics as st
from pathlib import Path

ARMS = ["RoPE", "Vanilla", "ConvAngle", "NoBottleneck", "GateAngle", "SRoPEGen"]
DESC = {"RoPE": "index position (floor reference)",
        "Vanilla": "MapFormer: r=2, no conv, no gate",
        "ConvAngle": "+ causal conv only",
        "NoBottleneck": "+ full rank only",
        "GateAngle": "+ sigmoid gate only",
        "SRoPEGen": "Selective RoPE: all three"}
PARAMS = {"RoPE": 199042, "Vanilla": 199490, "ConvAngle": 199683,
          "NoBottleneck": 207363, "GateAngle": 207683, "SRoPEGen": 215875}


def stat(d):
    d = [x for x in d if x is not None]
    n = len(d)
    if n < 2:
        return dict(m=float("nan"), sd=float("nan"), mde=float("nan"), n=n, pos=0)
    sd = st.stdev(d)
    return dict(m=st.mean(d), sd=sd, mde=2.8 * sd / n ** 0.5, n=n,
                pos=sum(1 for x in d if x > 0))


def fmt(x):
    return (f"{x['m']:+.3f} (sd {x['sd']:.3f}, MDE {x['mde']:.3f}, "
            f"{x['pos']}/{x['n']})") if x["m"] == x["m"] else "—"


def vd(x):
    if x["m"] != x["m"]:
        return "NO DATA"
    return ("DETECTABLE POSITIVE" if x["m"] > x["mde"] else
            "DETECTABLE NEGATIVE" if x["m"] < -x["mde"] else "UNMEASURED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    par = {}
    for v in ARMS:
        for s in range(16):
            f = Path(a.runs_dir) / "parity" / f"{v}_s{s}" / f"{v}_parity.json"
            if f.exists():
                par.setdefault(v, {})[s] = json.load(open(f))["acc"]
    tor = {}
    j = Path(a.repo) / "_SELECTIVE_TORUS.json"
    if j.exists():
        for k, rows in json.load(open(j)).items():
            _p, v, T = k.split("|")
            tor[(v, int(T))] = {int(s): x for s, x, _n in rows if x is not None}

    o = ["# Selective RoPE's angle generator vs MapFormer's", "",
         "Both papers put a content-dependent cumulative sum in the rotation phase,",
         "posted three days apart with neither citing the other. This swaps the",
         "generator and changes nothing else, with one arm per difference so any",
         "effect can be attributed.", "",
         "**Not parameter-matched, and it cannot be** — removing the rank bottleneck",
         "IS the design difference. Read every contrast against the params column.", "",
         "| arm | what it adds | params | vs MapFormer |", "|---|---|---|---|"]
    for v in ARMS:
        d = PARAMS[v] - PARAMS["Vanilla"]
        o.append(f"| {v} | {DESC[v]} | {PARAMS[v]:,} | "
                 + (f"{d:+,}" if v != "Vanilla" else "—") + " |")
    o.append("")

    for L in (16, 128, 256):
        o += [f"## Parity, L={L}" + (" (training length)" if L == 16 else
                                     " (extrapolation)"), "",
              "| arm | accuracy | vs MapFormer | verdict |", "|---|---|---|---|"]
        for v in ARMS:
            xs = [par.get(v, {}).get(s, {}).get(str(L)) for s in range(16)]
            xs = [x for x in xs if x is not None]
            base = [par.get("Vanilla", {}).get(s, {}).get(str(L)) for s in range(16)]
            d = stat([x - y for x, y in zip(xs, base)
                      if x is not None and y is not None]) if v != "Vanilla" else None
            o.append(f"| {v} | "
                     + (f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f}" if len(xs) > 1 else "—")
                     + " | " + (fmt(d) if d else "—") + " | "
                     + (vd(d) if d else "baseline") + " |")
        o.append("")

    if tor:
        o += ["## Torus paper task", "",
              "| arm | T=128 | T=512 | T=1024 | vs MapFormer @T=512 |",
              "|---|---|---|---|---|"]
        for v in ARMS:
            cells = []
            for T in (128, 512, 1024):
                xs = list(tor.get((v, T), {}).values())
                cells.append(f"{st.mean(xs):.3f}" if xs else "—")
            A, B = tor.get((v, 512), {}), tor.get(("Vanilla", 512), {})
            ss = sorted(set(A) & set(B))
            d = stat([A[s] - B[s] for s in ss]) if v != "Vanilla" else None
            o.append(f"| {v} | " + " | ".join(cells) + " | "
                     + (fmt(d) if d else "baseline") + " |")
        o.append("")

    o += ["## Reading it", "",
          "**ConvAngle is the cleanest single knob** (+193 params, essentially free),",
          "and it is the one with a directional prediction: smoothing the increment",
          "over a local window blurs a displacement that is exact, so on the torus —",
          "where the cumsum is a literal path integral — the conv should HURT. On",
          "parity, where the increment is a learned clock rate with no geometric",
          "meaning, it has no reason to.", "",
          "**NoBottleneck and GateAngle each cost ~8k parameters (+4%)**, so a small",
          "win from either is not obviously a win for the mechanism.", "",
          "## Scope", "",
          "This swaps the GENERATOR and keeps MapFormer's PLACEMENT: the angle is",
          "computed once from token embeddings before the blocks, not per-head and",
          "per-layer from the query. At 1 layer the two are close, since the query is",
          "itself a learned linear map of the token. **A negative result here does",
          "not refute Selective RoPE** — it is evidence about the generator's",
          "components in this setting, on tasks their paper does not run."]
    Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o)); print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

"""Local-revisit accuracy by gap N (moves since the latest D = 0 visit) and wrap-only accuracy, per run, from m1_out.json."""
import json
d = json.load(open("/home/prashr/mapformer/docs/audits/2026-10-14/cope_diagnosis/m1_out.json"))
keys = ["local N[1,8)", "local N[8,32)", "local N[32,64)", "local N[64,129)"]
print("run | acc | class | " + " | ".join(keys) + " | wrap-only (n)")
for r in d:
    s = r["strata"]
    print(f"{r['batch'][:3]} {r['arm']:9s} s{r['seed']} | {r['acc']:.3f} | {r['cls']:4s} | "
          + " | ".join(f"{s[k][0]:.3f}" for k in keys) + f" | {s['wrap_only'][0]:.3f} ({s['wrap_only'][1]})")

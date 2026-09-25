"""Plumbing test for analyze_dyck_mdepth: a MOCK batch dir whose checkpoints are the ladder's
D4-trained ones (symlinked) and whose JSONs carry the content keys the analysis asserts."""
import json, os, sys
sys.path.insert(0, sys.argv[1])
from mapformer.analyze_dyck_mdepth import COND, PATH_ARMS, run_name, SEEDS, LADDER
M = sys.argv[2]
for cond, (sub, depths, idx_arms, suffix, want) in COND.items():
    for L in depths:
        for arm in idx_arms + PATH_ARMS:
            nm = run_name(arm, L, suffix); src = run_name(arm.replace("_b32", ""), L, "")
            for s in SEEDS:
                d = f"{M}/{sub}/{nm}_s{s}"; os.makedirs(d, exist_ok=True)
                if not os.path.exists(f"{d}/{nm}.pt"):
                    os.symlink(f"{LADDER}/{src}_s{s}/{src}.pt", f"{d}/{nm}.pt")
                js = json.load(open(f"{LADDER}/{src}_s{s}/{src}.json"))
                js.update(want); js["rope_base"] = 32.0 if arm.endswith("_b32") else None
                js["train_ce_floor"] = 0.52; js["name"] = nm
                json.dump(js, open(f"{d}/{nm}.json", "w"))
for nm in ("RoPE-4L", "MapWM-4L_r2"):
    d = f"{M}/repro/{nm}_s0"; os.makedirs(d, exist_ok=True)
    for ext in ("pt", "json"):
        if not os.path.exists(f"{d}/{nm}.{ext}"):
            os.symlink(f"{LADDER}/{nm}_s0/{nm}.{ext}", f"{d}/{nm}.{ext}")
print("mock ok")

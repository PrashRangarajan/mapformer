"""ckpt_guard.check_not_stale over every code checkpoint, a planted stale copy, and the
strict loads patch 09 introduced in probe_code_depth0 / probe_code_accum (CPU only)."""
import glob, os, shutil, sys, tempfile
import torch
sys.path.insert(0, "/home/prashr")
from mapformer.ckpt_guard import check_not_stale, CheckpointLayoutError
from mapformer.train_hourglass_enwik8 import build
from mapformer.probe_code_depth0 import ARMS
R = "/home/prashr/mapformer/runs"
cks = sorted(p for d in ("code", "code2048", "code_ablate", "code_decay")
             for p in glob.glob(f"{R}/{d}/*.best.pt") + glob.glob(f"{R}/{d}/*.final.pt"))
ok = warn = 0; bad = []
for ck in cks:
    try:
        check_not_stale(ck); ok += 1
    except CheckpointLayoutError as e:
        bad.append(str(e))
print(f"check_not_stale: {ok}/{len(cks)} code checkpoints pass; raised on {len(bad)}")
for b in bad: print("  ", b)
# the quarantined 2026-09-22 stale checkpoint, restored under its original name in a temp dir
st = f"{R}/code_ablate/PoPE-NoSigma_s2.best.pt.stale"
with tempfile.TemporaryDirectory() as t:
    shutil.copy(st, f"{t}/PoPE-NoSigma_s2.best.pt"); shutil.copy(f"{R}/code_ablate/PoPE-NoSigma_s2.json", t)
    try:
        check_not_stale(f"{t}/PoPE-NoSigma_s2.best.pt"); print("stale copy: NOT caught (FAIL)")
    except CheckpointLayoutError as e:
        print("stale copy: caught ->", str(e).replace(t, "<tmp>"))
    os.remove(f"{t}/PoPE-NoSigma_s2.json")
    check_not_stale(f"{t}/PoPE-NoSigma_s2.best.pt")   # JSON missing: must warn, not pass silently
# strict loads, as the two probes now do them
n = 0
for arm in ARMS:
    for seed in (0, 1, 2):
        ck = f"{R}/code/{arm}_s{seed}.best.pt"
        if not os.path.exists(ck):
            print("MISSING", ck); continue
        blob = torch.load(ck, map_location="cpu", weights_only=False); c = blob["cfg"]
        for gs in (2048, 1024):
            m = build(c["model"], shorten=c["shorten"], dim=c["dim"], heads=c["heads"],
                      n_layers=c["n_layers"], grid_size=gs, bottleneck_r=c["bottleneck_r"])
            m.load_state_dict(blob["state_dict"]); n += 1          # strict
print(f"strict load_state_dict: {n}/{len(ARMS) * 3 * 2} (arm, seed, grid_size) loads succeeded")

"""Amendment-1 secondaries for LEAK_PREREG.md (declared before any result was read). Reads LEAK.json (the registered
per-run readouts) and the checkpoints."""
import json
import numpy as np
import torch

from mapformer.leak_eval import load, sequences, Probe, obj_acc
from mapformer.environment_newobj import N_SPECIAL, BLANK
from mapformer.model_codes import set_pool
from mapformer.stats_core import perm2_p, mde

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/leak/p0"; S = list(range(8)); ARMS = ("MapWM", "ActOnly", "NormStep")


@torch.no_grad()
def step_gains(m, toks, dev):
    tok = toks[:10].to(dev); x = m.token_emb(tok)
    d = (m.step(tok, x) if hasattr(m, "step") else m.action_to_lie(x)).float().flatten(2).pow(2).sum(-1)
    rms = lambda sel: float(d[sel].mean().sqrt()) if sel.any() else float("nan")
    a = rms(tok < BLANK)
    return rms(tok >= N_SPECIAL) / a, rms(tok == BLANK) / a


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    L = json.load(open(f"{REPO}/LEAK.json"))["res"]
    acc1 = {a: [r["test|x1"]["intact"] for r in L[a]] for a in ARMS}
    leak1 = {a: [r["test|x1"]["leak"] for r in L[a]] for a in ARMS}
    toks, revs = sequences("test")
    fl, gains, zob = {}, {}, []
    for a in ARMS:
        fl[a], gains[a] = [], []
        for s in S:
            m, b = load(f"{R}/{a}_s{s}/{a}.pt", a, dev); losses = b["losses"]
            fl[a].append(float(np.mean(losses[-max(1, len(losses) // 20):])))
            gains[a].append(step_gains(m, toks, dev))
            if a == "MapWM":                                   # (e) blank steps zeroed too, eval only
                p = Probe(m); set_pool(m, "test")
                orig = p._hook
                def hook(mod, inp, out, p=p):
                    if p.keep is None:
                        return out
                    return torch.where(p.keep[:, :, None, None], out, torch.zeros_like(out))
                p.keep = None
                ok = n = 0
                for i in range(0, len(toks), 20):
                    t = toks[i:i + 20].to(dev); r = revs[i:i + 20].to(dev); inp = t[:, :-1]
                    p.keep = inp < BLANK; lg = m(inp).float(); p.keep = None
                    tgt, msk = t[:, 1:], r[:, 1:] & (t[:, 1:] >= N_SPECIAL)
                    ok += int((((lg[..., 5 + 1000:5 + 2000].argmax(-1) + 1005) == tgt) & msk).sum()); n += int(msk.sum())
                zob.append(ok / n)
    print("== (d) final-5% loss per arm (mean, per seed) ==")
    for a in ARMS:
        print(f"  {a:8s} {np.mean(fl[a]):.4f}  " + " ".join(f"{x:.3f}" for x in fl[a]))
    x = sum((fl[a] for a in ARMS), []); y = sum((acc1[a] for a in ARMS), [])
    print(f"  r(final loss, x1 accuracy) over 24 runs: {np.corrcoef(x, y)[0, 1]:+.3f}")
    print("\n== (a) x1 contrast R - MapWM, with exact-t MDE (unpaired, pooled sd) ==")
    for a in ("ActOnly", "NormStep"):
        d = np.mean(acc1[a]) - np.mean(acc1["MapWM"]); p = perm2_p(acc1["MapWM"], acc1[a])["p"]
        sd = np.sqrt((np.var(acc1[a], ddof=1) + np.var(acc1["MapWM"], ddof=1)) / 2)
        print(f"  {a} - MapWM: {d:+.4f} (perm p {p:.4f}; MDE ~{mde(sd, 8) * np.sqrt(2):.4f}); expected if leak removed at no cost: "
              f"+{np.mean(leak1['MapWM']):.4f}")
    print("\n== (b) in-distribution leak L(x1): NormStep vs MapWM ==")
    for a in ARMS:
        print(f"  {a:8s} median {np.median(leak1[a]):+.4f} mean {np.mean(leak1[a]):+.4f}")
    print(f"  NormStep - MapWM: {np.mean(leak1['NormStep']) - np.mean(leak1['MapWM']):+.4f} perm p {perm2_p(leak1['MapWM'], leak1['NormStep'])['p']:.4f}")
    print("\n== (c) step gain: rms(object step) / rms(action step), blank / action ==")
    for a in ARMS:
        print(f"  {a:8s} object {np.mean([g[0] for g in gains[a]]):.4f}  blank {np.mean([g[1] for g in gains[a]]):.4f}")
    print(f"\n== (e) MapWM eval-only, object AND blank steps zeroed: {np.mean(zob):.4f} (vs object-only zeroed "
          f"{np.mean([r['test|x1']['zobj'] for r in L['MapWM']]):.4f}, intact {np.mean(acc1['MapWM']):.4f})")


if __name__ == "__main__":
    main()

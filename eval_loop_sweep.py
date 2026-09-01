"""Evaluate looped models across BOTH sequence length and eval-time loop count.

The finding this tests: a model trained at a fixed 4 passes is best at 4 passes on
its training length and best at 2 on a 4x longer one (0.794 vs 0.776, same
weights, eval-only change). Every pass past the second hurts out of distribution.
That makes the fixed count a length-specific choice baked in at training time --
so the question is whether sampling the count during training removes the
trade-off, keeps peak performance, or costs it.

Reports the whole (length x loop-count) surface, not a single number, because the
claim is about the SHAPE of the curve.
"""
import argparse, json, os
import numpy as np, torch, torch.nn.functional as F
from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP


def corrupt(tokens, p, n_actions, gen):
    if p <= 0: return tokens
    even = torch.zeros_like(tokens, dtype=torch.bool); even[:, 0::2] = True
    hit = (torch.rand(tokens.shape, generator=gen, dtype=torch.float) < p) & even
    return torch.where(hit, torch.randint(0, n_actions, tokens.shape, generator=gen), tokens)


@torch.no_grad()
def ev(model, env, T, n_trials, p, dev, seed):
    gen = torch.Generator().manual_seed(seed); np.random.seed(seed)
    ok = tot = 0
    for _ in range(n_trials):
        tok, _om, rev = env.generate_trajectory(T)
        tok = corrupt(tok.unsqueeze(0), p, env.N_ACTIONS, gen).to(dev)
        lp = F.log_softmax(model(tok[:, :-1]).float(), -1)
        pr = lp.argmax(-1)[0]; tg = tok[0, 1:]; m = rev[1:].to(dev)
        if m.sum() == 0: continue
        ok += (pr[m] == tg[m]).sum().item(); tot += int(m.sum())
    return ok / tot if tot else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--noises", nargs="+", type=float, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4])
    ap.add_argument("--lengths", nargs="+", type=int, default=[128,512,1024])
    ap.add_argument("--loops", nargs="+", type=int, default=[1,2,3,4,6])
    ap.add_argument("--n-trials", type=int, default=80)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device); res = {}
    for p in a.noises:
        tag = f"p{p:g}".replace(".", "")
        for v in a.variants:
            for s in a.seeds:
                ck = os.path.join(a.runs_dir, tag, f"{v}_s{s}", f"{v}.pt")
                if not os.path.exists(ck): continue
                b = torch.load(ck, map_location="cpu", weights_only=False); c = b["config"]
                m = VARIANT_MAP[v](vocab_size=c["vocab_size"], d_model=c["d_model"],
                                   n_heads=c["n_heads"], n_layers=c["n_layers"],
                                   grid_size=c["grid_size"])
                m.load_state_dict(b["model_state_dict"]); m = m.to(dev).eval()
                env = GridWorld(size=c["grid_size"], n_obs_types=c.get("n_obs_types",16),
                                p_empty=c.get("p_empty",0.5), seed=10000)
                loops = a.loops if hasattr(m, "n_loops") else [0]
                for T in a.lengths:
                    for k in loops:
                        if k: m.n_loops = k
                        acc = ev(m, env, T, a.n_trials, p, dev, 1234+s)
                        res.setdefault(f"{p}|{v}|{T}|{k}", []).append(acc)
                    print(f"p={p:g} {v:14s} s{s} T={T:5d} done", flush=True)
                del m; torch.cuda.empty_cache()
    def mean(p,v,T,k):
        r=[x for x in res.get(f"{p}|{v}|{T}|{k}",[]) if x is not None]
        return (np.mean(r), np.std(r,ddof=1) if len(r)>1 else 0.0, len(r)) if r else (None,0,0)
    o=["# Can the loop's length trade-off be trained away?","",
       "A model trained at a FIXED 4 passes peaks at 4 passes on its training length and",
       "at 2 on a 4x longer one -- every pass past the second hurts out of distribution.",
       "`LoopedSampled` draws the count from {2..6} each training batch instead, so the",
       "count becomes a runtime knob the model has been trained across.","",
       "Torus paper task, held-out map, evaluated under the noise it trained on.",""]
    for p in a.noises:
        o += [f"## p_action_noise = {p:g}",""]
        for v in a.variants:
            has=any(f"{p}|{v}|{a.lengths[0]}|{k}" in res for k in a.loops)
            if not has:
                r=[]
                for T in a.lengths:
                    mu,sd,n=mean(p,v,T,0); r.append(f"{mu:.3f} ± {sd:.3f}" if mu else "—")
                o += [f"**{v}** (no loop): " + " | ".join(f"T={T}: {c}" for T,c in zip(a.lengths,r)), ""]
                continue
            o += [f"**{v}** — accuracy by (sequence length × loops at eval)","",
                  "| loops at eval | " + " | ".join(f"T={T}" for T in a.lengths) + " |",
                  "|---"*(len(a.lengths)+1)+"|"]
            for k in a.loops:
                cells=[]
                for T in a.lengths:
                    mu,sd,n=mean(p,v,T,k)
                    cells.append(f"{mu:.3f} ± {sd:.3f}" if mu is not None else "—")
                o.append(f"| {k} | " + " | ".join(cells) + " |")
            o.append("")
            best=[]
            for T in a.lengths:
                cand=[(mean(p,v,T,k)[0],k) for k in a.loops if mean(p,v,T,k)[0] is not None]
                if cand: mu,k=max(cand); best.append(f"T={T}: {k} loops ({mu:.3f})")
            o += ["best count per length — " + " · ".join(best), ""]
    open(a.out,"w").write("\n".join(o)+"\n")
    json.dump(res, open(a.out.replace(".md",".json"),"w"), indent=2)
    print("\n".join(o))

if __name__ == "__main__":
    main()

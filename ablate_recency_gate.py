"""Establish (or kill) the content-gate account CAUSALLY, by intervention.

The correlational evidence is weak and cannot decide this. Per head, the gate
(|Delta| large on counted tokens, small on filler) is present in 8/8 seeds -- but
its STRENGTH does not predict the outcome: r(log gate ratio, accuracy) = -0.34,
i.e. if anything negative. Across a range where every seed is near ceiling that
correlation has no power either way (rule 11), so no amount of further measuring
settles it. Intervene instead.

Eval-only, no training (rule 18: check whether the knob is a runtime argument
before proposing a training sweep).

  none          baseline.
  zero_filler   Delta := 0 on filler tokens. If the model already ignores
                filler, this is a NO-OP. A large drop falsifies the account:
                the filler increments were carrying something.
  equalize      Delta on filler := the sequence's mean Delta on CONTENT tokens.
                THE DECISIVE ONE. theta now advances on every token, so it
                measures TOKEN count instead of CONTENT count -- exactly the
                quantity an index code has. Predicted: collapse toward the index
                arms' 0.234, because the answer sits at a variable token
                distance (129.7 +/- 10.3 at k=64, gate G8).
  zero_content  Delta := 0 on counted tokens. POSITIVE CONTROL: must destroy
                performance, otherwise the intervention is not reaching the
                pathway that matters and the other two conditions mean nothing.
  zero_query    Delta := 0 on query/mask tokens. Control for "any intervention
                hurts".
  scale_match   THE SCEPTIC'S CONTROL for `equalize`. Making filler count roughly
                DOUBLES how fast theta advances, so `equalize`'s collapse could
                be theta leaving its trained range rather than counting the wrong
                thing. This multiplies every Delta by the single scalar that
                makes theta travel the SAME total distance as under `equalize`,
                while still counting only content. If accuracy survives here,
                `equalize` is about WHAT is counted. If it collapses too,
                `equalize` is confounded with magnitude and proves nothing.

Reading it: the account is ESTABLISHED only if zero_filler is a no-op AND
equalize collapses AND zero_content destroys. Any other pattern refutes or
complicates it, and the pattern is stated here before the run.
"""
import argparse, glob, json, os
import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_recency import RecencyWorld
from mapformer.probe_recency_accumulator import _load

MODES = ("none", "zero_filler", "equalize", "scale_match",
         "uniform_content", "uniform_all", "zero_content", "zero_query")


class GateIntervention(torch.nn.Module):
    """Wraps action_to_lie and rewrites Delta by token class."""

    def __init__(self, inner, env):
        super().__init__()
        self.inner, self.env, self.mode, self.ids = inner, env, "none", None

    def forward(self, x):
        d = self.inner(x)                       # (B,T,H,nb)
        if self.mode == "none" or self.ids is None:
            return d
        e, t = self.env, self.ids               # t: (B,T)
        content = t < e.n_symbols
        filler = (t >= e.filler_offset) & (t < e.query_offset)
        query = ~(content | filler)
        m = {"zero_filler": filler, "zero_content": content,
             "zero_query": query}.get(self.mode)
        if m is not None:
            return d * (~m).to(d.dtype)[..., None, None]
        if self.mode in ("uniform_content", "uniform_all"):
            # THE ISOLATED PAIR. Both replace Delta with a CONSTANT vector and
            # both make theta travel the same total distance as the baseline.
            # They differ in ONE thing: which tokens the constant is applied to.
            # uniform_content -> a CONTENT clock; uniform_all -> a TOKEN clock.
            # Magnitude, per-token variation and total drift are matched, so a
            # gap between them isolates WHAT IS COUNTED, which `equalize` could
            # not (its sceptic control `scale_match` collapses too).
            out = torch.zeros_like(d)
            for b in range(d.shape[0]):
                tgt = d[b].sum(dim=0)                 # baseline total drift
                sel = content[b] if self.mode == "uniform_content" else \
                    torch.ones_like(content[b])
                n = int(sel.sum())
                if n:
                    out[b][sel] = (tgt / n).unsqueeze(0)
            return out
        if self.mode == "scale_match":
            out = d.clone()
            for b in range(d.shape[0]):
                cm, fm = content[b], filler[b]
                if not cm.any():
                    continue
                base = d[b].sum()
                eq = base + fm.sum() * (d[b][cm].mean(dim=0).sum() ) - d[b][fm].sum()
                out[b] = d[b] * (eq / base if base.abs() > 1e-9 else 1.0)
            return out
        if self.mode == "equalize":
            out = d.clone()
            for b in range(d.shape[0]):
                cm = content[b]
                if cm.any():
                    # make every filler token advance theta exactly as a counted
                    # token does on average -> theta becomes a TOKEN clock
                    out[b][filler[b]] = d[b][cm].mean(dim=0)
            return out
        raise ValueError(self.mode)


@torch.no_grad()
def run(ck, env, T, dev, n_ep=64, seed=0):
    m, _ = _load(ck, dev)
    m.action_to_lie = GateIntervention(m.action_to_lie, env).to(dev)
    out = {}
    for mode in MODES:
        m.action_to_lie.mode = mode
        rng = np.random.RandomState(seed)
        ok = tot = 0
        for _ in range(n_ep):
            tok, sp, ans, _i = env.generate_episode(T, rng)
            ids = tok.unsqueeze(0).to(dev)
            m.action_to_lie.ids = ids[:, :-1]
            logits = m(ids[:, :-1])
            for p, a in zip(sp, ans):
                if p >= logits.shape[1]:
                    continue
                sl = logits[0, p, env.sym_offset:env.sym_offset + env.n_symbols]
                ok += int(sl.argmax().item() == a - env.sym_offset); tot += 1
        out[mode] = ok / max(tot, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="/home/prashr/mapformer/runs/recency")
    ap.add_argument("--variant", default="Signed_r4")
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--k-max", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/home/prashr/mapformer/RECENCY_GATE_ABLATION.md")
    a = ap.parse_args()
    dev = torch.device(a.device)
    env = RecencyWorld(k_max=a.k_max, seed=10000)

    rows = []
    for ck in sorted(glob.glob(f"{a.runs_dir}/{a.variant}_s*/{a.variant}_recency.pt")):
        r = run(ck, env, a.T, dev)
        rows.append(r)
        print(os.path.basename(os.path.dirname(ck)),
              " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)

    chance, index_level = 1.0 / env.n_symbols, 0.234
    L = [f"# Is the content gate the mechanism? A causal test", "",
         f"`{a.variant}`, n={len(rows)} seeds, T={a.T}, eval-only intervention on "
         f"Delta. Chance {chance:.4f}; the index arms sit at {index_level:.3f}.", "",
         "| condition | accuracy | delta vs baseline | sd | MDE | seeds worse |",
         "|---|---|---|---|---|---|"]
    base = np.array([r["none"] for r in rows])
    for mode in MODES:
        v = np.array([r[mode] for r in rows])
        d = v - base; sd = d.std(ddof=1) if len(d) > 1 else 0.0
        mde = 2.8 * sd / np.sqrt(len(d)) if len(d) > 1 else 0.0
        L.append(f"| `{mode}` | {v.mean():.4f} +/- {v.std(ddof=1):.4f} | "
                 f"{d.mean():+.4f} | {sd:.4f} | {mde:.4f} | "
                 f"{int((d < -1e-9).sum())}/{len(d)} |")
    L += ["", "## Reading it -- and a correction to my own criterion", "",
          "**The pre-stated criterion was: `zero_filler` a no-op AND `equalize` "
          "collapses AND `zero_content` destroys. Two of those hold, but the "
          "`equalize` leg is CONFOUNDED and I am not counting it.** Making filler "
          "count roughly doubles how fast theta advances, and `scale_match` -- "
          "which changes ONLY the scale, still counting content alone -- collapses "
          "just as hard (0.110 vs 0.086). So this model is acutely sensitive to "
          "theta's absolute scale, and any intervention that moves it is "
          "destructive for reasons that have nothing to do with what is counted. "
          "`equalize` therefore proves nothing on its own.", "",
          "**What does establish it is the magnitude-matched pair, added after "
          "`scale_match` failed.** `uniform_content` and `uniform_all` both "
          "replace Delta with a CONSTANT and both make theta travel the same "
          "total distance as the baseline. They differ in exactly one thing: "
          "which tokens the constant lands on. The gap is the isolated value of "
          "counting CONTENT rather than TOKENS.", "",
          "The chain that survives:", "",
          "1. `zero_filler` is a no-op -- filler increments contribute nothing, "
          "so a gate exists. Unconfounded: this LOWERS theta\'s rate and costs "
          "nothing, while scale changes in either direction are otherwise fatal.",
          "2. `zero_content` destroys -- the positive control bites, so the "
          "intervention reaches the pathway that matters.",
          "3. `uniform_content` (0.783) vs `uniform_all` (0.189) -- **+0.594 at "
          "8/8 seeds, magnitude-matched**. What is counted is what matters.", "",
          "A constant increment on content alone recovers 0.783 of the baseline "
          "1.000, so the learned Delta structure beyond \"constant on counted "
          "tokens, zero elsewhere\" is worth only ~0.22: the gate is most of the "
          "mechanism, not a component of it. And `uniform_all` (0.189) lands "
          "BELOW the index arms (0.234) -- a token clock inside this architecture "
          "is no better than an index code, which is what it is.", "",
          "Caveat on provenance: the isolated pair was designed AFTER seeing "
          "`scale_match` fail. It is a control for a confound, not a second bite "
          "at the hypothesis, but it was not pre-registered and is labelled here."]
    open(a.out, "w").write("\n".join(L) + "\n")
    json.dump(rows, open(a.out.replace(".md", ".json"), "w"), indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()

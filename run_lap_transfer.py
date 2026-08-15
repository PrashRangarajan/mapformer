"""Does learning the lap task DEGRADE the cognitive map?

The lap probe showed Vanilla solves the lap task (exact 1.000 at K=4) by breaking
faithful path integration: per-lap theta drift 3.86 rad on a circuit whose true
displacement is exactly zero, with observation tokens carrying 19% of an action's
displacement. If theta no longer returns to the same value at the same place, the
matching mechanism that Match-Query depends on should be measurably worse.

DESIGN. A lap-only model would fail Match-Query simply because it never saw the
task format -- a confound, not evidence. So this is SEQUENTIAL on ONE model with
a SHARED vocabulary (actions 0-3 | obs 4..43 | MASK 44 | REWARD 45):

  phase 1  train on Match-Query        -> measure MQ accuracy, measure theta drift
  phase 2  continue training on LAP    -> measure lap exact
  phase 3  re-measure MQ accuracy and theta drift on the SAME model

CONTROL (required, else phase-3 changes could just be more training on anything):
an arm that continues phase 2 on MORE MATCH-QUERY instead of lap, for the same
number of steps. If MQ drops only in the lap arm, the degradation is attributable
to the lap task rather than to continued training or LR schedule effects.

Prediction: lap arm shows MQ accuracy DOWN and theta drift UP; control arm shows
neither. Reported either way.

STATED CHOICE: constant LR throughout (no decay schedule). The Match-Query sweep
used linear decay over its 200 epochs, but a decay-to-zero schedule would leave
phase 2 with LR=0 and nothing would happen. Constant LR keeps phase 2 live at the
cost of phase 1 not exactly reproducing the sweep's 0.888. Phase-1 MQ accuracy is
reported so the baseline can be judged; if it is near the 0.0625 chance level the
comparison is void and must not be read.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_match_query import MatchQueryGridWorld
from mapformer.environment_lap import LapWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.train_match_query import _losses as mq_losses, evaluate as mq_eval
from mapformer.train_lap import _losses as lap_losses, evaluate as lap_eval

_REPO = Path(__file__).resolve().parent
VOCAB, MASK_TOK, REWARD_TOK = 46, 44, 45


@torch.no_grad()
def theta_drift(model, lap_env, device, n_ep=60, seed=1):
    """Per-lap |theta| drift and the observation/action |Delta| ratio.

    The circuit's true displacement is exactly zero, so faithful path integration
    gives drift ~0. Large drift means observation tokens are displacing.
    """
    rng = np.random.RandomState(seed)
    drift, obs_d, act_d = [], [], []
    for _ in range(n_ep):
        t, _dp, _dl, info = lap_env.generate_lap_episode(rng)
        L = info["loop_len"]
        x = model.token_emb(t.unsqueeze(0).to(device))
        delta = model.action_to_lie(x)
        cos_a, sin_a = model.path_integrator(delta)
        th = torch.atan2(sin_a, cos_a)[0, 0]
        drift.append(np.mean([(th[2 * (k * L)] - th[0]).abs().max().item()
                              for k in (1, 2, 3)]))
        d = delta[0].norm(dim=-1)
        act_d.append(d[0::2].mean().item()); obs_d.append(d[1::2].mean().item())
    return (float(np.mean(drift)), float(np.mean(obs_d)),
            float(np.mean(act_d)),
            float(np.mean(obs_d) / max(np.mean(act_d), 1e-9)))


def train_mq(model, env, opt, rng, dev, steps, TE, TQ, bs):
    for _ in range(steps):
        toks, rev, sps, ans, _i = env.generate_match_batch(bs, TE, TQ, rng)
        Lm, Lo = mq_losses(model, env, toks, rev, sps, ans, dev)
        loss = Lm + Lo
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def train_lap(model, env, opt, rng, dev, steps, bs):
    for _ in range(steps):
        toks, valid, dp, dl, _i = env.generate_lap_batch(bs, rng)
        Ln, Ld = lap_losses(model, env, toks, valid, dp, dl, dev)
        loss = Ln + Ld
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def run_arm(arm, seed, args, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    mq = MatchQueryGridWorld(n_obs_types=16, mask_tok=MASK_TOK, vocab_size=VOCAB,
                             seed=seed)
    mq_t = MatchQueryGridWorld(n_obs_types=16, mask_tok=MASK_TOK, vocab_size=VOCAB,
                               seed=10000)
    lp = LapWorld(n_obs_types=40, reward_tok=REWARD_TOK, vocab_size=VOCAB, seed=seed)
    lp_t = LapWorld(n_obs_types=40, reward_tok=REWARD_TOK, vocab_size=VOCAB, seed=10000)

    model = VARIANT_MAP["Vanilla"](vocab_size=VOCAB, d_model=128, n_heads=2,
                                   n_layers=3, grid_size=64).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    rng = np.random.RandomState(seed)

    # phase 1: Match-Query
    train_mq(model, mq, opt, rng, dev, args.mq_steps, args.T_explore, args.T_query,
             args.batch_size)
    model.eval()
    a0, n0, _ = mq_eval(model, mq_t, args.T_explore, args.T_query, 8, 8, dev, 7000 + seed)
    d0 = theta_drift(model, lp_t, dev)
    model.train()

    # phase 2: lap  (or MORE match-query, for the control arm)
    if arm == "lap":
        train_lap(model, lp, opt, rng, dev, args.phase2_steps, args.batch_size)
    else:
        train_mq(model, mq, opt, rng, dev, args.phase2_steps, args.T_explore,
                 args.T_query, args.batch_size)

    # phase 3: re-measure
    model.eval()
    a1, n1, _ = mq_eval(model, mq_t, args.T_explore, args.T_query, 8, 8, dev, 7000 + seed)
    d1 = theta_drift(model, lp_t, dev)
    lap_ex = lap_eval(model, lp_t, 8, 32, dev, 8000 + seed)[2] if arm == "lap" else None
    model.train()

    return dict(arm=arm, seed=seed,
                mq_before=a0, mq_after=a1, mq_delta=a1 - a0,
                mq_nll_before=n0, mq_nll_after=n1,
                drift_before=d0[0], drift_after=d1[0],
                obs_act_before=d0[3], obs_act_after=d1[3],
                lap_exact=lap_ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--mq-steps", type=int, default=3000)
    ap.add_argument("--phase2-steps", type=int, default=800)
    ap.add_argument("--T-explore", type=int, default=512)
    ap.add_argument("--T-query", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(_REPO / "LAP_TRANSFER.md"))
    args = ap.parse_args()
    dev = torch.device(args.device)

    rows = []
    for seed in args.seeds:
        for arm in ("lap", "control"):
            r = run_arm(arm, seed, args, dev)
            rows.append(r)
            print(f"[{arm:7s} s{seed}] MQ {r['mq_before']:.3f} -> {r['mq_after']:.3f} "
                  f"({r['mq_delta']:+.3f})   drift {r['drift_before']:.2f} -> "
                  f"{r['drift_after']:.2f}   obs/act {r['obs_act_before']:.3f} -> "
                  f"{r['obs_act_after']:.3f}"
                  + (f"   lap_exact={r['lap_exact']:.3f}" if r['lap_exact'] is not None else ""),
                  flush=True)

    def agg(arm, key):
        v = [r[key] for r in rows if r["arm"] == arm and r[key] is not None]
        return (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0) if v else (float("nan"), 0)

    lines = ["# Does learning the lap task degrade the cognitive map?", "",
             "One model, shared vocabulary. Phase 1 Match-Query, phase 2 either LAP or",
             "(control) MORE Match-Query for the same number of steps, phase 3 re-measure.",
             "Match-Query chance = 0.0625. Lap circuit has exactly zero net displacement,",
             "so faithful path integration gives theta drift ~0.", "",
             "| arm | MQ before | MQ after | delta | theta drift before | after | obs/act Delta before | after |",
             "|---|---|---|---|---|---|---|---|"]
    for arm in ("lap", "control"):
        f = lambda k: agg(arm, k)[0]
        lines.append(f"| **{arm}** | {f('mq_before'):.3f} | {f('mq_after'):.3f} | "
                     f"{f('mq_delta'):+.3f} | {f('drift_before'):.2f} | {f('drift_after'):.2f} | "
                     f"{f('obs_act_before'):.3f} | {f('obs_act_after'):.3f} |")
    le = agg("lap", "lap_exact")
    lines += ["", f"Lap-arm lap `exact` after phase 2: **{le[0]:.3f}** "
              f"(random-boundary floor 0.250).", "",
              "## Per seed", "",
              "| arm | seed | MQ before | MQ after | delta | drift before | drift after |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['seed']} | {r['mq_before']:.3f} | "
                     f"{r['mq_after']:.3f} | {r['mq_delta']:+.3f} | "
                     f"{r['drift_before']:.2f} | {r['drift_after']:.2f} |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    json.dump(rows, open(str(args.out).replace(".md", ".json"), "w"), indent=2, default=str)
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

# READING -- angle magnitude is refuted too, and my predicted direction was backwards

**B1 REFUTED, with the sign inverted.** I registered that a LARGER omega base (slower low-frequency
blocks, so an unbounded cumulative angle stays in its trained range longer) would reduce the
collapse. The opposite holds, monotonically, at 2-4x beyond the training context:

| base | MapPoPE | MapWM |
|---|---|---|
| 512 | **3.823** | **1.109** |
| 2048 | 4.616 | 1.397 |
| 8192 | 4.922 | 1.662 |
| 32768 | 5.220 | 1.865 |

base 512 - base 2048 is -0.793 for MapPoPE (5/5, detectable) and -0.288 for MapWM (5/5, detectable);
base 32768 is detectably WORSE on both. So spreading the frequency schedule wider -- adding very slow
channels -- hurts extrapolation here. That is the reverse of the RoPE long-context intuition, where
raising the base is the standard extrapolation fix, and it is consistent with this project's earlier
refutation of the critical-dimension account (`LOCALISATION.md`): when position is a content-driven
cumulative sum rather than the token index, arguments about low-frequency channels do not carry over.

**B2 fires against the account again.** The control row moves the same way at similar relative size,
so the base is a general extrapolation knob for path integration, not an explanation of anything
PoPE-specific.

**B3: the collapse is robust to every knob tried.** MapPoPE never beats MapWM out of distribution at
any base: +2.71 / +3.22 / +3.26 / +3.35, 0/5 seeds at every one. Two pre-registered mechanism
accounts are now refuted (rank in amendment 1, angle magnitude here). **The mechanism is
unidentified.** The remaining suspect -- PoPE carrying one phase per element rather than per pair --
cannot be reached by any hyperparameter in this codebase; it needs a pair-frequency PoPE variant,
which per B3 I am NOT building, because at this point that would be constructing an arm to keep a
story alive rather than testing a registered prediction.

**Usable finding, separate from the mechanism.** For length extrapolation on this task, smaller is
better on both knobs: MapWM at base 512 reaches 1.109 at 2-4x and MapWM at r=1 (base 2048) reaches
0.912, against 1.397 for the r=2 / base-2048 default and 2.059 for index RoPE. The two have not been
combined -- r=1 at base 512 is untested and is not claimed.

## Batch provenance (audit note, 2026-09-19)

The contrasts here pair by seed index ACROSS run directories built on different days, not within one
batch -- the repo's standing rule 3 asks for one batch. Mitigating: the arms share code (only new
classes were added between the runs; the data and evaluation paths are byte-identical), the data
stream is seeded identically, and the primary readings lean on a parameter-matched inert twin
trained INSIDE the new batch. Unmitigated for the decay arms, which have no same-batch baseline.

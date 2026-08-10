# Publication venues & timeline for the MapFormer work

Compiled 2026-08-10. Dates past this were verified via web search where noted;
re-confirm on the official sites before relying on them (conference dates shift
and some 2027 CFPs were not yet posted).

Scope of the work: path-integration transformers (MapFormer) + state-correction
mechanisms (InEKF/Kalman, predictive coding), the Hourglass hierarchy analysis,
and the compositional-motif experiments. Two natural framings:
- **ML / architecture:** when structured priors help (action-noise, landmarks,
  OOD length, calibration) and when they are redundant (hierarchy = efficiency
  not capability; PC↔Kalman duality; hierarchy vs MapFormer dissociation).
- **NeuroAI / comp-neuro:** cognitive maps, grid/place cells, hippocampal
  reproduction, CSCG/clone structure, SO(2)/Lie-group position codes.

## Realistic calendar from 2026-08-10

| Deadline | Days out | Venue | Realistic? | Format |
|---|---|---|---|---|
| **Aug 22, 2026** | ~12 | **NeurReps 2026** (ext-abstract / Findings) | Tight, doable | 4 pg, non-archival |
| ~late Aug 2026 | ~2–3 wk | Other NeurIPS 2026 workshops (UniReps, Memory/Associative-Memory, NeuroAI) | Yes | 4–8 pg, non-archival |
| **Sept 24, 2026** (abs Sept 19) | ~6 wk | **ICLR 2027 main** | Yes, with multi-seed + controls done | 9 pg, archival |
| ~mid-Oct 2026 (TBA) | ~2 mo | **Cosyne 2027** abstract (conf Mar 11–16, Montréal) | Yes, low effort | abstract |
| ~Oct 2026 (verify) | ~2 mo | AISTATS 2027 | Yes | archival |
| ~Jan–Feb 2027 | ~5–6 mo | ICML 2027 / UAI 2027 / RLC 2027 | Yes, ample runway | archival |
| anytime | — | **TMLR** | Yes, rolling, no SOTA bar | archival |

Already PAST as of Aug 10: CCN 2026, AAAI 2026, RLC 2026, Bernstein 2026,
ICML 2026 & ICLR 2026 workshops, NeurIPS 2026 *main* track (May 2026).

## Recommended plan (two tracks, non-conflicting)

Workshops are non-archival, so a NeurReps abstract does NOT block submitting the
fuller paper to ICLR.

1. **Now → Aug 22 — NeurReps extended abstract (4 pg).** Only thing with a
   ~12-day fuse. Build it around the **hierarchy-vs-MapFormer dissociation**
   (compact, self-contained, results largely in hand, plain-vs-MapFormer
   control makes it rigorous). Label single-seed clearly. 9-pg proceedings
   track is NOT realistic in 12 days; the 4-pg extended abstract is.
2. **Now → Sept 24 — ICLR 2027 (archival, the real paper).** Six weeks is
   enough to finish multi-seed + training-length control + coarse-contribution
   diagnostic on the other server and write up the fuller story (correction
   wins in noise/landmark/OOD + calibration, plus the honest hierarchy/MapFormer
   dissociation).
3. **~Oct — Cosyne 2027 abstract.** Cheap; puts the grid-cell/navigation angle
   in front of the neuro community for the March meeting.
4. **Fallback / archival home — TMLR, anytime.** Natural venue for the rigorous,
   negatives-heavy analysis (no novelty bar).

## Full venue landscape

### Workshops (near-term entry; mostly non-archival)
| Workshop | Venue / timing | Fit |
|---|---|---|
| NeurReps (Symmetry & Geometry) | NeurIPS 2026, **Aug 22** | ★★★ Lie groups, grid cells — bullseye |
| UniReps (Unifying Representations) | NeurIPS 2026, ~late Aug | ★★★ rep. comparison bio/artificial |
| Associative Memory / Memory in ML | NeurIPS/ICLR | ★★★ attention-as-associative-retrieval |
| World Models | ICLR/NeurIPS (recurs) | ★★★ path-integration world model |
| InfoCog (Info-Theoretic Cognitive Systems) | NeurIPS (recurs) | ★★ calibration/NLL, predictive coding |
| NeuroAI / Foundation Models of the Brain | NeurIPS 2026, ~late Aug | ★★★ the neuro framing |
| Efficient / long-context sequence models | ICLR/NeurIPS | ★★ Hourglass efficiency result |

### NeuroAI / computational-neuroscience conferences
| Venue | Timing | Fit |
|---|---|---|
| Cosyne | conf Mar; abstract ~mid-Oct | ★★★ grid cells / navigation / path integration |
| CCN | conf Aug; deadline ~Apr | ★★★ cognitive maps, hippocampus, TEM-adjacent |
| RLDM | biennial | ★★ goal-directed / navigation results |
| Bernstein Conference | Germany, Sept/Oct; deadline ~summer | ★★ models of navigation |
| CNS / OCNS | conf Jul; deadline ~Feb | ★ mechanistic models |

### Main ML conferences (archival; anchor on a positive result)
| Venue | Next deadline (approx) | Fit |
|---|---|---|
| ICLR | **Sept 24, 2026** (2027) | ★★★ representations, architectures, NeuroAI |
| NeurIPS | ~May 2027 | ★★★ big NeuroAI presence |
| ICML | ~Jan 2027 | ★★ architectures/theory |
| AISTATS | ~Oct 2026 | ★★ Kalman/statistical-inference angle |
| RLC | ~Feb 2027 | ★★ goal-directed/navigation |
| UAI | ~Feb 2027 | ★★ InEKF/Bayesian-filtering framing |
| CPAL (Parsimony & Learning) | ~fall 2026 | ★★ geometry/structure-in-representations |

### Journals
| Venue | Fit |
|---|---|
| TMLR | ★★★ rolling, no novelty/SOTA bar — best home for the rigorous "when does it help" analysis + negatives |
| PLoS Computational Biology | ★★★ neuro-modeling story |
| eLife | ★★ if biological grounding deepens |
| Neural Computation | ★★ classic comp-neuro/ML theory |
| JMLR | ★★ if it becomes a large methods paper |

## NeurReps at a glance (for calibration)
- A NeurIPS **workshop**, not a standalone conference. 5th edition (founded 2022).
- ~60 accepted papers/year (61 in 2024, 64 in 2023), mostly posters + a few talks.
- 1–2 day event co-located with NeurIPS (Dec 11–12, 2026, Sydney).
- Tracks: **Proceedings** (9 pg, archival, PMLR) · **Extended Abstract** (4 pg,
  non-archival, arXiv-OK) · **Findings** (new, non-archival, no page limit).
- Right audience for the Lie-group/grid-cell content; acceptance not highly
  competitive. Credential weight: workshop-tier (proceedings track counts more).

## Candidate paper stories
1. **Hierarchy-vs-MapFormer dissociation** (compact, workshop-ready): hierarchy
   helps content pattern-completion; path integration helps spatial recall at
   OOD length; they are separable. Plain-vs-MapFormer control makes it rigorous.
   → NeurReps ext-abstract now; expand for a workshop or ICLR section.
2. **Structured priors: when they help** (main-conf scope): correction (InEKF /
   L1.5) earns its keep in noise / landmark / OOD-length / calibration regimes
   the paper did not test; honest negatives (hierarchy = efficiency not
   capability; PC↔Kalman duality) as analysis. → ICLR 2027 / TMLR.
3. **NeuroAI modeling** (comp-neuro scope): cognitive-map / grid-cell / clone
   structure results. → Cosyne abstract; PLoS Comp Bio for a full version.

## Immediate decision
Go for the NeurReps Aug-22 abstract? If yes, start now — multi-seed may not all
be in, so write around the single-seed dissociation + the plain-vs-MapFormer
control (clearly n=1), and let ICLR carry the multi-seed version.

## Sources
- NeurReps 2026: https://neurreps.org/ · past: https://www.neurreps.org/past-workshops
- NeurIPS 2026 workshops: https://neurips.cc/Conferences/2026/CallForWorkshops
- ICLR 2027 dates: https://iclr.cc/Conferences/2027/Dates · CfP: https://iclr.cc/Conferences/2027/CallForPapers
- Cosyne: https://www.cosyne.org/

# PAIRSPLIT -- results (pre-registration `PAIRSPLIT_PREREG.md`, commit 8de02d4)

`EMPair_r4` vs `EMPairConst_r4` at **n=48** (seeds 0-7 reused under a bitwise determinism
re-check, 8-47 fresh), to resolve the accuracy split PAIRCONST left open at n=8.

## The result: per-pair freedom DOES buy accuracy

| contrast | n | delta | sd | MDE | seeds + | verdict |
|---|---|---|---|---|---|---|
| **EMPair - EMPairConst (C1: freedom)** | 48 | **+0.091** | 0.164 | 0.066 | 34/48 | **DETECTABLE** |
| EMPairConst - P0 (C2: pathway) | 24 | +0.100 | 0.197 | 0.113 | 14/24 | unmeasured |
| EMPair - P0 (total) | 24 | +0.191 | 0.175 | 0.100 | 20/24 | DETECTABLE |

**S1 CONFIRMED**: freedom is real, detectable, and in the registered 0.05-0.15 band. S2 (freedom
buys nothing) and S3 (the n=8 estimate was low) are both not confirmed.

## The registered fresh-seed guard, which mattered

| seeds | delta | MDE | verdict |
|---|---|---|---|
| first 8 | +0.098 | 0.209 | unmeasured |
| **fresh 8-47** | **+0.089** | 0.069 | **DETECTABLE** |
| pooled 48 | +0.091 | 0.066 | DETECTABLE |

The freedom estimate is stable across the split (+0.098 -> +0.089) -- the first time in this line
that the first eight seeds did not overestimate. **They overestimated the OTHER two terms
instead**: the total fell +0.280 -> +0.191 and the pathway term +0.182 -> +0.100 as n went 8 ->
24. Fourth instance of the pattern, and worth noting that it landed on the terms I was NOT
guarding.

## The decomposition, CLOSED at n=48 (P0 extended, determinism bitwise)

| term | delta | MDE | seeds + | verdict | share |
|---|---|---|---|---|---|
| total, EMPair - P0 | **+0.215** | 0.068 | 42/48 | DETECTABLE | 100% |
| pathway, EMPairConst - P0 | **+0.124** | 0.071 | 34/48 | **DETECTABLE** | 58% |
| freedom, EMPair - EMPairConst | **+0.091** | 0.066 | 34/48 | **DETECTABLE** | 42% |

**Both halves are real.** Adding 2,048 parameters to the position pathway is worth +0.124 on its
own, changing no mechanism; making those parameters read the token -- per-pair freedom -- is
worth a further +0.091 and changes the mechanism completely.

The total has moved with n: +0.280 (n=8), +0.191 (n=24, P0 at 24 seeds), **+0.215 (n=48)**. Quote
the n=48 figure; the n=8 one was inflated by 30%.

## Reading this with the mechanism result

`PAIRCONST_RESULTS.md` already showed the mechanism attribution is clean and goes entirely to
freedom: the constant-origin control keeps the per-token rewind route (0.948, against P0's 0.964)
while EMPair abandons it (0.189), and only EMPair has per-pair phase spread (1.448 vs 0.000).

Together: **per-pair freedom changes HOW the model solves the task (decisively) and buys about
+0.09 of accuracy (detectably). Extra capacity in the position pathway buys a similar amount
without changing the mechanism at all.** Those are two different things and this line had been
conflating them since PAIRORIGIN.

r(final loss, accuracy) = -0.954 over the 96 runs, so these remain fit contrasts.

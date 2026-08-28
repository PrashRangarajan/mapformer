# enwik8 — PoPE x path-integration, and does the combination compose?

byte-level enwik8, seq 512, batch 16, 12k iters, lr 2e-4 (matching the existing flat9 run). Flat arms param-matched to 0.03%. **Lower bpc is better.** n=1 — this is a shape-finding run, not a claim.

| model | position | encoding | params | val bpc | vs RoPE | note |
|---|---|---|---|---|---|---|
| RoPE | index | RoPE | 28,634,880 | **1.5221** |  | baseline |
| Vanilla | path-int | RoPE | 28,636,672 | **1.5505** | +0.0284 |  |
| PoPE-Flat | index | PoPE | 28,639,488 | **1.5303** | +0.0082 |  |
| MapPoPE-Flat | path-int | PoPE | 28,642,048 | **1.5193** | -0.0027 | **the combination** |
| NoPE | none | none | 28,634,880 | **1.6113** | +0.0892 | null arm |
| MapPoPE-Hier | path-int | PoPE +hier | 29,404,288 | **1.6345** | +0.1124 | dim=896, +2.7% params |

> On NAVIGATION these compose: PoPE alone is at the floor, path-int alone
> 0.967, both 0.994. The question here is whether that holds on text, where
> each is worth only ~1-2% alone. Also watch NoPE: it collapsed to chance on
> navigation but is competitive on language in Selective RoPE's own tables.

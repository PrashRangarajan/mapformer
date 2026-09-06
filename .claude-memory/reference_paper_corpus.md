---
name: reference-paper-corpus
description: 28 positional-encoding papers stored locally at papers/ with an INDEX; grep the corpus instead of re-searching the web.
metadata:
  type: reference
---

Every source `mapformer_math.tex` cites lives in the repo at **`papers/`**:
`papers/txt/<key>.txt` (tracked, greppable), `papers/pdf/` (gitignored, 69 MB),
`papers/fetch.sh` (restores the PDFs), `papers/INDEX.md` (the manifest, with the
specific claim each reading confirms or corrects).

**All 28 were read first-hand on 2026-09-06.** No row of the relational map is
second-hand any more. Grep the corpus rather than re-searching:

    grep -n -i "data-dependent" papers/txt/hgrn.txt

Keys: mapformer srope pope grape mamba3 fox rope alibi xpos nope carope
jordan_rope liere alg_pe puranik_janestreet pj_rope mamba mamba2 gla path
deltanet rwkv7 cope tape mesanet titans hgrn hgrn2.

See [[reference-positional-landscape]] for what the corpus establishes, and
[[feedback-verify-before-relaying]] for why it exists.

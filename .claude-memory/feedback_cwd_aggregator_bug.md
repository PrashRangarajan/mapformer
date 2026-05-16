---
name: cd $REPO before every Python aggregator heredoc in run_*.sh
description: Run scripts cd to /home/prashr (so python3 -m mapformer.X resolves), then heredocs with relative paths resolve in the WRONG directory. Always cd to $REPO immediately before each Python heredoc.
type: feedback
originSessionId: be30e775-ba9b-48e1-a763-b2488b550411
---
In MapFormer run scripts (`run_*.sh`), the standard pattern is:
```bash
cd /home/prashr                                  # so python3 -m mapformer.X works
REPO=/home/prashr/mapformer
```
This means the shell's cwd is `/home/prashr`. Any `python3 -u <<PYEOF` heredoc whose Python code reads relative paths like `runs/{v}/...` or `paper_figures/{v}_s0.json` will resolve those paths against `/home/prashr/`, NOT against `/home/prashr/mapformer/`. The aggregator silently finds nothing, every row in the result table becomes `—`, and the failure is invisible until someone reads the MD.

**Why:** This has bitten us three times in 2026-05-14/05-15 sessions: `make_place_cell_figure.py` saved its PNG to the wrong dir, `run_post_pipeline_analysis.sh`'s probe/eval aggregator returned all `—`, and `run_level15_novelenv_fillin.sh` produced a TEM_NOVEL_ENV_RESULTS.md with every cell blank despite all the checkpoints existing. Each time, the fix is a 1-line `cd "$REPO"` before the heredoc.

**How to apply:**
- In every new `run_*.sh`, insert `cd "$REPO"` on the line immediately above each `python3 -u <<PYEOF` block (or any other relative-path command), even when an earlier `cd "$REPO"` already happened. Bash subshells / pipelines can reset cwd unexpectedly.
- Alternative if the heredoc must run from `/home/prashr` (rare): pass paths through `$REPO` interpolation BEFORE the heredoc starts (unquote the heredoc terminator: `<<PYEOF` not `<<'PYEOF'`) and read absolute paths inside. Quoting the terminator preserves `$REPO` as literal text — paths stay relative.
- Existing scripts patched 2026-05-15: `run_level15_novelenv_fillin.sh`, `run_post_pipeline_analysis.sh`, `run_tem_novel_envs.sh`, `run_tem_background_baselines.sh`, `run_multienv_clean_baselines.sh`, `run_multiclass_multiseed.sh`, `run_vanilla_nodrop_control.sh`. If older scripts need to be re-run, audit them first.
- Quick audit command: `grep -B1 "python3 -u <<" run_*.sh | grep -v "cd \"\\$REPO\"" | grep "python3"` — flags heredocs without a preceding cd.

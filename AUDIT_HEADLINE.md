# Audit: do the reported numbers match the data?

Recomputed from the per-seed JSON each evaluator wrote, and diffed against the markdown. Tolerance 0.0015 (docs print 3 decimals).

**23 pass, 0 mismatch, 0 skipped.**

| check | claimed | recomputed | |
|---|---|---|---|
| A paper-task fresh-map Vanilla | 0.9670 | 0.9674 | PASS |
| A paper-task fresh-map MapPoPE-Flat | 0.9940 | 0.9938 | PASS |
| A paper-task fresh-map RoPE | 0.5300 | 0.5295 | PASS |
| A paper-task fresh-map PlainFlat | 0.5340 | 0.5343 | PASS |
| A paper-task fresh-map PoPE-Flat | 0.5090 | 0.5086 | PASS |
| H MiniGrid T=1024 MapWM-Flat | 0.8230 | 0.8225 | PASS |
| H MiniGrid T=1024 MapWM-Hier | 0.8930 | 0.8930 | PASS |
| H MiniGrid T=1024 MapPoPE-Flat | 0.9190 | 0.9193 | PASS |
| H MiniGrid T=1024 MapPoPE-Hier | 0.9420 | 0.9423 | PASS |
| H MiniGrid T=1024 RoPE-Flat | 0.8270 | 0.8272 | PASS |
| H MiniGrid T=1024 RoPE-Hier | 0.9240 | 0.9235 | PASS |
| H MiniGrid T=1024 PoPE-Flat | 0.9530 | 0.9535 | PASS |
| H MiniGrid T=1024 PoPE-Hier | 0.9550 | 0.9551 | PASS |
| I knob baseline position effect | 0.4380 | 0.4383 | PASS |
| I knob allcombined position effect | -0.0760 | -0.0765 | PASS |
| I knob rotate position effect | 0.0500 | 0.0502 | PASS |
| I knob allocentric position effect | 0.4880 | 0.4877 | PASS |
| TABLE headline MiniGrid T=512 encoding | 0.0350 | 0.0350 | PASS |
| TABLE headline MiniGrid T=512 hierarchy | 0.0220 | 0.0216 | PASS |
| TABLE headline MiniGrid T=512 position | -0.0050 | -0.0048 | PASS |
| TABLE headline MiniGrid T=1024 encoding | 0.0760 | 0.0760 | PASS |
| TABLE headline MiniGrid T=1024 hierarchy | 0.0480 | 0.0478 | PASS |
| TABLE headline MiniGrid T=1024 position | -0.0210 | -0.0205 | PASS |

## What a PASS does and does not mean

It means the printed number is what the saved per-seed data computes to. It does NOT mean the experiment was well designed: every one of the 48 files in `archive/void/` would have passed this check on the day it was written. Design errors — a shortcut in the task, an undertrained arm, a confounded factor — are invisible here and need the gates and ablations instead.

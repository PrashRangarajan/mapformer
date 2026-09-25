# next_exp -- Dyck position effect at MATCHED nesting depth (prepared 2026-09-25, NOT launched, NOT committed)

Deliverables (copy into /home/prashr/mapformer):
- DYCK_MDEPTH_PREREG.md     pre-registration draft (commit it BEFORE launching)
- train_dyck.patch          opt-in --train-L / --train-D / --train-D-set; defaults reproduce runs/dyck_ladder bitwise
  (train_dyck.py = the patched file; `patch /home/prashr/mapformer/train_dyck.py < train_dyck.patch` applies cleanly)
- dyck_mdepth_common.py     A2f (feasible-closer A2) + the ladder's eval cells with the sampler's feasibility mask
- validate_dyck_mdepth.py   gates G1-G4; exits non-zero on failure (the driver runs it first)
- analyze_dyck_mdepth.py    readout, branches, secondaries; refuses to overwrite an existing results file
- run_dyck_mdepth.sh        lib_driver.sh driver, 210 runs, MAXPG 2/GPU; `bash -n` clean; DRV_DRYRUN=1 tested

Launch (after copying + committing):
  cd /home/prashr/mapformer && setsid nohup bash run_dyck_mdepth.sh > /dev/null 2>&1 &
  log: dyck_mdepth.log; marker: .dyck_mdepth_done; results: DYCK_MDEPTH_RESULTS.md/.json, DYCK_MDEPTH_GATES.md/.json

Evidence produced while gating (scratch only, never to be reused as runs):
- gates/DYCK_MDEPTH_GATES.md    gate report, PASS
- gate_repro/, gate_patched2/   repro + smoke runs (seed 0); patched defaults: 0 weights differ vs runs/dyck_ladder
- mock_runs/, mock_out/         analysis plumbing test on the ladder's checkpoints (reproduces the ladder's L32 D12 table)
- pkg/mapformer/                package copy used for every gate; identical to the repo except the new/patched files

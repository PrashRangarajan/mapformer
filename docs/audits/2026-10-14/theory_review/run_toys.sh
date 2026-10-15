#!/bin/bash
cd /home/prashr/mapformer/docs/audits/2026-10-14/theory_review
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
xargs -P 20 -L 1 -I{} sh -c 'f=$(echo "{}" | sed "s/.*--out //"); python3 toy_probe.py {} > ${f%.json}.log 2>&1'  < jobs.txt
touch toy_runs/ALL.done

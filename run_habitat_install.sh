#!/usr/bin/env bash
# Step 2 of the Habitat build: an ISOLATED py3.9 env for habitat-sim.
#
# habitat-sim is conda-only (not on PyPI) and py3.9-only. The main env is
# py3.12 + torch 2.6.0+cu124 and must not be touched. So habitat lives in its
# own env and is used ONLY to generate + tokenise trajectories, which are cached
# to disk; training then happens in the main env on that buffer. Same split
# MiniGridWorld_Cached already uses.
#
# 'headless' build: the server has no DISPLAY (libEGL is present).
set -euo pipefail
export PATH=/home/prashr/miniconda3/bin:$PATH
source /home/prashr/miniconda3/etc/profile.d/conda.sh
conda create -y -n habitat python=3.9 cmake=3.14.0 2>&1 | tail -3
conda install -y -n habitat -c aihabitat -c conda-forge \
  habitat-sim=0.3.3=py3.9_headless_linux_acbe6f4922e68145e401e55c30f9dfea460a3f24 2>&1 | tail -5
echo INSTALL_DONE; touch /home/prashr/mapformer/runs/.habitat_installed

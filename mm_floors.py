"""Gate floors (mm_gate_v3, unchanged) on the held-out secondary scoring split of the Memory Maze batch:
train-part0 indices 1600-2899, fits on 300-599 as in gate v3 (MEMORY_MAZE_PREREG.md). CPU only.
  python3 -I mm_floors.py --zip <15x15>/train-part0.zip --out floors_held.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_gate_v3 as g3  # noqa: E402

if __name__ == "__main__":
    g3.SCORE = (1600, 2900)
    g3.main()

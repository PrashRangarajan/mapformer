"""Two post hoc Fisher tests quoted in COPE_DIAGNOSIS.md (counts from m4_out.txt and m7_init_vs_final_out.txt).
Heads within a run are not independent, so the second p is optimistic."""
from scipy.stats import fisher_exact
print("runs with a NONE query type: sinusoidal table starts (Hi, Lo, LoS) 9/24 vs zero starts (WRAP_COPE CoPEF + GENERALITY) 11/16:",
      fisher_exact([[9, 15], [11, 5]]))
print("heads with both move gates < 0.1 at the end: OFF 18/32 vs LATTICE 0/21:", fisher_exact([[18, 14], [0, 21]]))

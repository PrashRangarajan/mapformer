"""Regenerate the RESULTS_INDEX.md catalogue: group every top-level *.md by line of work and
star the ones whose first 12 lines carry a CORRECTED / RETRACTED / WITHDRAWN / SUPERSEDED / VOID /
STALE banner.

    python3 docs/tools/catalog_results_index.py

Prints the grouped lists plus UNCLASSIFIED (files matching no group -- add a pattern for each, or
they are silently missing from the index). Used to build the 2026-09-24 catalogue; kept because
RESULTS_INDEX.md was last regenerated 2026-09-11 and is missing the Dyck / Bach / code /
rank-matched / rank-separation / loop-rank files (2026-09-30: added the CANCEL_, TEXTWORLD_, CTXSTEP_ and
CONTEXT_STEP groups). Output is pasted by hand into the catalogue section of RESULTS_INDEX.md; the
hand-written tables above it are not touched by this script. It chdirs to the repo itself (rule 25).
"""
import glob,re,os,collections
os.chdir('/home/prashr/mapformer')
files=sorted(glob.glob('*.md'))
groups=[
 ('Torus paper task, recipe and reproduction', r'^(PAPER_(?!FIG4)|PAPER2X2|_PAPER2X2|PAPERTASK|INDEX_BASELINE|BASELINE_TABLE|REVISIT_|AUDIT_HEADLINE|N3_AUDIT|NOISE_CLEAN|HORIZON_|FREQ_CONTROL|ROPE_CANONICAL|ROPE_CONVERGE|RECIPE_POWER|_RECIPE_|OOD_GRID|PERSCALE_OMEGA|OMEGA_RESCALE|LONG_SEQ|PER_VISIT|ZERO_SHOT|DRIFT_PROBE|CLOCK_SCAN|TOPOLOGY|GENERALIZATION|DETAILED_RESULTS|KNOB_SWEEP|ALLOCENTRIC|H12_|TIMING)'),
 ('Rank, generator and accumulator', r'^(RANK_|RANK[0-9]|FAST_ATTN_RANK|PAPER_FIG4|ACTION_GEOMETRY|LEARNED_RANK|DXR_|ND_GATES|SELECTIVE_ROPE|_SELECTIVE|GATE_PROBE|CONV_KERNEL|MAPPOPE_R4|LOCALISATION|ACCUMULATOR|THEORY_NARRATIVE|THEORY_NUMBERS|THEORY_SEARCH)'),
 ('Sign, clock/map and recency', r'^(SIGN_|_SIGN|MONOTONE|_MONOTONE|RECENCY_(?!EM|T3)|GATED_|FLIPFLOP|MQAR|FORGET_|LAMBDA_TRACE|COUNTER_|TEM_RECENCY)'),
 ('EM vs WM and the position kernel', r'^(EM_|AUDIT_2026|THEORY_KERNEL|TALE_OF|REC_EM|RECENCY_EM|N5_|_N5|DOF_|_DOF|D5_|MAGONLY|WARM_|UNFREEZE|NOLEAK|SEARCH_|SPREAD|PAIR|MATCH_QUERY_EM|VOCAB_EM|MINIGRID_EM|AP_KERNEL|HOPFIELD|PAPER_FIG4_EM)'),
 ('Match-Query, loop and algorithmic tasks', r'^(MATCH_QUERY|MATCH_GATES|LOOP_|LOOPED_|REFINE_|MQ_|L15_LOOP|_L15_LOOP|FRONTIER|ALGORITHMIC|RECURSIVE|HIER_PARITY|ADDITION_|SAMEBLOCK)'),
 ('Hierarchy, compositional and planner tasks', r'^(COMPOSITIONAL|COMP_HEADROOM|HIER_|HIERGOAL|AGGREGATE|BOUNDED_MEMORY|ROUTE_ATTN|SPACETIME|ABLATE_COMPOSITIONAL|CORRECTION_COMPOSITIONAL|PLANNER|ROOMS_GOAL|DISSOCIATION|CSCG|STITCH|MAP_QUERY|LAP_)'),
 ('Family tree', r'^(FAMILY_TREE|ABLATE_FAMILY)'),
 ('MiniGrid, MiniWorld, Habitat', r'^(MINIGRID|MINIWORLD|ALIASING|VISITS_TEST|POSITION_EFFECT|CROSSOVER_CONVERGED|CONTINUOUS_ALLOC|DAGGER|DOORKEY|HABITAT|PERCEPTION)'),
 ('Dyck-2 and the cancellation knob (H3): depth substitution', r'^(DYCK_|CANCEL_)'),
 ('Navigation told in words and the context-dependent step', r'^(TEXTWORLD_|CTXSTEP_|CONTEXT_STEP)'),
 ('Indirect Indexing', r'^INDIRECT_'),
 ('Bach chorales, decay envelope and the MapPoPE collapse', r'^(JSB|AUG_|DECAY_|T1_|T2_|T3_|T3GEN|TORUS_T3|RECENCY_T3|CROSS_|MAESTRO|THEORY_MAPPOPE|MAPPOPE_VS_POPE|POPE_WRAPPING)'),
 ('PoPE ablation, code and enwik8', r'^(ABLATE_PREREG|ABLATE_RESULTS|CODE_|ENWIK8|BF16|LANGUAGE_LANDSCAPE)'),
 ('Level 1.5 / InEKF / PC / TEM / grid cells (April-August lines)', r'^(L15_ABLATION|NOISE_REFINE|EXTRAHEAD|LEVEL15|LM200|CORRECTED_LM200|CAPACITY_|GSF_|NOBYPASS|NODROP|V3_|V4_|R_T_|CLONE_|CASCADE|SESSION_HIER|BUMP_TOKEN|NUMBERLINE|MULTICLASS|MULTISEED_FOLLOW|TEM_(?!RECENCY)|HIPPOCAMPAL|CNAV_|DOG_|VECTOR_NAV|VOCAB_SWEEP|RESULTS_PAPER)'),
 ('Project meta, reports and infrastructure', r'^(GUARDS|KNOWN_BUGS|HOURGLASS_README|README|CLAUDE|RESULTS_INDEX|PUBLICATION|REPORT|RESULTS_SUMMARY|SESSION_2026|MINIWORLD_TODO)'),
]
out=collections.OrderedDict((g,[]) for g,_ in groups); un=[]
ban=re.compile(r'(CORRECTED|RETRACTED|WITHDRAWN|SUPERSEDED|VOID|STALE)')
for f in files:
    for g,p in groups:
        if re.match(p,f):
            head=''.join(open(f,errors='ignore').readlines()[:12])
            b = bool(ban.search(head))
            out[g].append(f[:-3]+('*' if b else ''))
            break
    else: un.append(f)
print('UNCLASSIFIED',un)
for g,l in out.items():
    print(f'### {g} ({len(l)})\n'); print(', '.join('`%s`'%x if not x.endswith('*') else '`%s`*'%x[:-1] for x in l)); print()

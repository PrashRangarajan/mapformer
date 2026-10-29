"""Regenerate the RESULTS_INDEX.md catalogue: group every top-level *.md by line of work and
star the ones whose first 12 lines carry a CORRECTED / RETRACTED / WITHDRAWN / SUPERSEDED / VOID /
STALE banner.

    python3 docs/tools/catalog_results_index.py

Prints the grouped lists plus UNCLASSIFIED (files matching no group -- add a pattern for each, or
they are silently missing from the index). Used to build the 2026-09-24 catalogue; kept because
RESULTS_INDEX.md was last regenerated 2026-09-11 and is missing the Dyck / Bach / code /
rank-matched / rank-separation / loop-rank files (2026-09-30: added the CANCEL_, TEXTWORLD_, CTXSTEP_ and
CONTEXT_STEP groups; 2026-10-03: a LEAK_ / NEWOBJ / WHAT_WHERE group, and a second pass listing the
docs-level notes `docs/*.md` and `docs/lit/*.md`, which have no top-level file). Output is in the
catalogue's own format and is pasted by hand into the catalogue section of RESULTS_INDEX.md; the
hand-written tables above it are not touched by this script. It chdirs to the repo itself (rule 25).
Exits 1 if any top-level file is unclassified.

2026-10-29 (cleanup A5): also scans the archive folders (`archive/**` and `archive_stale/`, recursively). An archived
file is listed in its line's group by its repo-relative path and marked `[a]`; archived files whose name matches no
group go to "Archived, other lines" instead of failing the run (only top-level files must classify). Added the groups
for the batches registered after 2026-10-09 (accounts, clock side/speed, delta-free, exptorus, factorial, G3 SSM,
gauge, generality, image torus, lrsweep, mechanism, Memory Maze, RANKCLOCK / RANKRECIPE / RANKWD, ring budget / dial,
RNN torus, robust, signroute, signsep, SRoPE torus, torus lrsweep / start, wrap cope / existence).
"""
import glob,re,os,collections
os.chdir('/home/prashr/mapformer')
files=sorted(glob.glob('*.md'))
arch=sorted(f for f in glob.glob('archive/**/*.md',recursive=True)+glob.glob('archive_stale/**/*.md',recursive=True)
           if os.path.basename(f) not in ('README.md','MOVED.md') and not os.path.basename(f).startswith('RESULTS_INDEX_'))
groups=[
 ('Torus paper task, recipe and reproduction', r'^(PAPER_(?!FIG4)|PAPER2X2|_PAPER2X2|PAPERTASK|INDEX_BASELINE|BASELINE_TABLE|REVISIT_|AUDIT_HEADLINE|N3_AUDIT|NOISE_CLEAN|HORIZON_|FREQ_CONTROL|ROPE_CANONICAL|ROPE_CONVERGE|RECIPE_POWER|_RECIPE_|OOD_GRID|PERSCALE_OMEGA|OMEGA_RESCALE|LONG_SEQ|PER_VISIT|ZERO_SHOT|DRIFT_PROBE|CLOCK_SCAN|TOPOLOGY|GENERALIZATION|DETAILED_RESULTS|KNOB_SWEEP|ALLOCENTRIC|H12_|TIMING)'),
 ('Rank, generator and accumulator', r'^(RANK_|RANK[0-9]|FAST_ATTN_RANK|PAPER_FIG4|ACTION_GEOMETRY|LEARNED_RANK|DXR_|ND_GATES|SELECTIVE_ROPE|_SELECTIVE|GATE_PROBE|CONV_KERNEL|MAPPOPE_R4|MAPPOPE_PAIR|SCORE_RANK|GAIN_GRAIN|DELTA_FREE|_DELTA_FREE|RANKCLOCK|RANKRECIPE|RANKWD|LOCALISATION|ACCUMULATOR|THEORY_NARRATIVE|THEORY_NUMBERS|THEORY_SEARCH)'),
 ('Sign, clock/map and recency', r'^(SIGN_|SIGNROUTE|SIGNSEP|_SIGN|ACCOUNTS_|CLOCK_SIDE|CLOCK_SPEED|EXPTORUS|GAUGE_|GENERALITY|LRSWEEP|TORUS_LRSWEEP|MECHANISM_|RING_BUDGET|RING_DIAL|ROBUST_|SROPE_TORUS|TORUS_START|WRAP_COPE|WRAP_EXISTENCE|MONOTONE|_MONOTONE|RECENCY_(?!EM|T3)|GATED_|FLIPFLOP|MQAR|FORGET_|LAMBDA_TRACE|COUNTER_|TEM_RECENCY)'),
 ('EM vs WM and the position kernel', r'^(EM_|AUDIT_2026|THEORY_KERNEL|TALE_OF|REC_EM|RECENCY_EM|N5_|_N5|DOF_|_DOF|D5_|MAGONLY|WARM_|UNFREEZE|NOLEAK|SEARCH_|SPREAD|PAIR|MATCH_QUERY_EM|VOCAB_EM|MINIGRID_EM|AP_KERNEL|HOPFIELD|PAPER_FIG4_EM)'),
 ('Match-Query, loop and algorithmic tasks', r'^(MATCH_QUERY|MATCH_GATES|LOOP_|LOOPED_|REFINE_|MQ_|L15_LOOP|_L15_LOOP|FRONTIER|ALGORITHMIC|RECURSIVE|HIER_PARITY|ADDITION_|SAMEBLOCK)'),
 ('Depth, loops and recurrent baselines on the torus (factorial, SSM, RNN)', r'^(FACTORIAL|G3_SSM|RNN_TORUS)'),
 ('Navigation from pixels (image torus, Memory Maze)', r'^(IMAGE_TORUS|MEMORY_MAZE)'),
 ('Hierarchy, compositional and planner tasks', r'^(COMPOSITIONAL|COMP_HEADROOM|HIER_|HIERGOAL|AGGREGATE|BOUNDED_MEMORY|ROUTE_ATTN|SPACETIME|ABLATE_COMPOSITIONAL|CORRECTION_COMPOSITIONAL|PLANNER|ROOMS_GOAL|DISSOCIATION|CSCG|STITCH|MAP_QUERY|LAP_)'),
 ('Family tree', r'^(FAMILY_TREE|ABLATE_FAMILY)'),
 ('MiniGrid, MiniWorld, Habitat', r'^(MINIGRID|MINIWORLD|ALIASING|VISITS_TEST|POSITION_EFFECT|CROSSOVER_CONVERGED|CONTINUOUS_ALLOC|DAGGER|DOORKEY|HABITAT|PERCEPTION)'),
 ('Dyck-2 and the cancellation knob (H3): depth substitution', r'^(DYCK_|CANCEL_)'),
 ('Navigation told in words and the context-dependent step', r'^(TEXTWORLD_|CTXSTEP_|CONTEXT_STEP)'),
 ('New objects, what/where and the what-to-where leak', r'^(LEAK_|NEWOBJ|WHAT_WHERE|TW_|GAIN_PHASE)'),
 ('Indirect Indexing', r'^INDIRECT_'),
 ('Bach chorales, decay envelope and the MapPoPE collapse', r'^(JSB|AUG_|DECAY_|T1_|T2_|T3_|T3GEN|TORUS_T3|RECENCY_T3|CROSS_|MAESTRO|THEORY_MAPPOPE|MAPPOPE_VS_POPE|POPE_WRAPPING)'),
 ('PoPE ablation, code and enwik8', r'^(ABLATE_PREREG|ABLATE_RESULTS|CODE_|ENWIK8|BF16|LANGUAGE_LANDSCAPE|TINYSTORIES)'),
 ('Level 1.5 / InEKF / PC / TEM / grid cells (April-August lines)', r'^(L15_ABLATION|NOISE_REFINE|EXTRAHEAD|LEVEL15|LM200|CORRECTED_LM200|CAPACITY_|GSF_|NOBYPASS|NODROP|V3_|V4_|R_T_|CLONE_|CASCADE|SESSION_HIER|BUMP_TOKEN|NUMBERLINE|MULTICLASS|MULTISEED_FOLLOW|TEM_(?!RECENCY)|HIPPOCAMPAL|CNAV_|DOG_|VECTOR_NAV|VOCAB_SWEEP|RESULTS_PAPER)'),
 ('Project meta, reports and infrastructure', r'^(GUARDS|KNOWN_BUGS|HOURGLASS_README|README|CLAUDE|RESULTS_INDEX|PUBLICATION|REPORT|RESULTS_SUMMARY|SESSION_2026|MINIWORLD_TODO)'),
]
out=collections.OrderedDict((g,[]) for g,_ in groups); un=[]
ban=re.compile(r'(CORRECTED|RETRACTED|WITHDRAWN|SUPERSEDED|VOID|STALE)')
def fmt(name,b,a):
    return '`%s`'%name+('*' if b else '')+(' [a]' if a else '')
arch_other=[]
for f in files+arch:
    a = f in arch
    base=os.path.basename(f)
    head=''.join(open(f,errors='ignore').readlines()[:12])
    b = bool(ban.search(head))
    for g,p in groups:
        if re.match(p,base):
            out[g].append(fmt(f[:-3],b,a))
            break
    else:
        (arch_other if a else un).append(fmt(f[:-3],b,a) if a else f)
print('UNCLASSIFIED',un)
print(f'top-level *.md: {len(files)}; archived *.md (archive/, archive_stale/): {len(arch)}\n')
for g,l in out.items():
    print(f'**{g}** ({len(l)})\n'); print(', '.join(l)); print()
print(f'**Archived, other lines** ({len(arch_other)})\n'); print(', '.join(arch_other)); print()
docs=sorted(glob.glob('docs/*.md'))+sorted(glob.glob('docs/lit/*.md'))
dl=[]
for f in docs:
    head=''.join(open(f,errors='ignore').readlines()[:12])
    dl.append('`%s`'%f+('*' if ban.search(head) else ''))
print(f'**Docs-level notes** ({len(dl)})\n'); print(', '.join(dl)); print()
raise SystemExit(1 if un else 0)

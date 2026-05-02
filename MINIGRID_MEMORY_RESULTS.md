# MiniGrid-MemoryS13: Vanilla vs Level 1.5 vs RoPE

Different topology than DoorKey: starting room → narrow hallway → 
choice room with two objects. Re-run after fixing MiniGridWorld_Cached's 
kwarg-propagation bug (was crashing with 'p_transition_noise unexpected').

| Variant | T=128 OOD | T=512 OOD | T=1024 OOD |
|---|---|---|---|
| **Vanilla** | 0.956 (NLL 0.112) | 0.763 (NLL 1.561) | 0.674 (NLL 2.327) |
| **Level15** | 0.963 (NLL 0.095) | 0.897 (NLL 0.329) | 0.809 (NLL 0.703) |
| **RoPE** | 0.918 (NLL 0.192) | 0.796 (NLL 0.796) | 0.731 (NLL 1.502) |

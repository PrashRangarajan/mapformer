# READING

**Neither encoding learns pointer arithmetic as arithmetic.** Both solve the trained range and
collapse the moment the shift leaves it: at |k| in [16,30] PoPE scores 0.024 and MapPoPE 0.039
(chance 0.019), and at |k| in [21,30] both are at zero. The PoPE paper says the task "requires
models to learn to independently manipulate the content and positional information of tokens and to
apply pointer arithmetic operations" -- what they in fact learn is the operation for the 31 shift
values seen in training. This is a statement about the task and both encodings, not about either one.

**The one place path integration is ahead is robustness to a changed offset, and the control is
what shows it.** The length condition needs a bigger block, so a pad control was run: the SAME
in-distribution examples at block 68 instead of 56. That control is where the gap is largest
(MapPoPE 0.430 vs PoPE 0.149, +0.284, 6/7 seeds, MDE 0.231 -- detectable), and it is larger than the
length effect it was meant to control for (+0.203, MDE 0.193). So the honest reading is:

- there is no length-generalisation claim here -- the pad change alone accounts for the difference;
- both arms are badly damaged by simply padding to a different block (PoPE 0.962 -> 0.149,
  MapPoPE 0.920 -> 0.430), which is a fragility neither paper reports;
- path integration is roughly 3x more robust to that change. Its position is a cumulative sum over
  tokens, so extra pad tokens shift every angle by a constant that attention differences cancel;
  an index code sees genuinely different absolute positions.

**Answer to "is there a PoPE task where path integration does better": not in distribution.**
Bach Chorales is a null (0/5 seeds better); Indirect Indexing ties at sufficient budget (8/8 vs 7/8)
and differs only in speed. Under distribution shift it is ahead on offset robustness, a condition
the paper does not test, and both arms are far below their in-distribution accuracy there.


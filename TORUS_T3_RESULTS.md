# Paper's own OOD protocol (Table 2, 1D-2D grid navigation)

Appendix B verbatim: trained at l=128, lgrid=64, pempty=0.5; **OOD-d**: l=64, pempty=0.2, lgrid=32; **OOD-s**: l=512, pempty=0.8, lgrid=128.
'dense'/'sparse' = OBJECT density (pempty), not visit density.

The Table 2 caption instead gives OOD-s as l=256; both lengths are reported since the paper is internally inconsistent.

Paper's 2D results -- MapWM: IID 0.99, OOD-d 0.99, OOD-s 0.96. MapEM-os: IID 1.0, OOD-d 0.99, OOD-s 0.97.

| variant | IID  l=128 g=64  pe=0.5 | OOD-d l=64 g=32  pe=0.2 | OOD-s l=256 g=128 pe=0.8 | OOD-s l=512 g=128 pe=0.8 | ext-s l=1024 g=128 pe=0.8 | ext-s l=2048 g=128 pe=0.8 |
|---|---|---|---|---|---|---|
| MapPoPE-Flat | 1.000 ± 0.001 | 0.992 ± 0.004 | 0.998 ± 0.002 | 0.990 ± 0.003 | 0.979 ± 0.004 | 0.963 ± 0.005 |
| MapPoPE_T3 | 0.999 ± 0.001 | 0.992 ± 0.003 | 0.996 ± 0.003 | 0.978 ± 0.024 | 0.961 ± 0.042 | 0.942 ± 0.046 |
| MapPoPE_T3inert | 1.000 ± 0.001 | 0.993 ± 0.004 | 0.998 ± 0.002 | 0.991 ± 0.004 | 0.979 ± 0.005 | 0.962 ± 0.005 |

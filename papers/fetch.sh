#!/usr/bin/env bash
# Re-fetch every paper referenced by mapformer_math.tex.
# PDFs are gitignored (bulk); the extracted .txt files ARE tracked, so a
# fresh clone can grep the corpus without re-downloading.
#   usage: bash papers/fetch.sh [key ...]     (no args = all)
set -u
cd "$(dirname "$0")"
mkdir -p pdf txt

declare -A P=(
  # --- the two the whole note is about
  [mapformer]=2511.19279  [srope]=2511.17388
  # --- read first-hand before this session
  [pope]=2509.10534       [grape]=2512.07805   [mamba3]=2603.15569
  [fox]=2503.02130
  # --- in-frame rows that were UNVERIFIED
  [rope]=2104.09864       [alibi]=2108.12409   [xpos]=2212.10554
  [nope]=2305.19466       [carope]=2507.23083  [jordan_rope]=2605.04217
  [mamba]=2312.00752      [mamba2]=2405.21060  [gla]=2312.06635
  # --- out-of-frame rows that were UNVERIFIED
  [path]=2505.16381       [deltanet]=2406.06484 [rwkv7]=2503.14456
  [cope]=2405.18719       [tape]=2501.00712    [mesanet]=2506.05233
  [titans]=2501.00663
  # --- the unverified prior-art paragraph
  [grazzi]=2411.12537     [sarrof]=2405.17394    [hgrn]=2311.04823       [hgrn2]=2404.07904
  # --- adjacent, found while resolving
  [pj_rope]=2606.05345    [liere]=2406.10322   [alg_pe]=2312.16045
)

keys=("$@"); [ ${#keys[@]} -eq 0 ] && keys=("${!P[@]}")
for k in "${keys[@]}"; do
  id="${P[$k]:-}"; [ -z "$id" ] && { echo "unknown key $k"; continue; }
  if [ ! -s "pdf/$k.pdf" ]; then
    curl -sSL --max-time 120 -A "mapformer-research/1.0" \
         -o "pdf/$k.pdf" "https://arxiv.org/pdf/$id" || { echo "FAIL $k"; continue; }
    sleep 4
  fi
  if [ ! -s "txt/$k.txt" ]; then
    pdftotext -q "pdf/$k.pdf" "txt/$k.txt" 2>/dev/null
  fi
  printf "%-14s %-12s pdf %6sK  txt %6s lines\n" "$k" "$id" \
    "$(( $(stat -c%s "pdf/$k.pdf" 2>/dev/null || echo 0) / 1024 ))" \
    "$(wc -l < "txt/$k.txt" 2>/dev/null || echo 0)"
done

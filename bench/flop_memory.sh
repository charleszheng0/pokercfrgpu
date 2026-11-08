#!/bin/bash
# Memory of a full flop tree (2 bet sizes, 1 raise size) for three board textures,
# with and without suit isomorphism and 16-bit storage. Builds the tree only (no solve).
set -u
BIN="$(dirname "$0")/../build/pokersolve"
OOP="TT-22,AJs-A2s,KTs+,K9s-K6s,QTs+,Q9s,J9s+,T8s+,97s+,86s+,75s+,64s+,54s,AJo-A8o,KTo+,QTo+,JTo"
IP="AA-22,A2s+,K7s+,Q8s+,J8s+,T8s+,97s+,86s+,76s,65s,A8o+,KTo+,QTo+,JTo"
printf "%-10s %-10s %-4s %-7s %10s %s\n" board texture iso bits memory_MB "cards solved"
for spec in "Ks 9d 5c:rainbow" "Ks 9s 5d:two-tone" "Ks 9s 5s:monotone"; do
  board="${spec%%:*}"; tex="${spec##*:}"
  for opt in "0 0" "1 0" "1 1"; do
    iso="${opt% *}"; c16="${opt#* }"
    out=$("$BIN" --board "$board" --pot 10 --stack 95 --oop "$OOP" --ip "$IP" --bets 33,75 --raises 60 \
      --iso "$iso" --compress "$c16" --mem_gb 64 --iters 0 --target 100 2>&1)
    mb=$(echo "$out" | sed -nE 's/.*memory ([0-9.]+) MB.*/\1/p')
    solved=$(echo "$out" | sed -nE 's/.*solved ([0-9]+ of [0-9]+).*/\1/p')
    printf "%-10s %-10s %-4s %-7s %10s %s\n" "$board" "$tex" "$iso" "$([ "$c16" = 1 ] && echo 16 || echo 32)" "$mb" "$solved"
  done
done

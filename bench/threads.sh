#!/bin/bash
# Thread load balance: wall time and CPU use ((user+sys)/real) for a fixed iteration count.
# Usage: bench/threads.sh [NEW_BIN] [OLD_BIN] [RUNS] [THREADS...]
#   NEW_BIN  default build/pokersolve
#   OLD_BIN  optional; when given, old and new runs alternate (the machine is noisy)
#   RUNS     runs per binary and case (default 3)
#   THREADS  thread counts to try (default "8")
# Prints one line per run, then the median per binary/case/threads.
set -u
DIR="$(cd "$(dirname "$0")/.." && pwd)"
NEW="${1:-$DIR/build/pokersolve}"
OLD="${2:-}"
RUNS="${3:-3}"
shift $(( $# < 3 ? $# : 3 ))
THREADS=("${@:-8}")
[ ${#THREADS[@]} -eq 0 ] && THREADS=(8)

OOP="TT-22,AJs-A2s,KTs+,K9s-K6s,QTs+,Q9s,J9s+,T8s+,97s+,86s+,75s+,64s+,54s,AJo-A8o,KTo+,QTo+,JTo"
IP="AA-22,A2s+,K7s+,Q8s+,J8s+,T8s+,97s+,86s+,76s,65s,A8o+,KTo+,QTo+,JTo"
TURN=(--board "Ks 9d 5c 2h" --pot 10 --stack 95 --oop "$OOP" --ip "$IP" --bets 33,75 --raises 60 --iters 60 --target 0)
FLOP=(--board "Ks 9s 5s" --pot 10 --stack 95 --oop "$OOP" --ip "$IP" --bets 50 --raises none --max_raises 0 --iters 10 --target 0)

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
BINS=(new)
[ -n "$OLD" ] && BINS=(old new)

run() {  # bin case threads -> appends "real cpu expl" to the result file
  local bin="$1" name="$2" t="$3" exe args
  [ "$bin" = old ] && exe="$OLD" || exe="$NEW"
  if [ "$name" = turn ]; then args=("${TURN[@]}"); else args=("${FLOP[@]}"); fi
  local real user sys expl TIMEFORMAT="%R %U %S"
  { time "$exe" "${args[@]}" --threads "$t" >"$TMP/log" 2>&1; } 2>"$TMP/time"
  read -r real user sys <"$TMP/time"
  expl=$(sed -nE 's/.*exploitability ([0-9.]+)%.*/\1/p' "$TMP/log" | tail -1)
  local cpu
  cpu=$(awk -v r="$real" -v u="$user" -v s="$sys" 'BEGIN{printf "%.2f", (u+s)/r}')
  printf "%-5s %-4s t=%-2s real %7.2f s  cpu %5s  expl %s%%\n" "$name" "$bin" "$t" "$real" "$cpu" "$expl"
  echo "$real $cpu $expl" >>"$TMP/$name.$bin.$t"
}

for name in turn flop; do
  for t in "${THREADS[@]}"; do
    for ((r = 0; r < RUNS; ++r)); do
      for bin in "${BINS[@]}"; do run "$bin" "$name" "$t"; done
    done
  done
done

echo
printf "%-5s %-4s %-4s %10s %8s %s\n" case bin thr median_s cpu expl%
for name in turn flop; do
  for t in "${THREADS[@]}"; do
    for bin in "${BINS[@]}"; do
      f="$TMP/$name.$bin.$t"
      med=$(sort -n "$f" | awk '{a[NR]=$0} END{print a[int((NR+1)/2)]}')
      read -r real cpu expl <<<"$med"
      printf "%-5s %-4s %-4s %10.2f %8s %s\n" "$name" "$bin" "$t" "$real" "$cpu" "$expl"
    done
  done
done

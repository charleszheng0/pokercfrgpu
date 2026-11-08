#!/bin/bash
# Keep filling the training database. Each round: 30 river spots, then 6 turn spots,
# each with its own seed, so rounds never repeat. Stop with Ctrl-C (or kill); every
# finished solve is already committed to the database.
#   scripts/train.sh [db] [rounds] [first_round]
# first_round defaults to a number from the clock, so a restart never repeats old boards.
set -u
cd "$(dirname "$0")/.."
DB="${1:-data/spots.db}"
ROUNDS="${2:-1000}"
FIRST="${3:-$(( $(date +%s) % 1000000 ))}"
mkdir -p "$(dirname "$DB")"
# Run a frozen copy of the runner, so rebuilding build/ during a long run is safe.
BIN="${POKERBATCH:-data/bin/pokerbatch}"
mkdir -p "$(dirname "$BIN")"
[ -n "${POKERBATCH:-}" ] || cp build/pokerbatch "$BIN"
echo "runner: $BIN  database: $DB  first round: $FIRST"
for ((r = FIRST; r < FIRST + ROUNDS; r++)); do
  echo "=== round $r $(date '+%H:%M:%S')"
  "$BIN" --db "$DB" --count 30 --street river --seed $((r * 1000 + 1)) || exit 1
  "$BIN" --db "$DB" --count 6 --street turn --seed $((r * 1000 + 2)) || exit 1
done

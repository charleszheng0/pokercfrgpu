#!/bin/bash
# Value-network loop on a Colab GPU: on-path data with GPU label solves, retrain, score.
# Seeds: models/vnet_r3.net and models/vnet_data.tar.gz (the laptop rounds). Prints SUMMARY lines.
set -e
cd "$(dirname "$0")/.."
cmake --build build -j4 --target pokervnet_gpu 2>&1 | grep -E "error" || true
B=./build/pokervnet_gpu
mkdir -p data && tar -xzf models/vnet_data.tar.gz -C data
cp models/vnet_r3.net data/vnet_r3.net
ROUNDS=${ROUNDS:-3}
FLOPS=${FLOPS:-40}
prev=data/vnet_r3.net
for r in $(seq 5 $((4 + ROUNDS))); do
  t0=$(date +%s)
  $B gen3 --gpu 1 --count $FLOPS --seed $((40 + r)) --net $prev --out data/vnet3 > gen_r$r.log
  t1=$(date +%s)
  labels=$(grep -c "label" gen_r$r.log)
  lsec=$(grep "label" gen_r$r.log | awk '{s+=$(NF-1)} END {printf "%.2f", s/NR}')
  $B train --data data/vnet,data/vnet2,data/vnet3 --net data/vnet_r$r.net --valid_every 6 --epochs 30 > train_r$r.log
  t2=$(date +%s)
  score=$($B eval --gpu 1 --preset "BB 3-bet vs BTN" --flop Ac9d8c --net data/vnet_r$r.net --only_nn 1 | grep "flop-locked")
  echo "SUMMARY round $r: $labels labels in $((t1 - t0)) s (mean $lsec s per label solve), train $((t2 - t1)) s, $(grep 'network: valid' train_r$r.log), $score"
  prev=data/vnet_r$r.net
done
best=$prev
for job in "BB 3-bet vs BTN|Ac9d8c" "BB 3-bet vs BTN|QcJd5d" "SB 3-bet vs BTN|Tc9d4h"; do
  pr=${job%%|*}; fl=${job##*|}
  $B eval --gpu 1 --preset "$pr" --flop $fl --net $best --dl_iters 300 > eval_$fl.log
  echo "SUMMARY eval $fl:"; grep -E "value net |check-down |DCFR at|DCFR full|whole-tree" eval_$fl.log
done

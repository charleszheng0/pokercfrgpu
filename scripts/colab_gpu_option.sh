#!/bin/bash
# Colab check of the --gpu option (PR #5): build the _gpu programs with CUDA and run each one next
# to its CPU program on the same spots. Every database is a scratch file in /tmp.
# Prints CHECK lines; the last cell of notebooks/colab_gpu_option.ipynb collects them.
set -e
cd "$(dirname "$0")/.."
T=/tmp/gpucheck
rm -rf $T && mkdir -p $T

cmake -S . -B build -DPOKER_CUDA=ON -DCMAKE_BUILD_TYPE=Release > $T/cmake.log
cmake --build build -j4 --target tests pokersolve pokerbatch pokersolve_gpu pokerbatch_gpu pokerlab_gpu > $T/build.log 2>&1 ||
  { grep -E "error" $T/build.log | head -20; echo "CHECK build FAILED"; exit 1; }
echo "CHECK build ok: $(ls build | grep -E '_gpu$' | tr '\n' ' ')"

./build/tests --quick > $T/tests.log 2>&1 && echo "CHECK tests --quick: $(tail -1 $T/tests.log)" ||
  { tail -15 $T/tests.log; echo "CHECK tests FAILED"; }

# 1. pokersolve vs pokersolve_gpu, same turn spot, same target.
OOP="TT-22,AJs-A2s,KTs+,K9s-K6s,QTs+,Q9s,J9s+,T8s+,97s+,86s+,75s+,64s+,54s,AJo-A8o,KTo+,QTo+,JTo"
IP="AA-22,A2s+,K7s+,Q8s+,J8s+,T8s+,97s+,86s+,76s,65s,A8o+,KTo+,QTo+,JTo"
ARGS=(--board Ks9d5c2h --pot 100 --stack 950 --oop "$OOP" --ip "$IP" --bets 33,75 --raises 60 --max_raises 4 --target 0.5 --iters 2000)
./build/pokersolve "${ARGS[@]}" > $T/solve_cpu.log
./build/pokersolve_gpu "${ARGS[@]}" --gpu 1 > $T/solve_gpu.log
for k in cpu gpu; do
  echo "CHECK pokersolve $k: $(grep -E '^solver' $T/solve_$k.log) | $(grep '^iter' $T/solve_$k.log | tail -1) | $(grep '^EV' $T/solve_$k.log)"
done

# 2. pokerbatch vs pokerbatch_gpu: same seed = same boards. River, turn, and one 3-bet flop.
echo "BB 3-bet vs BTN|Ac9d8c|test|0" > $T/flops.txt
for k in cpu gpu; do
  B=./build/pokerbatch; G=""
  [ $k = gpu ] && B=./build/pokerbatch_gpu && G="--gpu 1"
  s0=$(date +%s)
  $B --db $T/batch_$k.db --count 4 --street river --seed 11 $G > $T/batch_${k}_river.log 2>&1
  $B --db $T/batch_$k.db --count 2 --street turn --seed 12 $G > $T/batch_${k}_turn.log 2>&1
  $B --db $T/batch_$k.db --street flop --flops $T/flops.txt --iters 800 $G > $T/batch_${k}_flop.log 2>&1
  echo "CHECK pokerbatch $k: $(( $(date +%s) - s0 )) s total"
  sqlite3 $T/batch_$k.db "select 'CHECK   $k solve', street, board, status, iterations, round(seconds,2), round(exploitability_pct,3),
    (select count(*) from spots p where p.solve_id = s.id), config like '%backend=cuda%' from solves s order by id;"
done
for k in cpu gpu; do
  sqlite3 $T/batch_$k.db "select 'CHECK   $k categories', category, count(*) from spots group by category;"
done
grep -h "GPU not used\|needs a CUDA build" $T/batch_gpu_*.log | sed 's/^/CHECK   gpu note: /' || true

# 3. pokerlab_gpu --gpu 1: one turn solve through the web API.
./build/pokerlab_gpu 8090 $T/batch_gpu.db --gpu 1 > $T/lab.log 2>&1 &
LAB=$!
sleep 2
echo "CHECK pokerlab limits: $(curl -s localhost:8090/api/limits)"
ID=$(curl -s -X POST localhost:8090/api/solve --data-urlencode "board=Ks 9d 5c 2h" -d pot=10 -d stack=95 \
  --data-urlencode "oop=TT-22,AJs-A2s,KTs+,QTs+,AJo-A8o" --data-urlencode "ip=AA-22,A2s+,K7s+,A8o+,KTo+" \
  -d bets=33,75 -d raises=60 -d target=0.5 | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
for i in $(seq 1 120); do
  ST=$(curl -s "localhost:8090/api/status?id=$ID")
  echo "$ST" | grep -q '"state":"done"' && break
  sleep 1
done
echo "CHECK pokerlab status: $(echo "$ST" | grep -o '"state":"[a-z]*"\|"note":"[^"]*"\|"iteration":[0-9]*\|"expl_pct":[0-9.]*\|"seconds":[0-9.]*\|"backend":"[a-z]*"' | tr '\n' ' ')"
echo "CHECK pokerlab root node: $(curl -s "localhost:8090/api/node?id=$ID&path=" | head -c 160)"
kill $LAB
echo "CHECK done"

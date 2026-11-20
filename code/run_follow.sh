#!/bin/bash
cd "$(dirname "$0")"
run_one() { g=$1; e=$2; n=$3; out="../results/followup_${g}_${e}.csv"; [ -f "$out" ] && exit 0
  python3 followup_experiments.py $g $e $n > ../results/log_fu_${g}_${e}.txt 2>&1; }
export -f run_one
printf "opstats 0 20\nopstats 1 20\nopstats 3 20\npgwo 0 20\npgwo 1 20\npgwo 3 20\nexactloop 0 30\nexactloop 1 30\nexactloop 3 30\n" | xargs -P2 -L1 bash -c 'run_one $0 $1 $2'

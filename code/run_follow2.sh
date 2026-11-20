#!/bin/bash
cd "$(dirname "$0")"
run_one() { e=$1; a=$2; out="../results/followup_exactloop_${e}_${a}.csv"; [ -f "$out" ] && exit 0
  python3 followup_experiments.py exactloop $e 30 $a > ../results/log_fu_exact_${e}_${a}.txt 2>&1; }
export -f run_one
for e in 0 1 3; do for a in GWO MSIPSO CD-MSIPSO; do echo "$e $a"; done; done | xargs -P2 -L1 bash -c 'run_one $0 $1'

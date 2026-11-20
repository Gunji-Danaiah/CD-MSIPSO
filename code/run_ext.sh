#!/bin/bash
# extended confirmatory runs 60..199 for the key arms (resumable: skips finished jobs)
cd "$(dirname "$0")"
run_one() { e=$1; a=$2; out="../results/revision_ext2_${e}_${a}.csv"; [ -f "$out" ] && exit 0
  python3 run_revision_experiments.py $e 60 200 ext2_${e}_${a} $a > ../results/log_ext2_${e}_${a}.txt 2>&1; }
export -f run_one
cat ../results/jobs_ext.txt | xargs -P2 -L1 bash -c 'run_one $0 $1'

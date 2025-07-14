#!/usr/bin/env bash
set -euo pipefail

Ns=(100 140 180 220 260 300 340)
taus=(0.5 0.6)

for N in "${Ns[@]}"; do
  for tau in "${taus[@]}"; do
    echo "Running experiment with N=${N}, tau=${tau}"
    python test.py \
      --N "$N" \
      --tau "$tau"
      
    echo "-> finished N=${N}, tau=${tau}"
    echo
  done
done

python ../result/result.py
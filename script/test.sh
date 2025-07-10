#!/usr/bin/env bash
set -euo pipefail

Ns=(20 40 60)
taus=(0.5 0.6 0.7 0.8)

for N in "${Ns[@]}"; do
  for tau in "${taus[@]}"; do
    echo "Running experiment with N=${N}, tau=${tau}"
    python test.py \
      --N "$N" \
      --tau "$tau" \
      
    echo "-> finished N=${N}, tau=${tau}"
    echo
  done
done
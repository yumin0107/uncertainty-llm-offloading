#!/usr/bin/env bash
set -euo pipefail

Ns=(80 90 100 110 120)
taus=(0.6)

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
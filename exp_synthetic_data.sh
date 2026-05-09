#!/usr/bin/env bash

mkdir -p results/synthetic_data
mkdir -p logs/synthetic_data

METHODS=(
  epsilon_diagnosis
  baro
  circa
  score_ordering
  smooth_traversal
  traversal
  cholesky
  counterfactual
  rcd
  rcg_0
  stable_rca
)


for METHOD in "${METHODS[@]}"; do
  echo "=========================================="
  echo "Running method: ${METHOD}"
  echo "=========================================="

  python main.py \
    --experiment_mode synthetic_data \
    --methods "${METHOD}" \
    --number_trials 20 \
    --batch_mode \
    --aggregate_method mean \
    --results_path "results/synthetic_data/results_synthetic_data_${METHOD}.npy" \
    > "logs/synthetic_data/results_synthetic_data_${METHOD}.log" 2>&1

  if [ $? -eq 0 ]; then
    echo "Finished method: ${METHOD}"
  else
    echo "Failed method: ${METHOD}. Check logs/synthetic_data/results_synthetic_data_${METHOD}.log"
  fi

  echo ""
done
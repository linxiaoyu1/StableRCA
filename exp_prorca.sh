#!/usr/bin/env bash

set -u

NUMBER_TRIALS=5
RESULT_DIR="results/prorca"
LOG_DIR="logs/prorca"

mkdir -p "${RESULT_DIR}"
mkdir -p "${LOG_DIR}"

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
  echo "Running ProRCA method: ${METHOD}"
  echo "=========================================="

  python main.py \
    --experiment_mode pro_rca \
    --methods "${METHOD}" \
    --number_trials "${NUMBER_TRIALS}" \
    --batch_mode \
    --aggregate_method mean \
    --results_path "${RESULT_DIR}/results_prorca_${METHOD}.npy" \
    --verbose \
    > "${LOG_DIR}/results_prorca_${METHOD}.log" 2>&1

  if [ $? -eq 0 ]; then
    echo "Finished method: ${METHOD}"
  else
    echo "Failed method: ${METHOD}. Check ${LOG_DIR}/results_prorca_${METHOD}.log"
  fi

  echo ""
done
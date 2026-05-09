#!/usr/bin/env bash

set -u

NUMBER_TRIALS=5
RESULT_DIR="results/sockshop"
LOG_DIR="logs/sockshop"

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
  echo "Running Sock-Shop method: ${METHOD}"
  echo "=========================================="

  python main.py \
    --experiment_mode sock_shop \
    --methods "${METHOD}" \
    --number_trials "${NUMBER_TRIALS}" \
    --use_all_sock_shop_anomaly_samples \
    --batch_mode \
    --aggregate_method mean \
    --results_path "${RESULT_DIR}/results_sockshop_${METHOD}.npy" \
    > "${LOG_DIR}/results_sockshop_${METHOD}.log" 2>&1

  if [ $? -eq 0 ]; then
    echo "Finished method: ${METHOD}"
  else
    echo "Failed method: ${METHOD}. Check ${LOG_DIR}/results_sockshop_${METHOD}.log"
  fi

  echo ""
done
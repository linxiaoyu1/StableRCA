#!/usr/bin/env bash

set -u

NUMBER_TRIALS=1
RCAEval_DATA_PATH="/home/xiaoyulin/data/RCAEval_v2"
RESULT_DIR="results/rcaeval"
LOG_DIR="logs/rcaeval"

mkdir -p "${RESULT_DIR}"
mkdir -p "${LOG_DIR}"

METHODS=(
  rcg_0
  traversal
  smooth_traversal
  circa
  counterfactual
  epsilon_diagnosis
  rcd
  score_ordering
  cholesky
)

for METHOD in "${METHODS[@]}"; do
  echo "=========================================="
  echo "Running RCAEval method: ${METHOD}"
  echo "=========================================="

  python main.py \
    --experiment_mode rcaeval \
    --methods "${METHOD}" \
    --batch_mode \
    --aggregate_method mean \
    --rcaeval_data_path "${RCAEval_DATA_PATH}" \
    --use_all_rcaeval_anomaly_samples \
    --number_trials "${NUMBER_TRIALS}" \
    --results_path "${RESULT_DIR}/results_rcaeval_${METHOD}.npy" \
    > "${LOG_DIR}/results_rcaeval_${METHOD}.log" 2>&1

  if [ $? -eq 0 ]; then
    echo "Finished method: ${METHOD}"
  else
    echo "Failed method: ${METHOD}. Check ${LOG_DIR}/results_rcaeval_${METHOD}.log"
  fi

  echo ""
done
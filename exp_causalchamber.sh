#!/usr/bin/env bash

set -u

NUMBER_TRIALS=5
RESULT_DIR="results/causalchamber"
LOG_DIR="logs/causalchamber"

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
  echo "Running CausalChamber method: ${METHOD}"
  echo "=========================================="

  python main.py \
    --experiment_mode causal_chamber \
    --methods "${METHOD}" \
    --number_trials "${NUMBER_TRIALS}" \
    --batch_mode \
    --aggregate_method mean \
    --selection_weight_ratio 0.1 \
    --results_path "${RESULT_DIR}/results_causalchamber_${METHOD}.npy" \
    > "${LOG_DIR}/results_causalchamber_${METHOD}.log" 2>&1

  if [ $? -eq 0 ]; then
    echo "Finished method: ${METHOD}"
  else
    echo "Failed method: ${METHOD}. Check ${LOG_DIR}/results_causalchamber_${METHOD}.log"
  fi

  echo ""
done
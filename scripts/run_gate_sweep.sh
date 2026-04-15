#!/usr/bin/env bash
set -euo pipefail

# Example:
# CODE_ROOT=/workspace/interaction \
# DIGIR_ROOT=/workspace/DIGIR \
# DATA_ROOT=/data/interaction \
# RUNS_ROOT=/data/interaction_runs \
# GPUS=8 \
# bash scripts/run_gate_sweep.sh

CODE_ROOT="${CODE_ROOT:-/path/to/interaction}"
DIGIR_ROOT="${DIGIR_ROOT:-/path/to/DIGIR}"
DATA_ROOT="${DATA_ROOT:-/path/to/interaction_data}"
RUNS_ROOT="${RUNS_ROOT:-/path/to/interaction_runs}"
DATA_FILE="${DATA_FILE:-interaction_digir_all_12loc_h8_f12.pkl}"
GPUS="${GPUS:-8}"

RATIOS=(0.0 0.2 0.4 0.6 0.8 1.0)

cd "${CODE_ROOT}"

for r in "${RATIOS[@]}"; do
  echo "=== Running gate_fixed_ratio=${r} ==="
  torchrun --standalone --nproc_per_node="${GPUS}" train_digir_full.py \
    --digir_root "${DIGIR_ROOT}" \
    --data_root "${DATA_ROOT}" \
    --save_root "${RUNS_ROOT}" \
    --data "${DATA_FILE}" \
    --save "gate_ratio_${r}.pt" \
    --coord_frame per_agent \
    --batch_by_location \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --train_subset 5000 \
    --eval_batches 0 \
    --k 5 \
    --seed 42 \
    --lambda_rule 1e-3 \
    --map_margin 3.0 \
    --gate_fixed_ratio "${r}" \
    --log_gate_stats
done

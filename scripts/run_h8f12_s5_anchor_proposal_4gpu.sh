#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/bitwxy/interaction}
RUN_DIR=${RUN_DIR:-/data/sdb/bitwxy/interaction_runs/homotopy_official_mamba_geo_kg}
DATA_ROOT=${DATA_ROOT:-/data/sdb/bitwxy/interaction_data}
GPU_LIST=${GPU_LIST:-4,5,6,7}
NPROC=${NPROC:-4}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-12}
MASTER_PORT=${MASTER_PORT:-29660}
TORCHRUN=${TORCHRUN:-/home/bitwxy/miniconda3/envs/wxy/bin/torchrun}

NAME=${NAME:-anchor_S5_h8_f12_s5_r24_cv02_graph001_lr5e5_bs${BATCH_SIZE}_${NPROC}gpu}
LOG_PATH="$RUN_DIR/${NAME}.log"

mkdir -p "$RUN_DIR"
cd "$REPO_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $NAME"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] log=$LOG_PATH"

CUDA_VISIBLE_DEVICES="$GPU_LIST" "$TORCHRUN" \
  --standalone \
  --master_port "$MASTER_PORT" \
  --nproc_per_node="$NPROC" \
  train_digir_full_goal_cascade_homotopy_official_mamba_geo_kg.py \
  --data_root "$DATA_ROOT" \
  --save_root "$RUN_DIR" \
  --data interaction_digir_all_12loc_h8_f12_s5.pkl \
  --save "${NAME}.pt" \
  --save_latest "${NAME}_latest.pt" \
  --motion_features xyhs \
  --coord_frame per_agent \
  --batch_by_location \
  --epochs 60 \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --train_subset 999999 \
  --eval_batches 0 \
  --eval_every 4 \
  --k 6 \
  --route_modes 24 \
  --enable_anchor_proposals \
  --anchor_traj_weight 0.35 \
  --anchor_score_weight 0.7 \
  --seed 42 \
  --lambda_rule 0 \
  --map_margin 3.0 \
  --geo_corridor_embed_weight 1.0 \
  --geo_corridor_dist_weight 2.0 \
  --geo_path_goal_weight 0.2 \
  --ablate_gate none \
  --log_gate_stats \
  --lambda_interaction_graph 0.01 \
  --interaction_dist_threshold 2.5 \
  --cv_residual_weight 0.2 \
  --disable_interaction_spatial_modulation \
  > "$LOG_PATH" 2>&1

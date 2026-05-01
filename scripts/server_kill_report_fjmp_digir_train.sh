#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/bitwxy/miniconda3/envs/wxy/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/bitwxy/miniconda3/envs/wxy/bin/torchrun}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/bitwxy/interaction}"
FJMP_CODE_ROOT="${FJMP_CODE_ROOT:-/data/sdb/bitwxy/baselines/FJMP}"
RAW_ROOT="${RAW_ROOT:-/data/sdb/bitwxy/interaction_raw}"
DATA_ROOT="${DATA_ROOT:-/data/sdb/bitwxy/interaction_data}"
RUN_ROOT="${RUN_ROOT:-/data/sdb/bitwxy/interaction_runs/homotopy_official_mamba_geo_kg}"
FJMP_DIGIR_DATA="${FJMP_DIGIR_DATA:-${DATA_ROOT}/interaction_fjmp_digir_h10_f30.pkl}"

echo "== Kill current 1000-epoch DIGIR training =="
mapfile -t PIDS < <(ps -eo pid=,cmd= | awk '/train_digir/ && /--epochs[ =]1000/ && !/awk/ {print $1}')
if (( ${#PIDS[@]} == 0 )); then
  echo "No train_digir process with --epochs 1000 found."
else
  printf 'PIDs: %s\n' "${PIDS[*]}"
  for pid in "${PIDS[@]}"; do
    pgid="$(ps -o pgid= -p "$pid" | tr -d ' ' || true)"
    if [[ -n "${pgid}" ]]; then
      kill -- "-${pgid}" 2>/dev/null || kill "$pid" 2>/dev/null || true
    else
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 5
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      pgid="$(ps -o pgid= -p "$pid" | tr -d ' ' || true)"
      if [[ -n "${pgid}" ]]; then
        kill -9 -- "-${pgid}" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
      else
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi
  done
fi

echo
echo "== Best ADE currently visible in logs/checkpoints =="
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import re

root = Path("/data/sdb/bitwxy/interaction_runs")
patterns = [
    re.compile(r"best\s+minADE[=:]\s*([0-9.]+)", re.I),
    re.compile(r"best_ade[=:]\s*([0-9.]+)", re.I),
    re.compile(r"minADE_5:\s*([0-9.]+)", re.I),
]
hits = []
for p in root.rglob("*.log"):
    try:
        text = p.read_text(errors="ignore")
    except Exception:
        continue
    for pat in patterns:
        vals = [float(x) for x in pat.findall(text)]
        if vals:
            hits.append((min(vals), str(p), pat.pattern))
if hits:
    hits.sort(key=lambda x: x[0])
    print(f"best_log_ADE={hits[0][0]:.6f} file={hits[0][1]}")
else:
    print("No ADE found in logs.")

try:
    import torch
    ckpts = []
    for p in root.rglob("*.pt"):
        if "latest" not in p.name and "best" not in p.name:
            continue
        try:
            ckpt = torch.load(p, map_location="cpu")
            ade = ckpt.get("best_ade", None) if isinstance(ckpt, dict) else None
            if ade is None and isinstance(ckpt, dict) and isinstance(ckpt.get("metrics"), dict):
                ade = ckpt["metrics"].get("minADE_5")
            if ade is not None:
                ckpts.append((float(ade), str(p)))
        except Exception:
            pass
    if ckpts:
        ckpts.sort(key=lambda x: x[0])
        print(f"best_ckpt_ADE={ckpts[0][0]:.6f} file={ckpts[0][1]}")
    else:
        print("No ADE found in checkpoints.")
except Exception as exc:
    print(f"Checkpoint ADE scan skipped: {exc}")
PY

echo
echo "== Ensure FJMP preprocessing exists =="
if [[ ! -d "${RAW_ROOT}/preprocess/train_interaction" || -z "$(find "${RAW_ROOT}/preprocess/train_interaction" -name '*.p' -print -quit 2>/dev/null)" ]]; then
  echo "FJMP preprocess output missing; running official FJMP preprocessing first."
  cd "${FJMP_CODE_ROOT}"
  rm -rf dataset_INTERACTION
  ln -s "${RAW_ROOT}" dataset_INTERACTION
  "${PYTHON_BIN}" fjmp_preprocess_interaction.py
fi

echo "FJMP preprocess output found."

echo
echo "== Convert FJMP preprocess to DIGIR pkl =="
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/convert_fjmp_preprocess_to_digir.py \
  --fjmp_root "${RAW_ROOT}" \
  --output "${FJMP_DIGIR_DATA}" \
  --max_vehicles 10 \
  --max_kg_nodes 200

echo
echo "== Start DIGIR training on FJMP-converted data =="
mkdir -p "${RUN_ROOT}"
TRAIN_LOG="${RUN_ROOT}/homotopy_official_mamba_geo_kg_fjmp_h10_f30_bs4_k6.log"
nohup "${TORCHRUN_BIN}" --standalone --nproc_per_node=8 train_digir_full_goal_cascade_homotopy_official_mamba_geo_kg.py \
  --data_root "${DATA_ROOT}" \
  --save_root "${RUN_ROOT}" \
  --data "$(basename "${FJMP_DIGIR_DATA}")" \
  --save homotopy_official_mamba_geo_kg_fjmp_h10_f30_bs4_k6.pt \
  --save_latest homotopy_official_mamba_geo_kg_fjmp_h10_f30_bs4_k6_latest.pt \
  --motion_features xyhs \
  --coord_frame per_agent \
  --batch_by_location \
  --epochs 100 \
  --batch_size 4 \
  --num_workers 16 \
  --lr 9e-5 \
  --weight_decay 1e-4 \
  --train_subset 999999 \
  --eval_batches 0 \
  --eval_every 4 \
  --k 6 \
  --seed 42 \
  --lambda_rule 0 \
  --map_margin 3.0 \
  --geo_corridor_embed_weight 1.0 \
  --geo_corridor_dist_weight 2.0 \
  --geo_path_goal_weight 0.2 \
  --ablate_gate none \
  --log_gate_stats \
  > "${TRAIN_LOG}" 2>&1 &

echo $! > "${RUN_ROOT}/homotopy_official_mamba_geo_kg_fjmp_h10_f30_bs4_k6.pid"
echo "training_pid=$(cat "${RUN_ROOT}/homotopy_official_mamba_geo_kg_fjmp_h10_f30_bs4_k6.pid")"
echo "log=${TRAIN_LOG}"

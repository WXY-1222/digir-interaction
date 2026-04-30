#!/usr/bin/env bash
set -euo pipefail

# Reproduce the official FJMP INTERACTION preprocessing.
# Expected raw layout:
#   ${RAW_ROOT}/{train,val,test_multi-agent,maps}
# Outputs:
#   ${RAW_ROOT}/mapping_{train,val,test}.pkl
#   ${RAW_ROOT}/{train_reformatted,val_reformatted,test_reformatted}
#   ${RAW_ROOT}/preprocess/{train_interaction,val_interaction,test_interaction}

PYTHON_BIN="${PYTHON_BIN:-/home/bitwxy/miniconda3/envs/wxy/bin/python}"
FJMP_ROOT="${FJMP_ROOT:-/data/sdb/bitwxy/baselines/FJMP}"
RAW_ROOT="${RAW_ROOT:-/data/sdb/bitwxy/interaction_raw}"
LOG_FILE="${LOG_FILE:-${FJMP_ROOT}/fjmp_preprocess_interaction.log}"

cd "${FJMP_ROOT}"

if [[ ! -d "${RAW_ROOT}/train" || ! -d "${RAW_ROOT}/val" || ! -d "${RAW_ROOT}/test_multi-agent" || ! -d "${RAW_ROOT}/maps" ]]; then
  echo "Raw INTERACTION data is incomplete under ${RAW_ROOT}" >&2
  exit 1
fi

rm -rf dataset_INTERACTION
ln -s "${RAW_ROOT}" dataset_INTERACTION

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

p = Path("fjmp_utils.py")
s = p.read_text()
old = "import horovod.torch as hvd \n"
new = "try:\n    import horovod.torch as hvd\nexcept Exception:\n    hvd = None\n"
if old in s:
    p.write_text(s.replace(old, new))
PY

"${PYTHON_BIN}" - <<'PY'
import av2
import lanelet2
import torch
print("dependency check ok")
print("cuda available:", torch.cuda.is_available())
PY

echo "Starting FJMP INTERACTION preprocessing..."
echo "FJMP_ROOT=${FJMP_ROOT}"
echo "RAW_ROOT=${RAW_ROOT}"
echo "LOG_FILE=${LOG_FILE}"

nohup "${PYTHON_BIN}" fjmp_preprocess_interaction.py > "${LOG_FILE}" 2>&1 &
echo $! > "${FJMP_ROOT}/fjmp_preprocess_interaction.pid"
echo "pid=$(cat "${FJMP_ROOT}/fjmp_preprocess_interaction.pid")"
echo "tail -f ${LOG_FILE}"

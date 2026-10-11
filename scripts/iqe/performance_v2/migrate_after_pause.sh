#!/usr/bin/env bash
# Manual entrypoint only. It never signals/stops the current Base.
set -euo pipefail
cd /mnt/project/DriveVLA-M0-iqe-perf-v2-3855e06
IQE_PY=/root/miniconda3/envs/ddp/bin/python
IQE_RUNTIME=${1:-configs/iqe/performance_v2/combined_visual_bf16.yaml}
IQE_CERT=${2:-reports/iqe_perf_v2_3855e06/BEST_RUNTIME_QUALIFICATION.json}
IQE_UPDATES=${3:-1000}
IQE_OUTPUT=${4:-/mnt/project/iqe-formal-runtime-v2/continuation_$(date -u +%Y%m%dT%H%M%SZ)}
# A live-PID scan, not a historical PID. Inaccessible source host fails closed.
ssh training-vla-zt2 "$IQE_PY -" <<'PY'
from pathlib import Path
import json
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=str((p/'cwd').resolve())
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 if cwd=='/mnt/project/DriveVLA-M0-iqe-20261009' and ('iqe.cli train-base' in cmd or 'iqe.performance.run' in cmd):active.append({'pid':int(p.name),'cmd':cmd,'cwd':cwd})
assert not active,'ACTIVE_BASE_STILL_RUNNING; no duplicate continuation: '+json.dumps(active)
PY
IQE_CHECKPOINT=$($IQE_PY - <<'PY'
from pathlib import Path
from iqe.io import read_json,file_hash
from iqe.contracts import require
root=Path('/mnt/project/DriveVLA-M0-iqe-20261009/outputs/iqe/main/query_base')
receipts=[(read_json(p),p) for p in root.glob('step_*.pt.COMPLETE.json')]
require(bool(receipts),'no complete Base checkpoint')
r,p=max(receipts,key=lambda v:v[0]['optimizer_step']);checkpoint=Path(str(p).removesuffix('.COMPLETE.json'))
require(checkpoint.is_file() and file_hash(checkpoint)==r['checksum'],'latest receipt corrupt; no silent fallback')
print(checkpoint)
PY
)
# Run on a policy-authorized empty eight-GPU host. The runner rechecks resources,
# certificate, source dependencies, full state, capacity and duplicate output lock.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec "$IQE_PY" -u -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m iqe.performance.run --mode continuation \
  --config configs/iqe/performance_v2/source.yaml --runtime "$IQE_RUNTIME" \
  --qualification "$IQE_CERT" --checkpoint "$IQE_CHECKPOINT" \
  --output "$IQE_OUTPUT" --updates "$IQE_UPDATES" --warmup 10 --checkpoint-every 1000

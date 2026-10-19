"""Read-only revalidation of the accepted smoke and its saved evidence hashes."""
import hashlib
import json
import math
from pathlib import Path

root=Path(__file__).resolve().parent
run=root/json.loads((root/'LATEST.json').read_text())['run']
receipt=json.loads((run/'VALIDATION.json').read_text())
config=json.loads((run/'CONFIG.json').read_text())
steps=[json.loads(x) for x in (run/'steps.jsonl').open()]
rollouts=[json.loads(x) for x in (run/'rollouts.jsonl').open()]
train={json.loads(x)['token'] for x in (root.parent/'splits/log_reason_90_10_v1/train/TRAINING_PAIRS.jsonl').open()}
for name,digest in receipt['evidence_sha256'].items():
    assert hashlib.sha256((run/name).read_bytes()).hexdigest()==digest,name
for name,digest in config['code_hashes'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
assert len(steps)==receipt['updates']==2 and len(rollouts)==receipt['rollouts']==12
assert {r['token'] for r in rollouts} <= train
assert len({(r['group'],r['candidate']) for r in rollouts})==len(rollouts)
for row in rollouts:
    assert len(row['trajectory'])==8 and all(len(p)==3 and all(math.isfinite(v) for v in p) for p in row['trajectory'])
    assert all(0<=v<=1 for v in row['score']['metrics'].values())
for step in steps:
    assert all(math.isfinite(v) and v>0 for v in step['gradient_norms'].values())
    assert all(v>0 for v in step['max_parameter_changes'].values())
    assert step['ratio_min']==step['ratio_max']==1
    assert step['rewards']==[r['score']['metrics']['PDMS'] for r in rollouts if r['token']==step['token']]
assert any(r['policy_update']>0 for r in rollouts)
print('PASS: immutable evidence hashes, actual parameter-update receipts, training-only rollouts and reward correspondence.')

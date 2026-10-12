"""Retain one verified best G1 inference state using the frozen native loader."""
from pathlib import Path
import hashlib
import json
import os
import resource
import sys
import time

ROOT = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')
OUT = ROOT/'navtest_recovery_20261012T1341Z'
FROZEN = Path('/mnt/project/VLA-Drive-structured-world-fgtr-io-formal-frozen-20261009')
sys.path.insert(0, str(FROZEN))
from tools.structured_world.infer_checkpoint import load_model
from tools.structured_world.build_cache import atomic_json, digest
import torch

start = time.monotonic()
torch.set_num_threads(2)
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
pin = json.loads((OUT/'G1_60K_BEST_NATIVE_PIN.json').read_text())
dest, tag = Path(pin['destination']), pin['tag']
disk = os.statvfs(dest)
assert disk.f_bavail*disk.f_frsize > 16*(1<<30), 'Insufficient space for one necessary best-model export'
assert not (dest/'model_fp32.pt').exists()
model, source, checkpoint = load_model(dest, tag, 'cpu')
assert all(p.device.type == 'cpu' for p in model.parameters())
tmp = dest/'model_fp32.tmp'
torch.save(model.state_dict(), tmp)
tmp.replace(dest/'model_fp32.pt')
(dest/'model_fp32.pt').chmod(0o444)
receipt = dict(training_identity=source['identity'], checkpoint=tag,
    checkpoint_files_sha256=checkpoint['file_sha256'], sha256=digest(dest/'model_fp32.pt'),
    frozen_evaluation_source='0c54f1c49e7ed2a76b22826b654d6c72625f5b6d',
    origin='Unchanged frozen load_model on CPU: verified native ZeRO FP32 masters and strict state loading',
    dtype='Native FP32 parameters and unchanged buffers', GPU_workers=0)
atomic_json(dest/'FP32_EXPORT_COMPLETE.json', receipt)
for name, value in pin['files'].items():
    p = dest/'checkpoints'/tag/name
    st = p.stat()
    assert (st.st_dev, st.st_ino, st.st_size) == (value['device'], value['inode'], value['bytes'])
for name in pin['files']:
    (dest/'checkpoints'/tag/name).unlink()
retained = dict(status='RETAINED_BEST_OBSERVED_G1_INFERENCE_STATE', source=pin['source'],
    target=str(dest/'model_fp32.pt'), sha256=receipt['sha256'],
    bytes=(dest/'model_fp32.pt').stat().st_size, contains_optimizer_state=False,
    PDMS=89.28227233694238, q0_PDMS=88.42849940326299,
    selection='Highest completed observed G1 final PDMS at seed42, not a multi-seed or significance claim',
    temporary_native_hardlinks_released=len(pin['files']), original_source_files_changed=False,
    regenerated_export=True, CPU_seconds_wall=time.monotonic()-start,
    peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, GPU_workers=0,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
atomic_json(dest/'RETENTION.json', retained)
atomic_json(OUT/'G1_60K_BEST_RETENTION.json', retained)
print(json.dumps(retained), flush=True)

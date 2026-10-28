"""Exact-owner sharing for the finite, read-only post-head diagnostic only."""
import json
from pathlib import Path
import subprocess
import time

from .contracts import fingerprint

MODULE = 'tools.qwen_world_postcot.diagnose_post_head'
NAMESPACE = 'world_post_head_readonly_v1'
LEASE_ROOT = Path('/mnt/project/runtime_post_head_diagnostic_workers')
TRAINERS = {'tools.qwen_world_postcot.train_il', 'tools.qwen_world_postcot.train_rl',
            'tools.qwen_world_postcot.train_scope_rl'}


def snapshot(uuid):
    rows=[]
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines():
        pid,device=line.split(',')
        if device.strip()!=uuid:continue
        root=Path('/proc')/pid.strip();argv=[v for v in (root/'cmdline').read_bytes().decode().split('\0') if v]
        module=argv[argv.index('-m')+1] if '-m' in argv else None
        rows.append(dict(pid=int(pid),start_ticks=int((root/'stat').read_text().rsplit(')',1)[1].split()[19]),
            module=module,argv_sha256=fingerprint(argv),
            world_training=module in TRAINERS and '--output' in argv and
                argv[argv.index('--output')+1].startswith('/mnt/project/qwen-world-postcot-20261025/')))
    return sorted(rows,key=lambda r:r['pid'])


def validate(spec,current,*,now=None):
    record=spec.get('diagnostic_training_cohost')
    if (spec.get('module')!=MODULE or spec.get('shared_lease_namespace')!=NAMESPACE
            or len(spec.get('gpu_uuids',[]))!=1 or not isinstance(record,dict)
            or record.get('uuid')!=spec['gpu_uuids'][0] or record.get('namespace')!=NAMESPACE
            or not record.get('no_eviction') or not record.get('training_unchanged')):
        raise ValueError('Only the fixed read-only diagnostic may use the typed cohost lease')
    if not 0 <= (time.time() if now is None else now)-record['at'] <= 300:
        raise ValueError('Expired training-cohost admission; resample, do not reuse stale owners')
    if (not current or current!=record['owners'] or any(not r['world_training'] for r in current)):
        raise ValueError('Exact known World-training occupants changed; do not share unknown work')
    return LEASE_ROOT

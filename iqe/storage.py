"""Capacity planning is read-only. No checkpoint retention or deletion is implicit."""
from pathlib import Path
import math
import shutil
from .io import atomic_json, file_hash, read_json, BlockedError
from .contracts import require


def capacity_plan(directory, checkpoint_bytes, save_every, final_step, current_step=0, safety_bytes=500*1024**3, *, free_bytes=None):
    require(checkpoint_bytes>0 and save_every>0 and final_step>=current_step>=0 and safety_bytes>=0,'invalid checkpoint capacity budget')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(directory).free if free_bytes is None else free_bytes
    saves=math.ceil(final_step/save_every)-current_step//save_every
    # Reserve one full extra temporary file for atomic writing.
    required=(saves+1)*checkpoint_bytes+safety_bytes
    return {'status':'READY' if free>=required else 'BLOCKED_STORAGE_CAPACITY',
            'checkpoint_bytes':checkpoint_bytes,'save_every_steps':save_every,'current_step':current_step,
            'final_step':final_step,'remaining_saves':saves,'planned_additional_checkpoint_bytes':saves*checkpoint_bytes,
            'atomic_write_headroom_bytes':checkpoint_bytes,'safety_margin_bytes':safety_bytes,'free_bytes':free,
            'required_free_bytes':required,'retention_dry_run':True,'deletion_performed':False,
            'protected':['current resume point','best','all existing scientific milestones','all user checkpoints']}


def require_capacity(report):
    if report['status']!='READY':
        raise BlockedError('BLOCKED_STORAGE_CAPACITY: no automatic checkpoint deletion or job launch')
    return report

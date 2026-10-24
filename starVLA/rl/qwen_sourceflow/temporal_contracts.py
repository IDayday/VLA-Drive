"""Independent identity/admission for temporal-DINO IL, never an old IL resume."""
import json
from pathlib import Path
import socket
import os
from .contracts import file_sha, fingerprint

ROOT=Path('/mnt/project/qwen-sourceflow-temporal-dino-20261024')
TASK='qwen_sourceflow_temporal_dino_20261024'
TEACHER_ID='bda28c73a00fa2d47df2720dd03007f09d006f34482e406b0418a2112ae6924e'
PARENT=Path('/mnt/project/qwen-sourceflow-native-20261018/runs/IL_seed42_v2/checkpoints/update060000')


def require_temporal_dispatch():
    policy=json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    host=socket.gethostname().removesuffix('-worker-0')
    if host not in policy['task_authorizations'][TASK]['allowed_hosts'] or host not in policy['allowed_hosts']:
        raise PermissionError('Unauthorized temporal-DINO task host: '+host)
    if policy.get('global_dispatch_prohibitions',{}).get(host,{}).get('new_task_dispatch_allowed') is False:
        raise PermissionError('Withdrawn temporal-DINO task host: '+host)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible:
        allowed=policy['task_authorizations'][TASK]['authorized_gpu_indices'][host]
        if any(not value.isdigit() or int(value) not in allowed for value in visible.split(',')):
            raise PermissionError('GPU range withdrawn from temporal-DINO task')
    return host


def temporal_sources():
    from .contracts import source_files
    return source_files()

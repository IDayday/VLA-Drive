"""A separate user-requested Test observation, never a DEV parent selection."""
import json
from pathlib import Path

from .contracts import file_sha, fingerprint

SCHEMA = 'qwen_world_compare_user_interim_NAVTEST_v1'


def validate_registration(path, meta, *, data_sha256, protocol_sha256):
    value = json.loads(Path(path).read_text())
    body = dict(value)
    if body.pop('sha256', None) != fingerprint(body):
        raise ValueError('Changed interim Test registration')
    if (value.get('schema') != SCHEMA or value.get('user_instruction') != '评测最近的ckpts'
            or value.get('seed') != 2101 or value.get('count') != 12146
            or value.get('inference') != 'single_candidate_cleanx_K10_shift5'
            or value.get('Test_observed') is not True
            or value.get('used_for_parent_selection') is not False
            or value.get('used_for_actor_update') is not False):
        raise ValueError('Interim Test must be explicitly registered without selection/update')
    parent = meta['IL_parent']
    expected = dict(architecture_id=meta['architecture_id'], model_sha256=meta['model_sha256'],
        native_checkpoint=parent['path'], native_model_sha256=parent['model_sha256'],
        native_files_sha256=parent['seal']['files_sha256'], updates=parent['seal']['updates'],
        export_sha256=file_sha(Path(value['export'])/'export.json'),
        data_sha256=data_sha256, protocol_sha256=protocol_sha256)
    if (meta.get('qualification') is not False or meta.get('RL_parent') is not None
            or parent['identity'].get('qualification') is not False
            or parent['identity'].get('architecture_id') != meta['architecture_id']
            or meta['architecture_id'] not in ('A','B','C','D')
            or any(value.get(k) != v for k,v in expected.items())):
        raise ValueError('Interim Test does not bind the pinned formal IL policy/data/protocol')
    return dict(registration_sha256=value['sha256'], updates=value['updates'],
                Test_observed=True, used_for_parent_selection=False, used_for_actor_update=False)

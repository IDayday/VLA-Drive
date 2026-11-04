"""Recognize a specifically audited frozen screening cohost, never signal it.

This does not broaden the task's host grant or the two-process limit. Command,
host, source bytes and measured memory must match the local admission receipt.
"""
import json
import os
from pathlib import Path
import socket
from starVLA.rl.qwen_world_compare.contracts import file_sha

RECEIPT=Path('/mnt/project/d46-aug-migration-20261104/operations/COHOST_ADMISSION.json')
CONDITION_RECEIPT=RECEIPT.with_name('COHOST_CONDITION_EDIT_ADMISSION.json')


def registered_readonly_owner(args,used_mib,*,receipt=None,host=None):
    if os.environ.get('WORLD_COMPARE_TASK')!='d46_aug_migration_20261104':return False
    if receipt is None:
        return any(registered_readonly_owner(args,used_mib,receipt=p,host=host)
                   for p in (RECEIPT,CONDITION_RECEIPT))
    try:
        value=json.loads(Path(receipt or RECEIPT).read_text())
        command=[x for x in args if x];host=host or socket.gethostname().removesuffix('-worker-0')
        schema=value.get('schema')
        if schema not in ('AUG_registered_frozen_cohost_v1','AUG_registered_frozen_condition_edit_v1') or value.get('may_signal_cohost')is not False:return False
        if not isinstance(used_mib,int) or not 0<used_mib<=value['maximum_observed_owner_mib']:return False
        if not any(row['host']==host and row['command']==command for row in value['commands']):return False
        if not all(file_sha(path)==sha for path,sha in value['source_files'].items()):return False
        auth=json.loads(Path(value['authorization']).read_text())
        if schema=='AUG_registered_frozen_condition_edit_v1':
            # The audited cohost only optimizes a per-scene input delta, with
            # assertions that every model parameter stays frozen. Exact argv,
            # source hashes, two-context ceiling and memory admission still apply.
            return host in auth['allowed_hosts'] and value.get('model_parameters_frozen')is True and '--base-source'in command
        return host in auth['allowed_hosts'] and '--screen-only'in command
    except(OSError,ValueError,KeyError,TypeError):return False

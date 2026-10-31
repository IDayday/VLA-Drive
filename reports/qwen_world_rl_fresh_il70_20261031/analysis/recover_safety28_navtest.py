"""One exact approved NAVTEST recovery; no native or live cadence state edits."""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

R=Path('/mnt/project/qwen-world-rl-optimization-20261030')
sys.path.insert(0,str(R/'analysis'))
import navtest_cadence_owner_v2 as C
LABEL='PPO_SAFETY_EP_u028000_NAVTEST'
EXPECTED='0a36d568437ccc38556eff885627c09b9546055a399e45aafdb3513868786017'
ORIGINAL_O=C.O
O=R/'owners/SAFETY28_EXPORT_RECOVERY'


def main():
    assert hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest()==EXPECTED
    O.mkdir(exist_ok=True)
    guard=(O/'recovery.lock').open('a')
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=C.read_json(ORIGINAL_O/'STATE.json')['evaluations'][LABEL]
    assert previous['status']=='FAILED_REQUIRES_REVIEW'
    assert previous['arm']=='PPO_SAFETY_EP' and previous['updates']==28000
    assert 'export-only' in previous['error'] and 'exit status 1' in previous['error']
    assert not (R/'evaluation_specs'/f'{LABEL}.json').exists()
    assert not (R/'jobs'/('EVAL_'+LABEL)).exists()
    export=C.D.TASK_STORAGE/'exports'/LABEL
    staging=export.with_name(export.name+'.building')
    quarantine=export.with_name(export.name+'.failed_export_20261031')
    assert not export.exists() and staging.is_dir() and not (staging/'COMPLETE').exists()
    assert not quarantine.exists()
    log=R/'exports'/LABEL/'export.log'
    original_error=log.read_text()
    assert 'PytorchStreamWriter failed' in original_error and 'FileExistsError' in original_error
    C.atomic_json(O/'FAILED_NODE_BEFORE_RECOVERY.json',previous)
    C.atomic_json(O/'FAILED_EXPORT_BEFORE_RECOVERY.json',dict(
        label=LABEL,export_log=str(log),export_log_sha256=C.sha(log),
        staging=str(staging),files=[dict(path=str(p.relative_to(staging)),bytes=p.stat().st_size)for p in staging.rglob('*')if p.is_file()],
        quarantined_to=str(quarantine),native_checkpoint=previous['checkpoint'],
        native_complete_sha256=previous['checkpoint_complete_sha256'],scientific_failure=False,
        training_changed=False,native_signals=0,successful_scenes_to_repeat=0))
    os.rename(staging,quarantine)
    C.O=O  # Independent administrative state; the live cadence is read-only.
    node=copy.deepcopy(previous)
    node.update(status='READY',prior_failure=previous['error'],recovery_reason='CPU export write failure; unpublished staging retained atomically')
    C.atomic_json(O/'STATE.json',dict(schema=C.SCHEMA,status='READY_ONE_EXACT_LABEL',evaluations={LABEL:node},events=[],results=[]))
    owner=C.CadenceOwner()
    C.atomic_json(O/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),source_file=str(Path(__file__).resolve()),source_sha256=C.sha(__file__),started_at=C.now(),role='CPU single approved28k export/evaluation recovery; no native actions',reused_source=C.__file__,reused_source_sha256=EXPECTED))
    try:
        owner.require_allowed(LABEL)
        # Share the original cadence export mutex during the one recovery.
        with C.D.locked(ORIGINAL_O/'CPU_EXPORT.lock'):
            owner.evaluate(LABEL)
        result=owner.state['evaluations'][LABEL]
        owner.state['status']='COMPLETE_ONE_LABEL_RECOVERY' if result['status']=='COMPLETE' else 'RECOVERY_REQUIRES_REVIEW'
        owner.flush()
        C.atomic_json(O/'RESULT.json',dict(status=owner.state['status'],at=C.now(),label=LABEL,result=result,native_changed=False,original_cadence_state_written=False))
        return 0 if result['status']=='COMPLETE' else 2
    finally:
        owner.pool.shutdown(wait=False)
        owner.events.close()


if __name__=='__main__':raise SystemExit(main())

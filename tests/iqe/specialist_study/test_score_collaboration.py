import multiprocessing
from pathlib import Path
import time

import pytest

from iqe.io import atomic_json, file_hash, read_json, ContractError
from iqe.internal_baseline.score_worker import score_item
from iqe.specialist_study.score_assist import compatible_job, pending_items


def _backend(request, response):
    counter = Path(response).parent / 'backend_calls.txt'
    with open(counter, 'a') as f:
        f.write('call\n')
    time.sleep(.05)
    atomic_json(response, {'score_valid': True, 'score': .75})


def _call(job, item):
    assert score_item(job, item, _backend)


def item_fixture(tmp_path):
    context = tmp_path / 'context.bin'; context.write_bytes(b'context')
    trajectory = tmp_path / 'trajectory.bin'; trajectory.write_bytes(b'trajectory')
    return {'response': str(tmp_path / 'response.json'), 'request': {
        'scene': {'scene_id': 'real_fixture_label', 'metric_context_ref': str(context),
                  'metric_context_hash': file_hash(context)},
        'candidate': {'trajectory_ref': str(trajectory), 'trajectory_hash': file_hash(trajectory)}}}


def test_concurrent_record_lock_runs_backend_once(tmp_path):
    item = item_fixture(tmp_path); job = {'attempts_per_invocation': 2}
    ctx = multiprocessing.get_context('fork')
    workers = [ctx.Process(target=_call, args=(job, item)) for _ in range(4)]
    for p in workers: p.start()
    for p in workers:
        p.join(10)
        assert p.exitcode == 0
    assert (tmp_path / 'backend_calls.txt').read_text().splitlines() == ['call']
    assert len(list((tmp_path / 'attempts/response').glob('attempt_*.json'))) == 1


def test_coordinator_rejects_stale_completed_trajectory(tmp_path):
    item = item_fixture(tmp_path); job = {'attempts_per_invocation': 2}
    assert score_item(job, item, _backend)
    Path(item['request']['candidate']['trajectory_ref']).write_bytes(b'changed')
    with pytest.raises(ContractError, match='trajectory changed'):
        score_item(job, item, _backend)


def test_helper_excludes_legacy_and_corrupt_protocol(tmp_path):
    worker = tmp_path / 'worker.py'; worker.write_text('unchanged backend')
    job = tmp_path / 'JOB.json'; atomic_json(job, {'attempts_per_invocation': 2, 'items': []})
    assert compatible_job(job, worker) is None
    marker = {'schema': 'record_locked_official_score_v1', 'job_hash': file_hash(job),
              'worker_code_hash': file_hash(worker)}
    atomic_json(tmp_path / 'COOPERATIVE_PROTOCOL.json', marker)
    assert compatible_job(job, worker)['items'] == []
    worker.write_text('changed backend')
    with pytest.raises(ContractError, match='worker code changed'):
        compatible_job(job, worker)


def test_helper_partition_preserves_all_records_without_duplication(tmp_path):
    items = [{'response': str(tmp_path / f'{i}.json')} for i in range(17)]
    atomic_json(tmp_path / '5.COMPLETE.json', {'status': 'COMPLETE'})
    all_items = [x for i in range(4) for x in pending_items({'items': items}, i, 4)]
    assert len(all_items) == 16
    assert {x['response'] for x in all_items} == {x['response'] for i, x in enumerate(items) if i != 5}

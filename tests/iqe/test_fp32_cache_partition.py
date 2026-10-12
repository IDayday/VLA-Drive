"""Storage orchestration tests; no claim of new real-image qualification."""
import importlib.util
from pathlib import Path
import sys

import pytest

from iqe.io import ContractError, atomic_json, digest, file_hash

path = Path(__file__).resolve().parents[2] / 'scripts/iqe/fp32_cache/build_phase.py'
spec = importlib.util.spec_from_file_location('iqe_cache_phase', path)
phase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase)


def fixture(tmp_path):
    rows = [{'scene_id': str(i), 'role': 'incremental_fit', 'observation_hash': str(i)} for i in range(20)]
    full = {'schema': 'iqe_fp32_cache_plan_v1', 'scope': 'full', 'fit_count': 20,
            'requested_scenes': 20, 'rows': rows, 'records_per_shard': 4, 'fit_sha256': 'full',
            'checkpoint_retention_budget': {'parent_bytes': 10, 'remaining_count': 60},
            'seed': 42, 'global_batch': 32, 'max_optimizer_steps': 100000}
    full['plan_hash'] = digest(full)
    p = tmp_path / 'full.json'
    atomic_json(p, full, immutable=True)
    reuse = {'17': rows[17], '19': rows[19]}
    return full, reuse, p


def test_partition_preserves_original_training_and_reuse(tmp_path):
    full, reuse, p = fixture(tmp_path)
    a = phase.partition(full, reuse, target_bytes=1300, record_bytes=100, full_path=p, full_checksum=file_hash(p))
    b = phase.partition(full, reuse, target_bytes=1300, record_bytes=100, full_path=p, full_checksum=file_hash(p))
    assert a == b
    assert a['fit_count'] == 20 and a['requested_scenes'] == 10
    assert [r['scene_id'] for r in a['rows']] == ['0', '1', '2', '3', '4', '5', '6', '7', '17', '19']
    assert a['checkpoint_retention_budget'] == full['checkpoint_retention_budget']
    assert a['phase']['missing_from_full'] == 10
    assert not a['phase']['full_population_complete']
    assert phase.validate_partition(a) == full
    assert full['requested_scenes'] == len(full['rows']) == 20


@pytest.mark.parametrize('bad', ['foreign_id', 'role', 'observation'])
def test_foreign_reuse_rejected(tmp_path, bad):
    full, reuse, p = fixture(tmp_path)
    if bad == 'foreign_id': reuse = {'wrong': reuse['17']}
    elif bad == 'role': reuse = {'17': {**reuse['17'], 'role': 'final_test'}}
    else: reuse = {'17': {**reuse['17'], 'observation_hash': 'changed'}}
    with pytest.raises(ContractError):
        phase.partition(full, reuse, target_bytes=1300, record_bytes=100, full_path=p, full_checksum=file_hash(p))


def test_phase_cannot_relax_reserve_or_claim_full(tmp_path):
    full, reuse, p = fixture(tmp_path)
    a = phase.partition(full, reuse, target_bytes=1300, record_bytes=100, full_path=p, full_checksum=file_hash(p))
    a['phase']['full_population_complete'] = True
    with pytest.raises(ContractError): phase.validate_partition(a)
    a['phase']['full_population_complete'] = False
    a['checkpoint_retention_budget'] = {'parent_bytes': 10, 'remaining_count': 0}
    with pytest.raises(ContractError): phase.validate_partition(a)


def test_original_manifest_change_rejected(tmp_path):
    full, reuse, p = fixture(tmp_path)
    a = phase.partition(full, reuse, target_bytes=1300, record_bytes=100, full_path=p, full_checksum=file_hash(p))
    p.write_text('{}')
    with pytest.raises(ContractError): phase.validate_partition(a)


def test_native_partial_worker_exit_is_not_full_success():
    assert phase.result_status([3, 3, 0], True, 10, 20) == 'PHASE_COMPLETE_FULL_POPULATION_PENDING'
    assert phase.result_status([3, 3, 0], True, 20, 20) == 'FULL_POPULATION_COMPLETE'
    assert phase.result_status([0, 0], False, 20, 20) == 'PARTIAL_OR_FAILED'
    assert phase.result_status([2, 0], True, 20, 20) == 'PARTIAL_OR_FAILED'


def test_too_small_budget_does_not_create_empty_phase(tmp_path):
    full, reuse, p = fixture(tmp_path)
    with pytest.raises(ContractError):
        phase.partition(full, reuse, target_bytes=500, record_bytes=100, full_path=p, full_checksum=file_hash(p))


def test_multiple_hosts_have_disjoint_complete_assignment(tmp_path):
    full, reuse, p = fixture(tmp_path)
    a = phase.partition(full, reuse, target_bytes=1900, record_bytes=100, full_path=p, full_checksum=file_hash(p))
    parts = phase.split_partitions(a, reuse, 2)
    left, right = ({r['scene_id'] for r in part['rows']} for part in parts)
    assert not left & right
    assert left | right == {r['scene_id'] for r in a['rows']}
    assert set(reuse) <= left and not set(reuse) & right
    for part in parts:
        assert phase.validate_partition(part) == full
        assert part['phase']['aggregate_plan_hash'] == a['plan_hash']
        assert not part['phase']['full_population_complete']


def test_duplicate_phase_rows_rejected(tmp_path):
    full, reuse, p = fixture(tmp_path)
    a = phase.partition(full, reuse, target_bytes=1300, record_bytes=100, full_path=p, full_checksum=file_hash(p))
    a['rows'][1] = a['rows'][0]
    with pytest.raises(ContractError): phase.validate_partition(a)


def queue_module(monkeypatch):
    monkeypatch.setitem(sys.modules, 'build_phase', phase)
    spec = importlib.util.spec_from_file_location('iqe_cache_queue', path.with_name('queue_phase.py'))
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    return queue


def test_dispatch_never_uses_occupied_or_unobservable_gpu(monkeypatch):
    queue = queue_module(monkeypatch)
    snap = {'gpus': [{'memory_mib': 0, 'utilization': 0} for _ in range(8)], 'owners': []}
    assert queue.idle(snap)
    snap['owners'] = ['123, GPU-other']
    assert not queue.idle(snap)
    snap['owners'] = []
    snap['gpus'][0]['memory_mib'] = 39400  # owner may be in another PID namespace
    assert not queue.idle(snap)
    snap['gpus'][0]['memory_mib'] = 0
    snap['gpus'][0]['utilization'] = 80
    assert not queue.idle(snap)


def test_cohort_reserves_all_hosts_and_subtracts_only_completed_bytes(tmp_path, monkeypatch):
    queue = queue_module(monkeypatch)
    full, reuse, p = fixture(tmp_path)
    jobs = [{'root': str(tmp_path/'a')}, {'root': str(tmp_path/'b')}]
    for j in jobs: Path(j['root']).mkdir()
    first = Path(jobs[0]['root'])/'shard_0.bin'; first.write_bytes(b'x'*100)
    atomic_json(str(first)+'.COMPLETE.json', {})
    (Path(jobs[1]['root'])/'shard_1.bin').write_bytes(b'x'*300)  # incomplete is not reusable
    (Path(jobs[0]['root'])/'reuse_old.bin').write_bytes(b'x'*200)  # separately accounted hardlink
    monkeypatch.setattr(queue, 'capacity_check', lambda root, need, **kw: {'required_bytes': need, **kw})
    result = queue.cohort_capacity(full, {'representative_record_max_bytes': 100}, 200, tmp_path, jobs)
    assert result['required_bytes'] == 2100-200-100
    assert result['temporary_bytes'] == 16*2*1024**3
    assert result['reserve_bytes'] == 10*60+(180+200)*1024**3


def test_controller_failure_signals_only_owned_children(tmp_path):
    class Child:
        def __init__(self, done): self.done, self.signaled, self.pid = done, False, 99
        def poll(self): return 0 if self.done else None
        def terminate(self): self.signaled = True
        def wait(self, timeout): self.done = True
    live, done = Child(False), Child(True)
    assert phase.pause_owned_workers(tmp_path, [(live, None, []), (done, None, [])], 'failure') == []
    assert live.signaled and not done.signaled
    from iqe.io import read_json
    assert read_json(tmp_path/'CONTROL.json')['pause']

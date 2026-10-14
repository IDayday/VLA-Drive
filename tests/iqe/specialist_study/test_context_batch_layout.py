import pytest
from iqe.io import digest, ContractError
from iqe.specialist_study.context_batch_layout import (
    original_batch_map, initial_shard_batch_map, OriginalBatchReplay, within_tolerance,
)


def plan(batches):
    p = {'batch_composition_unchanged': True, 'cache_batch': 4,
         'tasks': [{'task_id': '0', 'batches': [{'ids': b} for b in batches]}]}
    p['identity'] = digest(p)
    return p


def test_original_tail_and_order_retained():
    batches = original_batch_map(plan([['a', 'b', 'c', 'd'], ['e', 'f']]))
    assert batches['f'] == ('e', 'f')
    assert batches['b'] == ('a', 'b', 'c', 'd')
    assert 'unknown' not in batches


def test_bad_manifest_and_duplicate_are_rejected():
    p = plan([['a', 'b']]); p['tasks'][0]['batches'][0]['ids'].reverse()
    with pytest.raises(ContractError): original_batch_map(p)
    with pytest.raises(ContractError): original_batch_map(plan([['a'], ['a']]))


def test_tolerance_not_relaxed_for_batch_shape():
    e = {'raw_max_abs': 1e-7, 'world_max_abs': 0., 'action_max_abs': 0.}
    assert within_tolerance(e)
    assert not within_tolerance(e | {'world_max_abs': .00002})
    assert not within_tolerance(e | {'action_max_abs': float('nan')})


def legacy_receipts(ids, shards=2):
    launches = [{'shard': i, 'command': ['python', '-m', 'iqe.internal_baseline.cli',
                'cache-features', '--config', 'source.yaml', '--shard', str(i), '--shards', str(shards)]}
                for i in range(shards)]
    receipts = {i: {'status': 'COMPLETE', 'new': len(ids[i::shards]),
                    'scenes': len(ids[i::shards])} for i in range(shards)}
    return launches, receipts


def test_inherited_first_extraction_replays_stride_before_grouping_and_tail():
    ids = list('abcdefghijk')
    launches, receipts = legacy_receipts(ids)
    batches = initial_shard_batch_map(ids, 4, launches, receipts)
    assert batches['a'] == ('a', 'c', 'e', 'g')
    assert batches['j'] == ('j',)
    assert batches['k'] == ('i', 'k')
    assert set(batches) == set(ids)


@pytest.mark.parametrize('failure', ['partial', 'resumed', 'missing', 'duplicate', 'mixed', 'sample_limit'])
def test_inherited_layout_rejects_unattested_original_composition(failure):
    ids = list('abcdefghijk'); launches, receipts = legacy_receipts(ids)
    if failure == 'partial': receipts[0]['status'] = 'RUNNING'
    elif failure == 'resumed': receipts[0]['new'] -= 1
    elif failure == 'missing': del receipts[1]
    elif failure == 'duplicate': launches.append(launches[0])
    elif failure == 'mixed': launches[1]['command'][-1] = '3'
    else: launches[0]['command'] += ['--max-samples', '4']
    with pytest.raises(ContractError): initial_shard_batch_map(ids, 4, launches, receipts)


def test_accessed_inherited_payload_is_verified_and_corruption_rejected(tmp_path):
    from iqe.io import atomic_json, file_hash
    layout = OriginalBatchReplay.__new__(OriginalBatchReplay)
    layout.batches = {'a': ('a',)}; layout.verified_reused = set()
    layout.reused_source = tmp_path/'source'; layout.reused_destination = tmp_path/'destination'
    for root in (layout.reused_source, layout.reused_destination):
        (root/'features').mkdir(parents=True)
        (root/'features/a.pt').write_bytes(b'original fp32 payload')
    path = layout.reused_source/'features/a.pt'
    receipt = {'scene_id': 'a', 'key': 'observation-contract-key', 'checksum': file_hash(path)}
    layout.reused_records = {'a': receipt | {'bytes': path.stat().st_size}}
    for root in (layout.reused_source, layout.reused_destination):
        atomic_json(root/'features/a.pt.COMPLETE.json', receipt)
    assert layout.batch_for('a') == ('a',)
    layout.verified_reused.clear()
    (layout.reused_destination/'features/a.pt').write_bytes(b'wrong but still a payload')
    with pytest.raises(ContractError): layout.batch_for('a')
    with pytest.raises(ContractError): layout.batch_for('unknown')

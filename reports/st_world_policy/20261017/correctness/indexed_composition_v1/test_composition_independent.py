"""Independent bounded CPU-only fixtures; no production assets or CUDA."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest
import torch

from test_indexed_composition import setup, exercise, descriptor, FakeBudget, FakeExecutor
from test_fixed_provider_training import sign
from tools.st_world_policy import qualify_indexed_provider as q
from starVLA.dataloader.st_world_indexed_provider import IndexedTeacherCacheReader


@pytest.mark.parametrize('mutation', ['zero_elements', 'one_element', 'contradictory_summary',
                                    'negative_requested', 'nan_read', 'bool_read'])
def test_resigned_incomplete_measurement_cannot_admit(tmp_path, mutation):
    _, plan = setup(tmp_path); report = exercise(plan)
    if mutation in ('zero_elements', 'one_element'):
        count = 0 if mutation == 'zero_elements' else 1
        comparisons = report['legacy_cached_reference'] + report['reference_reordered4'] + [
            item for rows in report['cases'].values() for row in rows for item in row['comparisons']]
        for comparison in comparisons: comparison['difference']['elements'] = count
    elif mutation == 'contradictory_summary':
        report['legacy_reference_summary']['bitwise_equal'] = False
        report['legacy_reference_summary']['max_abs'] = 1.0
    elif mutation == 'negative_requested':
        slots = report['compute_stats']['compute_image_slots']
        report['compute_stats']['requested_images'] = -1
        report['compute_stats']['padding_image_slots'] = slots + 1
    elif mutation == 'nan_read': report['process_rchar_bytes'] = float('nan')
    else: report['process_rchar_bytes'] = True
    # Canonical identity signing already rejects NaN. Leave this mutation
    # unsigned and require the actual consumer to reject it as well.
    if mutation != 'nan_read': sign(report)
    # Direct consumer validation isolates schema rejection from atomic_json's
    # serialization policy (which correctly refuses NaN in its own writer).
    d = {'requests':plan['requests'], 'request_index':plan['request_index'],
         'sources':[{'index':plan['source_index'],'cache_identity':plan['current_identity']}],
         'teacher_config':plan['teacher_config'], 'execution':plan['execution'],
         'limits':plan['limits'], 'provider':plan['provider']}
    with pytest.raises((ValueError,TypeError)):
        q.validate_receipt(plan, report, d)


@pytest.mark.parametrize('resource_name', ['time','rss','allocated','reads'])
def test_runtime_resource_guard_rejects_each_exceeded_budget(monkeypatch, resource_name):
    monkeypatch.setattr(q, '_rchar', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    budget = q.Budget({'max_seconds':900,'ram_peak_bytes':8*q.GiB,
        'gpu_peak_bytes':16*q.GiB,'max_logical_input_bytes':64*q.GiB,
        'max_image_forwards':8,'max_batch_calls':1})
    if resource_name == 'time': budget.start -= 901
    elif resource_name == 'rss':
        monkeypatch.setattr(q.resource,'getrusage',lambda _:SimpleNamespace(ru_maxrss=(8*q.GiB)//1024+1))
    elif resource_name == 'allocated':
        monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
        monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:16*q.GiB+1)
    else: budget.mmap_bytes = 64*q.GiB+1
    with pytest.raises((TimeoutError,MemoryError)): budget.check()


def test_single_last_tensor_element_difference_is_not_hidden(tmp_path):
    _, plan = setup(tmp_path); req, src = q.open_pair(plan)
    try:
        source = IndexedTeacherCacheReader(src, expected_identity=plan['current_identity'])
        original = source.read_image
        def altered(key):
            z, mask = original(key); z = z.clone()
            z[-1,-1,-1] = torch.nextafter(torch.tensor(0.),torch.tensor(1.))
            return z, mask
        source.read_image = altered
        result = q.qualify_layouts(plan, req, source, FakeExecutor(plan),
            rng=lambda:None, budget=FakeBudget())
        assert result['fixed_policy_summary']['bitwise_equal']
        assert result['legacy_reference_summary']['bitwise_equal'] is False
        assert all(x['difference']['elements'] == math.prod(plan['contract']['output_grid_hw'])*plan['contract']['channels']
                   for x in result['legacy_cached_reference'])
    finally: req.close(); src.close()


def test_nonconstant_miss_features_keep_key_and_slot_alignment(tmp_path):
    _, plan = setup(tmp_path,scene_count=2)
    values = {key:0. for key in plan['hit_keys']}
    values.update({key:float(index+1) for index,key in enumerate(plan['miss_keys'])})
    class KeyExecutor(FakeExecutor):
        def compute_batch(self, records, **kwargs):
            out = super().compute_batch(records, **kwargs)
            for index,row in enumerate(records): out.features[index].fill_(values[row['key']])
            return out
    req,src=q.open_pair(plan)
    try:
        result=q.qualify_layouts(plan,req,IndexedTeacherCacheReader(src,expected_identity=plan['current_identity']),
            KeyExecutor(plan),rng=lambda:None,budget=FakeBudget())
        assert result['fixed_policy_summary']['bitwise_equal'] and result['legacy_reference_summary']['bitwise_equal']
        assert result['all_masks_equal'] and result['sampler_equal']
        assert len({result['reference_digests'][k]['features'] for k in plan['miss_keys']}) == len(plan['miss_keys'])
    finally:req.close();src.close()

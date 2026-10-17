"""Additional receipt-only boundaries against the fixed consumer."""
from copy import deepcopy

import pytest

from test_indexed_composition import setup, exercise
from test_fixed_provider_training import sign
from tools.st_world_policy import qualify_indexed_provider as q


@pytest.mark.parametrize('mutation', ['coherent_extra_batch', 'cold_hit_miss_swap', 'bool_tensor_disk'])
def test_arithmetic_consistency_is_not_actual_coverage(tmp_path, mutation):
    _, plan = setup(tmp_path); result = exercise(plan)
    descriptor = {'requests':plan['requests'], 'request_index':plan['request_index'],
        'sources':[{'index':plan['source_index'],'cache_identity':plan['current_identity']}],
        'teacher_config':plan['teacher_config'], 'execution':plan['execution'],
        'limits':plan['limits'], 'provider':plan['provider']}
    q.validate_receipt(plan, result, descriptor)
    bad = deepcopy(result)
    if mutation == 'coherent_extra_batch':
        # All simple sums still agree, but the exact reference/provider schedule
        # cannot have executed this extra fully padded batch.
        bad['batch_calls'] += 1; bad['rng_forward_checks'] += 1
        bad['image_forwards'] += 8
        bad['compute_stats']['calls'] += 1
        bad['compute_stats']['compute_image_slots'] += 8
        bad['compute_stats']['padding_image_slots'] += 8
    elif mutation == 'cold_hit_miss_swap':
        counts = bad['cases']['cold'][0]['counts']
        assert counts['computed_images'] > 0
        counts['computed_images'] -= 1; counts['immutable_hits'] += 1
    else:
        bad['tensor_disk_bytes'] = False
    sign(bad)
    with pytest.raises(ValueError): q.validate_receipt(plan,bad,descriptor)

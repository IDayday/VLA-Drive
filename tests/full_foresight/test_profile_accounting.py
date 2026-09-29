from tools.full_foresight.summarize_profiles import concurrent_rate


def test_concurrent_cost_uses_actual_common_interval():
    first = [{'seconds': 2., 'ended_unix': t, 'batch_scenes': 32} for t in (2., 4., 6., 8.)]
    second = [{'seconds': 3., 'ended_unix': t, 'batch_scenes': 32} for t in (4., 7., 10.)]
    result = concurrent_rate([first, second])
    assert result['seconds'] == 7.
    assert result['completed_scenes_per_job'] == [96, 64]
    assert result['samples_per_s'] == 160 / 7
    shifted = [{**r, 'ended_unix': r['ended_unix'] + 20} for r in second]
    assert not concurrent_rate([first, shifted])['valid']

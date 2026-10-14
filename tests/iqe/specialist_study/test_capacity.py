from iqe.specialist_study.capacity import assess


def test_hidden_owner_is_not_free_even_when_memory_fits():
    x = assess('0, GPU0, 20599, 81920, 99', '6335, GPU0, 1282', 0)
    assert not x['ready']
    assert x['unexplained_mib'] == 19317
    assert 'UNIDENTIFIED_PHYSICAL_OCCUPANCY' in x['reasons']
    assert not x['process_signals']


def test_idle_device_and_small_known_owner():
    assert assess('1, GPU1, 1, 81920, 0', '', 1)['ready']
    assert assess('1, GPU1, 1024, 81920, 10', '42, GPU1, 1020', 1)['ready']


def test_busy_or_full_device_blocks_without_kill():
    assert not assess('1, GPU1, 78000, 81920, 0', '42, GPU1, 77999', 1)['ready']
    assert not assess('1, GPU1, 1024, 81920, 99', '42, GPU1, 1020', 1)['ready']

"""Independent CPU-only comparator review; every source is synthetic."""
import json
from pathlib import Path

import pytest
from PIL import Image

from tests.st_world.test_future_metadata_reference import built_fixture
from tools.st_world_policy import compare_future_metadata as reference
from tools.st_world_policy import prepare_future_metadata as future
from starVLA.dataloader.st_world_targets import atomic_json


@pytest.mark.parametrize('mutation', ['requested_time', 'frame_token', 'frame_timestamp',
                                      'missing_view', 'extra_field', 'numeric_type'])
def test_whole_scene_schema_changes_cannot_match(tmp_path, mutation):
    args, root, _, _ = built_fixture(tmp_path)
    manifest = json.loads((root/'TEACHER_REQUESTS.json').read_text())
    scene = manifest['scenes'][0]
    frame = scene['frames'][1]
    if mutation == 'requested_time': frame['requested_s'] = .6
    elif mutation == 'frame_token': frame['frame_token'] = 'wrong'
    elif mutation == 'frame_timestamp': frame['timestamp_us'] += 1
    elif mutation == 'missing_view': frame['views'][0] = None
    elif mutation == 'extra_field': frame['unregistered'] = True
    else: scene['timestamp_us'] = float(scene['timestamp_us'])
    manifest = future.signed({k:v for k,v in manifest.items() if k != 'identity'})
    atomic_json(root/'TEACHER_REQUESTS.json', manifest)
    done = json.loads((root/'COMPLETE.json').read_text())
    done['files']['TEACHER_REQUESTS.json'] = future.file_sha256(root/'TEACHER_REQUESTS.json')
    done['request_identity'] = manifest['identity']
    atomic_json(root/'COMPLETE.json', done)
    args.complete_sha256 = future.file_sha256(root/'COMPLETE.json')
    result = reference.compare(args)
    assert result['status'] == 'REFERENCE_MISMATCH'
    assert result['mismatch_count'] > 0
    assert all(row['field'].startswith('scenes[0]') for row in result['first_mismatches'])


@pytest.mark.parametrize('field,value', [('source_bytes_read', 4*future.GiB+1), ('elapsed_s', 450.01)])
def test_build_cost_over_half_rejects_before_data_reads(tmp_path, monkeypatch, field, value):
    args, root, _, _ = built_fixture(tmp_path)
    done = json.loads((root/'COMPLETE.json').read_text())
    done['cost'][field] = value
    atomic_json(root/'COMPLETE.json', done)
    args.complete_sha256 = future.file_sha256(root/'COMPLETE.json')
    original = future.BudgetedReader.__init__
    def restricted(self, path, budget):
        assert Path(path).suffix not in ('.png', '.jpg', '.pkl', '.pt')
        original(self, path, budget)
    monkeypatch.setattr(future.BudgetedReader, '__init__', restricted)
    with pytest.raises(ValueError, match='half'):
        reference.compare(args)


def test_comparator_cannot_write_decode_or_use_new_selector(tmp_path, monkeypatch):
    args, root, _, _ = built_fixture(tmp_path)
    before = {str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    def forbidden(*args, **kwargs): raise AssertionError('Forbidden reference dependency')
    monkeypatch.setattr(future, 'write_chunks', forbidden)
    monkeypatch.setattr(future.TargetLogLookup, 'scene', forbidden)
    monkeypatch.setattr(Image.Image, 'load', forbidden)
    import torch
    monkeypatch.setattr(torch, 'load', forbidden)
    result = reference.compare(args)
    after = {str(p.relative_to(tmp_path)):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert result['status'] == 'REFERENCE_MATCH' and before == after
    assert result['reference_code_objects_reused'] is True
    assert result['prior_failed_invocations_accounted'] is False
    assert result['outer_cumulative_invocation_ledger_required'] is True


def test_reference_code_object_keeps_legacy_defaults_and_private_globals(tmp_path):
    args, root, plan, _ = built_fixture(tmp_path)
    budget = future.PreparationBudget(root, plan['limits'])
    function = reference.reference_scene_function(budget)
    original = reference.legacy.build_target_scene
    assert function.__code__ is original.__code__
    assert function.__defaults__ == original.__defaults__
    assert function.__kwdefaults__ == original.__kwdefaults__
    assert function.__globals__ is not original.__globals__
    assert function.__globals__['image_reference'] is not original.__globals__['image_reference']
    assert function.__globals__['np'] is original.__globals__['np']
    assert function.__globals__['resolve_camera_path'] is original.__globals__['resolve_camera_path']

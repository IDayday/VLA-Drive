"""Independent CPU fixtures for bounded future metadata; no real data access."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
from PIL import Image

from tools.st_world_policy import prepare_future_metadata as f
from tools.st_world_policy.prepare_history import build_target_scene
from starVLA.dataloader.st_world_future_provenance import LogCoverage, validate_ledger
from starVLA.dataloader.st_world_history import VIEWS, file_sha256
from starVLA.dataloader.st_world_metadata_index import _atomic_rows
from starVLA.dataloader.st_world_schema import identity_hash
from starVLA.dataloader.st_world_targets import atomic_json
from tests.st_world.test_prepare_future_metadata import setup


def _resign(value):
    return f.signed({k: v for k, v in value.items() if k != 'identity'})


def _limits(**changes):
    result = dict(workers=1, read_mib_per_second=64., max_seconds=60.,
                  max_source_read_bytes=8*f.GiB, max_metadata_bytes=32*f.MiB,
                  max_rss_bytes=4*f.GiB, max_raw_log_bytes=128*f.MiB)
    result.update(changes)
    return result


def test_independent_random_frame_selection_reference_and_label_independence(tmp_path):
    rng = np.random.default_rng(731)
    primary = tmp_path/'primary'; primary.mkdir()
    fallback = tmp_path/'fallback'; fallback.mkdir()
    roots = [str(primary), str(fallback)]
    frames = []
    for i in range(30):
        # Include inside/outside exact 50 ms boundaries and per-view missingness.
        jitter = 0 if i == 0 else int(rng.choice([-50001, -50000, -49999, 0, 49999, 50000, 50001]))
        cams = {}
        for j, view in enumerate(VIEWS):
            if i and rng.random() < .18:
                continue
            name = f'{i}_{j}.png'; root = primary if (i+j)%2 else fallback
            Image.new('RGB', (32, 18), (i, j, 11)).save(root/name)
            cams[view] = {'data_path': name}
        frames.append(dict(token=f't{i}', timestamp=10_000_000+i*500_000+jitter, cams=cams,
                           gt_trajectory='UNREAD', future_actors='UNREAD'))
    lookup = f.TargetLogLookup(frames)
    budget = f.PreparationBudget(tmp_path, _limits())
    memo = f.ImageMemo(budget)
    checked = 0
    for frame in frames:
        if len(frame['cams']) != 3:
            continue
        expected = build_target_scene(frames, frame['token'], 'log', 'train', roots)
        actual = lookup.scene(frame['token'], 'log', 'train', roots, memo)
        assert actual == expected
        # Add/replace target-like dictionary members without changing legal RGB inputs.
        changed = deepcopy(frames)
        for row in changed:
            row['gt_trajectory'] = [999, float('nan')]
            row['future_actors'] = {'secret': object()}
        assert f.TargetLogLookup(changed).scene(frame['token'], 'log', 'train', roots, memo) == actual
        checked += 1
    assert checked >= 10
    assert budget.snapshot()['source_read_categories']['image_or_other'] > 0


@pytest.mark.parametrize('shift,found', [(-50001, False), (-50000, True), (50000, True), (50001, False)])
def test_tolerance_boundaries_and_early_future_frame(tmp_path, shift, found):
    Image.new('RGB', (16, 9)).save(tmp_path/'image.png')
    cams = {name: {'data_path': 'image.png'} for name in VIEWS}
    frames = [dict(token='past', timestamp=500000, cams=cams),
              dict(token='now', timestamp=1000000, cams=cams),
              dict(token='next', timestamp=1500000+shift, cams=cams)]
    memo = f.ImageMemo(f.PreparationBudget(tmp_path, _limits()))
    actual = f.TargetLogLookup(frames).scene('now', 'l', 'train', [str(tmp_path)], memo)
    assert actual == build_target_scene(frames, 'now', 'l', 'train', [str(tmp_path)])
    assert (actual['frames'][1]['frame_token'] == 'next') is found
    assert actual['frames'][0]['frame_token'] == 'now'
    assert all(row['frame_token'] != 'past' for row in actual['frames'])


def test_build_does_not_decode_ego_or_run_teacher(tmp_path, monkeypatch):
    args, run, _, _ = setup(tmp_path)
    import torch
    def forbidden(*a, **k):
        raise AssertionError('Training target or teacher must not be decoded')
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(Image.Image, 'load', forbidden)
    result = f.build(run)
    assert result['status'] == 'COMPLETE'
    assert result['ego_tensor_decoded'] is False
    assert result['teacher_extractions'] == result['tensor_disk_bytes'] == 0


@pytest.mark.parametrize('field', ['source_student_identity', 'source_index_sha256',
                                 'partition_sha256', 'selection_identity', 'registration_identity'])
def test_each_ledger_source_binding_is_effective(tmp_path, field):
    args, run, _, _ = setup(tmp_path); f.build(run)
    manifest = json.loads((Path(args.output)/'TEACHER_REQUESTS.json').read_text())
    manifest[field] = 'f'*64
    with pytest.raises(ValueError, match='source/registration'):
        validate_ledger(manifest)


@pytest.mark.parametrize('corruption', ['log_order', 'scene_count', 'token_order', 'split', 'semantic', 'duplicates'])
def test_ledger_body_coverage_not_just_path_or_pin(tmp_path, corruption):
    args, run, _, _ = setup(tmp_path); f.build(run)
    manifest = json.loads((Path(args.output)/'TEACHER_REQUESTS.json').read_text())
    coverage = LogCoverage()
    for scene in manifest['scenes']:
        coverage.add(scene['token'], scene['log'])
    ref = manifest['source_logs_ledger']; path = Path(ref['path'])
    body = json.loads(path.read_text())
    if corruption == 'log_order': body['logs'].reverse()
    elif corruption == 'scene_count': body['logs'][0]['scenes'] += 1
    elif corruption == 'token_order': body['logs'][0]['tokens_sha256'] = identity_hash(['other'])
    elif corruption == 'split': body['split'] = 'dev'
    elif corruption == 'semantic': body['logs'][0]['bytes'] += 1
    elif corruption == 'duplicates':
        encoded = path.read_text().replace('"split": "train"', '"split": "train", "split": "train"')
        path.write_text(encoded)
    if corruption != 'duplicates':
        body = body if corruption == 'semantic' else _resign(body)
        atomic_json(path, body)
        ref['identity'] = body['identity']
    ref['sha256'] = file_sha256(path)
    with pytest.raises(ValueError): validate_ledger(manifest, coverage=coverage.records())


def test_log_coverage_preserves_interleaved_log_and_token_order():
    coverage = LogCoverage()
    for token, log in [('c', 'l2'), ('a', 'l1'), ('b', 'l2'), ('d', 'l1')]:
        coverage.add(token, log)
    assert coverage.records() == [dict(log='l2', scenes=2, tokens_sha256=identity_hash(['c', 'b'])),
                                  dict(log='l1', scenes=2, tokens_sha256=identity_hash(['a', 'd']))]


def test_partial_write_remains_evidence_and_counts_on_resume(tmp_path):
    budget = f.PreparationBudget(tmp_path, _limits(max_metadata_bytes=8))
    with pytest.raises(f.BudgetExceeded, match='DISK'):
        f.write_chunks(tmp_path/'nested'/'record.json', [b'12345678', b'9'], budget)
    assert not (tmp_path/'nested'/'record.json').exists()
    files = list((tmp_path/'nested').glob('*.partial'))
    assert len(files) == 1 and files[0].read_bytes() == b'12345678'
    resumed = f.PreparationBudget(tmp_path, _limits(max_metadata_bytes=8))
    assert resumed.disk_bytes == 8
    with pytest.raises(f.BudgetExceeded, match='DISK'): resumed.reserve(1)


def test_read_budget_is_precharged_not_post_read(tmp_path):
    path = tmp_path/'raw.pkl'; path.write_bytes(b'abcdefgh')
    budget = f.PreparationBudget(tmp_path, _limits(max_source_read_bytes=3))
    with f.BudgetedReader(path, budget) as reader:
        assert reader.read(3) == b'abc'
        with pytest.raises(f.BudgetExceeded, match='READ_BYTES'): reader.read(1)
        assert reader.tell() == 3
    assert budget.bytes == 3


def test_aggregate_pacing_counts_both_raw_and_image_reads(tmp_path, monkeypatch):
    clock = [1.]
    monkeypatch.setattr(f.time, 'monotonic', lambda: clock[0])
    waits = []
    def advance(seconds):
        waits.append(seconds); clock[0] += seconds
    monkeypatch.setattr(f.time, 'sleep', advance)
    budget = f.PreparationBudget(tmp_path, _limits(read_mib_per_second=1.))
    budget.read_allowance(f.MiB, 'raw_log')
    budget.read_allowance(f.MiB, 'image_or_other')
    budget.write_allowance(f.MiB)
    assert sum(waits) == pytest.approx(3.)
    assert budget.bytes == 2*f.MiB and budget.written == f.MiB
    assert budget.read_categories == {'raw_log': f.MiB, 'image_or_other': f.MiB}


def test_row_budget_not_disabled_for_nested_metadata(tmp_path):
    budget = f.PreparationBudget(tmp_path, _limits())
    path = tmp_path/'row.json'
    f.write_row_document(path, {'schema': 'test'}, lambda: iter([{'token': 't', 'nested': {'v': 'x'*(65536)}}]), budget)
    with pytest.raises(ValueError, match='bound|large|budget'):
        _atomic_rows(path, {'scenes'}, lambda *a: None)


@pytest.mark.parametrize('corruption', ['unknown_token', 'wrong_timestamp', 'wrong_current_image'])
def test_raw_current_contract_changes_never_publish_complete(tmp_path, corruption):
    args, run, _, _ = setup(tmp_path)
    raw = Path(args.raw_log_root)/'log_a.pkl'; frames = pickle.loads(raw.read_bytes())
    if corruption == 'unknown_token': frames[0]['token'] = 'unknown'
    elif corruption == 'wrong_timestamp': frames[0]['timestamp'] += 1
    else: frames[0]['cams']['CAM_F0']['data_path'] = 'b.png'
    raw.write_bytes(pickle.dumps(frames))
    with pytest.raises((ValueError, KeyError)): f.build(run)
    assert not (Path(args.output)/'COMPLETE.json').exists()


@pytest.mark.parametrize('field,value', [('status','PARTIAL'), ('schema','wrong'),
                                        ('plan_identity','0'*64), ('scenes',999), ('logs',999)])
def test_corrupted_complete_control_fields_are_rejected(tmp_path, field, value):
    args, run, _, _ = setup(tmp_path); f.build(run)
    path = Path(args.output)/'COMPLETE.json'; receipt = json.loads(path.read_text())
    receipt[field] = value; atomic_json(path, receipt)
    with pytest.raises(ValueError): f.build(run)


def test_symlinked_log_shard_directory_is_rejected_before_write(tmp_path):
    args, run, _, _ = setup(tmp_path)
    external = tmp_path/'outside_admitted_root'; external.mkdir()
    (Path(args.output)/'log_shards').symlink_to(external, target_is_directory=True)
    with pytest.raises((ValueError, f.BudgetExceeded)):
        f.build(run)
    assert not list(external.iterdir())


def test_hardlinked_completed_shard_is_rejected_before_reuse(tmp_path):
    args, run, _, _ = setup(tmp_path); f.build(run)
    shard = next((Path(args.output)/'log_shards').glob('*.json'))
    external = tmp_path/'external_alias.json'
    external.hardlink_to(shard)
    before = external.read_bytes()
    with pytest.raises((ValueError, f.BudgetExceeded)):
        f.build(run)
    assert external.read_bytes() == before


def test_ledger_growth_is_rejected_before_unbounded_read_text(tmp_path, monkeypatch):
    from starVLA.dataloader import st_world_future_provenance as provenance
    args, run, _, _ = setup(tmp_path); f.build(run)
    manifest = json.loads((Path(args.output)/'TEACHER_REQUESTS.json').read_text())
    ledger = Path(manifest['source_logs_ledger']['path'])
    # Retain the original vulnerable-window regression without requiring a
    # corrected implementation to keep its removed second hash/read sequence.
    original_sha, original_read = getattr(provenance, 'file_sha256', file_sha256), Path.read_text
    activated = [False]
    def grow_after_hash(path):
        digest = original_sha(path)
        if Path(path) == ledger:
            activated[0] = True
            with ledger.open('ab') as stream:
                stream.write(b' ' * provenance.MAX_LEDGER_BYTES)
        return digest
    def reject_unbounded_large_read(path, *args, **kwargs):
        if path == ledger:
            assert path.stat().st_size <= provenance.MAX_LEDGER_BYTES, 'Unbounded read after ledger grew past cap'
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(provenance, 'file_sha256', grow_after_hash, raising=False)
    monkeypatch.setattr(Path, 'read_text', reject_unbounded_large_read)
    try:
        result = validate_ledger(manifest)
    except ValueError:
        return
    assert activated == [False], 'A changed ledger was accepted after the old hash/read window'
    assert result['sha256'] == manifest['source_logs_ledger']['sha256']


@pytest.mark.parametrize('mutation', ['growth', 'atomic_replace'])
def test_ledger_single_read_bound_and_after_read_mutation_rejection(tmp_path, monkeypatch, mutation):
    from starVLA.dataloader import st_world_future_provenance as provenance
    args, run, _, _ = setup(tmp_path); f.build(run)
    manifest = json.loads((Path(args.output)/'TEACHER_REQUESTS.json').read_text())
    ledger = Path(manifest['source_logs_ledger']['path'])
    original_open = Path.open
    changed, read_sizes = [False], []
    class Reader:
        def __init__(self, raw): self.raw = raw
        def __enter__(self): return self
        def __exit__(self, *args): self.raw.close()
        def __getattr__(self, name): return getattr(self.raw, name)
        def read(self, count=-1):
            assert 0 < count <= provenance.MAX_LEDGER_BYTES+1, 'Ledger read must be bounded'
            read_sizes.append(count)
            data = self.raw.read(count)
            if not changed[0]:
                changed[0] = True
                if mutation == 'growth':
                    with original_open(ledger, 'ab') as stream: stream.write(b' ' * provenance.MAX_LEDGER_BYTES)
                else:
                    replacement = ledger.with_suffix('.replacement')
                    with original_open(replacement, 'wb') as stream: stream.write(data)
                    replacement.replace(ledger)
            return data
    def intercept(path, mode='r', *args, **kwargs):
        raw = original_open(path, mode, *args, **kwargs)
        return Reader(raw) if path == ledger and mode == 'rb' else raw
    monkeypatch.setattr(Path, 'open', intercept)
    with pytest.raises(ValueError): validate_ledger(manifest)
    assert changed == [True] and read_sizes

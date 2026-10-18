"""Small independent files only; no real metadata census or source image scan."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO = Path('/mnt/project/VLA-Drive-st-temporal-profile-prepare-20261018')
HERE = REPO / 'reports/st_world_policy/20261018/full_history_train_resume_preparation_v1'
sys.path[:0] = [str(HERE), str(REPO), str(REPO / 'tests/st_world')]
spec = importlib.util.spec_from_file_location('history_resume_author_fixture', HERE / 'test_resume.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def replace_committed(p, record):
    root, _, _, _, _, census, _ = p
    raw = fixture.record_bytes(record)
    (root / 'history_shards' / census['rows'][0]['name']).write_bytes(raw)
    census['rows'][0].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    p[4] = record


def reuse(p, c, *, roots=None):
    root, images, frames, current, _, census, limits = p
    command = ['build', '--student-root', str(root.parent), '--output', str(root),
               '--split', 'train', '--max-scenes', '2', '--record-layout', 'packed_log_v1']
    with fixture.resume_original_writer(writer=fixture.writer, original=fixture.original,
            bounded=fixture.bounded, history=fixture.history, schema=fixture.schema,
            record_bytes=fixture.record_bytes, census=census, expected_command_args=command):
        fixture.bounded.PreparationBudget(root, limits)
        return fixture.original.build_history_record(frames, 't2', 'log', current,
            roots or [images], raw_log_sha256='a' * 64)


def test_previously_missing_image_now_present_is_rejected(tmp_path):
    p = list(fixture.package(tmp_path))
    image = Path(p[4]['frames'][0]['views'][0]['image']['path'])
    saved = image.read_bytes()
    image.unlink()
    old = fixture.history.build_history_record(p[2], 't2', 'log', p[3], [p[1]], raw_log_sha256='a'*64)
    assert old['frames'][0]['views'][0]['valid'] is False
    replace_committed(p, old)
    image.write_bytes(saved)
    fresh = fixture.history.build_history_record(p[2], 't2', 'log', p[3], [p[1]], raw_log_sha256='a'*64)
    assert fresh['frames'][0]['views'][0]['valid'] is True
    with pytest.raises(ValueError):
        reuse(p, None)


def test_new_higher_priority_image_path_is_rejected(tmp_path):
    p = fixture.package(tmp_path)
    primary = tmp_path / 'primary'
    primary.mkdir()
    old = Path(p[4]['frames'][0]['views'][0]['image']['path'])
    (primary / old.name).write_bytes(old.read_bytes())
    with pytest.raises(ValueError):
        reuse(p, None, roots=[primary, p[1]])


def test_duplicate_committed_record_rejected(tmp_path):
    p = list(fixture.package(tmp_path))
    root, _, _, _, record, census, _ = p
    raw = fixture.record_bytes(record) * 2
    (root / 'history_shards' / census['rows'][0]['name']).write_bytes(raw)
    census['rows'][0].update(bytes=len(raw), scenes=2, sha256=hashlib.sha256(raw).hexdigest())
    census['scenes'] = 2
    with pytest.raises(ValueError, match='Duplicate'):
        fixture.cache(p)


def test_wrong_log_rejected_even_with_recomputed_self_identity(tmp_path):
    p = list(fixture.package(tmp_path))
    record = copy.deepcopy(p[4])
    record['log'] = 'foreign-role-log'
    record['identity'] = fixture.schema.identity_hash({k: v for k, v in record.items() if k != 'identity'})
    replace_committed(p, record)
    with pytest.raises(ValueError, match='wrong shard log'):
        fixture.cache(p)


def test_reused_and_new_records_use_original_immutable_writer(tmp_path):
    from tools.st_world_policy.history_shards import HistoryShards
    p = list(fixture.package(tmp_path))
    root, images, frames, current, old, census, limits = p
    # A legitimate committed boundary is a completed log, not an arbitrary
    # prefix within one unflushed native shard.
    (root / 'SELECTED_SCENES.json').write_text(json.dumps({'scenes':[
        {'token': 't2', 'log': 'log'}, {'token': 't3', 'log': 'log2'}]}))
    current3 = dict(current, token='t3', log='log2', timestamp=1500000,
                    image_paths=[str(images/f'3_{v}.png') for v in fixture.history.VIEWS])
    new = fixture.history.build_history_record(frames, 't3', 'log2', current3, [images], raw_log_sha256='a'*64)
    reused = reuse(p, None)
    before = {path.name: path.read_bytes() for path in (root/'history_shards').iterdir()}
    freshroot = tmp_path/'fresh'
    freshroot.mkdir()
    values = []
    for out, first in [(root, reused), (freshroot, old)]:
        writer = HistoryShards(out, fixture.bounded.PreparationBudget(out, limits), max_shards=8)
        for record in (first, new):
            writer.add(out/'history'/f"{record['token']}.json", record)
        values.append(writer.manifest({'scenes': [{'token':'t2'}, {'token':'t3'}]}))
    assert values[0] == values[1]
    assert {x.name: x.read_bytes() for x in (root/'history_shards').iterdir()} == {
        x.name: x.read_bytes() for x in (freshroot/'history_shards').iterdir()}
    for name, raw in before.items():
        assert (root/'history_shards'/name).read_bytes() == raw
    writer = HistoryShards(root, fixture.bounded.PreparationBudget(root, limits), max_shards=8)
    conflicting = copy.deepcopy(old)
    conflicting['preprocess'] = 'conflicting payload'
    writer.add(root/'history/t2.json', conflicting)
    with pytest.raises(ValueError, match='Immutable artifact conflict'):
        writer.flush()


def test_overlay_role_and_independent_admission_restrictions():
    import run_resume_once as bridge
    controller = bridge.load_controller()
    plan = json.loads((HERE/'FULL_ROLE_PLAN.json').read_text())
    a = json.loads((HERE/'ADMISSION_PROPOSAL.json').read_text())
    a.update(status='APPROVED_FULL_CAUSAL_HISTORY_METADATA_ONLY',
             reviewer='/root/independent_fixture', issued_at_unix=1,
             latest_start_unix=100, valid_until_unix=20000,
             hostname=controller.socket.gethostname())
    cp = dict(schema='st_checkpoint_storage_commitments_v1', status='CURRENT',
              remaining_bytes=1, free_floor_bytes=1<<40, checked_at_unix=1,
              valid_until_unix=30000)
    checked = []
    def pinned(pin):
        checked.append(pin)
        return plan if pin == a['plan'] else {}
    inherited = SimpleNamespace(W_COMPAT=a['w_source_compatibility'], pinned=pinned)
    spec = plan['roles']['train']
    assert controller.validate(a, cp, 'train', spec, inherited, now=10) == [10, 11]
    assert all(pin in checked for pin in plan['versioned_attempt']['old_failures'])
    wrong = copy.deepcopy(a)
    wrong.update(role='dev', limits=controller.limits('dev', [10, 11]))
    with pytest.raises(ValueError, match='same-input resume'):
        controller.validate(wrong, cp, 'dev', spec, inherited, now=10)
    wrong = copy.deepcopy(a)
    wrong['reviewer'] = '/root/temporal_profile_prepare'
    with pytest.raises(ValueError, match='cannot admit itself'):
        controller.validate(wrong, cp, 'train', spec, inherited, now=10)

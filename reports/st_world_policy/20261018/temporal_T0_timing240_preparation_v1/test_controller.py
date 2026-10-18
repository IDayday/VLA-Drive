"""Bounded CPU behavior tests. Fixtures are not resource or training evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('timing_binding', HERE / 'controller.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def c():
    return m.controller()


@pytest.fixture
def configured(c, monkeypatch):
    a = json.loads((HERE / 'RESOURCE_ADMISSION_TEMPLATE.json').read_text())
    a.update(status='ADMITTED', reviewer='/root/independent_resource_fixture',
             gpu_uuids=['GPU-fixture-' + str(i) for i in range(4)], cpu_affinity=list(range(8)),
             valid_from_unix=100, valid_until_unix=10000,
             correctness_review={'path': '/fixture/correctness.json', 'sha256': 'b' * 64})
    cp = dict(remaining_bytes=a['other_commitments_bytes'], free_floor_bytes=1 << 40, valid_until_unix=10000)
    review = dict(schema='st_T0_timing240_correctness_review_v1', status='PASS',
                  reviewer='/root/independent_correctness_fixture', source_sha=c.SOURCE,
                  controller_sha256=c.digest(HERE / 'controller.py'),
                  launcher_sha256=c.digest(HERE / 'launch_once.sh'), inputs_sha256=m.INPUT_SHA)
    original_digest, original_read = c.digest, c.read
    def digest(path):
        return {'/fixture/admission.json': 'a' * 64, '/fixture/correctness.json': 'b' * 64}.get(str(path)) or original_digest(path)
    def read(path):
        if str(path) == '/fixture/admission.json': return a
        if str(path) == '/fixture/correctness.json': return review
        if str(path) == a['checkpoint_commitments']['path']: return cp
        return original_read(path)
    monkeypatch.setattr(c, 'digest', digest)
    monkeypatch.setattr(c, 'read', read)
    monkeypatch.setattr(time, 'time', lambda: 101)
    monkeypatch.setattr(c.os, 'sched_getaffinity', lambda _: set(range(16)))
    return c, a, cp, review


def test_fixed_timing_command(c):
    assert c.command()[:4] == ['timeout', '--signal=TERM', '--kill-after=180s', '3720s']
    assert '--nproc_per_node=4' in c.command() and '--resume' not in c.command()
    assert c.INPUT == HERE / 'prepared_inputs/INPUTS.json'
    assert c.RUN_ID == 'T0_fulltrain_seed42_profile240_world4_micro1_global32_v1'
    assert c.RESERVE == 40 << 30
    with pytest.raises(ValueError): c.command(True)


def test_recipe_and_fresh_base(c):
    inputs = c.read(c.INPUT)
    cfg = c.read(inputs['pins']['config']['path'])
    c.validate_recipe(cfg)
    assert cfg['execution']['stop_after'] == cfg['runtime']['save_every'] == 240
    assert cfg['execution']['max_updates'] == 20000 and cfg['optimizer']['warmup'] == 1000
    assert inputs['consumption']['cumulative_scene_exposures'] == 7680
    assert inputs['consumption']['per_rank_scene_exposures'] == [1920] * 4
    parent = c.read(inputs['parent_capacity_inputs']['path'])
    for name in ('base_binding', 'data_manifest', 'target_contract'):
        assert inputs['pins'][name] == parent['pins'][name]
    assert 'continuation' not in cfg and 'resume' not in inputs['native_argv']
    assert cfg['data']['history_root'] is None


def test_metadata_clone_changes_only_consumption_and_pins(c):
    inputs = c.read(c.INPUT)
    parent = c.read(inputs['parent_capacity_inputs']['path'])
    old = c.read(parent['pins']['teacher_manifest']['path'])
    new = c.read(inputs['pins']['teacher_manifest']['path'])
    old.pop('identity'); new.pop('identity')
    for key in ('consumption', 'recipe'):
        old['temporal_profile_admission'].pop(key)
        new['temporal_profile_admission'].pop(key)
    assert old == new
    old = c.read(parent['pins']['provider_pca_scope']['path'])
    new = c.read(inputs['pins']['provider_pca_scope']['path'])
    for value in (old, new):
        value.pop('identity'); value.pop('provider_descriptor')
    assert old == new


@pytest.mark.parametrize('group,key,value', [
    ('execution', 'scope', 'FORMAL'), ('execution', 'stop_after', 5000),
    ('execution', 'max_updates', 240), ('execution', 'world_size', 8),
    ('execution', 'global_batch', 16), ('execution', 'micro_batch', 2),
    ('execution', 'sample_limit', 64), ('execution', 'seed', 43),
    ('execution', 'precision_policy', 'fp32'), ('runtime', 'max_seconds', 3601),
    ('runtime', 'profile_mode', 'trace'), ('runtime', 'save_every', 120),
    ('runtime', 'gradient_checkpointing', True), ('optimizer', 'warmup', 20),
    ('optimizer', 'lr', 1e-4), ('space_time', 'history', True),
    ('space_time', 'current_weight', .5), ('space_time', 'sampled_positions', 48),
])
def test_scientific_or_budget_drift_rejected(c, group, key, value):
    cfg = c.read(c.read(c.INPUT)['pins']['config']['path'])
    cfg[group][key] = value
    with pytest.raises(RuntimeError): c.validate_recipe(cfg)


def test_requires_both_independent_actual_reviews(configured):
    c, a, cp, review = configured
    result, pin = c.admission('/fixture/admission.json', 'a' * 64, 'prepare')
    assert result == a and pin['sha256'] == 'a' * 64
    assert c.OTHER_COMMITMENTS == cp['remaining_bytes']
    assert c.storage_requirement(0) == (1 << 40) + cp['remaining_bytes'] + ((768 + 10 + 4 + 40) << 30)
    assert c.storage_requirement(35 << 30) == c.storage_requirement(0) - (35 << 30)


@pytest.mark.parametrize('key,value', [
    ('status', 'TEMPLATE_NOT_ADMITTED'), ('reviewer', '/root'),
    ('reviewer', '/root/temporal_t0_timing_prepare'), ('gpu', '0'),
    ('checkpoint_reservation_bytes', 80 << 30), ('retained_evaluation_reservation_bytes', 0),
    ('retained_cache_reservation_bytes', 0), ('automatic_resume', True),
    ('automatic_retry', True), ('maximum_new_checkpoint_directories', 2),
    ('correctness_review', None), ('valid_until_unix', 4180),
])
def test_bad_admission_rejected(configured, key, value):
    c, a, _, _ = configured
    a[key] = value
    with pytest.raises((RuntimeError, TypeError)): c.admission('/fixture/admission.json', 'a' * 64, 'prepare')


@pytest.mark.parametrize('key,value', [('status', 'NOT_REVIEWED'), ('reviewer', '/root'),
    ('reviewer', '/root/temporal_t0_timing_prepare'), ('controller_sha256', '0' * 64),
    ('inputs_sha256', '0' * 64), ('source_sha', '0' * 40)])
def test_invalid_correctness_is_not_waived(configured, key, value):
    c, _, _, review = configured
    review[key] = value
    with pytest.raises(RuntimeError): c.admission('/fixture/admission.json', 'a' * 64, 'run')


def test_resume_and_missing_admission_rejected(c, configured):
    with pytest.raises(RuntimeError): c.storage_requirement(0)
    c, _, _, _ = configured
    with pytest.raises(RuntimeError): c.admission('/fixture/admission.json', 'a' * 64, 'resume')


def test_one_checkpoint_limit(configured, monkeypatch, tmp_path):
    c, _, _, _ = configured
    c.admission('/fixture/admission.json', 'a' * 64, 'run')
    monkeypatch.setattr(c, 'OUT', tmp_path)
    slots = tmp_path / 'runs' / c.RUN_ID / 'checkpoints'
    (slots / 'partial_one').mkdir(parents=True)
    (slots / 'partial_two').mkdir()
    with pytest.raises(RuntimeError, match='Only one checkpoint'): c.storage()


def test_source_drift_fails_closed():
    raw = m.CAPACITY.read_text()
    with pytest.raises(ValueError): m.adapted_capacity(raw.replace('stop_after=4,', 'stop_after=5,'))
    assert hashlib.sha256(m.CAPACITY.read_bytes()).hexdigest() == m.CAPACITY_SHA


def test_help_and_bad_cli_never_dispatch():
    result = subprocess.run([sys.executable, '-B', str(HERE / 'controller.py'), '--help'], capture_output=True, text=True)
    assert result.returncode == 0 and '{preflight,prepare,run}' in result.stdout
    for args in (['--action', 'resume'], ['--skip-review'], ['--stop-after', '100000']):
        result = subprocess.run([sys.executable, '-B', str(HERE / 'controller.py'), *args], capture_output=True, text=True)
        assert result.returncode != 0 and 'usage:' in result.stderr


def test_inherited_supervisor_remains_exact(c):
    assert c.execute.__code__.co_filename.endswith('full_population_profile_launch_v1/controller.py')
    assert c.monitor_training.__code__.co_filename == c.execute.__code__.co_filename
    assert c.READ_CAP == 1 << 40 and c.RSS_LIMIT == 192 << 30
    assert len(c.GPU_LOCKS) == 4


def test_native_recipe_accepts_exact_prepared_metadata():
    code = ('import sys,json; from pathlib import Path; '
            'sys.path.insert(0,"/mnt/project/VLA-Drive-st-temporal-profile-runtime-v1-20261018"); '
            'from tools.st_world_policy.temporal_formal import validate_recipe; '
            f'p=Path({str(HERE / "prepared_inputs")!r}); '
            'd=json.loads((p/"PROVIDER.json").read_text()); '
            'c=json.loads((p/"training_config.json").read_text()); '
            'validate_recipe(d["temporal_profile_admission"],c); print("NATIVE_RECIPE_OK")')
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith('NATIVE_RECIPE_OK')

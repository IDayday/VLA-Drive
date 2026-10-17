"""Independent CPU counterexamples; only tiny temporary metadata is used."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from test_provider_profile_upgrade import f, sign
from tools.st_world_policy import teacher_provider as tp, qualify_indexed_provider as q
from starVLA.dataloader.st_world_metadata_index import build_index
from starVLA.dataloader.st_world_targets import atomic_json
from test_approval import run as registration_fixture, put, git


@pytest.mark.parametrize('count', [256, 257])
def test_actual_indexed_scene_ceiling(f, tmp_path, count):
    manifest = deepcopy(f.manifest)
    proto = manifest['scenes'][0]
    manifest['scenes'] = [dict(deepcopy(proto), token='independent-' + str(i)) for i in range(count)]
    sign(manifest)
    req = tmp_path/'requests.json'; atomic_json(req, manifest)
    index = tmp_path/'index'
    build_index(index, requests=req, expected_request_identity=manifest['identity'], contract=f.a.fit.contract, max_scenes=300)
    selected = deepcopy(f.selected)
    selected['scenes'] = [dict(token=x['token'], log=x['log'], current_record_sha256='a'*64, ego_record_sha256='b'*64) for x in manifest['scenes']]
    selectedpath = tmp_path/'selected.json'; atomic_json(selectedpath, selected)
    recipe = deepcopy(f.recipe); recipe['execution']['sample_limit'] = count
    recipepath = tmp_path/'recipe.json'; atomic_json(recipepath, recipe)
    admission = deepcopy(f.admission)
    admission.update(data_manifest=tp.pin(selectedpath), recipe=tp.pin(recipepath))
    def prepare():
        return tp.prepare_indexed(index, [f.a.source_index], f.p['teacher_config']['path'], f.resultpath,
            tmp_path/'provider.json', limits=f.p['limits'], profile_admission=admission)
    if count == 257:
        with pytest.raises(ValueError, match='256'): prepare()
        assert not (tmp_path/'provider.json').exists()
    else:
        d = prepare()
        _, _, provider = tp.load_descriptor(tp.pin(tmp_path/'provider.json'))
        try:
            assert len(provider.manifest['scenes']) == 256
            assert d['profile_admission']['new_population_tensor_qualified'] is False
        finally: tp.close_provider_metadata(provider)


@pytest.mark.parametrize('change', ['lr', 'data_root', 'patch_count', 'precision', 'seed', 'workers', 'formal'])
def test_actual_live_config_not_just_recipe_pin(f, tmp_path, change):
    from tools.st_world_policy.training_config import validate_config
    path = tp.configure_smoke(f.recipepath, f.path, f.a.fit.root, tmp_path/'configured', fit_cache_index=f.a.fit_index)
    cfg = json.loads(Path(path).read_text())
    if change == 'lr': cfg['optimizer']['lr'] *= 2
    elif change == 'data_root': cfg['data']['root'] = str(tmp_path/'foreign')
    elif change == 'patch_count': cfg['space_time']['sampled_positions'] += 1
    elif change == 'precision': cfg['execution']['precision_policy'] = 'bf16_direct'
    elif change == 'seed': cfg['execution']['seed'] += 1
    elif change == 'workers': cfg['runtime']['loader_workers'] = 1
    else: cfg['execution']['scope'] = 'FORMAL'; cfg['execution']['sample_limit'] = 0
    with pytest.raises(ValueError): validate_config(cfg)


def test_archived_math_must_match_original_git_object_even_if_repinned(f):
    path = f.archive/q.MATH_PATHS[0]
    original = path.read_bytes()
    try:
        path.write_bytes(original+b'\n# uncommitted mathematical drift\n')
        m = deepcopy(f.migration)
        m['mathematics'][0]['old'] = tp.pin(path)
        sign(m)
        with pytest.raises(ValueError, match='actual commit'):
            q.validate_control_migration(f.p, m)
    finally: path.write_bytes(original)


@pytest.mark.parametrize('kind', ['omit_math', 'new_control_pin', 'old_control_pin'])
def test_source_inventory_cannot_be_relaxed_or_rebound(f, kind):
    m = deepcopy(f.migration)
    if kind == 'omit_math': m['mathematics'].pop()
    elif kind == 'new_control_pin': m['controls'][0]['new']['sha256'] = 'a'*64
    else: m['controls'][0]['old']['sha256'] = 'b'*64
    sign(m)
    with pytest.raises(ValueError): q.validate_control_migration(f.p, m)


def test_identical_manifest_bytes_at_another_path_are_not_registered_identity(f, tmp_path):
    alias = tmp_path/'same.json'; alias.write_bytes(f.selectedpath.read_bytes())
    assert tp.pin(alias)['sha256'] == tp.pin(f.selectedpath)['sha256']
    with pytest.raises(ValueError, match='path/SHA'):
        tp.validate_training_manifest({'descriptor': tp.pin(f.path)}, tp.pin(alias))


def test_real_training_run_reaches_manifest_guard_before_provider_or_dataset(f, tmp_path, monkeypatch):
    from tools.st_world_policy import train_world as train
    path = tp.configure_smoke(f.recipepath, f.path, f.a.fit.root, tmp_path/'configured', fit_cache_index=f.a.fit_index)
    cfg = json.loads(Path(path).read_text())
    cfg['runtime']['output_root'] = str(tmp_path/'output')
    atomic_json(path, cfg)
    contract = tmp_path/'contract.json'; atomic_json(contract, cfg['space_time']['teacher'])
    wrong = dict(tp.pin(f.selectedpath), sha256='f'*64)
    reg = {'run_id': 'fixture', 'artifacts': {'config': tp.pin(path),
        'teacher_manifest': tp.pin(f.path), 'target_contract': tp.pin(contract), 'data_manifest': wrong}}
    regpath = tmp_path/'registration.json'; atomic_json(regpath, reg)
    monkeypatch.setattr(train, 'check_formal_approval', lambda *a, **kw: {'formal_launch_allowed': False})
    monkeypatch.setattr(train, 'verified_runtime', lambda *a: {})
    monkeypatch.setattr(train.torch.cuda, 'set_device', lambda *a: None)
    monkeypatch.setattr(train.dist, 'init_process_group', lambda *a, **kw: None)
    monkeypatch.setattr(train.dist, 'destroy_process_group', lambda *a: None)
    def forbidden(*a, **kw): raise AssertionError('Provider/dataset must not be reached')
    monkeypatch.setattr(tp, 'load_descriptor', forbidden)
    for k, v in [('RANK','0'),('WORLD_SIZE',str(cfg['execution']['world_size'])),('LOCAL_RANK','0')]: monkeypatch.setenv(k,v)
    for resume in (False, True):
        with pytest.raises(ValueError, match='Registered PROFILE dataset'):
            train.run(SimpleNamespace(registration=str(regpath), approval=None, resume=resume, acknowledge_stop=False))


def test_resume_keeps_registered_source_and_exact_signature_gate():
    from tools.st_world_policy import train_world as train
    from tools.st_world_policy import check_formal_approval as gate
    source = Path(train.__file__).read_text()
    tree = ast.parse(source)
    run = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'run')
    approval = next(x for x in ast.walk(run) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == 'check_formal_approval')
    ownership = next(x for x in ast.walk(run) if isinstance(x, ast.FunctionDef) and x.name == 'own_run')
    assert approval.lineno < ownership.lineno
    assert '"source_sha": registration["source_sha"]' in source
    assert 'read_json(out / "identity.json") != {"identity": signature, **identity}' in source
    assert 'resolve_latest(out / "checkpoints", identity=signature' in source
    gate_source = Path(gate.__file__).read_text()
    assert 'source_fingerprint(repo) != expected' in gate_source
    assert 'Runtime source must be committed and clean' in gate_source


def test_true_cli_migration_writes_only_explicit_proposal(f, tmp_path):
    import os, sys
    output = tmp_path/'migration.json'
    result = subprocess.run([sys.executable, '-m', 'tools.st_world_policy.qualify_indexed_provider',
        'prepare-profile-migration', '--parent-descriptor', str(f.parentpath),
        '--parent-runtime', str(f.archive), '--output', str(output)], capture_output=True, text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'), timeout=30)
    assert result.returncode == 0, result.stderr + result.stdout
    actual = json.loads(output.read_text())
    assert actual == f.migration
    assert actual['formal_approved'] is False and actual['new_population_tensor_qualified'] is False


def test_profile_source_gate_still_rejects_dirty_and_committed_code_drift(registration_fixture):
    from tools.st_world_policy.check_formal_approval import check_formal_approval, ApprovalError
    repo, reg, _, record = registration_fixture
    record['execution'].update(scope='PROFILE', sample_limit=32, stop_after=240)
    record['weights_usable_for_formal_init'] = False
    record['artifacts']['config'] = put(Path(record['artifacts']['config']['path']), {'execution': record['execution']})
    put(reg, record)
    assert check_formal_approval(reg, None, repo=repo)['status'] == 'DIAGNOSTIC_ONLY'
    (repo/'train.py').write_text('LOSS_WEIGHT = 2\n')
    with pytest.raises(ApprovalError, match='committed and clean'):
        check_formal_approval(reg, None, repo=repo)
    git(repo, 'add', '.'); git(repo, 'commit', '-m', 'fixture changed execution')
    with pytest.raises(ApprovalError, match='source changed'):
        check_formal_approval(reg, None, repo=repo)

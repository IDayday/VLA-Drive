"""Small metadata guard tests; no actual population/model/qualification claims."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.st_world_policy import teacher_provider as tp, temporal_formal as tf
from tools.st_world_policy.training_config import validate_config
from tools.st_world_policy.check_formal_approval import validate_execution
from starVLA.dataloader.st_world_targets import atomic_json
from test_provider_profile_upgrade import f
from test_provider_full_index_profile import full

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'reports/st_world_policy/20261018/temporal_formal5k_preparation_v1/REFERENCE_RECIPE.json'


@pytest.fixture
def case(tmp_path, monkeypatch):
    # Decision is frozen by its exact production SHA. Only this local fixture
    # substitutes its small pin; production has no parameter to change it.
    decision = dict(status='RECIPE_SELECTED_NOT_FORMAL_DISPATCH',
        weights=dict(current=1., future=tf.FUTURE_WEIGHT, delta=tf.DELTA_WEIGHT))
    dp = tmp_path / 'decision.json'; atomic_json(dp, decision)
    monkeypatch.setattr(tf, 'DECISION_SHA', tp.pin(dp)['sha256'])
    original = json.loads(REFERENCE.read_text())
    rp = tmp_path / 'reference.json'; atomic_json(rp, original)
    hp = tmp_path / 'history' / 'COMPLETE.json'
    atomic_json(hp, dict(status='COMPLETE', scenes=tf.POPULATION,
        history_identity='a'*64, history_manifest_sha256='b'*64))
    def make(cell, scope='FORMAL', profile_mode='none', world_size=4, micro_batch=1):
        hist = (dict(history_complete=tp.pin(hp), history_identity='a'*64,
                     history_manifest_sha256='b'*64) if tf.CELLS[cell][0] else None)
        cfg = tf.selected_recipe(original, cell, output_root=str(tmp_path / 'runs'),
            history_root=str(hp.parent) if hist else None, scope=scope, profile_mode=profile_mode,
            world_size=world_size, micro_batch=micro_batch, max_seconds=28800 if scope=='FORMAL' else 3600)
        recipe = tmp_path / (cell + '.json'); atomic_json(recipe, cfg)
        a = dict(schema=tf.SCHEMA if scope=='FORMAL' else tf.PROFILE_SCHEMA, scope=scope, cell=cell, decision=tp.pin(dp),
            implementation=tp.pin(tf.__file__), parent_descriptor=original['data']['teacher_provider']['descriptor'],
            source_migration=dict(path='/fixture/MIGRATION.json',sha256='c'*64),
            data_manifest=dict(path='/fixture/SELECTED.json',sha256='d'*64),
            recipe=tp.pin(recipe), reference_recipe=tp.pin(rp), max_scenes=tf.POPULATION,
            max_updates=5000 if scope=='FORMAL' else 256, population={},
            consumption=tf.consumption(cfg['execution']['stop_after'],scope=scope,
                world_size=world_size,micro_batch=micro_batch,timing=profile_mode=='timing'), history=hist,
            new_population_tensor_qualified=False, formal_approved=False)
        descriptor = tmp_path / (cell + '-descriptor.json')
        atomic_json(descriptor, {('formal_admission' if scope=='FORMAL' else 'temporal_profile_admission'):a})
        configured = deepcopy(cfg)
        configured['data']['teacher_provider'] = dict(descriptor=tp.pin(descriptor),
            pca_channel_scope=dict(path='/fixture/SCOPE.json',sha256='e'*64))
        return SimpleNamespace(cfg=cfg, configured=configured, a=a, descriptor=descriptor)
    return SimpleNamespace(make=make, original=original, hp=hp, rp=rp, tmp=tmp_path)


@pytest.mark.parametrize('cell', list(tf.CELLS))
def test_actual_reference_native_config_routes(case, cell):
    c = case.make(cell)
    assert tf.validate_recipe(c.a, c.cfg) == c.cfg
    st = validate_config(c.configured); validate_execution(c.configured['execution'])
    assert (st.history, st.future, st.delta) == tf.CELLS[cell]
    assert st.future_weight == (tf.FUTURE_WEIGHT if st.future else 0.)
    assert st.delta_weight == (tf.DELTA_WEIGHT if st.delta else 0.)
    assert c.cfg['optimizer'] == case.original['optimizer']
    assert c.cfg['data']['pca_path'] == case.original['data']['pca_path']
    assert c.a['formal_approved'] is False


def test_actual_full_sampler_exposure_and_tail():
    c = tf.consumption()
    assert c['cumulative_scene_exposures'] == 159992
    assert c['per_rank_scene_exposures'] == [39998] * 4
    assert c['gradient_accumulation_steps'] == 8
    for value in (0, 5001, True, 5000.):
        with pytest.raises(ValueError): tf.consumption(value)


@pytest.mark.parametrize('mode,updates', [('none',4),('timing',240)])
@pytest.mark.parametrize('cell', ['T0','T1','T2','T3','T4'])
def test_distinct_global32_profile_route(case, cell, mode, updates):
    c = case.make(cell, 'PROFILE', mode)
    validate_config(c.configured)
    assert c.cfg['execution']['stop_after'] == updates
    assert c.cfg['execution']['sample_limit'] == 101592
    assert c.cfg['execution']['max_updates'] == 20000
    assert c.a['consumption']['cumulative_scene_exposures'] == 32*updates
    assert c.cfg['execution']['micro_batch'] == 1
    assert c.a['formal_approved'] is False
    wrong = deepcopy(c.configured); wrong['execution'].update(scope='FORMAL',sample_limit=0,stop_after=5000)
    with pytest.raises(ValueError): validate_config(wrong)
    for budget in (5,32,256,5000):
        wrong = deepcopy(c.configured); wrong['execution']['stop_after'] = budget
        with pytest.raises(ValueError): validate_config(wrong)


@pytest.mark.parametrize('world,micro', [(1,1),(2,1),(4,1),(8,1),(4,4),(8,4)])
def test_explicit_topology_keeps_global32_and_epoch_tail(case, world, micro):
    c = case.make('T2',world_size=world,micro_batch=micro)
    validate_config(c.configured)
    counts = c.a['consumption']
    assert sum(counts['per_rank_scene_exposures']) == 159992
    assert counts['gradient_accumulation_steps'] * world * micro == 32
    assert c.cfg['execution']['global_batch'] == 32


@pytest.mark.parametrize('kind', ['world','micro','global','scope','sample_limit','stop','schedule','warmup',
    'weight','delta','history','width','W','K','precision','loader','continuation','output','optimizer'])
def test_config_mutations_rejected(case, kind):
    c = case.make('T4'); cfg = deepcopy(c.configured)
    if kind in ('world','micro','global'):
        cfg['execution'][{'world':'world_size','micro':'micro_batch','global':'global_batch'}[kind]] *= 2
    elif kind == 'scope': cfg['execution']['scope'] = 'PROFILE'
    elif kind == 'sample_limit': cfg['execution']['sample_limit'] = 32
    elif kind == 'stop': cfg['execution']['stop_after'] = 5001
    elif kind == 'schedule': cfg['execution']['max_updates'] = 5000
    elif kind == 'warmup': cfg['optimizer']['warmup'] = 50
    elif kind == 'weight': cfg['space_time']['future_weight'] = 1.
    elif kind == 'delta': cfg['space_time']['delta_weight'] = 1.
    elif kind == 'history': cfg['data']['history_root'] = '/other'
    elif kind == 'width': cfg['space_time']['teacher']['channels'] = 256
    elif kind == 'W': cfg['space_time']['world_queries_per_view'] = 96
    elif kind == 'K': cfg['space_time']['sampled_positions'] = 48
    elif kind == 'precision': cfg['execution']['precision_policy'] = 'fp32'
    elif kind == 'loader': cfg['runtime']['loader_workers'] = 1
    elif kind == 'continuation': cfg['continuation'] = {}
    elif kind == 'output': cfg['runtime']['output_root'] = '/changed'
    else: cfg['optimizer']['lr'] *= 2
    with pytest.raises(ValueError): validate_config(cfg)


@pytest.mark.parametrize('kind', ['population','update','approval','qualified','source','decision','consumption','missing','partial','subset','identity'])
def test_admission_or_history_changes_rejected(case, kind):
    c = case.make('T3'); a = deepcopy(c.a)
    if kind == 'population': a['max_scenes'] = 32
    elif kind == 'update': a['max_updates'] = 20000
    elif kind == 'approval': a['formal_approved'] = True
    elif kind == 'qualified': a['new_population_tensor_qualified'] = True
    elif kind == 'source': a['implementation']['sha256'] = '0'*64
    elif kind == 'decision': a['decision']['sha256'] = '0'*64
    elif kind == 'consumption': a['consumption']['cumulative_scene_exposures'] = 160000
    elif kind == 'missing': a['history'] = None
    elif kind == 'identity': a['history']['history_identity'] = '0'*64
    else:
        h = tp.pinned(a['history']['history_complete'])
        if kind == 'partial': h['status'] = 'PARTIAL'
        else: h['scenes'] = 32
        atomic_json(case.hp, h); a['history']['history_complete'] = tp.pin(case.hp)
    with pytest.raises((ValueError, FileNotFoundError)): tf.validate_recipe(a, c.cfg)


def test_t2_no_history_and_profile_no_promotion(case):
    c = case.make('T2'); a = deepcopy(c.a)
    a['history'] = case.make('T1').a['history']
    with pytest.raises(ValueError, match='T2'): tf.validate_recipe(a, c.cfg)
    with pytest.raises(ValueError):
        tp.validate_option(c.configured['data']['teacher_provider'], loader_workers=0,
            teacher_root=None, pca_path=c.cfg['data']['pca_path'], scope='PROFILE', config=c.configured)
    with pytest.raises(ValueError, match='relabeled'):
        tp.validate_option(c.configured['data']['teacher_provider'], loader_workers=0,
            teacher_root=None, pca_path=c.cfg['data']['pca_path'], scope='SMOKE', config=c.configured)
    d = dict(formal_admission=c.a, profile_admission={})
    with pytest.raises(ValueError, match='combine'): tp._validate_indexed_descriptor(d)


def test_history_attachment_original_role_and_current_guards(case):
    from tools.st_world_policy.temporal_composition import validate_dataset_attachment
    c = case.make('T1'); h = c.a['history']
    source = dict(split='train', identity='c'*64, index_sha256='d'*64, partition_sha256='e'*64)
    manifest = dict(schema='st_history_manifest_v1', identity=h['history_identity'],
        scenes=[{}]*tf.POPULATION, view_names=['CAM_F0','CAM_L0','CAM_R0'], split='train',
        source_student_identity=source['identity'], source_index_sha256=source['index_sha256'],
        partition_sha256=source['partition_sha256'])
    history = SimpleNamespace(root=case.hp.parent, manifest=manifest,
        index={'x':dict(log='l',current_record_sha256='f'*64)})
    provider = SimpleNamespace(temporal_composition=h, manifest=manifest)
    rows = [dict(token='x',log='l',current_record_sha256='f'*64)]
    validate_dataset_attachment(history,source,rows,provider)
    for key,value in [('split','dev'),('identity','0'*64),('index_sha256','0'*64)]:
        wrong = dict(source); wrong[key] = value
        with pytest.raises(ValueError): validate_dataset_attachment(history,wrong,rows,provider)
    with pytest.raises(ValueError): validate_dataset_attachment(history,source,[dict(rows[0],log='other')],provider)


def test_native_prepare_missing_history_fails_without_publication(case, tmp_path, monkeypatch):
    # No synthetic fake COMPLETE is passed into a production-like prepare.
    template = dict(schema=tp.INDEXED_SCHEMA,
        profile_admission=dict(schema=tp.FULL_PROFILE_SCHEMA))
    template['identity'] = tp.identity_hash(template)
    path = tmp_path / 'template.json'; atomic_json(path, template)
    args = SimpleNamespace(full_profile_descriptor=str(path), source_migration=str(case.rp),
        decision=str(case.rp), history_complete=None, cell='T3', scope='FORMAL', world_size=4, micro_batch=1, profile_mode='none', max_seconds=28800,
        pca_fit_cache_root='/missing', pca_fit_cache_index='/missing',
        training_output_root=str(tmp_path/'runs'), output=str(tmp_path/'prepared'))
    def forbidden(*a,**kw): raise AssertionError('No DB/model read before missing dependency refusal')
    monkeypatch.setattr(tp,'open_metadata_index',forbidden)
    with pytest.raises(ValueError, match='Missing real full history'): tf.prepare(args)
    assert not Path(args.output).exists()
    assert list(tmp_path.glob('.prepared.staging-*/PREPARATION_FAILED.json'))


def test_explicit_admission_keeps_original_full_order_validator(full):
    # Exercise the actual SQLite/streaming full-population validator using its
    # existing 258-row fixture, with an explicit admission rather than the old
    # diagnostic field. No production population limit is patched or weakened.
    request = tp.open_metadata_index(full.d['request_index'])
    source = tp.open_metadata_index(full.d['sources'][0]['index'])
    try:
        d = deepcopy(full.d); a = d.pop('profile_admission')
        tp._validate_full_population(d, request, source, admission=a)
        wrong = deepcopy(a); wrong['max_scenes'] -= 1
        with pytest.raises(ValueError): tp._validate_full_population(d, request, source, admission=wrong)
        wrong = deepcopy(a); wrong['data_manifest']['sha256'] = '0'*64
        with pytest.raises(ValueError): tp._validate_full_population(d, request, source, admission=wrong)
    finally:
        request.close(); source.close()


def test_formal_cannot_promote_small_full_index_fixture(full, case):
    c = case.make('T2')
    original = deepcopy(case.original)
    original['data']['teacher_provider']['descriptor'] = full.parent['qualification_plan']  # rejected reference
    atomic_json(case.rp, original)
    a = deepcopy(c.a); a['reference_recipe'] = tp.pin(case.rp)
    with pytest.raises(ValueError, match='original exact32'):
        tf.validate_recipe(a, c.cfg)
    # The exact101592 admission boundary also rejects synthetic smaller complete
    # populations even when all their original qualification metadata is valid.
    a = deepcopy(c.a); a['max_scenes'] = full.n
    with pytest.raises(ValueError, match='admission'): tf.validate_recipe(a, c.cfg)

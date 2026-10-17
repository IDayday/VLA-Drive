"""Small metadata-only integration; no pretrained model or teacher tensors."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from tools.st_world_policy import teacher_provider as tp
from tools.st_world_policy import temporal_composition as tc
from tools.st_world_policy.training_config import validate_config


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True))
    return tp.pin(path)


@pytest.fixture
def case(tmp_path):
    # A committed actual-config shape; all paths/content below are synthetic.
    original = json.loads((Path(__file__).resolve().parents[2] /
        'reports/st_world_policy/20261017/indexed_training2_handoff_v1/TRAINING_CONFIG.json').read_text())
    original['data'].update(root=str(tmp_path/'current'), pca_path=None,
        teacher_provider={'descriptor':{'path':str(tmp_path/'PARENT.json'), 'sha256':'a'*64}, 'pca_channel_scope':None})
    manifest = dict(schema='st_history_manifest_v1', split='train', source_student_identity='c'*64,
        source_index_sha256='d'*64, partition_sha256='e'*64, view_names=['CAM_F0','CAM_L0','CAM_R0'],
        scenes=[dict(token=str(i),log='log'+str(i),current_record_sha256='f'*64) for i in range(32)])
    manifest['identity'] = tp.identity_hash(manifest)
    hm = write(tmp_path/'history/HISTORY_MANIFEST.json', manifest)
    hp = write(tmp_path/'history/COMPLETE.json', dict(status='COMPLETE', scenes=32,
        history_identity=manifest['identity'], history_manifest_sha256=hm['sha256']))
    request = dict(identity='1'*64, split='train', source_index_sha256='d'*64,
        scenes=deepcopy(manifest['scenes']))
    requests = write(tmp_path/'REQUESTS.json', request)
    index = dict(request_identity=request['identity'])
    plan = dict(schema='st_indexed_composition_plan_v1', scenes=32,
        scene_ids=[r['token'] for r in manifest['scenes']], requests=requests, request_index=index,
        training_admission=dict(max_scenes=32,max_updates=4,exact_request_identity=request['identity']))
    descriptor = write(tmp_path/'PARENT.json', dict(schema=tp.INDEXED_SCHEMA,
        qualification_plan=write(tmp_path/'PLAN.json',plan), requests=requests,request_index=index))
    original['data']['teacher_provider']['descriptor'] = descriptor
    ref = write(tmp_path/'REF.json', original)
    cfg = deepcopy(original)
    cfg['execution'].update(scope='PROFILE', stop_after=4)
    cfg['data'].pop('teacher_provider')
    parent = write(tmp_path/'T2.json', cfg)
    selected = write(tmp_path/'SELECTED.json', dict(schema='st_fixed_scene_list_v1',split='train',
        source_index_hash='d'*64,scenes=deepcopy(manifest['scenes'])))
    a = dict(schema=tp.PROFILE_SCHEMA, scope='PROFILE', parent_descriptor=original['data']['teacher_provider']['descriptor'],
        source_migration=write(tmp_path/'M.json',{'synthetic':True}), data_manifest=selected,
        recipe=parent, reference_recipe=ref, max_scenes=256, max_updates=256,
        new_population_tensor_qualified=False, formal_approved=False)
    assert tp.validate_profile_recipe(a, cfg) == cfg
    return SimpleNamespace(**locals())


def composition(case, cell='T3'):
    cfg=deepcopy(case.cfg); a=deepcopy(case.a)
    c=dict(schema=tc.SCHEMA, cell=cell, parent_recipe=case.parent, history_complete=case.hp,
        history_identity=case.manifest['identity'], history_manifest_sha256=case.hm['sha256'], delta_weight=float(cell=='T4'))
    cfg['space_time'].update(cell=cell,history=True,delta=cell=='T4',delta_weight=c['delta_weight'])
    cfg['data']['history_root']=str(case.tmp_path/'history')
    a.update(recipe=write(case.tmp_path/(cell+'.json'),cfg),temporal_composition=c)
    return cfg,a


@pytest.mark.parametrize('cell',['T3','T4'])
def test_explicit_composition_keeps_parent_guard_and_config(cell, case):
    cfg,a=composition(case,cell)
    assert tp.validate_profile_recipe(a,cfg)==cfg
    d=dict(profile_admission=a)
    descriptor=write(case.tmp_path/(cell+'_provider.json'),d)
    cfg['data']['teacher_provider']={'descriptor':descriptor,'pca_channel_scope':None}
    assert validate_config(cfg).cell==cell
    # T2 admission cannot be reused by adding history or swapping cell.
    with pytest.raises(ValueError):tp.validate_profile_recipe(case.a,cfg)


@pytest.mark.parametrize('kind',['teacher','future_weight','steps','optimizer','precision','batch','root','history','delta','cell','extra','formal'])
def test_reject_undeclared_change(case,kind):
    cfg,a=composition(case,'T4')
    if kind=='teacher':cfg['space_time']['teacher']['channels']=256
    elif kind=='future_weight':cfg['space_time']['future_weight']=.5
    elif kind=='steps':cfg['execution']['stop_after']=5
    elif kind=='optimizer':cfg['optimizer']['lr']*=2
    elif kind=='precision':cfg['execution']['precision_policy']='other'
    elif kind=='batch':cfg['execution']['global_batch']=4
    elif kind=='root':cfg['data']['root']='/wrong'
    elif kind=='history':cfg['data']['history_root']='/wrong'
    elif kind=='delta':a['temporal_composition']['delta_weight']=.5
    elif kind=='cell':a['temporal_composition']['cell']='T2'
    elif kind=='extra':a['temporal_composition']['ignore_history']=True
    else:a['formal_approved']=True
    with pytest.raises(ValueError):tp.validate_profile_recipe(a,cfg)


@pytest.mark.parametrize('kind',['none','source','partition','role','view','identity','current','log','missing','root'])
def test_history_attachment_rejects_mismatch(case,kind):
    cfg,a=composition(case)
    h=SimpleNamespace(root=case.tmp_path/'history',manifest=deepcopy(case.manifest))
    h.index={r['token']:r for r in h.manifest['scenes']}
    source=dict(split='train',identity='c'*64,index_sha256='d'*64,partition_sha256='e'*64)
    provider=SimpleNamespace(temporal_composition=a['temporal_composition'],manifest=deepcopy(case.manifest))
    rows=deepcopy(case.manifest['scenes'])
    tc.validate_dataset_attachment(h,source,rows,provider)
    if kind=='none':h=None
    elif kind=='source':source['identity']='b'*64
    elif kind=='partition':provider.manifest['partition_sha256']='b'*64
    elif kind=='role':h.manifest['split']='dev'
    elif kind=='view':h.manifest['view_names'].reverse()
    elif kind=='identity':h.manifest['identity']='b'*64
    elif kind=='current':h.index['0']['current_record_sha256']='b'*64
    elif kind=='log':h.index['0']['log']='foreign'
    elif kind=='missing':h.index.pop('0')
    else:h.root=case.tmp_path/'foreign'
    with pytest.raises(ValueError):tc.validate_dataset_attachment(h,source,rows,provider)


def test_history_binding_rejects_failed_or_changed_receipt(case):
    cfg,a=composition(case)
    case.hp['sha256']='f'*64
    a['temporal_composition']['history_complete']=case.hp
    with pytest.raises(ValueError):tp.validate_profile_recipe(a,cfg)


def test_t2_legacy_does_not_require_history_attachment():
    tc.validate_dataset_attachment(None,{},[],SimpleNamespace())


def test_source_wiring_uses_bound_preview_and_consumed_reader():
    import ast
    root=Path(__file__).resolve().parents[2]
    dataset=ast.parse((root/'starVLA/dataloader/st_world_dataset.py').read_text())
    calls=[n for n in ast.walk(dataset) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
           and n.func.id=='validate_dataset_attachment']
    assert len(calls)==1
    assert ast.unparse(calls[0])=='validate_dataset_attachment(self.history, self.source, self.rows, teacher_provider)'
    text=(root/'tools/st_world_policy/teacher_provider.py').read_text()
    assert 'provider.temporal_composition = deepcopy(composition)' in text
    assert "or cfg['data']['history_root'] is not None" in text


@pytest.mark.parametrize('count',[0,33,64,256])
def test_direct_validator_rejects_population_expansion(case,count):
    cfg,a=composition(case)
    parent=deepcopy(case.cfg);parent['execution']['sample_limit']=count
    a['temporal_composition']['parent_recipe']=write(case.tmp_path/'T2-expanded.json',parent)
    cfg['execution']['sample_limit']=count
    a['recipe']=write(case.tmp_path/'T3-expanded.json',cfg)
    with pytest.raises(ValueError,match='<=32'):
        tc.validate_composition_recipe(a,cfg)


def test_direct_validator_rejects_full_profile(case):
    cfg,a=composition(case);a['schema']=tp.FULL_PROFILE_SCHEMA
    with pytest.raises(ValueError,match='non-full'):
        tc.validate_composition_recipe(a,cfg)


@pytest.mark.parametrize('mutation',['replace','reorder','log','source','request_identity','count'])
def test_direct_validator_binds_original_qualified_population(case,mutation):
    cfg,a=composition(case)
    selected=tp.pinned(case.selected)
    if mutation=='replace':selected['scenes'][0]['token']='unqualified'
    elif mutation=='reorder':selected['scenes'].reverse()
    elif mutation=='log':selected['scenes'][0]['log']='wrong-log'
    elif mutation=='source':selected['source_index_hash']='0'*64
    elif mutation=='count':selected['scenes'].pop()
    else:
        d=tp.pinned(case.descriptor);d['request_index']['request_identity']='0'*64
        a['parent_descriptor']=write(case.tmp_path/'WRONG_PARENT.json',d)
        original=deepcopy(case.original);original['data']['teacher_provider']['descriptor']=a['parent_descriptor']
        a['reference_recipe']=write(case.tmp_path/'WRONG_REF.json',original)
    a['data_manifest']=write(case.tmp_path/'CHANGED_SELECTED.json',selected)
    with pytest.raises(ValueError):tc.validate_composition_recipe(a,cfg)


def preparation_case(case,monkeypatch):
    """Real publication/recipe checks; external index/PCA/history IO is synthetic."""
    from starVLA.dataloader import st_world_history as history_module
    from tools.st_world_policy import provider_pca,training_config
    args=SimpleNamespace(parent_descriptor=case.descriptor['path'],reference_config=case.ref['path'],
        data_manifest=case.selected['path'],history_complete=case.hp['path'],
        source_migration=case.a['source_migration']['path'],pca_fit_cache_root=str(case.tmp_path/'fit'),
        pca_fit_cache_index=str(case.tmp_path/'fit_index'),output=str(case.tmp_path/'prepared'),
        training_output_root=str(case.tmp_path/'runs'),max_seconds=60,cell='T3')
    def preview(d):
        a=d['profile_admission'];tp.validate_profile_recipe(a,tp.pinned(a['recipe']))
        return d,None,SimpleNamespace(temporal_composition=a['temporal_composition'],manifest=deepcopy(case.manifest))
    def history(root,identity):
        return SimpleNamespace(root=Path(root),manifest=deepcopy(case.manifest),
            index={r['token']:deepcopy(r) for r in case.manifest['scenes']})
    def scope(pca,fit,descriptor,output,**kwargs):
        write(output,{'provider_descriptor':descriptor})
    monkeypatch.setattr(tp,'validate_descriptor',preview)
    monkeypatch.setattr(tp,'close_provider_metadata',lambda _:None)
    monkeypatch.setattr(history_module,'HistoryReader',history)
    monkeypatch.setattr(provider_pca,'register_provider_scope',scope)
    # The synthetic scope is intentionally not a real PCA/index qualification.
    monkeypatch.setattr(training_config,'validate_config',lambda cfg:tp.validate_profile_recipe(
        tp.pinned(cfg['data']['teacher_provider']['descriptor'])['profile_admission'],cfg))
    return args


@pytest.mark.parametrize('failure',['descriptor','attachment','scope','config'])
def test_failed_prepare_preserves_staging_and_allows_corrected_retry(case,monkeypatch,failure):
    from tools.st_world_policy import provider_pca,training_config
    args=preparation_case(case,monkeypatch)
    def fail(*a,**kw):raise ValueError('injected '+failure)
    target,name={'descriptor':(tp,'validate_descriptor'),'attachment':(tc,'validate_dataset_attachment'),
        'scope':(provider_pca,'register_provider_scope'),'config':(training_config,'validate_config')}[failure]
    good=getattr(target,name);monkeypatch.setattr(target,name,fail)
    with pytest.raises(ValueError,match='injected'):tc.prepare(args)
    out=Path(args.output);assert not out.exists() and not out.is_symlink()
    failed=list(case.tmp_path.glob('.prepared.staging-*'));assert len(failed)==1
    assert (failed[0]/'PREPARATION_FAILED.json').is_file()
    prior={p.name:p.read_bytes() for p in failed[0].iterdir()}
    # A corrected dependency root must not collide with the failed version.
    args.history_complete=write(case.tmp_path/'corrected_history/COMPLETE.json',tp.pinned(case.hp))['path']
    monkeypatch.setattr(target,name,good)
    result=tc.prepare(args);assert out.is_symlink()
    assert {p.name:p.read_bytes() for p in failed[0].iterdir()}==prior
    assert (out/'PREPARATION_COMPLETE.json').is_file()
    assert tp.pinned(result['config'])['data']['history_root']==str(case.tmp_path/'corrected_history')
    assert tc.prepare(args)==result  # Same inputs reuse the published version.
    assert len(list(case.tmp_path.glob('.prepared.staging-*')))==2
    args.max_seconds=61
    with pytest.raises(ValueError,match='different immutable inputs'):tc.prepare(args)


def test_existing_partial_namespace_is_not_overwritten(case,monkeypatch):
    args=preparation_case(case,monkeypatch);out=Path(args.output)
    old=write(out/'T3_RECIPE.json',{'historical_failed_version':True})
    with pytest.raises(FileNotFoundError):tc.prepare(args)
    assert tp.pinned(old)=={'historical_failed_version':True}

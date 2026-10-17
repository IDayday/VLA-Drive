"""Bounded synthetic metadata/tensors only; no real PROFILE qualification claim."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from tools.st_world_policy import teacher_provider as tp, qualify_indexed_provider as q
from starVLA.dataloader.st_world_metadata_index import build_index
from starVLA.dataloader.st_world_schema import identity_hash
from starVLA.dataloader.st_world_targets import atomic_json
from test_indexed_composition import setup, exercise, descriptor
from test_training_config import config


def sign(d):
    d['identity']=identity_hash({k:v for k,v in d.items() if k!='identity'});return d


@pytest.fixture(scope='module')
def f(tmp_path_factory):
    root=tmp_path_factory.mktemp('profile')
    a,p=setup(root);r=exercise(p);parent,parentpath=descriptor(p,r,root)
    archive=root/'archived_runtime';archive.mkdir()
    current=Path(tp.__file__).resolve().parents[2]
    for rel in set(q.CONTROL_PATHS+q.MATH_PATHS):
        target=archive/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(current/rel,target)
    # Explicit control-only difference; no protected mathematics changes.
    with (archive/q.CONTROL_PATHS[0]).open('a') as file:file.write('\n# archived fixture control revision\n')
    subprocess.run(['git','init','-q',str(archive)],check=True)
    subprocess.run(['git','add','.'],cwd=archive,check=True)
    subprocess.run(['git','-c','user.name=fixture','-c','user.email=fixture@invalid','commit','-qm','fixture source'],cwd=archive,check=True)
    p['source_files']=[tp.pin(archive/Path(row['path']).relative_to(current)) for row in p['source_files']]
    p['runner_source_sha256']=tp.pin(archive/'tools/st_world_policy/qualify_indexed_provider.py')['sha256'];sign(p)
    planpath=Path(parent['qualification_plan']['path']);atomic_json(planpath,p)
    r['plan_identity']=p['identity'];sign(r);resultpath=Path(parent['qualification']['path']);atomic_json(resultpath,r)
    parent.update(qualification_plan=tp.pin(planpath),qualification=tp.pin(resultpath));sign(parent);atomic_json(parentpath,parent)
    migrationpath=root/'MIGRATION.json';migration=q.prepare_control_migration(parentpath,archive,migrationpath)
    manifest=json.loads(a.requests.read_text());proto=manifest['scenes'][0]
    manifest['scenes']=[dict(deepcopy(proto),token='profile-'+str(i)) for i in range(65)];sign(manifest)
    requestpath=root/'profile_requests.json';atomic_json(requestpath,manifest)
    index=root/'profile_index';build_index(index,requests=requestpath,expected_request_identity=manifest['identity'],contract=a.fit.contract,max_scenes=256)
    selected=dict(schema='st_fixed_scene_list_v1',split='train',source_index_hash=manifest['source_index_sha256'],
        scenes=[dict(token=x['token'],log=x['log'],current_record_sha256='a'*64,ego_record_sha256='b'*64) for x in manifest['scenes']])
    selectedpath=root/'SELECTED.json';atomic_json(selectedpath,selected)
    old=config();old['execution'].update(scope='SMOKE',family='temporal',stop_after=2,sample_limit=1)
    old['runtime'].update(gradient_checkpointing=False,loader_workers=0,profile_mode='none',
        official_execution={'preset':'drivedreamer_f35fe783'})
    from starVLA.dataloader.st_world_dataset import projected_teacher_contract
    from starVLA.model.modules.st_world.config import SpaceTimeConfig
    from tools.st_world_policy.official_runtime import REFERENCE_RECIPE
    old['space_time']=SpaceTimeConfig(cell='T2', spatial_parent='S-P2',
        teacher=projected_teacher_contract(a.fit.contract,pca_identity=a.receipt['identity'],dimension=2),
        world_queries_per_view=48,sampled_positions=96,future=True,future_weight=1.).to_dict()
    old['data']['pca_path']=str(a.pca)
    effective={**{k:old['execution'][k] for k in ('world_size','micro_batch','global_batch','max_updates')},
        **old['optimizer'],'gradient_checkpointing':False}
    old['runtime']['official_execution'].update(schema='st_official_drivedreamer_execution_v1',
        scope='research_stage',adaptation_reason='Explicit CPU fixture only; not training approval.',
        declared_recipe_overrides=sorted(k for k,v in REFERENCE_RECIPE.items() if effective[k]!=v))
    old['data'].update(teacher_root=None,teacher_provider={'descriptor':tp.pin(parentpath),'pca_channel_scope':None})
    oldpath=root/'original_recipe.json';atomic_json(oldpath,old)
    recipe=deepcopy(old);recipe['execution'].update(scope='PROFILE',stop_after=256,sample_limit=65)
    recipe['data'].pop('teacher_provider');recipe['data']['teacher_root']=str(a.current.root);recipepath=root/'profile_recipe.json';atomic_json(recipepath,recipe)
    admission={'schema':tp.PROFILE_SCHEMA,'scope':'PROFILE','parent_descriptor':tp.pin(parentpath),
        'source_migration':tp.pin(migrationpath),'data_manifest':tp.pin(selectedpath),'recipe':tp.pin(recipepath),
        'reference_recipe':tp.pin(oldpath),'max_scenes':256,'max_updates':256,
        'new_population_tensor_qualified':False,'formal_approved':False}
    path=root/'PROFILE_PROVIDER.json'
    d=tp.prepare_indexed(index,[a.source_index],p['teacher_config']['path'],resultpath,path,
        limits=p['limits'],profile_admission=admission)
    return SimpleNamespace(**locals())


def test_new_profile_exact_population_old_evidence_unchanged(f):
    d,_,provider=tp.load_descriptor(tp.pin(f.path))
    try:
        assert len(provider.manifest['scenes'])==65
        assert d['profile_admission']['new_population_tensor_qualified'] is False
        assert d['qualification']==f.parent['qualification']
        assert provider.contract==f.a.fit.contract and provider.max_cached_bytes==f.p['limits']['max_cached_bytes']
    finally:tp.close_provider_metadata(provider)
    tp.validate_training_manifest({'descriptor':tp.pin(f.path)},tp.pin(f.selectedpath))
    with pytest.raises(ValueError,match='dataset path/SHA'):
        tp.validate_training_manifest({'descriptor':tp.pin(f.path)},dict(tp.pin(f.selectedpath),sha256='0'*64))


@pytest.mark.parametrize('mutation',['population','source','budget','teacher','recipe','flags','times'])
def test_profile_wrong_binding_rejected(f,tmp_path,mutation):
    d=deepcopy(f.d)
    if mutation=='population':
        selected=deepcopy(f.selected);selected['scenes'].reverse();p=tmp_path/'wrong.json';atomic_json(p,selected)
        d['profile_admission']['data_manifest']=tp.pin(p)
    elif mutation=='source':d['sources'][0]['cache_identity']='9'*64
    elif mutation=='budget':d['limits']['max_cached_bytes']+=1
    elif mutation=='teacher':d['execution']['contract_identity']='9'*64
    elif mutation=='recipe':
        recipe=deepcopy(f.recipe);recipe['optimizer']['lr']*=2;p=tmp_path/'badrecipe.json';atomic_json(p,recipe)
        d['profile_admission']['recipe']=tp.pin(p)
    elif mutation=='flags':d['profile_admission']['new_population_tensor_qualified']=True
    else:d['profile_admission']['max_updates']=257
    sign(d)
    with pytest.raises(ValueError):tp.validate_descriptor(d)


@pytest.mark.parametrize('path,value', [('stop_after',257),('sample_limit',257),('max_updates',256),('scope','FORMAL')])
def test_profile_recipe_hard_scope_bounds(f,path,value):
    cfg=deepcopy(f.recipe);cfg['execution'][path]=value
    with pytest.raises(ValueError):tp.validate_profile_recipe(f.admission,cfg)


@pytest.mark.parametrize('field',['source_sha','commit','math','extra_control','parent_result'])
def test_migration_rejects_unapproved_math_or_historical_bindings(f,field):
    m=deepcopy(f.migration)
    if field=='commit':m['parent_commit']='0'*40
    elif field=='math':m['mathematics'][0]['new']['sha256']='0'*64
    elif field=='extra_control':m['controls'].append(m['mathematics'][0])
    elif field=='parent_result':m['parent_result']['sha256']='0'*64
    else:m['parent_plan']['sha256']='0'*64
    sign(m)
    with pytest.raises(ValueError):q.validate_control_migration(f.p,m)


def test_old_smoke_default_rejects_historical_code_without_migration(f):
    with pytest.raises(ValueError,match='source'):
        q.validate_receipt(f.p,f.r,f.parent)
    q.validate_receipt(f.p,f.r,f.parent,control_migration=f.migration)


def test_no_new_source_check_boolean_api(f):
    with pytest.raises(TypeError):q.validate_receipt(f.p,f.r,f.parent,skip_source_check=True)
    with pytest.raises(ValueError):q.validate_plan(f.p,control_migration=f.migration)


def test_configured_profile_reuses_frozen_pca_and_real_config_validation(f,tmp_path):
    from tools.st_world_policy.training_config import validate_config
    before={p.name:p.read_bytes() for p in f.a.pca.iterdir() if p.is_file()}
    output=tp.configure_smoke(f.recipepath,f.path,f.a.fit.root,tmp_path/'configured',fit_cache_index=f.a.fit_index)
    cfg=json.loads(Path(output).read_text());validate_config(cfg)
    assert cfg['execution']['scope']=='PROFILE' and cfg['execution']['stop_after']==256
    assert cfg['optimizer']==f.old['optimizer'] and cfg['space_time']==f.old['space_time']
    assert cfg['runtime']['loader_workers']==0
    assert all((f.a.pca/k).read_bytes()==v for k,v in before.items())
    lineage=json.loads(Path(output).with_name('PROVIDER_CONFIG_LINEAGE.json').read_text())
    assert lineage['new_population_tensor_qualified'] is False and lineage['formal_approved'] is False


def test_profile_cannot_be_relabelled_smoke(f):
    with pytest.raises(ValueError,match='relabeled'):
        tp.validate_option({'descriptor':tp.pin(f.path),'pca_channel_scope':None},loader_workers=0,
            teacher_root=None,pca_path=None,scope='SMOKE')

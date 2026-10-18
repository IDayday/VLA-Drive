"""Metadata-only behavior tests; these are not real training qualification."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('binding',Path(__file__).with_name('bind_prepared.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def put(path,value):
    path.write_text(json.dumps(value,sort_keys=True))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:type('R',(),{'returncode':0})())
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:b'')
    stage=tmp_path/'.prepared.staging-fixture';stage.mkdir()
    (tmp_path/'prepared').symlink_to(stage.name)
    key='T0_capacity4'; control=tmp_path/'control'
    expected=dict(cell='T0',scope='PROFILE',world_size=4,micro_batch=1,profile_mode='none',
                  max_seconds=3600,training_output_root=str(control/'runs'))
    base=put(tmp_path/'BASE.json',{'provenance':'unit-fixture'})
    selected=dict(path=str(tmp_path/'SELECTED_NOT_OPENED.json'),sha256='0'*64)
    consumption=dict(cumulative_optimizer_updates=4,cumulative_scene_exposures=128,
                     world_size=4,micro_batch=1,global_batch=32,gradient_accumulation_steps=8)
    provider=dict(temporal_profile_admission=dict(scope='PROFILE',cell='T0',max_scenes=101592,
                  max_updates=256,formal_approved=False,data_manifest=selected,
                  consumption=consumption,history=None))
    pp=put(stage/'PROVIDER.json',provider); scope=put(stage/'PROVIDER_CHANNEL_SCOPE.json',{})
    target=put(stage/'TARGET_CONTRACT.json',{})
    cfg=dict(execution=dict(scope='PROFILE',family='temporal',world_size=4,micro_batch=1,
             global_batch=32,max_updates=20000,stop_after=4,sample_limit=101592,seed=42,
             precision_policy='S0_BF16_execution_FP32_master_Adam_ZeRO2_TF32_off'),
      optimizer={'warmup':1000},runtime=dict(output_root=str(control/'runs'),profile_mode='none',
             max_seconds=3600,save_every=4,loader_workers=0,gradient_checkpointing=False,deterministic=True),
      space_time={'cell':'T0'},data={'teacher_provider':{'descriptor':pp,'pca_channel_scope':scope}})
    cp=put(stage/'training_config.json',cfg); recipe=put(stage/'RECIPE.json',cfg)
    result=dict(scope='PROFILE',updates=4,training_started=False,formal_approved=False,
                new_teacher_tensor_bytes=0,diagnostic_weights_allowed=False,consumption=consumption,
                config=cp,teacher_manifest=pp)
    done=dict(schema='st_temporal_formal5k_preparation_v1',inputs=expected,result=result,
              artifacts=[pp,scope,target,cp,recipe])
    put(stage/'PREPARATION_COMPLETE.json',done)
    case=dict(preparation_output=str(tmp_path/'prepared'),expected_inputs=expected,updates=4,
              training_output_root=str(control/'runs'),profile_mode='none',cell='T0',run_id='fixture',
              control_output=str(control),native_training_argv=['NOT_RUN'])
    plan=dict(runtime={'path':str(tmp_path),'source_sha':'0'*40},cases={key:case},
              execution_files={},data_manifest=selected,base_binding=base)
    return plan,key,stage,done,cfg


def test_clean_preparation_is_not_admission(fixture):
    plan,key,*_=fixture
    result=m.bind(plan,key)
    assert result['status']=='METADATA_BOUND_NOT_ADMITTED'
    assert result['formal_launch_allowed'] is False
    assert result['training_started'] is False
    assert result['weights_usable_for_formal_init'] is False
    assert not Path(plan['data_manifest']['path']).exists()  # Binder cannot rescan population.


@pytest.mark.parametrize('field,value',[('training_started',True),('formal_approved',True),
                                        ('new_teacher_tensor_bytes',1),('updates',240)])
def test_invalid_completion_rejected(fixture,field,value):
    plan,key,stage,done,cfg=fixture
    done['result'][field]=value;put(stage/'PREPARATION_COMPLETE.json',done)
    with pytest.raises(ValueError):m.bind(plan,key)


def test_changed_artifact_rejected(fixture):
    plan,key,stage,*_=fixture
    (stage/'training_config.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):m.bind(plan,key)


def test_scheduler_reduction_rejected_even_with_valid_hash(fixture):
    plan,key,stage,done,cfg=fixture
    cfg['execution']['max_updates']=240
    cp=put(stage/'training_config.json',cfg)
    done['result']['config']=cp
    done['artifacts']=[cp if Path(p['path']).name=='training_config.json' else p for p in done['artifacts']]
    put(stage/'PREPARATION_COMPLETE.json',done)
    with pytest.raises(ValueError,match='original20k'):m.bind(plan,key)


def test_history_cell_cannot_use_current_only_receipt(fixture):
    plan,key,stage,done,cfg=fixture
    plan['cases'][key]['cell']='T4'
    with pytest.raises(ValueError,match='Wrong cell'):m.bind(plan,key)


def test_missing_complete_rejected(fixture):
    plan,key,stage,*_=fixture
    (stage/'PREPARATION_COMPLETE.json').unlink()
    with pytest.raises(FileNotFoundError):m.bind(plan,key)


def test_dirty_runtime_rejected(fixture,monkeypatch):
    plan,key,*_=fixture
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:b' M source.py')
    with pytest.raises(ValueError,match='clean'):m.bind(plan,key)


def test_document_rejects_symlink_and_oversize(tmp_path):
    p=tmp_path/'x.json';p.write_text('{}');s=tmp_path/'alias.json';s.symlink_to(p.name)
    with pytest.raises(ValueError,match='Canonical'):m.document(s)
    with p.open('wb') as stream:stream.truncate((8<<20)+1)
    with pytest.raises(ValueError,match='small'):m.document(p)

"""Read-only reference comparator fixtures, never real logs/images or teachers."""
import json
import os
from pathlib import Path
import pickle
import sys
import subprocess
from types import SimpleNamespace

from PIL import Image
import pytest

from tests.st_world.test_prepare_future_metadata import setup
from tools.st_world_policy import prepare_future_metadata as f
from tools.st_world_policy import prepare_history as legacy
from tools.st_world_policy import compare_future_metadata as reference
from starVLA.dataloader import st_world_history as history
from starVLA.dataloader.st_world_targets import atomic_json


def built_fixture(tmp_path,*,build_seconds=450.,build_bytes=4*f.GiB):
    args,_,_,_=setup(tmp_path)
    args.output=str(tmp_path/'qualification')
    args.max_seconds=build_seconds; args.max_source_read_bytes=build_bytes
    args.max_metadata_bytes=32*f.MiB
    plan=f.plan(args); root=Path(args.output)
    reg=f.signed({'schema':f.REG_SCHEMA,'plan_identity':plan['identity'],'plan_sha256':f.file_sha256(root/'PLAN.json'),
        'phase':'data_preparation','approved_scope':plan['scope'],'gpu_authorized':False,
        'teacher_extraction_authorized':False,'training_authorized':False})
    reg_path=tmp_path/'qualified_registration.json'; atomic_json(reg_path,reg)
    run=SimpleNamespace(plan=str(root/'PLAN.json'),registration=str(reg_path),registration_sha256=f.file_sha256(reg_path))
    done=f.build(run)
    compare=SimpleNamespace(**vars(run),plan_sha256=f.file_sha256(root/'PLAN.json'),complete_sha256=f.file_sha256(root/'COMPLETE.json'))
    return compare,root,plan,done


def test_complete_fieldwise_reference_and_ledger_without_output_writes(tmp_path,monkeypatch):
    args,root,_,_=built_fixture(tmp_path)
    before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(Image.Image,'load',lambda *_: (_ for _ in ()).throw(AssertionError('No decode')))
    # The comparator must use the old selector, not the new optimized selector.
    monkeypatch.setattr(f.TargetLogLookup,'scene',lambda *_a,**_k: (_ for _ in ()).throw(AssertionError('Not independent')))
    result=reference.compare(args)
    after={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert before==after
    assert result['status']=='REFERENCE_MATCH' and result['mismatch_count']==0
    assert result['scenes']==2 and result['logs']==2 and result['time_view_slots']==30
    assert result['missing_view_slots']==1 and result['checked_leaf_fields']>150
    assert result['new_cache_files_written']==0 and result['tensor_disk_bytes']==0
    assert result['combined_observed_source_bytes']<=8*f.GiB
    assert result['combined_observed_seconds']<=900
    assert result['outer_cumulative_invocation_ledger_required'] is True


@pytest.mark.parametrize('field,value',[
    ('sha256','0'*64),('original_wh',[31,18]),('bytes',7),('mtime_ns',0)])
def test_per_image_metadata_difference_reported_on_whole_scene_set(tmp_path,field,value):
    args,root,_,_=built_fixture(tmp_path)
    manifest=json.loads((root/'TEACHER_REQUESTS.json').read_text())
    manifest['scenes'][0]['frames'][1]['views'][0][field]=value
    manifest=f.signed({k:v for k,v in manifest.items() if k!='identity'})
    atomic_json(root/'TEACHER_REQUESTS.json',manifest)
    done=json.loads((root/'COMPLETE.json').read_text())
    done['files']['TEACHER_REQUESTS.json']=f.file_sha256(root/'TEACHER_REQUESTS.json')
    done['request_identity']=manifest['identity']; atomic_json(root/'COMPLETE.json',done)
    args.complete_sha256=f.file_sha256(root/'COMPLETE.json')
    result=reference.compare(args)
    assert result['status']=='REFERENCE_MISMATCH' and result['mismatch_count']==1
    assert result['first_mismatches'][0]['field'].endswith('.'+field+('[0]' if field=='original_wh' else ''))


@pytest.mark.parametrize('build_seconds,build_bytes',[(900.,4*f.GiB),(450.,8*f.GiB)])
def test_two_arms_cannot_each_claim_whole_budget(tmp_path,build_seconds,build_bytes):
    args,_,_,_=built_fixture(tmp_path,build_seconds=build_seconds,build_bytes=build_bytes)
    with pytest.raises(ValueError,match='4 GiB/450'): reference.compare(args)


def test_original_code_objects_globals_and_exception_semantics(tmp_path):
    args,root,p,_=built_fixture(tmp_path)
    budget=f.PreparationBudget(root,p['limits'])
    old_image=legacy.image_reference; old_sha=history.file_sha256; old_image_module=history.Image
    fn=reference.reference_scene_function(budget)
    assert fn.__code__ is legacy.build_target_scene.__code__
    assert fn.__globals__ is not legacy.build_target_scene.__globals__
    assert fn.__globals__['image_reference'].__code__ is history.image_reference.__code__
    frames=pickle.loads((Path(p['raw_log_root'])/'log_a.pkl').read_bytes())
    actual=fn(frames,'a','log_a','train',p['sensor_roots'])
    assert actual==legacy.build_target_scene(frames,'a','log_a','train',p['sensor_roots'])
    frames[1]['timestamp']=1490000
    frames.insert(2,dict(frames[1],token='tie',timestamp=1510000))
    for function in (fn,legacy.build_target_scene):
        with pytest.raises(ValueError,match='Ambiguous'):
            function(frames,'a','log_a','train',p['sensor_roots'])
    assert legacy.image_reference is old_image and history.file_sha256 is old_sha and history.Image is old_image_module


def test_original_raw_and_ledger_disagreement_is_not_a_match(tmp_path):
    args,_,p,_=built_fixture(tmp_path)
    raw=Path(p['raw_log_root'])/'log_a.pkl'; raw.write_bytes(raw.read_bytes()+b' ')
    result=reference.compare(args)
    assert result['status']=='REFERENCE_MISMATCH'
    assert any(row['field'].startswith('raw_log_ledger.') for row in result['first_mismatches'])


def test_mandatory_input_pins_and_header_read_bound(tmp_path):
    args,root,p,_=built_fixture(tmp_path)
    args.complete_sha256='0'*64
    with pytest.raises(ValueError,match='checksum'): reference.compare(args)
    budget=f.PreparationBudget(root,p['limits'])
    with reference.HeaderReader(tmp_path/'a.png',budget) as reader:
        with pytest.raises(ValueError,match='header'): reader.read(reference.HEADER_BYTES+1)
        with pytest.raises(ValueError,match='unbounded'): reader.read()


def test_actual_comparator_cli_only_emits_bounded_json(tmp_path,monkeypatch,capsys):
    args,root,_,_=built_fixture(tmp_path)
    argv=['compare_future_metadata']
    for key,value in vars(args).items(): argv+=['--'+key.replace('_','-'),str(value)]
    monkeypatch.setattr(sys,'argv',argv)
    assert reference.main()==0
    output=capsys.readouterr().out
    assert len(output.encode())<f.MiB and json.loads(output)['status']=='REFERENCE_MATCH'
    assert not (root/'REFERENCE.json').exists()


def test_python_m_subprocess_reference_cli(tmp_path):
    args,root,_,_=built_fixture(tmp_path)
    command=[sys.executable,'-m','tools.st_world_policy.compare_future_metadata']
    for key,value in vars(args).items(): command+=['--'+key.replace('_','-'),str(value)]
    result=subprocess.run(command,cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),
        text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr+result.stdout
    assert len(result.stdout.encode())<f.MiB
    assert json.loads(result.stdout)['status']=='REFERENCE_MATCH'
    assert not (root/'REFERENCE.json').exists()

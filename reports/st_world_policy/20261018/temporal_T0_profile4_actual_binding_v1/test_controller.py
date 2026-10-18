"""CPU binding tests; no GPU process, trainer, model or metadata preparation."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('T0_actual',HERE/'controller.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


@pytest.fixture
def c():return m.controller()


@pytest.fixture
def configured(c,monkeypatch):
    a=json.loads((HERE/'RESOURCE_ADMISSION_TEMPLATE.json').read_text())
    a.update(status='ADMITTED',reviewer='/root/independent_fixture',gpu_uuids=['GPU-fixture-'+str(i) for i in range(4)],
             cpu_affinity=list(range(8)),valid_from_unix=100,valid_until_unix=10000)
    cp=dict(remaining_bytes=a['other_commitments_bytes'],free_floor_bytes=1<<40,valid_until_unix=10000)
    real_digest=c.digest
    monkeypatch.setattr(c,'digest',lambda p:'a'*64 if str(p)=='/fixture/admission.json' else real_digest(p))
    monkeypatch.setattr(c,'read',lambda p:a if str(p)=='/fixture/admission.json' else cp)
    monkeypatch.setattr(m.time,'time',lambda:101)
    monkeypatch.setattr(m.os,'sched_getaffinity',lambda _:set(range(16)))
    return c,a,cp


def test_command_and_four_locks(c):
    argv=c.command()
    assert argv[:4]==['timeout','--signal=TERM','--kill-after=180s','3720s']
    assert '--nproc_per_node=4' in argv and '--resume' not in argv
    assert len(c.GPU_LOCKS)==4
    assert c.GPU=='0,1,2,3'
    with pytest.raises(ValueError):c.command(True)


def test_exact_native_recipe_no_relocation(c):
    inputs=json.loads(m.INPUT.read_text());cfg=json.loads(Path(inputs['pins']['config']['path']).read_text())
    c.validate_recipe(cfg)
    assert c.registered_config(cfg)==cfg
    assert c.registered_config(cfg) is not cfg
    for group,key,value in [('execution','stop_after',240),('execution','global_batch',2),
      ('execution','world_size',1),('execution','max_updates',240),('runtime','profile_mode','timing'),
      ('runtime','max_seconds',3601),('space_time','history',True)]:
        bad=copy.deepcopy(cfg);bad[group][key]=value
        with pytest.raises(RuntimeError):c.validate_recipe(bad)


def test_valid_admission_only_configures_bounds(configured):
    c,a,cp=configured
    result,pin=c.admission('/fixture/admission.json','a'*64,'prepare')
    assert result==a and pin['sha256']=='a'*64
    assert c.OTHER_COMMITMENTS==cp['remaining_bytes']
    assert c.EXPECTED_UUIDS==a['gpu_uuids']
    assert not c.OUT.exists()


@pytest.mark.parametrize('key,value',[('status','TEMPLATE_NOT_ADMITTED'),
 ('gpu','0'),('reviewer','/root/temporal_profile_prepare'),('reviewer','/root'),
 ('logical_read_limit_bytes',(1<<40)+1),('rss_limit_bytes',64<<30),('checkpoint_reservation_bytes',81<<30),
 ('automatic_resume',True),('retained_cache_reservation_bytes',0),('cpu_affinity',[0,1]),
 ('gpu_uuids',['GPU-same']*4),('valid_until_unix',4180)])
def test_negative_admission(configured,key,value):
    c,a,cp=configured
    a[key]=value
    with pytest.raises(RuntimeError):c.admission('/fixture/admission.json','a'*64,'prepare')


def test_resume_rejected_and_commitments_must_match(configured):
    c,a,cp=configured
    with pytest.raises(RuntimeError):c.admission('/fixture/admission.json','a'*64,'resume')
    cp['remaining_bytes']+=1
    with pytest.raises(RuntimeError):c.admission('/fixture/admission.json','a'*64,'run')


def test_storage_counts_full_cache_and_metadata(configured):
    c,a,cp=configured;c.admission('/fixture/admission.json','a'*64,'prepare')
    total=(1<<40)+cp['remaining_bytes']+(768<<30)+(10<<30)+(80<<30)
    assert c.storage_requirement(0)==total
    assert c.storage_requirement(40<<30)==total-(40<<30)
    for used in (-1,(80<<30)+1):
        with pytest.raises(RuntimeError):c.storage_requirement(used)


def test_missing_admission_cannot_compute_storage(c):
    with pytest.raises(RuntimeError):c.storage_requirement(0)


def test_gpu_identity_and_busy_rejection(configured,monkeypatch):
    c,a,cp=configured;c.admission('/fixture/admission.json','a'*64,'prepare')
    lines=['%s, 81920, 0, 81920, 0'%x for x in a['gpu_uuids']]
    monkeypatch.setattr(c.subprocess,'check_output',lambda *args,**kwargs:'\n'.join(lines))
    assert len(c.gpu_record(True))==4
    lines[1]=lines[0]
    with pytest.raises(RuntimeError,match='physical'):c.gpu_record(True)
    lines[1]='GPU-fixture-1, 81920, 1000, 80920, 99'
    with pytest.raises(RuntimeError,match='busy'):c.gpu_record(True)


def test_rank_zero_through_three_telemetry(c,tmp_path):
    for rank in range(4):(tmp_path/f'teacher_provider_rank{rank}.json').write_text(json.dumps({'rank':rank}))
    assert set(c.rank_telemetry(tmp_path))=={'0','1','2','3'}


def test_original_monitor_functions_remain_bound(c):
    assert c.monitor_training.__code__.co_filename==str(m.DONOR)
    assert c.execute.__code__.co_filename==str(m.DONOR)
    assert c.prepare.__code__.co_filename==str(m.DONOR)
    assert c.OwnedReadGuard.sample.__code__.co_filename==str(m.GUARD_SOURCE)
    assert c.READ_OWNED.__code__.co_filename==str(m.IO_SOURCE)

"""Narrow author tests of T0 metadata binding; never execute a metadata CLI/child."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('T0_controller',HERE/'prepare_once.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Reads:
    def raw(self,*args):return b'pin already tested by the donor'
    def document(self,pin):return {'status':'COMPLETE','scene_count':101592}


@pytest.fixture
def values(monkeypatch):
    c=m.controller()
    p=json.loads((HERE/'CPU_PLAN.json').read_text())
    a=json.loads((HERE/'CPU_ADMISSION_TEMPLATE.json').read_text())
    a.update(status='APPROVED_T0_GLOBAL32_METADATA_PREPARATION_ONLY',reviewer='/root/independent_fixture',
             hostname='testhost',issued_at_unix=100,latest_start_unix=200,valid_until_unix=2000)
    monkeypatch.setattr(c.socket,'gethostname',lambda:'testhost')
    monkeypatch.setattr(c.os,'sched_getaffinity',lambda _:set(range(16)))
    return c,p,a


def test_bound_metadata_only_commands_and_outputs(values):
    c,p,a=values
    c.validate(a,p,None,Reads(),101)
    commands=c.commands(p)
    assert [x[0] for x in commands]==['tools.st_world_policy.qualify_indexed_provider',
                                     'tools.st_world_policy.temporal_formal']
    argv=commands[1][1]
    assert argv[argv.index('--cell')+1]=='T0'
    assert argv[argv.index('--scope')+1]=='PROFILE'
    assert '--history-complete' not in argv
    assert '--world-size' in argv and '--micro-batch' in argv
    assert c.ROOT==Path(p['output_root'])
    assert c.COMMIT=='38f86a15a92171e583bb5019cce89a63fb7a1d43'
    assert c.LIMITS==dict(wall_seconds=1500,cleanup_seconds=40,read_bytes=64<<30,rss_bytes=4<<30)
    assert c.OUTPUT_LIMIT==8<<20
    assert len(c.OUTPUTS)==7 and 'configured/TARGET_CONTRACT.json' in c.OUTPUTS
    assert not c.ROOT.exists()


@pytest.mark.parametrize('field,value',[('gpu',True),('automatic_retry',True),
 ('cpu_affinity',[2,3]),('reviewer','/root/temporal_profile_prepare'),('reviewer','/root'),
 ('checkpoint_reservation_created',True)])
def test_wrong_admission_rejected(values,field,value):
    c,p,a=values;a[field]=value
    with pytest.raises(ValueError):c.validate(a,p,None,Reads(),101)


def test_changed_budget_and_expired_lease_rejected(values):
    c,p,a=values
    a['limits']={**a['limits'],'read_bytes':65<<30}
    with pytest.raises(ValueError):c.validate(a,p,None,Reads(),101)
    a['limits']=p['limits'];a['valid_until_unix']=1640
    with pytest.raises(ValueError,match='insufficient'):c.validate(a,p,None,Reads(),101)


def test_parent_cannot_be_partial_current_population(values):
    c,p,a=values
    class Partial(Reads):
        def document(self,pin):return {'status':'COMPLETE','scene_count':32}
    with pytest.raises(ValueError,match='Full TRAIN'):c.validate(a,p,None,Partial(),101)


def test_controller_cannot_create_formal_or_history_run(values):
    c,p,a=values
    assert 'history_complete' not in p['inputs']
    assert p['formal_permission'] is False and p['training_started'] is False
    assert p['max_optimizer_updates']==4 and p['sample_exposures']==128
    assert p['control_reservation']['proposed_total_reserved_bytes']==10106<<20
    assert p['control_reservation']['remaining_bytes']==134<<20

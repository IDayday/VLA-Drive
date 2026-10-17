"""Affected control paths only; no real W, checkpoint, teacher, DB, GPU or loader."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'matched_spatial_probe_controls_v1'
spec=importlib.util.spec_from_file_location('independent_matched_subject',OWNER/'control.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

def read(path):return json.loads(Path(path).read_text())

def test_prepared_three_arms_all_role_and_checkpoint_bindings():
    delivery=read(OWNER/'DELIVERY.json')
    inventory=read(c.INVENTORY)
    assert hashlib.sha256((OWNER/'control.py').read_bytes()).hexdigest()=='c5c8521379b36c979bf8743f12c2a2baf80df5ca5d21941ba2fe9d453540c38c'
    assert hashlib.sha256(c.INVENTORY.read_bytes()).hexdigest()==c.INVENTORY_SHA
    populations={}
    cache_ids=[]
    for arm,tokens in [('SP1',144),('SP2',144),('SP3',288)]:
        row=next(x for x in inventory['arms'] if x['arm']=='S-P'+arm[-1])
        folder=OWNER/'prepared'/arm
        prep=read(folder/'PREPARATION.json');fit=read(folder/'FIT_REGISTRATION.json')
        assert prep['checkpoint_identity']==row['checkpoint']['sha256']
        assert prep['checkpoint_tag']==row['checkpoint']['tag']
        assert prep['runtime']=={'path':str(c.RUNTIME),'source_sha':c.SOURCE}
        assert fit['source_sha']==c.SOURCE and fit['arm']==row['arm'] and fit['device']=='cpu'
        assert fit['protocol']['sha256']==inventory['protocol']['sha256']
        assert fit['caches']==prep['caches'] and prep['dispatch'] is False
        for split,n in [('dev',16),('train',64)]:
            b=read(folder/(split+'_BINDING.json'));sig=read(folder/(split+'_EXPECTED_SIGNATURE.json'))
            assert b['checkpoint_identity']==row['checkpoint']['sha256'] and b['arm']==row['arm']
            assert b['checkpoint_tag']==row['checkpoint']['tag'] and b['source_sha']==c.SOURCE
            assert b['hidden_dim']==2048 and b['memory_tokens']==tokens
            assert b['observations']['split']==split
            assert b['observations']['manifest']['sha256']==inventory['populations'][split]['pin']['sha256']
            assert sig['binding']==b and sig['checkpoint']==row['checkpoint']
            assert len(sig['rows'])==n and len({r['log'] for r in sig['rows']})==n
            assert sig['targets_read'] is False and sig['observations']['teacher_cache_read'] is False
            assert sig['precision']=='FP32_native_master_TF32_off'
            ids=[r['token'] for r in sig['rows']]
            if split in populations:assert ids==populations[split]
            populations[split]=ids
            cache_ids.append(sig['identity'])
        for relative,expected in delivery['files'].items():
            if relative.startswith('prepared/'+arm+'/'):
                assert hashlib.sha256((OWNER/relative).read_bytes()).hexdigest()==expected
    assert len(set(cache_ids))==6

def test_runtime_delta_has_only_approved_storage_files():
    result=c.verify_runtime()  # git metadata and two small source files only
    assert result['source_sha']==c.SOURCE and result['compute_parent']==c.OLD_SOURCE
    assert set(result['only_changed_files'])==set(c.STORAGE)
    assert result['probe_mathematics_changed'] is False

def test_original_validator_accepts_cpu_stages_rejects_wrong_runtime_checkpoint_role():
    for arm in ('SP1','SP2','SP3'):
        p,row,s=c.controls(arm,supervisor=True)
        prep=read(OWNER/'prepared'/arm/'PREPARATION.json')
        for stage in ('dev_plan','train_plan','fit'):
            admission=read(OWNER/'prepared'/arm/(stage+'_ADMISSION_TEMPLATE.json'))
            admission.update(status='APPROVED',reviewer='independent_fixture',hostname=s.socket.gethostname(),
                             valid_until_unix=100000,cpu_affinity=[6,7],matched_controller=c.self_pin())
            assert s.validate(admission,prep,stage,1000)==s.LIMITS[stage.rsplit('_',1)[-1]]
            for key,val in [('checkpoint_identity','0'*64),('checkpoint_tag','wrong'),
                            ('runtime',{'path':str(c.OLD_RUNTIME),'source_sha':c.OLD_SOURCE})]:
                changed=copy.deepcopy(prep);changed[key]=val
                with pytest.raises(ValueError):s.validate(admission,changed,stage,1000)
            with pytest.raises(ValueError):s.validate(admission,prep,'dev_extract' if stage=='fit' else 'fit',1000)

def test_original_stage_sequence_rejects_failed_or_foreign_previous(monkeypatch):
    _,_,s=c.controls('SP2',supervisor=True)
    assert s.STAGES==('dev_plan','dev_extract','train_plan','train_extract','fit')
    assert s.PREVIOUS==dict(zip(s.STAGES[1:],s.STAGES[:-1]))
    admission={'preparation':{'path':'/fixture/prep','sha256':'a'*64}}
    for stage,previous in s.PREVIOUS.items():
        record={'schema':'st_SP0_probe_stage_result_v1','status':'STOPPED_NO_AUTO_RETRY',
                'stage':previous,'preparation':admission['preparation']}
        monkeypatch.setattr(s,'read',lambda *a,**k:(record,{'path':'/fixture/prior','sha256':'b'*64}))
        with pytest.raises(ValueError,match='Previous fixed stage'):s.stage_inputs(admission,stage)
        record['status']='COMPLETE';record['preparation']={'path':'/foreign','sha256':'c'*64}
        with pytest.raises(ValueError,match='Previous fixed stage'):s.stage_inputs(admission,stage)
    for key in ('manual_retry','previous_stage_result','previous_artifact_reconciliation','storage_compatibility'):
        with pytest.raises(ValueError):s.stage_inputs({**admission,key:{}},'train_plan')

@pytest.mark.parametrize('second,expected', [([],None),([{'pid':99}],'OWNED_DESCENDANTS_AFTER_LEADER_EXIT')])
def test_actual_patched_monitor_rediscovery_with_synthetic_owner(tmp_path,second,expected):
    _,_,s=c.controls('SP3',supervisor=True)
    class Owner:
        anchor={'sid':123,'start_ticks':1};verified=1
        def __init__(self,child):self.child=child;self.snapshots=iter([[{'pid':123}],second,[]]);self.finishes=[]
        def discover(self):return next(self.snapshots,[])
        def leader_done(self):return True
        def finish(self,**kwargs):self.finishes.append(kwargs);return 0
        def close(self):pass
    class Selector:
        def register(self,*args):pass
        def get_map(self):return {}
        def select(self,*args):return []
        def close(self):pass
    child=SimpleNamespace(pid=123,stdout=io.BytesIO())
    owners=[]
    def owner(c):o=Owner(c);owners.append(o);return o
    with patch.object(s,'pidfd_open',return_value=None),patch.object(s,'OwnedSession',side_effect=owner), \
         patch.object(s.selectors,'DefaultSelector',return_value=Selector()),patch.object(s,'write_once'), \
         patch.object(s,'own_group_signal') as signals,patch.object(s,'read_counter_snapshot',return_value=(None,0)):
        result=s._supervise(['fixture-only'],cwd=tmp_path,env={},cpus=[6,7],descriptors=(),attempt=tmp_path,
            wall=3,rss=1<<30,rss_reader=lambda _:0,popen=lambda *a,**k:child,stopped={'signal':None})
    assert result['limit_reason']==expected and result['exit_code']==0
    assert len(owners)==1 and len(owners[0].finishes)==1
    assert owners[0].finishes[0]['terminate']==(expected is not None)
    assert signals.call_count==(1 if expected else 0)

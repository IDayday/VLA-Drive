"""No model/GPU load: real old inventory, exact argv and mocked process group."""
import copy,json,sys
from pathlib import Path
from datetime import timedelta
from unittest.mock import Mock
import pytest,yaml
from tools.d46_priority_rl import train_control as c
from starVLA.rl.qwen_world_compare.contracts import file_sha,fingerprint


def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))

@pytest.fixture
def profile(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    train,source=c.original_module();inventory=train.code_identity()
    output=tmp_path/'runs/D46_D2';cp=output/'checkpoints/update001614_inner0';cp.mkdir(parents=True)
    cfg=dict(parent='D_IL46K',parent_checkpoint=str(tmp_path/'IL46_PARENT'),arm='D2')
    identity=dict(schema='d44_native_rl_v1',world=8,qualification=False,parent_model_sha256=c.MODEL_SHA,recipe=cfg,source_files=inventory)
    identity['sha256']=fingerprint(identity)
    put(cp/'identity.json',identity);put(output/'identity.json',identity)
    put(cp/'files.json',{'identity.json':file_sha(cp/'identity.json')})
    put(cp/'COMPLETE',dict(world=8,contract='qwen_world_abcd_research_checkpoint_v1',files_sha256=file_sha(cp/'files.json'),updates=1614,next_inner=0))
    cfgpath=tmp_path/'cfg.yaml';cfgpath.write_text(yaml.safe_dump(cfg));p=tmp_path/'profile.json'
    argv=['--config',str(cfgpath),'--output',str(output),'--resume',str(cp),'--stop-after-update','1700']
    c.prepare(argv,p)
    return train,source,inventory,argv,p,identity


def test_only600_control_alias_and_original_argv(profile,monkeypatch):
    train,source,inventory,argv,p,identity=profile;actual=[];old_alias=train.checkpoint.init_control_group;old_global=source.CONTROL_GROUP;oldargv=sys.argv[:]
    monkeypatch.setenv('WORLD_SIZE','8');monkeypatch.setenv('RANK','0');monkeypatch.setenv('TORCHELASTIC_RUN_ID','mock_same_invocation')
    monkeypatch.setattr(source.dist,'is_initialized',lambda:True);monkeypatch.setattr(source.dist,'get_world_size',lambda:8)
    group=Mock(return_value=object());monkeypatch.setattr(source.dist,'new_group',group)
    def original():
        actual.append(sys.argv[1:]);train.checkpoint.init_control_group();return 'unchanged_main'
    monkeypatch.setattr(train,'main',original)
    try:
        source.CONTROL_GROUP=None
        result=c.run(['--control-profile',str(p),'--control-profile-sha256',file_sha(p)]+argv)
        assert result=='unchanged_main'and actual==[argv]
        group.assert_called_once_with(backend='gloo',timeout=timedelta(seconds=600))
        assert train.code_identity()==inventory and train.checkpoint.init_control_group is old_alias and sys.argv==oldargv
        receipts=list(Path(argv[3]).glob('executions/control600_*/*/rank0.json'))
        value=json.loads(receipts[0].read_text());assert value['identity_sha256']==identity['sha256']and value['native_state_modified']is False
    finally:source.CONTROL_GROUP=old_global


def test_profile_tampering_and_original_source_difference_rejected(profile,monkeypatch):
    train,_,inventory,argv,p,_=profile;sha=file_sha(p);body=json.loads(p.read_text());body['protocol']['timeout_seconds']=900;put(p,body)
    with pytest.raises(ValueError,match='profile changed'):c.validate(p,sha,argv,train)
    with pytest.raises(ValueError,match='protocol'):c.validate(p,file_sha(p),argv,train)
    body['protocol']=c.PROTOCOL;body.pop('sha256');body['sha256']=fingerprint(body);put(p,body)
    changed=dict(inventory);changed[next(iter(changed))]='wrong'
    monkeypatch.setattr(train,'code_identity',lambda:changed)
    with pytest.raises(ValueError,match='byte-identical'):c.validate(p,file_sha(p),argv,train)


def test_alias_restored_after_original_main_failure(profile,monkeypatch):
    train,source,_,argv,p,_=profile;old=train.checkpoint.init_control_group
    monkeypatch.setattr(train,'main',Mock(side_effect=RuntimeError('original CLI error')))
    with pytest.raises(RuntimeError,match='original CLI'):
        c.run(['--control-profile',str(p),'--control-profile-sha256',file_sha(p)]+argv)
    assert train.checkpoint.init_control_group is old and old is source.init_control_group

"""Actual bounded CPU native8->16 qualification plus strict adapter guards."""
import copy,json,os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import yaml
from tools.d46_world16.qualify_native import qualify,make_actor
from starVLA.rl.d46_world16.runtime import World16Extension,World16Attempt,receipt,compare_restored_B1
from starVLA.rl.d46_world16.native import prepare,transition_identity,source_inventory,CONTROL_PROTOCOL

@pytest.fixture(scope='module')
def qualified(tmp_path_factory):
    root=Path(os.environ['WORLD16_CPU_QUALIFICATION_ROOT'])if 'WORLD16_CPU_QUALIFICATION_ROOT'in os.environ else tmp_path_factory.mktemp('w16')/'native'
    result=qualify(root);return root,result


def test_actual_native8_to16_READY_and_pending_DDP_exact(qualified,tmp_path):
    root,result=qualified
    assert result['actual_distributed_backend']=='gloo'and result['READY16_parameters_Adam_scheduler_exact']and result['terminal_JSON_all16']
    assert result['world8_control_timeout_seconds']==90 and result['world16_control_timeout_seconds']==600
    assert result['all16_adapters_Adam_scheduler_RNG_scene_stream_virtual8_and_material_bitexact']and result['post_native_B1_parity_all16']
    for rank in range(16):
        parity=json.loads((root/'continuous16'/f'native_B1_restore_parity_rank{rank}_u000002.json').read_text())
        assert parity['max_abs']==dict(normalized_action=0.,requested_trajectory=0.)
        terminal=json.loads((root/'continuous16'/f'result_rank{rank}.json').read_text())
        assert terminal['world16_virtual_sampler']['lane_count']==8 and'lanes'not in terminal['world16_virtual_sampler']
    a=torch.load(root/'old8/checkpoints/update000002_inner0/optimizer.pt',weights_only=False)
    b=torch.load(root/'continuous16/checkpoints/update000002_inner0/optimizer.pt',weights_only=False)
    assert a.keys()==b.keys()and a['scheduler']==b['scheduler']=={'last_epoch':2}
    assert sorted({int(s['step'])for s in b['optimizer']['state'].values()})==[2]
    own=torch.load(root/'continuous16/checkpoints/update000004_inner0/rank15.pt',weights_only=False)
    assert own['state']['scene_cursor']==128 and own['state']['valid_PG_candidates']==2048
    assert own['state']['world16_virtual_sampler']['committed_collections']==2
    original=[x.clone()for x in own['state']['world16_virtual_sampler']['lanes']]
    public=World16Extension.json_result(dict(own['state'],status='QUALIFICATION_STOPPED'))
    json.dumps(public,allow_nan=False)
    assert all(torch.equal(a,b)for a,b in zip(original,own['state']['world16_virtual_sampler']['lanes']))
    assert all(torch.is_tensor(x)for x in own['state']['world16_virtual_sampler']['lanes'])
    from starVLA.rl.d44_native_rl.recovery import checkpoint
    evidence=checkpoint.failure_evidence(tmp_path,World16Extension.json_result(own['state']),
        'intentional CPU serializer failure-path probe, not a training failure','serializer_fixture')
    failure=json.loads((Path(evidence)/'evidence.json').read_text())
    assert failure['state']['world16_virtual_sampler']['lane_sha256']==public['world16_virtual_sampler']['lane_sha256']
    assert all(torch.equal(a,b)for a,b in zip(original,own['state']['world16_virtual_sampler']['lanes']))
    from starVLA.rl.d44_native_rl.config import defaults46
    from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
    torch.manual_seed(42);cfg=defaults46('D0');cfg['checkpoint_policy']='all_off';actor=make_actor(cfg)
    trained=torch.load(root/'continuous16/checkpoints/update000004_inner0/optimizer.pt',weights_only=False)['adapters']
    restore_adapters(actor,trained)
    inputs=dict(input_ids=torch.tensor([[3,5,7,9]]),attention_mask=torch.ones(1,4,dtype=torch.long))
    errors=compare_restored_B1(actor,trained,inputs,torch.zeros(1,8),torch.randn((1,8,3),generator=torch.Generator().manual_seed(2101)))
    assert errors==dict(normalized_action=0.,requested_trajectory=0.)


def test_runtime_exact_recipe_identity_and_journal_origin(qualified,tmp_path,monkeypatch):
    root,_=qualified;execution=root/'continuous16';identity=json.loads((execution/'identity.json').read_text())
    cfg=tmp_path/'recipe.yaml';cfg.write_text(yaml.safe_dump(identity['recipe']))
    ext=World16Extension();args=SimpleNamespace(config=str(cfg),output=str(execution),resume=str(execution/'checkpoints/update000002_inner0'),qualification=True)
    assert ext.read_recipe(args)==identity['recipe'];ext.validate_world(16)
    assert identity['native_world_transition']['control_plane']==CONTROL_PROTOCOL
    from tools.d46_world16 import gang
    monkeypatch.setattr(gang.subprocess,'check_output',lambda command,**kwargs:'mock_frozen_SHA'if 'rev-parse'in command else'')
    plan=gang.build(execution,cfg,Path.cwd(),['training-mock0','training-mock1'],29785,4)
    for part in plan['participants']:
        assert '--nnodes=2'in part['command']and'--nproc_per_node=8'in part['command']and'--standalone'not in part['command']and'--rdzv_conf=timeout=120'in part['command']
    assert plan['no_local_dispatch']and plan['no_detached_subprocess']
    with pytest.raises(ValueError,match='world16'):ext.validate_world(8)
    actual=copy.deepcopy(identity);actual.pop('native_world_transition');actual['fresh_Adam']=True
    assert ext.identity(actual)==identity
    bad=copy.deepcopy(actual);bad['scope']['trainable_numel']+=1
    with pytest.raises(ValueError,match='closure'):ext.identity(bad)
    state=torch.load(execution/'checkpoints/update000002_inner0/rank0.pt',weights_only=False)['state']
    attempt=World16Attempt(execution,0,state,identity,args.resume)
    assert attempt.record['verified_native_origin_updates']==2
    bad=copy.deepcopy(state);bad['updates']=3
    with pytest.raises(ValueError,match='history gap'):World16Attempt(execution,0,bad,identity,args.resume)
    changed=copy.deepcopy(identity);changed['native_world_transition']['common_optimizer_sha256']='changed'
    with pytest.raises(ValueError,match='receipt'):receipt(execution,changed)


def test_converter_refuses_pending_and_second_conversion(qualified,tmp_path):
    root,_=qualified
    with pytest.raises(ValueError,match='READY'):prepare(root/'old8/checkpoints/update000001_inner1',tmp_path/'bad',qualification=True)
    with pytest.raises(ValueError,match='world8 READY'):prepare(root/'continuous16/checkpoints/update000004_inner0',tmp_path/'again',qualification=True)
    cp=root/'old8/checkpoints/update000002_inner0'
    with pytest.raises(ValueError,match='never enter formal'):prepare(cp,tmp_path/'no_formal',qualification=False)
    old=json.loads((cp/'identity.json').read_text());old['qualification']=False
    new=transition_identity(old,cp,json.loads((cp/'files.json').read_text()),source_inventory(),True)
    assert new['qualification']is True and new['native_world_transition']['source_qualification']is False


def test_world16_only_control_group_hook_preserves_default90(monkeypatch):
    from datetime import timedelta
    from unittest.mock import Mock
    from starVLA.rl.simwam_recipe import checkpoint as source
    calls=Mock(return_value=object())
    previous=source.CONTROL_GROUP
    try:
        monkeypatch.setattr(torch.distributed,'is_initialized',lambda:True)
        monkeypatch.setattr(torch.distributed,'get_world_size',lambda:16)
        monkeypatch.setattr(torch.distributed,'new_group',calls)
        source.CONTROL_GROUP=None;World16Extension.init_control_group()
        calls.assert_called_once_with(backend='gloo',timeout=timedelta(seconds=600))
        assert source.CONTROL_GROUP is calls.return_value
        source.CONTROL_GROUP=None;calls.reset_mock();source.init_control_group()
        calls.assert_called_once_with(backend='gloo',timeout=timedelta(seconds=90))
    finally:source.CONTROL_GROUP=previous

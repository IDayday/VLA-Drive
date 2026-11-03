"""Narrow CPU parent-contract/strict-load tests, not full-Qwen GPU qualification."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from starVLA.rl.d44_native_rl import config, deployment, parent
from starVLA.rl.qwen_world_compare.contracts import file_sha, fingerprint
from starVLA.rl.qwen_world_compare.deployment import prune_deployment

ROOT=Path('/mnt/project/d46-priority-rl-20261103')


@pytest.fixture(autouse=True)
def own_root(monkeypatch):monkeypatch.setenv('WORLD_COMPARE_ROOT',str(ROOT))


def test_actual_registration_three_policy_arms_and_legacy_configs():
    # Actual sealed metadata/source/library/processor/parity checks, no model hash.
    value=parent.audit_export(ROOT/'IL46_PARENT',verify=False)
    assert value['parent_kind']=='verified_deployment_export'
    assert value['model_sha256']==parent.D46_MODEL_SHA
    assert value['native_ancestry']['native_model_sha256']==parent.D46_NATIVE_SHA
    assert value['native_recovery_available']is False and value['training_readouts_available']is False
    assert value['seal']['kind']=='deployment_export_COMPLETE'
    for arm in ('D0','D2','D5'):
        c=config.defaults46(arm);assert config.validate(c)==c
        assert c['schema']=='d44_native_rl_v1'and c['parent']=='D_IL46K'
        bad=copy.deepcopy(c);bad['parent_registration']['file_sha256']='0'*64
        with pytest.raises(ValueError,match='registration'):config.validate(bad)
    for arm in ('D1','D4','D6'):
        with pytest.raises(ValueError):config.defaults46(arm)
    with pytest.raises(ValueError,match='no IL R16'):parent.load(ROOT/'IL46_PARENT','cpu',readouts=True)
    legacy=Path('/mnt/project/d44-native-rl-20261103/registered_configs')
    import yaml
    for file in legacy.glob('*.yaml'):
        raw=yaml.safe_load(file.read_text());assert config.read(file)==raw
    assert config.defaults('D0')['parent']=='D_IL44K'
    assert config.defaults('D0')['parent_checkpoint']=='/mnt/project/d44-native-rl-20261103/IL44_PARENT'


class TinyDecision(nn.Module):
    def __init__(self):
        super().__init__();self.action_input=nn.Linear(3,4);self.action_output=nn.Linear(4,3)
        self.action_position=nn.Parameter(torch.randn(8,4));self.p_embeddings=nn.Parameter(torch.randn(3,4));self.r_embeddings=nn.Parameter(torch.randn(16,4))
    def discard_training_only(self):self.register_parameter('r_embeddings',None)
    def forward(self,x):return self.action_output(self.action_input(x)+self.action_position+self.p_embeddings.mean(0))


class TinyPolicy(nn.Module):
    architecture_id='D'
    def __init__(self):
        super().__init__();self.decision=TinyDecision();self.readouts=nn.Linear(4,6);self.train_heads=True
    def forward(self,x):return self.decision(x)


def test_canonical_pruned_topology_strict_parameter_restore_and_no_aux_recreation():
    torch.manual_seed(19);original=prune_deployment(TinyPolicy(),copy_model=False)
    state={k:v.detach().clone()for k,v in original.state_dict().items()};x=torch.randn(1,8,3);expected=original(x)
    info=parent.audit_export(ROOT/'IL46_PARENT',verify=False)
    with patch.object(parent,'audit_export',return_value=info),patch.object(parent,'model_from_identity',side_effect=lambda *_args,**_kw:TinyPolicy()),patch.object(parent.torch,'load',return_value=state)as loader:
        model,loaded=parent.load(ROOT/'IL46_PARENT','cpu')
    assert loaded==info and not model.train_heads and not hasattr(model,'readouts')
    assert model.decision.r_embeddings is None and not any(p.requires_grad for p in model.parameters())
    assert torch.equal(model(x),expected)
    assert loader.call_args.kwargs==dict(map_location='cpu',weights_only=True,mmap=True)
    corrupt=dict(state);corrupt.pop('decision.action_output.weight')
    with patch.object(parent,'audit_export',return_value=info),patch.object(parent,'model_from_identity',side_effect=lambda *_args,**_kw:TinyPolicy()),patch.object(parent.torch,'load',return_value=corrupt):
        with pytest.raises((ValueError,RuntimeError)):parent.load(ROOT/'IL46_PARENT','cpu')


def test_native_export_loader_rejects_changed_parent_ancestry(tmp_path):
    # Tiny native inventory tests metadata admission only, never a formal seal.
    cfg=config.defaults46('D0');info=parent.audit_export(ROOT/'IL46_PARENT',verify=False)
    def save(identity):
        identity=dict(identity);identity['sha256']=fingerprint(identity)
        (tmp_path/'identity.json').write_text(json.dumps(identity))
        torch.save({'adapters':{}},tmp_path/'optimizer.pt');(tmp_path/'transaction.json').write_text('{}')
        files={name:file_sha(tmp_path/name)for name in ('identity.json','optimizer.pt','transaction.json')}
        (tmp_path/'files.json').write_text(json.dumps(files));(tmp_path/'COMPLETE').write_text(json.dumps({'files_sha256':file_sha(tmp_path/'files.json')}))
    identity=dict(schema='d44_native_rl_v1',architecture_id='D',recipe=cfg,parent=info,parent_model_sha256=parent.D46_MODEL_SHA)
    save(identity);assert deployment.inspect(tmp_path)[0]['parent_model_sha256']==parent.D46_MODEL_SHA
    wrong=copy.deepcopy(identity);wrong['parent']['native_recovery_available']=True;save(wrong)
    with pytest.raises(ValueError,match='provenance'):deployment.inspect(tmp_path)
    wrong=copy.deepcopy(identity);wrong['recipe']['parent_registration']['file_sha256']='0'*64;save(wrong)
    with pytest.raises(ValueError,match='registration'):deployment.inspect(tmp_path)

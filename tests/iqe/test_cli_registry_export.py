from dataclasses import asdict
import importlib
from pathlib import Path
from copy import deepcopy
import json
import subprocess
import sys
import pytest
import torch
from torch import nn
import yaml
from iqe.cli import COMMANDS, parser, main
from iqe.config import load_config, resolve
from iqe.registry import ExpertRegistry
from iqe.s0_adapter import S0Adapter
from iqe.io import atomic_torch, atomic_json, read_json, file_hash, digest
from iqe.export import export_bundle, load_bundle, activate_bundle, rollback
from iqe.model import IQEModel
from iqe.scorer import TrajectoryScorer
from iqe.selector import SelectionRule
from conftest import TensorAdapter


@pytest.mark.parametrize("command", COMMANDS)
def test_real_subcommand_help(command):
    with pytest.raises(SystemExit) as e:parser().parse_args([command,"--help"])
    assert e.value.code==0


def test_import_all_main_modules():
    root=Path(__file__).resolve().parents[2]/"iqe"
    for p in root.rglob("*.py"):
        if p.name in {"agent.py","__init__.py"}:continue
        importlib.import_module("iqe."+str(p.relative_to(root).with_suffix("")).replace("/","."))


def test_config_strict_unknown_unresolved_wrong_types(tmp_path,capsys):
    config=load_config("configs/iqe/main.yaml")
    for field,value in (("unknown",True),("lr", "fast"),("wta_across_experts",True)):
        c=deepcopy(config);c["expert_train"][field]=value
        path=tmp_path/(field+".yaml");path.write_text(yaml.safe_dump(c))
        with pytest.raises(ValueError):load_config(path)
        assert main(["preflight","--config",str(path)])==2
        result=json.loads(capsys.readouterr().out);assert result["status"]=="FAILED"
    with pytest.raises(ValueError,match="unresolved"):load_config("configs/iqe/main.yaml",resolved=True)
    contract={"query_architecture":{"width":256},"query_checkpoint":"actually_trained.pt"}
    resolved=resolve(config,contract,tmp_path/"resolved.json")
    assert load_config(tmp_path/"resolved.json",resolved=True)==resolved


def event(i=0,status="served",checkpoint="weights"):
    return {"expert_id":f"expert_{i}","created_round":i,"parent":"base","architecture":{"variant":"independent"},
            "checkpoint_hash":checkpoint,"train_manifest_hash":"manifest","shared_contract_hash":"contract","status":status,"gates":{}}


def test_registry_append_only_hash_and_wrong_order(tmp_path):
    r=ExpertRegistry(tmp_path/"registry.json","contract")
    r.append(event());r.append(event(1,"training","training"));r.append(event(1,"frozen"));r.append(event(1,"candidate_only"))
    assert len(r.read())==4 and set(r.candidate_pool())=={"expert_0","expert_1"}
    with pytest.raises(ValueError):r.append(event(3) | {"created_round":2})
    with pytest.raises(ValueError):r.append(event(1,"served","changed"))
    r.append(event(1,"rejected"));assert set(r.candidate_pool())=={"expert_0"}
    with pytest.raises(ValueError):ExpertRegistry(r.path,"wrong").read()


def test_wrong_checkpoint_architecture_rejected_before_model_loading(tmp_path):
    path=tmp_path/"bad.pt";atomic_torch(path,{"kind":"dit"})
    c={"binding_status":"LOCKED_QUERY_S0","query_checkpoint":str(path),"query_checkpoint_hash":file_hash(path),"framework_contract_hash":"f"}
    with pytest.raises(ValueError,match="architecture"):S0Adapter.load_and_validate_s0(c)
    c["query_checkpoint_hash"]="wrong"
    with pytest.raises(ValueError,match="hash"):S0Adapter.load_and_validate_s0(c)


class UnitFramework(nn.Module):
    def __init__(self):
        super().__init__()
        adapter=TensorAdapter()
        self.shared=adapter.shared
        self.action_model=nn.Module()
        self.action_model.expert=adapter.action
        self.action_model.prev_weight=0.0
    def strip_auxiliary_heads(self):return self


def test_unit_bundle_save_restore_gate_smoke_rollback_and_corruption(tmp_path,features,monkeypatch):
    # Explicit synthetic unit export; true Qwen/bundle integration has separate evidence.
    framework=UnitFramework().eval()
    c={"framework_contract_hash":"f","normalizer_hash":"n","metric":{},"source_config":{},"query_architecture":{"width":8},"source_root":"unit","source_commit":"unit"}
    model=IQEModel(S0Adapter(framework,c)).eval()
    model.append_expert("expert_1")
    model.scorer=TrajectoryScorer(8,8,[],16,1,2).eval()
    features.contract_hash=digest(c)
    registry=ExpertRegistry(tmp_path/"registry.json","contract");registry.append(event());registry.append(event(1))
    rule=SelectionRule(0,{}, {},dependency_hash="pool")
    smoke=tmp_path/"smoke"
    export_bundle(model,rule,registry,smoke,mode="smoke",gate_results={"passed":False},config_hash="conf",pool_hash="pool")
    with pytest.raises(ValueError,match="smoke"):activate_bundle(tmp_path,smoke)
    monkeypatch.setattr("iqe.query_base.build_query_framework",lambda *args,**kwargs:UnitFramework())
    restored,restored_rule,metadata=load_bundle(smoke)
    before=model.forward_candidates(None,features=features);after=restored.forward_candidates(None,features=features)
    assert torch.equal(before.raw,after.raw)
    torch.testing.assert_close(model.scorer(features,before.trajectories).values,restored.scorer(features,after.trajectories).values)
    for version in ("v1","v2"):
        p=tmp_path/version
        export_bundle(model,rule,registry,p,mode="full",gate_results={"passed":True},config_hash="conf",pool_hash="pool")
        activate_bundle(tmp_path,p)
    assert read_json(tmp_path/"ACTIVE.json")["path"].endswith("v2")
    assert rollback(tmp_path)["path"].endswith("v1")
    with pytest.raises(ValueError,match="Oracle"):export_bundle(model,rule,registry,tmp_path/"oracle",mode="smoke",gate_results={"passed":False},config_hash="c",pool_hash="p",selector_type="oracle")
    data=read_json(smoke/"BUNDLE.json");data["normalizer_hash"]="wrong";data["bundle_hash"]=digest({k:v for k,v in data.items() if k!="bundle_hash"});atomic_json(smoke/"BUNDLE.json",data)
    with pytest.raises(ValueError,match="normalizer"):load_bundle(smoke)


def test_all_ablation_configs_strict_and_same_framework():
    source=load_config('configs/iqe/main.yaml')['s0']
    for p in Path('configs/iqe/ablations').glob('*.yaml'):
        assert load_config(p)['s0']==source


def test_parser_errors_emit_json(capsys):
    with pytest.raises(SystemExit) as exc:
        parser().parse_args(['train-expert','--round','nonsense'])
    assert exc.value.code==2
    assert json.loads(capsys.readouterr().err)['exit_code']==2

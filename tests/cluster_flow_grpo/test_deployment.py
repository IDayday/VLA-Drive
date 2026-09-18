"""CPU state/launcher tests, not BF16 production qualification."""
from copy import deepcopy
import json
import socket
import sys
from pathlib import Path
import pytest
import torch
from scripts.cluster_flow_grpo import cluster, deployment
from scripts.cluster_flow_grpo.paired import validate_plan
from starVLA.rl.flow_grpo.checkpoint import directory_seal, validate_checkpoint
from starVLA.rl.flow_grpo.config import config_hash


def nodes(sizes):
    return [{"host": f"node{i}", "devices": list(range(n))} for i, n in enumerate(sizes)]


def test_projection_preserves_each_global_ranks_active_rng():
    rows = deployment.rng_projection(nodes([8, 8]), nodes([8, 6, 2]))
    assert len(rows) == 16
    for row in rows:
        assert row["indices"][row["target"]["local_rank"]] == row["source"]["local_rank"]
    assert rows[14]["indices"] == rows[15]["indices"] == [6, 7]
    assert rows[8]["indices"] == list(range(6))
    with pytest.raises(ValueError, match="same nonempty world"):
        deployment.rng_projection(nodes([8, 8]), nodes([8, 6]))
    with pytest.raises(ValueError, match="only splitting"):
        deployment.rng_projection(nodes([8, 8]), nodes([6, 8, 2]))


def test_actual_torchrun_heterogeneous_cpu_rank_assignment(tmp_path):
    worker = tmp_path/"worker.py"
    worker.write_text('''import os,json,datetime
from pathlib import Path
import torch
import torch.distributed as d
d.init_process_group("gloo",timeout=datetime.timedelta(seconds=30))
x=torch.tensor(d.get_rank()+1);d.all_reduce(x);assert x.item()==6
Path(os.environ["PROBE_OUTPUT"],f"rank{d.get_rank()}.json").write_text(json.dumps({k:int(os.environ[k]) for k in ["RANK","WORLD_SIZE","LOCAL_RANK","LOCAL_WORLD_SIZE"]}))
d.destroy_process_group()
''')
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
    spec = {"job_id": "heterogeneous_cpu", "nodes": [{"host": "local", "devices": [0, 1]},
            {"host": "local", "devices": [2]}], "master_addr": "127.0.0.1", "master_port": port,
            "entry": [str(worker)], "environment": {"PROBE_OUTPUT": str(tmp_path)},
            "timeout_seconds": 45, "control_dir": str(tmp_path/"control")}
    path = tmp_path/"spec.json"; path.write_text(json.dumps(spec))
    assert cluster.run(path)["exit_codes"] == [0, 0]
    rows = [json.loads((tmp_path/f"rank{i}.json").read_text()) for i in range(3)]
    assert [(r["RANK"], r["WORLD_SIZE"], r["LOCAL_RANK"], r["LOCAL_WORLD_SIZE"]) for r in rows] == [(0,3,0,2), (1,3,1,2), (2,3,0,1)]


def checkpoint(tmp_path):
    path = tmp_path/"source/update_000002"; path.mkdir(parents=True)
    cfg = {"runtime": {"deepspeed_stage": 0}, "checkpoint_contract": {"sha256": "F"}}
    state = {"schema": 2, "update": 2, "world_size": 4, "config_hash": config_hash(cfg),
             "boundary": "complete_rollout_update", "provenance": {"sft_sha256": "F"}}
    for name, value in [("trainer_state.json",state), ("rl_config.json",cfg),
                        ("normalization.json",{}), ("sft_parameter_manifest.json",{})]:
        (path/name).write_text(json.dumps(value))
    (path/"processor").mkdir(); (path/"processor/config.json").write_text("{}")
    (path/"config.yaml").write_text("fixture: true")
    model = torch.nn.Linear(2,1); optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1,2)).sum().backward(); optimizer.step()
    torch.save(model.state_dict(),path/"pytorch_model.bin")
    torch.save(optimizer.state_dict(),path/"optimizer.bin")
    torch.save({"step":2},path/"scheduler.bin")
    for rank in range(4):
        # Distinct values expose index swaps; these are CPU fixture RNG bytes.
        rng = [torch.tensor([rank*10+i],dtype=torch.uint8) for i in range(4)]
        torch.save({"rng":{"cuda":rng,"torch":torch.get_rng_state()},"streams":[{"cursor":17}],
                    "pending":None},path/f"rank_{rank}.pt")
        torch.save({"torch_cuda_manual_seed":rng},path/f"random_states_{rank}.pkl")
    (path/"checkpoint_files.json").write_text(json.dumps(directory_seal(path)))
    (path/"COMPLETE").write_text("CPU fixture")
    return path,cfg


def test_real_checkpoint_projection_sealed_idempotent_and_source_immutable(tmp_path):
    source,cfg = checkpoint(tmp_path); before = directory_seal(source)
    dest = tmp_path/"target/update_000002"
    deployment.project_checkpoint(source,dest,nodes([4]),nodes([2,2]),cfg)
    assert directory_seal(source) == before
    validate_checkpoint(dest,cfg,{"sft_sha256":"F"},4)
    for rank in range(4):
        old = torch.load(source/f"rank_{rank}.pt",weights_only=False)
        new = torch.load(dest/f"rank_{rank}.pt",weights_only=False)
        assert torch.equal(old["rng"]["cuda"][rank],new["rng"]["cuda"][rank%2])
        assert new["streams"] == old["streams"]
    completed = directory_seal(dest)
    deployment.project_checkpoint(source,dest,nodes([4]),nodes([2,2]),cfg)
    assert directory_seal(dest) == completed
    assert (dest/"pytorch_model.bin").read_bytes() == (source/"pytorch_model.bin").read_bytes()
    assert (dest/"optimizer.bin").read_bytes() == (source/"optimizer.bin").read_bytes()
    with pytest.raises(ValueError,match="identity conflict"):
        deployment.project_checkpoint(source,dest,nodes([4]),nodes([1,3]),cfg)
    torch.save({"corrupt":True},dest/"rank_0.pt")
    with pytest.raises(ValueError,match="integrity changed"):
        deployment.project_checkpoint(source,dest,nodes([4]),nodes([2,2]),cfg)
    assert directory_seal(source) == before


def test_projection_interruption_preserves_attempt_then_retries(tmp_path,monkeypatch):
    source,cfg = checkpoint(tmp_path); dest = tmp_path/"target/update_000002"
    real = deployment.atomic_json
    def fail(path,value):
        real(path,value)
        if Path(path).name == "deployment_rng.json": raise OSError("injected publish failure")
    with monkeypatch.context() as patch:
        patch.setattr(deployment,"atomic_json",fail)
        with pytest.raises(OSError,match="injected"):
            deployment.project_checkpoint(source,dest,nodes([4]),nodes([2,2]),cfg)
    assert not dest.exists()
    deployment.project_checkpoint(source,dest,nodes([4]),nodes([2,2]),cfg)
    assert len(list(dest.parent.glob(".update_000002.incomplete.attempt-*"))) == 1


def test_plan_enforces_user_gpu_caps_with_unequal_nodes():
    plan = {"groups":{"frozen_visual":{"nodes":nodes([8,6,2])},
                      "unfrozen_visual":{"nodes":[{"host":"u1","devices":list(range(8))},
                                                     {"host":"u2","devices":list(range(8))}]}},
            "resource_limits":{"node1":{"max_gpus":6,"reserved_devices":[6,7]},
                               "node2":{"max_gpus":4}}}
    assert validate_plan(plan) == 16
    plan["groups"]["frozen_visual"]["nodes"][1]["devices"][-1] = 6
    with pytest.raises(ValueError,match="resource limit"): validate_plan(plan)


@pytest.mark.parametrize("damage",[None,"native","recipe","source_layout","cutoff","gate"])
def test_deployment_descriptor_migration_requires_unchanged_recipe_and_gpu_gate(tmp_path,monkeypatch,damage):
    from scripts.cluster_flow_grpo import identity
    from starVLA.rl.flow_grpo import config,acceptance
    old = {"training_executable_sha256":"native", "orchestration_sources":{"old":"code"},
           "plan":{"groups":{"frozen_visual":{"config":"F","nodes":nodes([4])}}}}
    new = deepcopy(old);new["orchestration_sources"]={"new":"code"}
    new["plan"]["groups"]["frozen_visual"].update(nodes=nodes([2,2]),
        resume_layout={"source_nodes":nodes([4]),"through_update":100})
    cp=tmp_path/"frozen_visual/checkpoints/update_000100";cp.mkdir(parents=True);(cp/"COMPLETE").write_text("fixture")
    (tmp_path/"cluster_identity.json").write_text(json.dumps(old))
    monkeypatch.setattr(config,"resolve_config",lambda _ : ({},None))
    monkeypatch.setattr(acceptance,"enforce_training_budget",lambda _ : None)
    called=[]
    def binding(*args):
        called.append(True)
        if damage=="gate":raise ValueError("unqualified target")
    monkeypatch.setattr(identity,"validate_binding",binding)
    group=new["plan"]["groups"]["frozen_visual"]
    if damage=="native":new["training_executable_sha256"]="changed"
    if damage=="recipe":group["config"]="U"
    if damage=="source_layout":group["resume_layout"]["source_nodes"]=nodes([2,2])
    if damage=="cutoff":group["resume_layout"]["through_update"]=200
    if damage:
        with pytest.raises(ValueError):deployment.migrate_descriptor(tmp_path,new,{})
        assert json.loads((tmp_path/"cluster_identity.json").read_text())==old
    else:
        deployment.migrate_descriptor(tmp_path,new,{})
        assert called and json.loads((tmp_path/"cluster_identity.json").read_text())==new
        assert len(list((tmp_path/"deployment_history").glob("*/previous_identity.json")))==1

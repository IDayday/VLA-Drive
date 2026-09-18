"""CPU control-flow tests; these do not certify a new production world size."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from scripts.cluster_flow_grpo import cluster
from scripts.cluster_flow_grpo import parallel_evaluation as pe
from starVLA.rl.flow_grpo.contracts import digest
from starVLA.rl.flow_grpo.evaluation_transaction import evaluation_transaction, completed_evaluation


def test_multinode_command_keeps_actual_world_and_entry():
    spec = {"nodes": [{"devices": list(range(8))}] * 2,
            "master_addr": "127.0.0.1", "master_port": 29997,
            "entry": ["-m", "starVLA.rl.flow_grpo.cli", "train", "--max-updates", "2"]}
    command = cluster.command_for(spec, 1)
    assert command[command.index("--nnodes")+1] == "2"
    assert command[command.index("--node-rank")+1] == "1"
    assert command[command.index("--nproc-per-node")+1] == "8"
    assert command[-3:] == ["train", "--max-updates", "2"]
    assert "--standalone" not in command


@pytest.mark.parametrize("exit_code", [0, 7])
def test_real_local_supervision_records_child_exit(tmp_path, exit_code):
    spec = {"job_id": "cpu_fixture", "nodes": [{"host": "local", "devices": [0]}],
            "direct_command": [sys.executable, "-c", f"raise SystemExit({exit_code})"],
            "control_dir": str(tmp_path/"control"), "timeout_seconds": 15}
    path = tmp_path/"spec.json";path.write_text(json.dumps(spec))
    if exit_code:
        with pytest.raises(RuntimeError, match="peer failed"):
            cluster.run(path)
    else:
        assert cluster.run(path)["status"] == "PASS"
    state = json.loads((tmp_path/"control/node0.json").read_text())
    assert state["exit_code"] == exit_code


def test_status_write_failure_does_not_orphan_real_child(tmp_path):
    import subprocess
    spec={"job_id":"write_failure_fixture","nodes":[{"host":"local","devices":[0]}],
          "direct_command":[sys.executable,"-c","import time; time.sleep(30)"],
          "control_dir":str(tmp_path/"control"),"timeout_seconds":10}
    path=tmp_path/"spec.json";path.write_text(json.dumps(spec))
    code="""
from scripts.cluster_flow_grpo import cluster
real=cluster.subprocess.Popen
children=[]
def spawn(*args,**kwargs):
    child=real(*args,**kwargs);children.append(child);return child
def fail(*args,**kwargs):raise OSError('injected status write failure')
cluster.subprocess.Popen=spawn
cluster.write_json=fail
try:
    cluster.supervise(SPEC,0)
except OSError:
    assert len(children)==1 and children[0].poll() is not None
else:
    raise AssertionError('injection did not run')
""".replace("SPEC",repr(str(path)))
    subprocess.run([sys.executable,"-c",code],check=True,timeout=20)


def test_no_duplicate_gpu_slot(tmp_path):
    spec = {"nodes": [{"host": "local", "devices": [0, 0]}]}
    path=tmp_path/"spec.json";path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="duplicate GPU"):
        cluster.run(path)


def test_pair_plan_preserves_equal_global_batch_and_exclusive_slots():
    from scripts.cluster_flow_grpo.paired import validate_plan
    plan={"groups":{v:{"nodes":[{"host":h,"devices":list(range(8))} for h in hosts]}
                    for v,hosts in [("frozen_visual",["a","b"]),("unfrozen_visual",["c","d"])]}}
    assert validate_plan(plan)==16
    plan["groups"]["unfrozen_visual"]["nodes"][0]["host"]="a"
    with pytest.raises(ValueError,match="overlap"):validate_plan(plan)


def test_log_shards_cover_once_and_never_split_log():
    tokens = [f"t{i}" for i in range(21)]
    logs = {t: f"log{i//3}" for i,t in enumerate(tokens)}
    bins = pe.plan_shards(tokens, logs, 4)
    assert sorted(sum(bins, [])) == sorted(tokens)
    assert len(sum(bins, [])) == len(set(sum(bins, [])))
    assignment = {t:i for i,group in enumerate(bins) for t in group}
    for log in set(logs.values()):
        assert len({assignment[t] for t in tokens if logs[t]==log}) == 1
    assert bins == pe.plan_shards(tokens, logs, 4)


def test_real_parallel_control_reuse_merge_and_integrity(tmp_path, monkeypatch):
    monkeypatch.delenv("FLASH_ATTENTION_DETERMINISTIC",raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG",raising=False)
    tokens = ["a", "b", "c", "d"]
    logs = {"a":"log1", "b":"log1", "c":"log2", "d":"log2"}
    tokenfile=tmp_path/"tokens.json";tokenfile.write_text(json.dumps(tokens))
    cache={t:"a"*64 for t in tokens}
    index={t:str(tmp_path/logs[t]/"type"/t/"metric_cache.pkl") for t in tokens}
    monkeypatch.setattr(pe, "resolve_config", lambda _: ({}, None))
    monkeypatch.setattr(pe, "configure_numerics", lambda:None)
    monkeypatch.setattr(pe, "validate_evaluation_tokens", lambda *args:True)
    monkeypatch.setattr(pe, "build_cache_index", lambda _:index)
    monkeypatch.setattr(pe, "cache_index_snapshot", lambda _:index)
    monkeypatch.setattr(pe, "selected_cache_view", lambda index,tokens,root:root)
    def identity(cfg,sft,checkpoint,path,seed,split,*args):
        import os
        assert os.environ["FLASH_ATTENTION_DETERMINISTIC"]=="1"
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"]==":4096:8"
        selected=json.loads(Path(path).read_text())
        return {"schema_version":2,"metric_assets_sha256":digest({t:cache[t] for t in selected}),
                "checkpoint":str(checkpoint),"seed":seed,"split":split}
    monkeypatch.setattr(pe,"evaluation_identity",identity)
    called=[]
    def execute(command,slot,log):
        called.append(slot["gpu"])
        arg=lambda k:command[command.index(k)+1]
        path=Path(arg("--tokens"));selected=json.loads(path.read_text())
        ident=identity(None,None,arg("--checkpoint"),path,42,"rl_dev")
        def generate(root):
            physical=np.asarray([np.full((8,3),tokens.index(t)) for t in selected])
            pred=root/"predictions/rl_dev";pred.mkdir(parents=True)
            for t,x in zip(selected,physical):np.save(pred/f"{t}.npy",x)
            np.savez(root/"trajectories.npz",tokens=selected,physical=physical,
                     normalized=np.zeros((len(selected),8,4)))
            frame=pd.DataFrame({"token":selected,"log_name":[logs[t] for t in selected],
                                "score":[tokens.index(t)/4 for t in selected],"valid":True,
                                "two_frame_extended_comfort":np.nan})
            frame.to_csv(root/"original_protocol_scores.csv",index=False)
            (root/"metric_cache_identity.json").write_text(json.dumps({t:cache[t] for t in selected}))
            return {"identity":ident,"split":"rl_dev","scene_count":len(selected),"valid":len(selected),
                    "epdms":float(frame.score.mean()),"metric_cache_identity":ident["metric_assets_sha256"],
                    "aggregation":{"two_frame_available":0,"adjacent_mapping":{}}}
        evaluation_transaction(arg("--output-dir"),ident,selected,generate)
    output=tmp_path/"result"
    def run(checkpoint="source"):
        return pe.evaluate_parallel("config",checkpoint,output,"rl_dev",tokenfile,"data","cache",42,
            [{"host":"local","gpu":i} for i in range(2)],executor=execute)
    report=run()
    assert report["complete_split"] and report["scene_count"]==4 and report["epdms"]==.375
    assert len(called)==2
    assert run()==report and len(called)==2
    with pytest.raises(ValueError,match="identity conflict"):run("other_checkpoint")
    pred=output/"predictions/rl_dev/a.npy";pred.unlink()
    with pytest.raises(ValueError,match="prediction token set"):run()


def test_reject_duplicate_or_empty_tokens():
    for tokens in ([],["a","a"]):
        with pytest.raises(ValueError):pe.plan_shards(tokens,{"a":"log"},4)


def test_real_selected_cache_view_preserves_files_and_rejects_retarget(tmp_path):
    index={}
    for token in ["a","b"]:
        path=tmp_path/"original/log/type"/token/"metric_cache.pkl"
        path.parent.mkdir(parents=True);path.write_bytes(token.encode())
        index[token]=str(path)
    view=pe.selected_cache_view(index,["a"],tmp_path/"views")
    assert pe.selected_cache_view(index,["a"],tmp_path/"views")==view
    assert (view/"log/type/a/metric_cache.pkl").read_bytes()==b"a"
    link=view/"log/type/a";link.unlink();link.symlink_to(Path(index["b"]).parent,target_is_directory=True)
    with pytest.raises(ValueError,match="view link"):
        pe.selected_cache_view(index,["a"],tmp_path/"views")


def test_actual_numeric_optimizer_shards_support_sixteen_ranks():
    from scripts.cluster_flow_grpo.optimizer_evidence import ordered_shards
    paths=[Path(f"bf16_zero_pp_rank_{i}_mp_rank_00_optim_states.pt") for i in range(16)]
    assert ordered_shards(sorted(paths))==paths
    with pytest.raises(ValueError,match="missing/duplicate"):ordered_shards(paths[:10]+paths[11:])
    with pytest.raises(ValueError,match="missing/duplicate"):ordered_shards(paths+[paths[0]])


def test_diagnostic_queue_inherits_bounded_multinode_timeout(tmp_path, monkeypatch):
    from scripts.cluster_flow_grpo import diagnostics
    control=tmp_path/"u16_full_cont_control";control.mkdir()
    (control/"result.json").write_text(json.dumps({"status":"PASS"}))
    base={"job_id":"u16_full_cont","control_dir":str(control),"timeout_seconds":2400,
          "entry":["train","--config","production.yaml"],"nodes":[]}
    path=tmp_path/"base.json";path.write_text(json.dumps(base))
    monkeypatch.setattr(diagnostics,"resolve_config",lambda _:({"runtime":{"process_group_timeout":600}},None))
    calls=[]
    def execute(path):
        calls.append(json.loads(Path(path).read_text()))
        return {"status":"PASS"}
    monkeypatch.setattr(diagnostics,"run",execute)
    monkeypatch.setattr(sys,"argv",["diagnostics","--base-spec",str(path),"--steps","scale","rl_only"])
    diagnostics.main()
    assert len(calls)==2
    for spec in calls:
        assert "runtime.process_group_timeout=600" in spec["entry"]
        assert "runtime.accumulation_steps=1" in spec["entry"]
        assert spec["timeout_seconds"]==2400
        assert spec["entry"][spec["entry"].index("--max-updates")+1]=="1"


@pytest.mark.parametrize("change", [None, "value", "dtype"])
def test_fast_boundary_observer_keeps_native_exact_checks(tmp_path, change):
    import torch
    from scripts.cluster_flow_grpo.boundary_evidence import compare
    from scripts.flow_grpo import compare_boundaries as native
    dirs=[tmp_path/"left",tmp_path/"right"]
    for i,root in enumerate(dirs):
        root.mkdir();(root/"COMPLETE").write_text("complete")
        (root/"trainer_state.json").write_text(json.dumps({"update":2,"world_size":16}))
        (root/"rl_config.json").write_text(json.dumps({"runtime":{}}))
        value=torch.arange(10,dtype=torch.float32)
        if i and change=="value":value[3]+=1
        if i and change=="dtype":value=value.double()
        torch.save({"model":value,"rng":np.arange(3),"pending":{"inner_epoch":1}},root/"state.pt")
    expected=tmp_path/"native.json";actual=tmp_path/"fast.json"
    before=native.tensor_comparison
    if change:
        with pytest.raises(AssertionError):native.compare_boundaries(*dirs,expected)
        with pytest.raises(AssertionError):compare(*dirs,actual)
    else:
        native.compare_boundaries(*dirs,expected);compare(*dirs,actual)
    assert native.tensor_comparison is before
    a,b=[json.loads(p.read_text()) for p in [expected,actual]]
    assert a["status"]==b["status"]
    for key,row in a["files"]["state.pt"]["values"].items():
        assert row["allclose"]==b["files"]["state.pt"]["values"][key]["allclose"]


def test_final_dev_seed_schedule_uses_disjoint_gpus_and_all_fixed_seeds():
    import threading
    from scripts.cluster_flow_grpo.paired import evaluate_seeds
    slots=[{"host":str(i//8),"gpu":i%8} for i in range(16)]
    barrier=threading.Barrier(4);lock=threading.Lock();active=set();calls=[]
    def evaluate(checkpoint,label,split,seed,eval_slots):
        chosen={(s["host"],s["gpu"]) for s in eval_slots}
        assert len(chosen)==4
        with lock:
            assert not active & chosen
            active.update(chosen);calls.append(seed)
        if seed<46:barrier.wait(timeout=5)
        with lock:active.difference_update(chosen)
        return seed
    assert evaluate_seeds(evaluate,"checkpoint","sft","rl_dev",slots)==[42,43,44,45,46]
    assert sorted(calls)==[42,43,44,45,46]
    calls=[]
    def navtest(checkpoint,label,split,seed,eval_slots):
        assert eval_slots==slots;calls.append(seed);return seed
    assert evaluate_seeds(navtest,"checkpoint","last","navtest",slots)==[42,43,44,45,46]
    assert calls==[42,43,44,45,46]


def test_controller_lock_rejects_real_second_process_then_allows_restart(tmp_path):
    import subprocess
    code="from scripts.cluster_flow_grpo.cluster import exclusive_controller\nwith exclusive_controller("+repr(str(tmp_path))+"): pass"
    with cluster.exclusive_controller(tmp_path):
        result=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,timeout=10)
        assert result.returncode!=0 and "controller already active" in result.stderr
    subprocess.run([sys.executable,"-c",code],check=True,timeout=10)

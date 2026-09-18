"""Exercise the real paired CLI with injected model/cluster executors on CPU.

These synthetic artifacts test planning/publication/reuse, never RL quality.
The native checkpoint planner and evaluation transactions actually execute.
"""
import json
from pathlib import Path
import sys
import threading
import numpy as np
import pandas as pd
import pytest
from scripts.cluster_flow_grpo import paired
from starVLA.rl.flow_grpo.contracts import digest
from starVLA.rl.flow_grpo.evaluation_transaction import evaluation_transaction


@pytest.mark.parametrize("asynchronous", [False, True])
def test_paired_main_reuses_steps_last_best_and_complete_restart(tmp_path, monkeypatch, asynchronous):
    tokens=["a","b"];cache={t:"fixture" for t in tokens}
    for name in ("dev_tokens.json","navtest.json"):
        (tmp_path/name).write_text(json.dumps(tokens))
    plan={"output":str(tmp_path/"run"),"navtest_cache":"fixture_cache","groups":{}}
    for v,hosts in [("frozen_visual",["a","b"]),("unfrozen_visual",["c","d"])]:
        plan["groups"][v]={"config":v,"master_addr":hosts[0],"master_port":12345,
            "nodes":[{"host":h,"devices":list(range(8))} for h in hosts]}
    if asynchronous:
        plan["async_evaluation"]={"slots":[{"host":"evaluator","gpu":0}],"poll_seconds":.001}
    plan_file=tmp_path/"plan.json";plan_file.write_text(json.dumps(plan))
    def config(variant):
        return {"runtime":{"accumulation_steps":1},"sft_checkpoint":str(tmp_path/(variant+".pt")),
                "paths":{"split_manifest":str(tmp_path/"split.json"),
                         "test_list":str(tmp_path/"navtest.json"),"data_root":"fixture_data",
                         "metric_cache":"fixture_cache"}},None
    monkeypatch.setattr(paired,"resolve_config",config)
    monkeypatch.setattr(paired,"enforce_training_budget",lambda *a:None)
    monkeypatch.setattr(paired,"validate_binding",lambda *a:None)
    monkeypatch.setattr(paired,"resume_assets",lambda *a:{})
    monkeypatch.setattr(paired,"acceptance_context",lambda *a:{})
    monkeypatch.setattr(paired,"training_provenance",lambda *a:{})
    monkeypatch.setattr(paired.signal,"signal",lambda *a:None)
    monkeypatch.setattr(paired,"validate_checkpoint",lambda path,*a:json.loads((Path(path)/"trainer_state.json").read_text()))
    trained=[];generated=[];lock=threading.Lock()
    def train(spec_path,**kwargs):
        spec=json.loads(Path(spec_path).read_text());entry=spec["entry"]
        root=Path(entry[entry.index("--output-dir")+1]);target=int(entry[entry.index("--max-updates")+1])
        with lock:trained.append((root.name,target))
        for update in (range(100,target+1,100) if asynchronous else [target]):
            cp=root/"checkpoints"/f"update_{update:06d}";cp.mkdir(parents=True)
            (cp/"trainer_state.json").write_text(json.dumps({"update":update}))
            (cp/"COMPLETE").write_text("fixture")
    monkeypatch.setattr(paired,"cluster_run",train)
    monkeypatch.setattr(paired,"validate_export",lambda cp,dest:(Path(dest)/"COMPLETE").is_file())
    def export(command,*args):
        root=Path(command[command.index("--output-dir")+1]);root.mkdir()
        (root/"COMPLETE").write_text("fixture")
    monkeypatch.setattr(paired,"run_evaluator",export)
    def cpu_export(checkpoint,root):
        root=Path(root);root.mkdir(exist_ok=True);(root/"COMPLETE").write_text("fixture");return root
    monkeypatch.setattr(paired,"export_checkpoint",cpu_export)
    def evaluate(config,checkpoint,output,split,tokens_file,data_root,metric_cache,seed,slots,**kwargs):
        if asynchronous and not any(x["host"]=="evaluator" for x in slots):
            # The native final budget must finish before final eval borrows the
            # group's training allocation; dev sidecars use only evaluator.
            assert (tmp_path/"run"/config/"checkpoints/update_002000/COMPLETE").is_file()
        identity={"checkpoint":str(checkpoint),"split":split,"seed":seed,"metric_assets_sha256":digest(cache)}
        score=.5+(int(Path(checkpoint).name.replace("export_update",""))/10000 if "export_update" in str(checkpoint) else 0)
        def execute(root):
            with lock:generated.append((config,str(checkpoint),split,seed))
            values=np.zeros((2,8,3));pred=root/"predictions"/split;pred.mkdir(parents=True)
            for token,value in zip(tokens,values):np.save(pred/f"{token}.npy",value)
            np.savez(root/"trajectories.npz",tokens=tokens,physical=values,normalized=np.zeros((2,8,4)))
            pd.DataFrame({"token":tokens,"log_name":["l1","l2"],"score":score,"valid":True}).to_csv(root/"original_protocol_scores.csv",index=False)
            (root/"metric_cache_identity.json").write_text(json.dumps(cache))
            return {"identity":identity,"scene_count":2,"valid":2,"epdms":score,"split":split,
                    "metric_cache_identity":digest(cache),"complete_split":True}
        return evaluation_transaction(output,identity,tokens,execute)
    monkeypatch.setattr(paired,"evaluate_parallel",evaluate)
    monkeypatch.setattr(sys,"argv",["paired","--plan",str(plan_file)])
    paired.main()
    expected_trains=2 if asynchronous else 22
    assert len(trained)==expected_trains and len(generated)==60 and len(set(generated))==60
    for variant in plan["groups"]:
        root=tmp_path/"run";assert (root/variant/"EXPERIMENT_COMPLETE.json").is_file()
        loc=json.loads((root/(variant+"_evaluation")/"locations.json").read_text())
        assert loc["step2000/rl_dev/42"]==loc["last/rl_dev/42"]==loc["best/rl_dev/42"]
        for split in ["rl_dev","navtest"]:
            for seed in range(42,47):assert loc[f"last/{split}/{seed}"]==loc[f"best/{split}/{seed}"]
    monkeypatch.setattr(sys,"argv",["paired","--plan",str(plan_file),"--resume"])
    paired.main()
    assert len(trained)==expected_trains and len(generated)==60

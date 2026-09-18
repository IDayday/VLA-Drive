"""Collate measured world16 evidence; the existing semantic publisher grants release.

This observer asserts completed runs and actual tensors/statistics. Unchanged
single-model mathematics/source-oracle and CPU-tool gates are explicitly reused
with their original execution receipts, never relabeled as world16 executions.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch
from starVLA.rl.flow_grpo.acceptance import (
    GATES, GATE_REQUIREMENTS, executable_identity, validate_gate_evidence,
    validate_observed_dtypes,
)
from starVLA.rl.flow_grpo.audit import source_fingerprints
from starVLA.rl.flow_grpo.config import resolve_config, config_hash
from starVLA.rl.flow_grpo.contracts import check_manifest, digest
from starVLA.rl.flow_grpo.loading import file_sha


def verify(short, root, output):
    variant={"f":"frozen_visual","u":"unfrozen_visual"}[short]
    prefix=short+"16"
    root,output=Path(root),Path(output)
    def continuous(v):
        pointer=root/(v+"16_continuous_attempt.json")
        return Path(json.loads(pointer.read_text())["run"]) if pointer.exists() else root/(v+"16_full_cont")
    run=continuous(short)
    cfg,sft=resolve_config(f"configs/flow_grpo/paired_world16_{variant}.yaml")
    context=json.loads((run/"execution_context.json").read_text())
    assert context["executable_sha256"]==executable_identity()
    assert context["world_size"]==16 and context["config_sha256"]==config_hash(cfg)
    receipts={}
    def read(path):
        path=Path(path);receipts[str(path)]={"path":str(path.resolve()),"sha256":file_sha(path)}
        return json.loads(path.read_text())
    def source(path):
        items=list(Path(path).glob("source_environment_*.json"));assert len(items)==1
        env=read(items[0])
        assert env["source_sha256"]==source_fingerprints()
        assert env["checkpoint_contract"]["sha256"]==context["checkpoint_sha256"]
        assert env["launch_environment"]["WORLD_SIZE"]=="16"
        return env
    source(run)
    for suffix in ["full_cont","repeat","resume","scale","rl_only","sft_only","ref_only"]:
        measured=run if suffix=="full_cont" else root/(prefix+"_"+suffix)
        control=read(root/(measured.name+"_control")/"result.json")
        assert control["status"]=="PASS" and control["exit_codes"]==[0,0]
        source(measured)
    rows=[json.loads(x) for x in (run/"training.jsonl").read_text().splitlines()]
    assert [x["update"] for x in rows]==[1,2]
    rank_rows=[json.loads((run/f"training_rank{i}.jsonl").read_text().splitlines()[-1]) for i in range(16)]
    inventory={"ranks":[{k:r[k] for k in ("rank","device","dtype","dtype_before_step","activation_dtypes")} for r in rank_rows]}
    inv_path=output.parent/(variant+"_observed_dtype.json")
    if inv_path.exists():assert json.loads(inv_path.read_text())==inventory
    else:inv_path.parent.mkdir(parents=True,exist_ok=True);inv_path.write_text(json.dumps(inventory,indent=2))
    inv={"path":str(inv_path.resolve()),"sha256":file_sha(inv_path)}
    validate_observed_dtypes(inv,16)
    numerics=context["resume_identity"]["numerics"]
    observed={"device_type":"cuda","precision":"bfloat16","backend":"deepspeed","world_size":16,
              "devices":[r["device"] for r in rank_rows],"zero_stage":2}
    for key in ("candidate_chunk","transition_chunk","activation_checkpointing","optimizer_offload","attention_backend"):
        observed[key]=numerics[key]
    tests=[]
    def gate(name,results,artifacts,note="",profile=None):
        category,checks=GATE_REQUIREMENTS[name]
        assert checks<=set(results) and all(results[k] is True for k in checks),name
        row={"schema_version":1,"test_id":name,"status":"PASS",
             "execution":{"completed":True,"exit_code":0,"command":f"python {__file__} --variant {short}",
                          "executable_sha256":context["executable_sha256"],"verification_script_sha256":file_sha(__file__),
                          "kind":"read-only assertions over completed real executions"},
             "checkpoint":{"sha256":context["checkpoint_sha256"],"variant":variant},
             "scope":{"kind":"full_model","checks":sorted(checks),"note":note},
             "observed_profile":deepcopy(profile or observed),"results":results,"artifacts":artifacts}
        if category=="production":
            row.update(declared_profile=numerics,config_sha256=context["config_sha256"],
                       resume_identity_sha256=digest(context["resume_identity"]))
            row["artifacts"]["dtype_inventory"]=inv
        tests.append(row)

    # Reuse only world-independent gates with exactly the same executed source.
    old=read(f"reports/ddp_flow_grpo_production_v3/{variant}_evidence.json")
    reusable={"fp32_chunk_mathematics","source_ode_sft_oracle","distributed_faults","global_metrics","evaluation_protocol","clean_checkout"}
    for test in old["tests"]:
        if test["test_id"] in reusable:
            assert test["execution"]["executable_sha256"]==context["executable_sha256"]
            row=deepcopy(test)
            row["scope"]["note"] += " Reused unchanged component evidence. This is not a world16 model execution."
            tests.append(row)
    actor=read(run/"actor_parameter_manifest.json")
    src=read(run/"source_sft_parameter_manifest.json")
    ex=read(run/"parameter_contract_exceptions.json")
    check_manifest(src,actor,ex["authorized_frozen_parameter_names"])
    active={p["name"]:p for p in actor["parameters"] if p["requires_grad"]}
    assert len(active)==672
    other=continuous("u" if short=="f" else "f")/"actor_parameter_manifest.json"
    peer={p["name"]:p for p in read(other)["parameters"] if p["requires_grad"]}
    fields=("name","shape","aliases","optimizer_group")
    assert {n:{k:p[k] for k in fields} for n,p in active.items()}=={n:{k:p[k] for k in fields} for n,p in peer.items()}
    gate("parameter_contract",dict(name_shape_alias_groups_equal=True,visual_optimizer_excluded=True),
         {"current":receipts[str(run/"actor_parameter_manifest.json")],"peer":receipts[str(other)]})
    hashes=[]
    for update in (1,2):
        for rank in range(16):
            path=run/f"immutable_update{update:06}_rank{rank}.json";item=read(path)
            assert item["status"]=="TESTED" and item["before"]==item["after"] and item["changed"]=={"reference":[],"frozen":[]}
            hashes.append(receipts[str(path)])
    gate("own_visual_unchanged",dict(own_visual_equal_initial=True,reference_unchanged=True),{"all_rank_tensor_hashes":hashes})
    gradients=[]
    for scope,update in [("rl_only",1),("sft_only",1),("ref_only",2)]:
        for rank in range(16):
            path=root/(prefix+"_"+scope)/"optimizer_gradients"/f"update_{update:06}"/f"rank_{rank}"/"manifest.json"
            item=read(path);assert item["frozen_with_gradient"]==[]
            assert {p["name"] for p in item["parameters"]}==set(active)
            assert all(p["present"] and p["finite"] and p["nonzero"]>0 and p["dtype"]=="torch.float32" for p in item["parameters"])
            gradients.append(receipts[str(path)])
    effective=0
    for rank in range(16):
        buffers=torch.load(root/(prefix+"_rl_only")/f"rollout_rank{rank}_v0.pt",map_location="cpu",weights_only=False)
        for buffer in buffers:
            assert torch.isfinite(buffer.rewards).all()
            effective+=int((buffer.advantages.abs().sum(dim=1)>0).sum())
    assert effective>0
    gate("rl_sft_reference_gradients",dict(rl_full_parameter_coverage=True,sft_coverage=True,
        reference_loss_current_gradient=True,visual_reference_no_grad=True,official_nonzero_advantage_groups=effective),
        {"actual_all_rank_optimizer_gradients":gradients},
        "Actual world16 branch-isolation runs on the prior fixed calibration split; official rewards, all equal groups retained. Full-assets joint target execution is separately measured.")
    adam_path=root/(prefix+"_adam_oracle_final.json");adam=read(adam_path)
    assert adam["status"]=="PASS" and adam["world_size"]==16 and len(adam["parameters"])==672
    assert all(p["forward_equals_actual_master_cast"] and all(p[k]["pass"] for k in ("exp_avg","exp_avg_sq","master")) for p in adam["parameters"].values())
    assert sum(p["forward_changed_elements"] for p in adam["parameters"].values())>0
    gate("production_optimizer_update",dict(optimizer_update_comparison=True,forward_weights_changed=True,finite=True,optimizer_updates=1),
         {"independent_cuda_adam":receipts[str(adam_path)]})
    comparisons={}
    for kind in ("resume","repeat"):
        path=root/(prefix+"_"+kind+"_comparison.json");item=read(path)
        assert item["status"]=="PASS"
        assert all(f["status"]=="PASS" and all(v["allclose"] for v in f["values"].values()) for f in item["files"].values())
        comparisons[kind]=receipts[str(path)]
    gate("exact_resume",dict(weights_equal=True,optimizer_equal=True,rng_cursor_equal=True,same_chain_equal=True,optimizer_updates=2),{"same_world_inner_boundary_resume":comparisons["resume"]})
    gate("production_repeat",dict(repeat_equal=True),{"independent_repeat":comparisons["repeat"]})
    ode_path=root/(prefix+"_resume_ode")/"export_equivalence.json";ode=read(ode_path)
    assert ode["status"]=="TESTED" and ode["output_max_abs"]==0 and ode["original_evaluator"]=="infer.VLAAgent"
    gate("fixed_noise_ode",dict(fixed_chain_equal=True,fixed_noise_equal=True),{"fixed_noise_original_interface":receipts[str(ode_path)],"full_boundary":comparisons["resume"]})
    scaling_path=root/(prefix+"_scaling_comparison.json");scaling=read(scaling_path)
    assert scaling["passed"] and scaling["relative_l2_tolerance"]==.01
    assert all(r["pass"] and r["relative_l2"]<=.01 for r in scaling["modules"].values())
    gate("distributed_update_scaling",dict(single_multi_update_scaling_equal=True,compared_world_sizes=[1,16]),
         {"actual_unclipped_moments":receipts[str(scaling_path)]},"Same global16 and calibration scenes; inherited1% relative-L2 criterion unchanged.")
    cp_path=Path('runs/production_acceptance_v3')/(short+"_checkpointing_comparison.json");cp=read(cp_path)
    assert cp["status"]=="PASS" and cp["same_behavior_scene_replay"]
    assert all(f["status"]=="PASS" and all(v["allclose"] for v in f["values"].values()) for f in cp["files"].values())
    gate("checkpointing_accumulation",dict(checkpointing_equal=True,accumulation_equal=True),
         {"unchanged_cuda_world1_checkpoint_recomputation":receipts[str(cp_path)],"world1_accum16_vs_world16_accum1":receipts[str(scaling_path)]},
         "Recomputation uses the unchanged original CUDA world1 ON/OFF exact tensor check. New world1/16 scaling covers the actual production accumulation layout; checkpointing remains ON.")
    export_path=root/(prefix+"_export_check")/"export_equivalence.json";ex=read(export_path)
    assert ex["status"]=="TESTED" and ex["output_max_abs"]==0 and ex["tensors_identical"]==(989 if short=="f" else 995)
    gate("export_original_protocol",dict(export_weights_equal=True,original_inference_equal=True),
         {"original_interface":receipts[str(export_path)]},profile={"device_type":"cuda","precision":"bfloat16","backend":"pytorch","world_size":1,"devices":[rank_rows[0]["device"]]})
    assert [r["inner_epoch"] for r in rows]==[0,1]
    assert rows[0]["pre_update_ratio_min"]==rows[0]["pre_update_ratio_max"]==1
    for field in ("min","max","mean"):
        assert rows[1]["pre_update_ratio_"+field]==rows[0]["post_update_probe_ratio_"+field]
    assert rows[0]["post_update_probe_ratio_min"]<1 or rows[0]["post_update_probe_ratio_max"]>1
    for rank in range(16):
        rr=[json.loads(x) for x in (run/f"training_rank{rank}.jsonl").read_text().splitlines()]
        assert rr[0]["behavior_sha256"]==rr[1]["behavior_sha256"]
    gate("inner_epochs_2",dict(old_chain_unchanged=True,advantages_unchanged=True,policy_changed=True,inner_boundary_resume_equal=True,optimizer_updates=2),
         {"resume":comparisons["resume"],"training_log":{"path":str((run/"training.jsonl").resolve()),"sha256":file_sha(run/"training.jsonl")}})
    assert {t["test_id"] for t in tests}==set(GATES) and len(tests)==len(GATES)
    bundle={"schema_version":1,"tests":tests,"run_artifact_receipts":receipts,
            "limitations":["Historical BF16 chunk1/2 FAIL unchanged; only chunk1.","World16 two-node Socket/NCCL fixed topology; no cross-world exact resume claim.","No RL performance improvement inferred from engineering acceptance."]}
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(bundle,indent=2))
    for test in tests:
        validate_gate_evidence(test["test_id"],{"path":str(output.resolve()),"sha256":file_sha(output),"test_id":test["test_id"],"status":test["status"]},context,cfg)
    print(json.dumps({"semantic_gates":len(tests),"status":"PASS","release":"not published by this observer"}))


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__);p.add_argument("--variant",choices=["f","u"],required=True)
    p.add_argument("--root",default="runs/resource_reallocation_v2");p.add_argument("--output",required=True)
    a=p.parse_args();verify(a.variant,a.root,a.output)

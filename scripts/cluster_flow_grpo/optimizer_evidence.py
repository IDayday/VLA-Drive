"""Actual optimizer checks with numeric shard ordering for worlds >=10.

The existing world1/4 comparators sort shard filenames lexicographically, which
places rank10 before rank2. This observer keeps the same numerical criteria and
reads contiguous numeric rank IDs; it does not modify training or optimizer data.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import torch
from scripts.flow_grpo.check_adam_update import comparison
from starVLA.rl.flow_grpo.loading import weight_path, file_sha


def ordered_shards(paths):
    pairs=[]
    for path in paths:
        match=re.fullmatch(r"bf16_zero_pp_rank_(\d+)_mp_rank_00_optim_states.pt",Path(path).name)
        if match is None:raise ValueError(f"unexpected optimizer shard: {path}")
        pairs.append((int(match[1]),Path(path)))
    pairs.sort()
    if not pairs or [i for i,_ in pairs]!=list(range(len(pairs))):
        raise ValueError("missing/duplicate numeric optimizer rank")
    return [p for _,p in pairs]


def load_boundary(path):
    path=Path(path)
    if not (path/"COMPLETE").is_file():raise ValueError("incomplete optimizer boundary")
    # Lustre mmap page faults are much slower than bounded sequential reads.
    # This is page-cache warming, never a replacement for content checks.
    def warm(p):
        with p.open("rb") as stream:
            while stream.read(8*1024**2):
                pass
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(warm,[p for p in path.rglob("*.pt") if p.stat().st_size>1_000_000]))
    metadata=torch.load(next(path.glob("*/mp_rank_00_model_states.pt")),map_location="cpu",weights_only=False,mmap=True)
    paths=ordered_shards(path.glob("*/bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt"))
    shards=[torch.load(p,map_location="cpu",weights_only=False,mmap=True)["optimizer_state_dict"] for p in paths]
    world=json.loads((path/"trainer_state.json").read_text())["world_size"]
    if len(shards)!=world or any(s["zero_stage"]!=2 or max(s["partition_count"])!=world for s in shards):
        raise ValueError("optimizer shard world/stage mismatch")
    return metadata,shards


def update_learning_rate(training_row, options, group, cfg):
    """A saved optimizer already contains NEXT update's scheduled LR."""
    schedule = cfg.get("optimizer", {}).get("schedule")
    if "lr_used" not in training_row:
        if schedule:
            raise ValueError("scheduled Adam oracle requires measured lr_used")
        return options["lr"]
    used = float(training_row["lr_used"][group])
    if not math.isfinite(used) or used < 0:
        raise ValueError("invalid measured update LR")
    if schedule:
        from starVLA.rl.flow_grpo.stability import lr_multiplier
        initial = options["initial_lr"]
        update = training_row["update"]
        expected = initial * lr_multiplier(update - 1, schedule)
        saved_next = initial * lr_multiplier(update, schedule)
        if used != expected or options["lr"] != saved_next:
            raise ValueError("measured/saved LR differs from registered scheduler")
    elif used != options["lr"]:
        raise ValueError("constant LR differs from saved optimizer")
    return used


def adam(root):
    root=Path(root)
    cfg=json.loads((root/"rl_config.json").read_text())
    metadata,shards=load_boundary(root/"checkpoints/update_000001")
    source=torch.load(weight_path(cfg["sft_checkpoint"]),map_location="cpu",weights_only=False,mmap=True)
    source=source.get("module",source)
    folder=root/"optimizer_gradients/update_000001/rank_0"
    gradients={r["name"]:r for r in json.loads((folder/"manifest.json").read_text())["parameters"]}
    training_row=json.loads((root/"training_rank0.jsonl").read_text().splitlines()[0])
    norm=torch.tensor(training_row["pre_clip_grad_norm"],device="cuda",dtype=torch.float32)
    limit=cfg["optimizer"]["max_grad_norm"]
    scale=1/torch.clamp((norm+1e-6)/limit,min=1) if limit>0 else torch.ones_like(norm)
    result={"status":"PASS","world_size":len(shards),"numeric_rank_order":list(range(len(shards))),
            "master_bound_eps":8,"moment_relative_bound":4e-6,"clip_scale":float(scale),"parameters":{}}
    for group,shapes in enumerate(metadata["param_shapes"]):
        bases=[s["base_optimizer_state"] for s in shards]
        options=bases[0]["param_groups"][group]
        used_lr=update_learning_rate(training_row,options,group,cfg)
        result.setdefault("learning_rates",[]).append({"group":group,"used":used_lr,"saved_next":options["lr"]})
        ids=[b["param_groups"][group]["params"][0] for b in bases]
        states=[b["state"][i] for b,i in zip(bases,ids)]
        if any(int(s["step"])!=1 for s in states):raise ValueError("requires first Adam update")
        actual={**{k:torch.cat([s[k] for s in states]) for k in ("exp_avg","exp_avg_sq")},
                "master":torch.cat([s["single_partition_of_fp32_groups"][group] for s in shards])}
        offset=0
        for name,shape in shapes.items():
            count,key=math.prod(shape),name.removeprefix("policy.")
            info=gradients[key]
            if info["dtype"]!="torch.float32" or not info.get("file"):raise ValueError("no actual FP32 gradient")
            gradient=torch.load(folder/info["file"],map_location="cuda",weights_only=True).flatten()
            initial=source[key].to(device="cuda",dtype=torch.bfloat16).float().flatten()
            p=torch.nn.Parameter(initial.clone());p.grad=gradient*scale
            optimizer=torch.optim.AdamW([p],lr=used_lr,betas=options["betas"],eps=options["eps"],
                                       weight_decay=options["weight_decay"],foreach=True)
            optimizer.step()
            expected={"master":p.detach(),**{k:optimizer.state[p][k] for k in ("exp_avg","exp_avg_sq")}}
            row={k:comparison(expected[k],x[offset:offset+count].cuda(),master=k=="master") for k,x in actual.items()}
            master=actual["master"][offset:offset+count];forward=metadata["module"][name].flatten()
            row["forward_equals_actual_master_cast"]=torch.equal(forward,master.to(dtype=forward.dtype))
            row["master_changed_elements"]=int((master!=initial.cpu()).sum())
            row["forward_changed_elements"]=int((forward!=source[key].to(dtype=forward.dtype).flatten()).sum())
            result["parameters"][key]=row
            if not row["forward_equals_actual_master_cast"] or not all(row[k]["pass"] for k in actual):result["status"]="FAIL"
            offset+=count
            del p,gradient,initial,optimizer,expected
        alignment=2*len(shards)
        assert math.ceil(offset/alignment)==math.ceil(actual["master"].numel()/alignment)
    if set(result["parameters"])!=set(gradients):raise ValueError("optimizer coverage mismatch")
    return result


def moments(path):
    metadata,shards=load_boundary(path)
    result={}
    for group,shapes in enumerate(metadata["param_shapes"]):
        for field in ["exp_avg","exp_avg_sq"]:
            pieces=[]
            for s in shards:
                base=s["base_optimizer_state"];ids=base["param_groups"][group]["params"]
                assert len(ids)==1
                state=base["state"][ids[0]];assert int(state["step"])==1
                pieces.append(state[field])
            flat=torch.cat(pieces);offset=0
            for name,shape in shapes.items():
                n=math.prod(shape);result[(name,field)]=flat[offset:offset+n];offset+=n
            assert math.ceil(offset/(2*len(shards)))==math.ceil(flat.numel()/(2*len(shards)))
    return result


def scaling(left,right):
    for path in [left,right]:
        cfg=json.loads((Path(path)/"rl_config.json").read_text())
        assert cfg["runtime"]["noise_seed_schedule"]=="global_scene_v1"
        assert cfg["optimizer"]["max_grad_norm"]==1e6 and not cfg["runtime"]["optimizer_offload"]
    a,b=moments(left),moments(right)
    assert a.keys()==b.keys()
    rows=defaultdict(lambda:{"reference_squared_norm":0.,"difference_squared_norm":0.,"max_abs":0.,"numel":0})
    for name,field in a:
        row=rows[name.removeprefix("policy.").split(".")[0]+"/"+field]
        for x,y in zip(a[name,field].split(1_000_000),b[name,field].split(1_000_000)):
            assert torch.isfinite(x).all() and torch.isfinite(y).all()
            delta=x.double()-y.double()
            row["reference_squared_norm"]+=float(x.double().square().sum())
            row["difference_squared_norm"]+=float(delta.square().sum())
            row["max_abs"]=max(row["max_abs"],float(delta.abs().max()))
            row["numel"]+=x.numel()
    for row in rows.values():
        row["relative_l2"]=math.sqrt(row["difference_squared_norm"]/max(row["reference_squared_norm"],1e-300))
        row["pass"]=row["relative_l2"]<=.01
    return {"passed":all(r["pass"] for r in rows.values()),"relative_l2_tolerance":.01,
            "left":str(left),"right":str(right),"modules":dict(rows)}


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__);p.add_argument("mode",choices=["adam","scaling"])
    p.add_argument("--run");p.add_argument("--left");p.add_argument("--right");p.add_argument("--output",required=True)
    args=p.parse_args();torch.set_num_threads(4)
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    report=adam(args.run) if args.mode=="adam" else scaling(args.left,args.right)
    report["analysis_source_sha256"]=file_sha(__file__)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
    print(json.dumps({"passed":report.get("status")=="PASS" if args.mode=="adam" else report["passed"]}))
    assert report.get("status")=="PASS" if args.mode=="adam" else report["passed"]

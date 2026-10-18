"""Student-memory reader and unchanged repeat-8 Flow Matching training."""
import argparse
import json
from pathlib import Path
import time
import torch
from .train_representation import load_model,load_data,observation
from .runtime import numerical_policy,write_identity,source_sha,save_checkpoint,load_checkpoint
from starVLA.model.modules.decodable_st.training_state import TrainableDelta
from starVLA.teachers.decodable_st.source_registry import atomic_json


def original_fm(model,representation,ego,noise_generator,time_generator):
    condition=model.action_condition(representation)
    config=model.baseline.config.framework.action_model
    if (int(config.repeated_diffusion_steps),int(config.num_inference_timesteps),float(config.noise_beta_beta))!=(8,10,1.):
        raise ValueError("Audited original FM repeat/sampler/time contract changed")
    target=ego.to(condition.device,dtype=torch.float32).repeat(8,1,1)
    noise=torch.randn(target.shape,device=target.device,generator=noise_generator)
    u=torch.rand((len(target),),device=target.device,generator=time_generator)
    times=(config.noise_s-u.pow(1./config.noise_beta_alpha))/config.noise_s
    return model.baseline.action_model(condition.repeat(8,1,1),target,noise=noise,times=times)


def run(config,output,*,device="cuda",max_updates=2,resume=False,export_root=None):
    engineering=config.get("scope")=="engineering"
    if engineering and max_updates>4:raise ValueError("Engineering planning probe capped at four updates")
    if not engineering:
        acceptance=Path(config["representation_acceptance"])
        if not acceptance.is_file() or json.loads(acceptance.read_text()).get("status")!="PASS":
            raise ValueError("BLOCKED_REPRESENTATION_ACCEPTANCE")
    numerical_policy(42,device,.80);start=time.monotonic()
    rep_run=Path(config["representation_run"]);rep_id=json.loads((rep_run/"identity.json").read_text())
    model,base,_=load_model(rep_id["config"],device,rep_id["arm"],rep_id["seed"])
    state=torch.load(rep_run/"latest.pt",map_location="cpu",weights_only=False)
    if state["identity"]!=rep_id["identity"]:raise ValueError("Representation checkpoint identity")
    TrainableDelta(model,[x["name"] for x in rep_id["trainable_inventory"]]).load_state_dict(state["model"],strict=True)
    model.requires_grad_(False).eval();model.bridge.requires_grad_(True);model.reader_enabled=True
    if model.temporal_contract:model.temporal_contract.require_driving()
    tune_action=bool(config.get("train_DiT",False))
    if tune_action:model.baseline.action_model.requires_grad_(True)
    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    delta=TrainableDelta(model,[n for n,p in parameters])
    groups=[{"params":list(model.bridge.parameters()),"lr":1e-4,"name":"reader"}]
    if tune_action:groups.append({"params":list(model.baseline.action_model.parameters()),"lr":1e-5,"name":"original_DiT"})
    optimizer=torch.optim.AdamW(groups,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max_updates,eta_min=1e-6)
    identity=write_identity(output,{"schema":"decodable_planner_v1","source_sha":source_sha(),"config":config,
       "baseline_identity":base["identity"],"representation_identity":rep_id["identity"],"representation_progress":state["progress"],
       "trainable_inventory":[n for n,p in parameters],"FM_repeat":8,"inference_Euler_steps":10,
       "max_updates":max_updates,"precision":"FP32 TF32 disabled","action_dropout":"original train mode, p=.2"})
    data,selected,lookup=load_data(rep_id["config"])
    noise=torch.Generator(device=device).manual_seed(42011);times=torch.Generator(device=device).manual_seed(42012)
    sampler=torch.Generator().manual_seed(42013);order=torch.randperm(len(selected),generator=sampler).tolist();cursor=0;step=0
    if resume:
        saved=load_checkpoint(Path(output)/"latest.pt",model=delta,optimizer=optimizer,scheduler=scheduler,identity=identity)
        step=saved["progress"]["update"];cursor=saved["sampler"]["cursor"];order=saved["sampler"]["order"]
        sampler.set_state(saved["sampler"]["rng"]);noise.set_state(saved["extra"]["FM_noise"]);times.set_state(saved["extra"]["FM_time"])
    logs=open(Path(output)/"train.jsonl","a",buffering=1)
    global_batch=int(config.get("global_batch",1));last_obs=None;last_row=None
    try:
        while step<max_updates:
            optimizer.zero_grad(set_to_none=True);model.eval();model.baseline.action_model.train()
            losses=[]
            for micro in range(global_batch):
                if cursor==len(order):order=torch.randperm(len(selected),generator=sampler).tolist();cursor=0
                row=selected[order[cursor]];cursor+=1;example,targets=data[lookup[row["token"]]]
                obs=observation(example)
                with torch.no_grad():representation=model(obs)
                loss=original_fm(model,representation,targets["ego"][None],noise,times)/global_batch
                if not torch.isfinite(loss):raise FloatingPointError("Original FM loss nonfinite")
                loss.backward();losses.append(loss.item());last_obs,last_row=obs,row
            norm=torch.nn.utils.clip_grad_norm_([p for _,p in parameters],1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();step+=1
            logs.write(json.dumps({"update":step,"scene_exposures":step*global_batch,"original_FM":sum(losses),"gradient_norm":float(norm)})+"\n")
            stop=(Path(output)/"STOP_REQUESTED").exists()
            if step%100==0 or step==max_updates or stop:
                save_checkpoint(Path(output)/"latest.pt",model=delta,optimizer=optimizer,scheduler=scheduler,identity=identity,
                    progress={"update":step,"scene_exposures":step*global_batch,"phase":"planner_reader_DiT" if tune_action else "planner_reader"},
                    sampler={"order":order,"cursor":cursor,"rng":sampler.get_state()},extra={"FM_noise":noise.get_state(),"FM_time":times.get_state()})
            if stop:break
    finally:logs.flush();logs.close()
    result={"status":"INCONCLUSIVE" if engineering else "RUNNING","update":step,
        "seconds":time.monotonic()-start,"gpu_hours":(time.monotonic()-start)/3600,
        "scope":"real FM reader forward/backward only" if engineering else "planning training; evaluation still required",
        "planning_benefit":"NOT_RUN","peak_allocated_bytes":torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0}
    atomic_json(Path(output)/"RESULT.json",result)
    if export_root:
        from .export_student import export_loaded
        from tools.foresight.checkpoints import scene_noise
        model.eval();proposal_noise=scene_noise(last_row["token"],42,device)
        with torch.no_grad():trajectory=model.predict_action(last_obs,initial_noise=proposal_noise)
        export_root=Path(export_root);export_root.mkdir(parents=True,exist_ok=True)
        torch.save({"observation":last_obs,"noise":proposal_noise.cpu(),"trajectory":trajectory.cpu()},export_root/"current_only_probe.pt")
        export_loaded(model,export_root,{"baseline_identity":base["identity"],"representation_run_identity":rep_id["identity"],
                      "planner_run_identity":identity,"planner_updates":step,"scope":"engineering only; no planning score claim"})
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda");p.add_argument("--max-updates",type=int,default=2);p.add_argument("--resume",action="store_true")
    p.add_argument("--export-root");a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output,device=a.device,
        max_updates=a.max_updates,resume=a.resume,export_root=a.export_root)

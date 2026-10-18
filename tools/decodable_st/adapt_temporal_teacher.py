"""Bounded native future-head adaptation to independently audited ego(t0) labels."""
import argparse
import json
from pathlib import Path
import torch
from starVLA.teachers.decodable_st.sgdrive_adapter import load_sgdrive_decoder
from starVLA.teachers.decodable_st.source_registry import atomic_json
from starVLA.model.modules.decodable_st.losses import occupancy_loss
from .runtime import numerical_policy,write_identity,save_checkpoint


def run(config,output,device="cuda",max_updates=1000):
    label=Path(config["driving_label_manifest"])
    if not label.is_file():raise ValueError("BLOCKED_TEMPORAL_LABELS: audited occupied/free/unknown labels absent")
    manifest=json.loads(label.read_text())
    for k,v in {"split":"train","coordinate_system":"ego_t0_x_forward_y_left_z_up",
                "free_space_provenance_verified":True,"first_return_verified":True}.items():
        if manifest.get(k)!=v:raise ValueError("BLOCKED_TEMPORAL_LABELS: "+k)
    if manifest.get("horizon_seconds")!=1.:raise ValueError("This explicit adaptation uses a new 1.0s task")
    numerical_policy(42,device,.50)
    frozen,_=load_sgdrive_decoder(config["sgdrive_weights"],json.loads(Path(config["sgdrive_config"]).read_text()),future=True)
    # Explicitly rename as an adapted head, never claim unchanged native weights.
    head=frozen.decoder.to(device).train().requires_grad_(True)
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max_updates,eta_min=1e-5)
    identity=write_identity(output,{"schema":"SGDrive_target_domain_future_head_v1","config":config,
                                  "labels":manifest,"max_updates":max_updates,"native_backbone_frozen":True})
    from starVLA.model.modules.decodable_st.frozen_decoders import sgdrive_logits_layout
    rng=torch.Generator().manual_seed(42);rows=manifest["records"]
    for step in range(max_updates):
        row=rows[int(torch.randint(len(rows),(),generator=rng))]
        tensors=torch.load(row["target_file"],map_location="cpu",weights_only=True)
        if tensors["observed_free"].sum()==0:raise ValueError("No audited negative evidence: refusing all-occupied shortcut")
        hidden=tensors["current_only_teacher_future_hidden"].detach().to(device)
        labels=tensors["labels"].to(device);valid=tensors["valid"].to(device)
        optimizer.zero_grad();logits=sgdrive_logits_layout(head(hidden,[[100,100],[50,50]]))
        loss,_=occupancy_loss(logits,labels,valid);loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step()
        if (step+1)%200==0 or step+1==max_updates:
            save_checkpoint(Path(output)/"latest.pt",model=head,optimizer=optimizer,scheduler=scheduler,identity=identity,
                 progress={"update":step+1,"phase":"T-driving-head"},sampler={"rng":rng.get_state()})
        if (Path(output)/"STOP_REQUESTED").exists():break
    result={"status":"INCONCLUSIVE","update":step+1,"decoder_identity":"original structure, target-domain adapted parameters",
            "requires_dynamic_change_vs_persistence_evaluation":True}
    atomic_json(Path(output)/"RESULT.json",result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda");p.add_argument("--max-updates",type=int,default=1000);a=p.parse_args()
    run(json.loads(Path(a.config).read_text()),a.output,a.device,a.max_updates)

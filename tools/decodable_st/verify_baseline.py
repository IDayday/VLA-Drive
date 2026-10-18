"""Real-image same-noise baseline, appended-query and gradient checks."""
import argparse
import json
import time
from pathlib import Path
import torch
from starVLA.teachers.decodable_st.source_registry import atomic_json


def run(manifest_path,output,device="cuda"):
    from tools.foresight.checkpoints import load_student,scene_noise
    from starVLA.dataloader.decodable_st_dataset import DecodableDataset,collate_observations
    from starVLA.model.framework.ddp_decodable_st import DDPDecodableST
    from starVLA.model.modules.decodable_st.contracts import SpatialContract,TemporalContract
    manifest=json.loads(Path(manifest_path).read_text());start=time.monotonic()
    if manifest["status"]!="IDENTITY_VERIFIED":raise ValueError("BLOCKED_BASELINE_IDENTITY")
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    if device.startswith("cuda"):torch.cuda.set_per_process_memory_fraction(.60)
    identity=json.loads((Path(manifest["run"])/"identity.json").read_text())
    base=load_student(manifest["run"],manifest["tag"],identity,device=device,precision="fp32",strip=False)
    base.requires_grad_(False).eval()
    dataset=DecodableDataset(manifest["data"]["train"]["root"],indices=[0,1])
    obs,_,ids=collate_observations([dataset[0],dataset[1]])
    original=[{"image":list(im),"state":state.numpy(),"lang":nav} for im,state,nav in zip(obs.images,obs.ego_state,obs.navigation)]
    noise=torch.cat([scene_noise(x.scene_token,42,device) for x in ids])
    with torch.no_grad():
        reference_hidden=base.encode_current(original)
        reference_action=base.predict_action(original,initial_noise=noise)
        off=DDPDecodableST(base).to(device).eval()
        disabled=off.predict_action(obs,initial_noise=noise)
    torch.testing.assert_close(disabled,reference_action,rtol=0,atol=0)
    spatial=SpatialContract(256,256,256,(-40,-40,-1),(40,40,5.4))
    temporal=TemporalContract(1536,625,(100,100),16,None,"native_lidar_frame_unverified","local_point_hit","T-native-v1")
    model=DDPDecodableST(base,spatial_contract=spatial,temporal_contract=temporal).to(device).eval()
    with torch.no_grad():
        rep=model(obs);actual=model.predict_action(obs,initial_noise=noise)
    hidden_diff=(rep.original_action_hidden-reference_hidden["action_queries"]).abs()
    traj_diff=(actual-reference_action).abs()
    # Keep declared strict tolerance; record exact errors if native kernel shape
    # rounding fails it instead of silently broadening the acceptance.
    parity={"max_action_hidden_abs":hidden_diff.max().item(),"max_trajectory_abs":traj_diff.max().item()}
    try:
        torch.testing.assert_close(rep.original_action_hidden,reference_hidden["action_queries"],rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(actual,reference_action,rtol=1e-4,atol=1e-5)
        parity["status"]="PASS"
    except AssertionError as exc:
        parity.update(status="FAIL",error=str(exc))
    result={"status":parity["status"],"baseline_identity":manifest["identity"],"disabled_exact":True,"tail_parity":parity,
            "batch":2,"real_scenes":[x.scene_token for x in ids],"spatial_shape":list(rep.predicted_spatial_context.shape),
            "temporal_shape":list(rep.predicted_future_latent.shape),"seconds":time.monotonic()-start,
            "scope":"engineering parity; no learned representation or planning benefit claim"}
    if device.startswith("cuda"):
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    atomic_json(output,result)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cuda")
    a=p.parse_args();run(a.manifest,a.output,a.device)

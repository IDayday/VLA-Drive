"""Independent positive LiDAR evidence, with unknown free space left unknown.

This diagnoses geometric alignment and hit coverage; precision/IoU cannot be
measured from positive-only evidence and this tool never emits capability PASS.
"""
import argparse
import json
from pathlib import Path
import pickle
import time
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file,save_file
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
from starVLA.model.modules.structured_world.geometry import transform_points,crop_resize_intrinsics,geometric_fov
from starVLA.teachers.decodable_st.vigt_adapter import load_vigt_decoder
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256
from starVLA.model.modules.decodable_st.vigt_readout import ContextPatchBottleneck
from .runtime import numerical_policy,source_sha


@torch.no_grad()
def run(config,output,device="cuda"):
    numerical_policy(42,device,.45);start=time.monotonic();output=Path(output)
    if output.exists():raise FileExistsError("Use a new positive-evidence schema version")
    output.mkdir(parents=True)
    baseline=json.loads(Path(config["baseline_manifest"]).read_text())
    train=json.loads((Path(baseline["data"]["train"]["root"])/"index.json").read_text());allowed={r["token"] for r in train}
    selection=json.loads(Path(config["selection"]).read_text())
    if any(r["token"] not in allowed for r in selection):raise ValueError("Hit audit calibration must use training only")
    decoder,_=load_vigt_decoder(config["vigt_weights"],json.loads(Path(config["vigt_config"]).read_text()));decoder.to(device)
    bottleneck=ContextPatchBottleneck(2048,decoder.contract,(32,32)).to(device)
    if sha256(config["bottleneck_weights"])!=config["bottleneck_sha256"]:raise ValueError("PCA identity differs")
    bottleneck.load_state_dict(load_file(config["bottleneck_weights"]),strict=True);bottleneck.eval()
    generator=np.random.default_rng(42);results=[];tally={k:[0,0] for k in ("native","PCA_reconstructed","zero_context")}
    for row in selection:
        with (Path(config["raw_log_root"])/(row["log"]+".pkl")).open("rb") as f:frames=pickle.load(f)
        frame=next(f for f in frames if f["token"]==row["token"])
        path=Path(config["sensor_root"])/frame["lidar_path"]
        xyz=transform_points(LidarPointCloud.from_buffer(path.read_bytes(),"pcd").points[:3].T,frame["lidar2ego"])
        roi=(xyz[:,0]>=.5)&(xyz[:,0]<39.5)&(abs(xyz[:,1])<20)&(xyz[:,2]>=-.5)&(xyz[:,2]<4.9)
        roi &= ~((xyz[:,0]<3.5)&(abs(xyz[:,1])<1))
        ks=[];extrinsics=[];distortions=[]
        for camera in ("CAM_F0","CAM_L0","CAM_R0"):
            c=frame["cams"][camera]
            with Image.open(Path(config["sensor_root"])/c["data_path"]) as image:size=image.size
            k,_=crop_resize_intrinsics(c["cam_intrinsic"],size,(1024,576));ks.append(k)
            ext=np.eye(4);ext[:3,:3]=c["sensor2lidar_rotation"];ext[:3,3]=c["sensor2lidar_translation"]
            extrinsics.append(frame["lidar2ego"]@ext);distortions.append(c["distortion"])
        candidates=xyz[roi];fov=geometric_fov(candidates,np.asarray(ks),np.asarray(extrinsics),np.asarray(distortions))
        candidates=candidates[fov];indices=generator.permutation(len(candidates))[:2048]
        q=torch.tensor(candidates[indices],device=device,dtype=torch.float32)[None]
        native=load_file(str(Path(config["native_context_root"])/(row["token"]+".safetensors")))
        context=native["context"].to(device)
        predictions={"native":decoder(context,q),"PCA_reconstructed":decoder(bottleneck(context),q),"zero_context":decoder(torch.zeros_like(context),q)} if len(indices) else {}
        metrics={}
        for key,value in predictions.items():
            yes=int((value>=0).sum());count=value.numel();tally[key][0]+=yes;tally[key][1]+=count
            metrics[key]={"occupied_hit_recall_at_native_zero_logit":yes/count,"positive_hits":count,
                "logit_quantiles":torch.quantile(value.flatten(),value.new_tensor([.1,.5,.9])).tolist()}
        save_file({"xyz_ego":q.cpu(),**{key+"_logits":v.cpu() for key,v in predictions.items()}},str(output/(row["token"]+".safetensors")))
        results.append({**row,"ROI_hits":int(roi.sum()),"FOV_supported_hits":len(candidates),"evaluated_hits":len(indices),
            "metrics":metrics,"pcd_sha256":sha256(path),"no_height_correction_applied":True})
        atomic_json(output/"progress.json",{"completed":len(results),"requested":len(selection)})
    report={"status":"INCONCLUSIVE","source_sha":source_sha(),"rows":results,"scenes":len(results),
        "summary":{k:{"occupied_hit_recall_at_native_zero_logit":a/b if b else None,"positive_hits":b} for k,(a,b) in tally.items()},
        "coordinate_source":"Raw NAVSIM lidar2ego, original camera extrinsics/distortion; mature crop_resize_intrinsics/geometric_fov reused",
        "FOV_is_visibility":False,"free_space_labels":0,"unknown_is_negative":False,
        "independent_occupancy_precision_IoU":"NOT_MEASURABLE_WITH_POSITIVE_ONLY_EVIDENCE",
        "limitation":"Zero-context may label everything occupied and achieve high hit recall; this is not capability acceptance",
        "seconds":time.monotonic()-start,"gpu_hours":(time.monotonic()-start)/3600}
    atomic_json(output/"HIT_ALIGNMENT.json",report);return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda");a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output,a.device)

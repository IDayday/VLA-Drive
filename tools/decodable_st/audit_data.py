"""Real train/dev separation, physical timestamps and conservative LiDAR evidence."""
import argparse
import json
import pickle
from pathlib import Path
import time
import numpy as np
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256


def run(config,output):
    from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
    start=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(config["baseline_manifest"]).read_text())
    populations={}
    for split in ("train","dev"):
        root=Path(manifest["data"][split]["root"])
        populations[split]=json.loads((root/"index.json").read_text())
    train_logs={r["log"] for r in populations["train"]};dev_logs={r["log"] for r in populations["dev"]}
    train_tokens={r["token"] for r in populations["train"]};dev_tokens={r["token"] for r in populations["dev"]}
    if train_logs&dev_logs or train_tokens&dev_tokens:raise ValueError("Train/dev overlap")
    selected=json.loads(Path(config["selection"]).read_text());rows=[]
    for row in selected:
        if row["token"] not in train_tokens:raise ValueError("Calibration sample outside training split")
        with (Path(config["raw_log_root"])/(row["log"]+".pkl")).open("rb") as f:frames=pickle.load(f)
        current=next(x for x in frames if x["token"]==row["token"])
        times=np.array([int(x["timestamp"]) for x in frames]);delta=(times-int(current["timestamp"]))/1e6
        index=int(np.abs(delta-1.).argmin());future=frames[index]
        error=float(abs(delta[index]-1.));valid=error<=.05
        pcd=Path(config["sensor_root"])/current["lidar_path"]
        cloud=LidarPointCloud.from_buffer(pcd.read_bytes(),"pcd").points
        # nuPlan parser returns x,y,z,intensity,ring,lidar_id in precisely this order.
        lidar_ids,counts=np.unique(cloud[5].astype(int),return_counts=True)
        xyz=cloud[:3].T.astype(np.float64)
        transform=np.linalg.inv(current["ego2global"])@future["ego2global"]@future["lidar2ego"]
        # Algebraic static-world check, not a claim about point-cloud alignment quality.
        probe=np.array([[12.,3.,1.,1.],[5.,-8.,2.,1.]])
        world=probe@current["ego2global"].T
        future_points=world@np.linalg.inv(future["ego2global"]@future["lidar2ego"]).T
        compensated=future_points@transform.T
        static_error=float(np.max(np.abs(compensated-probe)))
        if static_error>1e-8:raise ValueError("Ego compensation algebra")
        front=(xyz[:,0]>=.5)&(xyz[:,0]<39.5)&(abs(xyz[:,1])<20)&(xyz[:,2]>=-.5)&(xyz[:,2]<4.9)
        # Only positive hit evidence is currently available. Do NOT invent ray
        # origins from the merged-cloud identity transform or label no-hit free.
        points=xyz[front].astype(np.float32)
        with (output/(row["token"]+".npz.tmp")).open("wb") as f:
            np.savez_compressed(f,occupied_xyz_ego=points,T_E0_from_Ltau=transform,
                               actual_delta_t=float(delta[index]),future_valid=valid)
        (output/(row["token"]+".npz.tmp")).replace(output/(row["token"]+".npz"))
        rows.append({**row,"timestamp_us":int(current["timestamp"]),"requested_horizon_seconds":1.,
              "actual_horizon_seconds":float(delta[index]),"time_error_seconds":error,"future_valid":valid,
              "static_transform_roundtrip_max_abs_m":static_error,"positive_points_front_roi":int(front.sum()),
              "point_cloud_points":len(xyz),"sensor_ids":dict(zip(map(str,lidar_ids),map(int,counts))),
              "height_percentiles_m":np.percentile(xyz[:,2],[1,10,50,90,99]).tolist(),
              "merged_lidar2ego_identity":bool(np.array_equal(current["lidar2ego"],np.eye(4))),
              "per_sensor_origins_available":False,"first_return_provenance_verified":False,
              "observed_free_labels":0,"no_hit_policy":"unknown","pcd_sha256":sha256(pcd),
              "published_occ_path_exists":bool(current.get("occ_gt_final_path")) and Path(current["occ_gt_final_path"]).is_file()})
    result={"status":"INCONCLUSIVE","split_verified":True,"train_scenes":len(train_tokens),"dev_scenes":len(dev_tokens),
            "train_logs":len(train_logs),"dev_logs":len(dev_logs),"log_overlap":0,"token_overlap":0,
            "rows":rows,"seconds":time.monotonic()-start,"calibration_population":"64 train-only logs",
            "coordinate_convention":"T_A_from_B, column vectors; ego x forward y left z up",
            "requested_1s_is_new_task_definition":True,"SGDrive_native_horizon_verified":False,
            "independent_occupancy_quality":"BLOCKED_OBSERVED_FREE_PROVENANCE",
            "reason":"MergedPointCloud supplies hit XYZ and sensor IDs but not audited per-sensor origins/first returns. "
                     "Its identity lidar2ego is a merged-coordinate transform, not five physical ray origins. "
                     "Unavailable published occupancy paths cannot establish occupied/free/unknown truth."}
    atomic_json(output/"DATA_COORDINATE_AUDIT.json",result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)

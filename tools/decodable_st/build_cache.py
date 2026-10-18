"""Bounded sparse teacher target cache. Never reconstruct teacher BEV from samples."""
import argparse
import json
from pathlib import Path
import time
import torch
from safetensors.torch import load_file
from starVLA.cache.decodable_st_cache import DecodableCache
from starVLA.model.modules.decodable_st.contracts import SampleIdentity,RepresentationTargets
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256


def run(config,output):
    from third_party.decodable_st.vigt.modules.implicit_decoder import grid_sample
    start=time.monotonic();root=Path(config["native_context_root"])
    audit=json.loads((root/"ROUNDTRIP.json").read_text());base=json.loads(Path(config["baseline_manifest"]).read_text())
    if audit["status"]!="PASS":raise ValueError("Native teacher cache must pass first")
    split=base["data"]["train"]["identity"]["identity"]
    c=json.loads(Path(config["vigt_config"]).read_text());lo=torch.tensor(c["roi_min"]);hi=torch.tensor(c["roi_max"])
    contract={"schema_version":"decodable_st_sparse_targets_v1","teacher_identity":config["teacher_identity"],
        "preprocess_identity":config["preprocess_identity"],"camera_ids":["CAM_F0","CAM_L0","CAM_R0"],
        "timestamps_contract":"actual current timestamp from frozen current record","target_horizon":0.,
        "coordinates":"ViGT native ego frame; target-domain z-origin alignment unresolved",
        "transform_convention":"T_A_from_B column vectors","roi":[c["roi_min"],c["roi_max"]],
        "normalization_identity":"native decoder distribution; sampled feature FP16, logits FP32",
        "split_identity":split,"full_context_is_not_reconstructed":True}
    cache=DecodableCache(output,contract);records=[];max_error=0.;bytes_per_scene=[]
    for row in audit["records"]:
        p=root/(row["token"]+".safetensors")
        if sha256(p)!=row["sha256"]:raise ValueError("Original teacher payload changed")
        t=load_file(str(p));xyz=t["xyz"]
        feature=grid_sample(t["context"],((xyz[:,:1024]-lo)/(hi-lo))[...,:2])
        max_error=max(max_error,(feature-feature.half().float()).abs().max().item())
        current=json.loads((Path(config["current_root"])/"current"/(row["token"]+".json")).read_text())
        target=RepresentationTargets({"spatial_features":feature[0].half(),"feature_xyz":xyz[0,:1024],
             "teacher_logits":t["logits"][0],"decode_xyz":xyz[0]},
            {"spatial_features":torch.ones(1024,1,dtype=torch.bool),"teacher_logits":torch.ones_like(t["logits"][0],dtype=torch.bool)},
            {"schema_version":contract["schema_version"],"timestamp_us":current["timestamp"],
             "label_semantics":"native teacher logits only; not independent occupancy truth","source_checksum":row["sha256"]})
        records.append((SampleIdentity(row["token"],row["log"],split),target))
        bytes_per_scene.append(sum(v.numel()*v.element_size() for v in target.values.values()))
        if len(records)==8:
            cache.write_shard(f"shard_{len(cache.manifest['shards']):04d}",records);records=[]
    if records:cache.write_shard(f"shard_{len(cache.manifest['shards']):04d}",records)
    count=cache.verify(payload_checks=len(cache.manifest["shards"]))
    t=time.monotonic()
    for row in audit["records"]:cache.read(SampleIdentity(row["token"],row["log"],split))
    read_seconds=time.monotonic()-t
    result={"status":"PASS","scope":"sparse cache integrity and quantization; not teacher/student capability",
        "identity":cache.key,"scenes":count,"payload_bytes_per_scene":sum(bytes_per_scene)/count,
        "projected_full_train_bytes":sum(bytes_per_scene)/count*base["data"]["train"]["scenes"],
        "feature_FP16_max_abs_error":max_error,"logits_precision":"unchanged FP32",
        "sequential_read_seconds":read_seconds,"build_seconds":time.monotonic()-start,
        "full_teacher_context_replaced_by_sparse_samples":False}
    atomic_json(Path(output)/"CACHE_BUILD.json",result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)

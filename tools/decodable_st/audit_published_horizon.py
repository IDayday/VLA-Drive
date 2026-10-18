"""Bind published SGDrive LiDAR-list indices to real training timestamps.

Only registered train tokens are examined. Full QA stays private and is never a
teacher or student prefix. A default protocol is not proof of an unlogged env.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time
import urllib.request
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256
from .runtime import source_sha

REV="c6379bea8aaf635883b33fcc99499b59d42aad1e"
URL=f"https://huggingface.co/SII-Whaleice/SGDrive/resolve/{REV}/sgdrive.jsonl"


def run(config,output):
    start=time.monotonic();output=Path(output)
    if output.exists():raise FileExistsError("Published-data audit must have an independent version")
    output.mkdir(parents=True)
    population=json.loads((Path(config["current_root"])/"index.json").read_text())
    train={r["token"] for r in population};seen=set();results=[];missing=[]
    request=urllib.request.Request(URL,headers={"Range":"bytes=0-4194303"});digest=hashlib.sha256();bytes_read=0;lines=0
    with urllib.request.urlopen(request,timeout=60) as response:
        if response.status!=206:raise ValueError("Bounded metadata request requires HTTP206")
        while len(results)<64:
            line=response.readline(131072)
            if not line or not line.endswith(b"\n"):break
            digest.update(line);bytes_read+=len(line);lines+=1
            item=json.loads(line)
            if item.get("token") not in train:continue
            paths=item.get("lidar_path",[])
            if len(paths)!=5:raise ValueError("Published data LiDAR-list contract differs")
            log=Path(paths[0]).parts[-3]
            if log in seen:continue
            raw=Path(config["raw_log_root"])/(log+".pkl")
            if not raw.is_file():missing.append(log);continue
            with raw.open("rb") as f:frames=pickle.load(f)
            by_token={r["token"]:r for r in frames}
            tokens=[Path(p).stem for p in paths]
            if tokens[0]!=item["token"] or any(t not in by_token for t in tokens):
                missing.append(log);continue
            t0=int(by_token[tokens[0]]["timestamp"])
            delta=[(int(by_token[t]["timestamp"])-t0)/1e6 for t in tokens]
            results.append({"token":item["token"],"log":log,"lidar_tokens":tokens,"delta_t_seconds":delta,
                "raw_log_sha256":sha256(raw),"source_record_sha256":hashlib.sha256(line).hexdigest()})
            seen.add(log);atomic_json(output/"progress.json",{"matched_training_logs":len(results),"bytes_read":bytes_read})
    if not results:raise ValueError("No eligible published train records with timestamp evidence")
    report={"status":"INCONCLUSIVE","source_sha":source_sha(),"dataset_url":URL,"revision":REV,
        "training_population_index_sha256":sha256(Path(config["current_root"])/"index.json"),
        "records":results,"missing_raw_logs":missing,"source_prefix_consumed_sha256":digest.hexdigest(),
        "source_lines_read":lines,"source_bytes_read":bytes_read,
        "released_code_default_future_idx":1,"verified_default_protocol_horizon_seconds":1.,
        "max_absolute_time_error_from_integer_seconds":max(abs(t-i) for row in results for i,t in enumerate(row["delta_t_seconds"])),
        "checkpoint_environment_future_idx":"UNRECORDED: release default + dataset evidence do not prove absence of an environment override",
        "native_checkpoint_exact_horizon_verified":False,"QA_answers_used_as_model_prefix":False,
        "seconds":time.monotonic()-start,"scope":"Published default task audit, not future prediction quality or training"}
    atomic_json(output/"HORIZON_AUDIT.json",report);return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)

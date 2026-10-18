import argparse
import json
from pathlib import Path
from starVLA.teachers.decodable_st.source_registry import sha256,atomic_json,object_hash


def audit(anchor_path,output):
    anchor=json.loads(Path(anchor_path).read_text())
    if anchor["registered_anchor"]!="original_GT_S0_common_100000_not_reselected":
        raise ValueError("User selected original-GT S0 common100k, no alternate anchor")
    from tools.foresight.checkpoints import checkpoint_identity
    identity,checkpoint=checkpoint_identity(anchor["run"],anchor["tag"])
    if checkpoint!=anchor["checkpoint"]:raise ValueError("S0 checkpoint differs from the preserved receipt")
    cfg=identity["config"]
    base=Path(cfg["framework"]["qwenvl"]["base_vlm"])
    qwen_config=json.loads((base/"config.json").read_text())
    tokenizer=Path(anchor["run"])/"tokenizer"
    if not tokenizer.is_dir():raise ValueError("Saved tokenizer missing")
    files={p.name:sha256(p) for p in tokenizer.iterdir() if p.is_file()}
    added=json.loads((tokenizer/"added_tokens.json").read_text())
    manifests={}
    for split in ("train","dev"):
        root=Path(anchor["data"][split]["current_root"])
        rows=json.loads((root/"index.json").read_text())
        manifests[split]={"root":str(root),"file_sha256":sha256(root/"index.json"),
            "scenes":len(rows),"logs":len({r["log"] for r in rows}),"identity":json.loads((root/"identity.json").read_text())}
    manifest={"schema_version":"decodable_baseline_v1","status":"IDENTITY_VERIFIED",
        "baseline":"S0_original_GT_common100k","selected_by_user":True,"source_commit":anchor["training_source_sha"],
        "checkpoint":checkpoint,"run":anchor["run"],"tag":anchor["tag"],"resolved_config":cfg,
        "qwen_config":qwen_config,"qwen_config_sha256":sha256(base/"config.json"),
        "actual_hidden_dim":qwen_config.get("text_config",qwen_config).get("hidden_size"),
        "tokenizer_file_hashes":files,"added_tokens":added,"data":manifests,
        "shared_training_gpu_hours":anchor["base_training_GPU_hours"],
        "historical_auxiliary_tasks":{k:v for k,v in cfg["foresight"].items() if k.startswith(("enable_","lambda_","future_"))},
        "evaluation":{"precision":"FP32 master and FP32 inference","tf32":False,"steps":10,
            "sampling":"sha256 foresight-action-v1:seed:token, CPU torch.randn","primary_seed":42,"candidates":1},
        "parity_status":"NOT_RUN"}
    manifest["identity"]=object_hash(manifest)
    output=Path(output)
    if output.exists():
        if json.loads(output.read_text())!=manifest:raise ValueError("Read-only baseline manifest changed")
    else:
        atomic_json(output,manifest);output.chmod(0o444)
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument("--anchor",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();r=audit(a.anchor,a.output);print(json.dumps({k:r[k] for k in ("status","identity","baseline","actual_hidden_dim")}))

if __name__=="__main__":main()

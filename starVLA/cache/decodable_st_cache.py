"""Identity-bound atomic shards; a manifest alone never proves valid payload."""
import os
from pathlib import Path
import json
import torch
from starVLA.teachers.decodable_st.source_registry import object_hash,sha256,atomic_json
from starVLA.model.modules.decodable_st.contracts import RepresentationTargets


REQUIRED=("schema_version","teacher_identity","preprocess_identity","camera_ids","timestamps_contract",
          "target_horizon","coordinates","transform_convention","roi","normalization_identity","split_identity")


class DecodableCache:
    def __init__(self,root,contract):
        self.root=Path(root);self.contract=contract
        missing=[k for k in REQUIRED if k not in contract]
        if missing:raise ValueError(f"Cache identity fields missing: {missing}")
        self.key=object_hash(contract);self.folder=self.root/self.key
        self.folder.mkdir(parents=True,exist_ok=True)
        self.manifest_path=self.folder/"manifest.json"
        if self.manifest_path.exists():
            self.manifest=json.loads(self.manifest_path.read_text())
            if self.manifest["contract"]!=contract or self.manifest["identity"]!=self.key:raise ValueError("Cache identity changed")
        else:self.manifest={"contract":contract,"identity":self.key,"shards":{}}

    def write_shard(self,name,records):
        if not name or Path(name).name!=name:raise ValueError("Invalid shard name")
        payload={}
        for identity,target in records:
            target.validate()
            if identity.scene_token in payload:raise ValueError("Duplicate scene in shard")
            payload[identity.scene_token]={"identity":vars(identity),"values":{k:v.detach().cpu() for k,v in target.values.items()},
                   "valid":{k:v.detach().cpu() for k,v in target.valid.items()},"metadata":dict(target.metadata)}
        dest=self.folder/(name+".pt");temp=dest.with_suffix(f".{os.getpid()}.tmp")
        with temp.open("wb") as f:
            torch.save({"cache_identity":self.key,"records":payload},f);f.flush();os.fsync(f.fileno())
        os.replace(temp,dest)
        receipt={"status":"complete","file":dest.name,"sha256":sha256(dest),"bytes":dest.stat().st_size,"tokens":sorted(payload)}
        # One writer per cache namespace. Multi-process builders use disjoint shard
        # namespaces and merge receipts explicitly after all workers finish.
        self.manifest["shards"][name]=receipt;atomic_json(self.manifest_path,self.manifest)
        return receipt

    def verify(self,payload_checks=1):
        seen=set()
        for i,(name,row) in enumerate(sorted(self.manifest["shards"].items())):
            p=self.folder/row["file"]
            if row["status"]!="complete" or not p.is_file() or p.stat().st_size!=row["bytes"]:raise ValueError(f"Incomplete cache shard {name}")
            if seen.intersection(row["tokens"]):raise ValueError("Duplicate scene across shards")
            seen.update(row["tokens"])
            if i<payload_checks and sha256(p)!=row["sha256"]:raise ValueError(f"Corrupt cache shard {name}")
        return len(seen)

    def read(self,identity):
        matches=[r for r in self.manifest["shards"].values() if identity.scene_token in r["tokens"]]
        if len(matches)!=1:raise ValueError("Scene missing or repeated in cache")
        row=matches[0];p=self.folder/row["file"]
        # Cache shards in the process after their first checksum check.
        if not hasattr(self,"_loaded"):self._loaded={}
        if p not in self._loaded:
            if sha256(p)!=row["sha256"]:raise ValueError("Corrupt cache payload")
            self._loaded.clear() # bounded one-shard working set
            self._loaded[p]=torch.load(p,map_location="cpu",weights_only=True)
        payload=self._loaded[p]
        if payload["cache_identity"]!=self.key:raise ValueError("Wrong payload identity")
        r=payload["records"][identity.scene_token]
        if r["identity"]!=vars(identity):raise ValueError("Scene/log/source identity mismatch")
        return RepresentationTargets(r["values"],r["valid"],r["metadata"])

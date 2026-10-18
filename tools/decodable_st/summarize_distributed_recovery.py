"""Compare real rank checkpoints, preserving strict failures and A/A evidence."""
import argparse
import json
from pathlib import Path
import torch
from .compare_recovery import compare
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256


def run(root,output):
    torch.set_num_threads(2);root=Path(root);evidence={};pairs={}
    for name,other in (("AA","repeated"),("AB_resume","resumed")):
        ranks=[]
        for rank in (0,1):
            a=root/"continuous"/f"rank{rank}.pt";b=root/other/f"rank{rank}.pt"
            for path in (a,b):evidence[str(path.relative_to(root))]=sha256(path)
            result=compare(torch.load(a,map_location="cpu",weights_only=False),torch.load(b,map_location="cpu",weights_only=False))
            rows=result["differing_float_tensors"]
            result.update(rank=rank,max_abs=max((x["max_abs"] for x in rows),default=0),
                max_normalized_l2=max((x["normalized_l2"] for x in rows),default=0),
                min_cosine=min((x["cosine"] for x in rows),default=1),
                scope="Real two-rank four-update execution with active LoRA; not formal tiny or long-run resume")
            ranks.append(result)
        pairs[name]=ranks
    receipts={name:json.loads((root/(name+".json")).read_text()) for name in ("continuous","resumed_prefix","resumed","repeated")}
    e1=all(not row["discrete_state_errors"] for rows in pairs.values() for row in rows)
    strict=all(row["strict_bitwise_equal"] for rows in pairs.values() for row in rows)
    synchronized=all(len({r["trainable_parameter_sha256"] for r in value["rows"]})==1 for value in receipts.values())
    result={"status":"PASS" if e1 and strict and synchronized else "INCONCLUSIVE" if e1 and synchronized else "FAIL",
        "discrete_state_equal":e1,"strict_bitwise_equal":strict,"rank_parameters_synchronized":synchronized,
        "pairs":pairs,"checkpoint_hashes":evidence,"receipt_hashes":{n:sha256(root/(n+".json")) for n in receipts},
        "receipts":receipts,"gpu_hours":sum(x["gpu_hours"] for x in receipts.values()),
        "scope":"Real S0/current images/DINO/native ViGT losses; continuous4, repeat4 and independent-process2+2. Unit feature scale and active LoRA are engineering-only.",
        "physical_device_note":"Continuous/repeat ranks on the same two A800s; interrupted/resumed ranks on another two A800s of the same host.",
        "limitation":"No full-stage or long-run equivalence assertion; strict floating differences retained without threshold inflation"}
    atomic_json(output,result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();run(a.root,a.output)

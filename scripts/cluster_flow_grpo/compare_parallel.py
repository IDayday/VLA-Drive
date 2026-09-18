"""Read-only exact token-level comparison of serial and log-sharded evaluations."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from starVLA.rl.flow_grpo.evaluation_transaction import completed_evaluation


def compare(left, right, output):
    roots=list(map(Path,[left,right]))
    archives=[np.load(p/"trajectories.npz",allow_pickle=False) for p in roots]
    tokens=archives[0]["tokens"].tolist()
    reports=[json.loads((p/"evaluation.json").read_text()) for p in roots]
    if reports[0]["identity"] != reports[1]["identity"]:
        raise ValueError("evaluation identity differs")
    for root,report in zip(roots,reports):
        if completed_evaluation(root,report["identity"],tokens) is None:
            raise ValueError("incomplete evaluation")
    result={"status":"PASS","scene_count":len(tokens),"scope":"all trajectories, all scene metrics, full adjacency map",
            "left":str(roots[0]),"right":str(roots[1]),"arrays":{},"metrics":{}}
    for key in ["tokens","normalized","physical"]:
        same=np.array_equal(archives[0][key],archives[1][key])
        result["arrays"][key]={"identical":bool(same)}
        if key!="tokens":
            result["arrays"][key]["max_abs"]=float(np.max(np.abs(archives[0][key]-archives[1][key])))
        if not same:result["status"]="FAIL"
    frames=[pd.read_csv(p/"original_protocol_scores.csv",float_precision="round_trip").set_index("token").loc[tokens] for p in roots]
    if list(frames[0].columns)!=list(frames[1].columns):
        raise ValueError("metric column set/order differs")
    for key in frames[0]:
        a,b=frames[0][key],frames[1][key]
        same=bool(((a==b)|(a.isna()&b.isna())).all())
        result["metrics"][key]={"identical":same}
        if not same:result["status"]="FAIL"
    result["aggregation_identical"]=reports[0]["aggregation"]==reports[1]["aggregation"]
    if not result["aggregation_identical"]:result["status"]="FAIL"
    result["scores"]=[r["epdms"] for r in reports]
    result["reported_seconds"]=[r["seconds"] for r in reports]
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2))
    if result["status"]!="PASS":raise AssertionError("serial/parallel evaluation differs; full results preserved")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--left",required=True);p.add_argument("--right",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();print(json.dumps(compare(a.left,a.right,a.output)))

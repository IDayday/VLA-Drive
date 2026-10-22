"""Paired log-cluster intervals for real fixed-endpoint decoding reports.

Ratios are computed after pooling counts, retaining the original scene/query
weights. One training seed and a tiny log sample do not establish stability.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256
from .runtime import source_sha


def run(left,right,output,draws=10000):
    a,b=(json.loads(Path(p).read_text()) for p in (left,right))
    for field in ("selection_sha256","baseline_identity"):
        if a[field]!=b[field]:raise ValueError("Comparison identity differs: "+field)
    if a["progress"]["update"]!=1000 or b["progress"]["update"]!=1000:
        raise ValueError("Registered common endpoint is 1000")
    aa,bb=({x["token"]:x for x in d["per_scene"]} for d in (a,b))
    if aa.keys()!=bb.keys():raise ValueError("Comparison scene population differs")
    logs=sorted({r["log"] for r in aa.values()})
    counts=np.zeros((2,len(logs),3),dtype=np.float64)
    for side,rows in enumerate((aa,bb)):
        for token,row in rows.items():
            if aa[token]["log"]!=bb[token]["log"]:raise ValueError("Log identity differs")
            m=row["native_teacher_agreement"]
            counts[side,logs.index(row["log"])]+=[m[k] for k in ("tp","fp","fn")]
    rng=np.random.default_rng(20261022)
    sampled=rng.integers(0,len(logs),size=(draws,len(logs)))
    resampled=counts[:,sampled].sum(2)
    ious=resampled[...,0]/resampled.sum(-1)
    if not np.isfinite(ious).all():raise ValueError("Bootstrap has undefined IoU; retain the invalid-population evidence")
    difference=ious[0]-ious[1]
    point=counts.sum(1);point=point[:,0]/point.sum(1)
    result={"status":"INCONCLUSIVE","evaluation_source_sha":source_sha(),
        "left_arm":a["arm"],"right_arm":b["arm"],"common_update":1000,
        "scenes":len(aa),"logs":len(logs),"bootstrap_draws":draws,"bootstrap_rng_seed":20261022,
        "left_IoU":float(point[0]),"right_IoU":float(point[1]),
        "paired_delta_IoU":float(point[0]-point[1]),
        "paired_log_cluster_percentile_CI95":np.quantile(difference,[.025,.975]).tolist(),
        "left_report_sha256":sha256(left),"right_report_sha256":sha256(right),
        "scope":"Native teacher agreement, pooled-count IoU; one new-head training seed, no independent GT/planning acceptance",
        "bootstrap_limitation":"Log resampling describes this fixed small evaluation population, not variation across VLA training seeds"}
    atomic_json(output,result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--left",required=True);p.add_argument("--right",required=True)
    p.add_argument("--output",required=True);a=p.parse_args();run(a.left,a.right,a.output)

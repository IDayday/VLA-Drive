"""Shareable numeric plots; never copies private scene imagery."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from starVLA.teachers.decodable_st.source_registry import atomic_json


def run(root,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    runs={"RS":"RS_gradient_v1","Rfrozen":"Rfrozen_tiny64_v1"}
    fig,axes=plt.subplots(1,3,figsize=(14,4));terms=("DINO","spatial_feature","spatial_decode")
    for arm,name in runs.items():
        rows=[json.loads(s) for s in (root/"runs"/name/"train.jsonl").read_text().splitlines()]
        if any(b["update"]<=a["update"] for a,b in zip(rows,rows[1:])):raise ValueError("Duplicate/nonmonotonic update records")
        with (output/(arm+"_LEARNING_CURVE.csv")).open("w") as f:
            writer=csv.DictWriter(f,fieldnames=("update","scene_exposures","lr","gradient_norm",*terms),lineterminator="\n");writer.writeheader()
            for r in rows:writer.writerow({**{k:r[k] for k in ("update","scene_exposures","lr","gradient_norm")},**r["losses"]})
        for ax,term in zip(axes,terms):
            ax.plot([r["update"] for r in rows],[r["losses"][term] for r in rows],label=arm,alpha=.8)
            ax.set(title=term,xlabel="optimizer updates",ylabel="raw loss (logged individual scene)")
            ax.axvline(200,color="grey",linestyle="--",linewidth=.7);ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle("Tiny64, common 1000-update endpoint; losses do not establish teacher quality")
    fig.tight_layout();fig.savefig(output/"LEARNING_CURVES.png",dpi=130);plt.close(fig)

    evaluations={}
    for arm,name in runs.items():
        evaluations[arm]={split:json.loads((root/"evaluations"/run/"DECODING_REPORT.json").read_text())
                         for split,run in [("train64",name),("dev16",arm+"_dev16_v1")]}
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,split in zip(axes,("train64","dev16")):
        labels=[];values=[]
        for arm in runs:
            r=evaluations[arm][split];labels.extend([arm,arm+" shuffled"])
            values.extend([r["native_teacher_agreement"]["occupied_iou"],r["negative_controls"]["scene_shuffled_student"]["occupied_iou"]])
        labels.append("train teacher mean");values.append(evaluations["RS"][split]["negative_controls"]["mean_training_teacher_context"]["occupied_iou"])
        ax.bar(labels,values,color=["#20639b","#91b5d3","#3c9d79","#a6d7c4","#999999"])
        ax.set(title=split,ylim=(0,1),ylabel="native-teacher occupied agreement IoU")
        ax.tick_params(axis="x",rotation=30)
    fig.suptitle("No independent GT free/unknown labels; not a safety or planning metric")
    fig.tight_layout();fig.savefig(output/"DECODING_CONTROLS.png",dpi=130);plt.close(fig)

    # Same scene weights; sample logs, never scene x inference-seed replicates.
    a={r["token"]:r for r in evaluations["RS"]["dev16"]["per_scene"]}
    b={r["token"]:r for r in evaluations["Rfrozen"]["dev16"]["per_scene"]}
    if a.keys()!=b.keys():raise ValueError("Unpaired held-out population")
    logs=sorted({r["log"] for r in a.values()});totals=[]
    for log in logs:
        pair=[]
        for model in (a,b):
            pair.append([sum(r["native_teacher_agreement"][k] for r in model.values() if r["log"]==log) for k in ("tp","fp","fn")])
        totals.append(pair)
    totals=np.asarray(totals,dtype=np.float64)
    rng=np.random.default_rng(42);sums=totals[rng.integers(0,len(logs),(10000,len(logs)))].sum(1)
    iou=sums[:,:,0]/np.maximum(sums.sum(2),1);delta=iou[:,0]-iou[:,1]
    point=evaluations["RS"]["dev16"]["native_teacher_agreement"]["occupied_iou"]-evaluations["Rfrozen"]["dev16"]["native_teacher_agreement"]["occupied_iou"]
    atomic_json(output/"DEV16_CLUSTER_BOOTSTRAP.json",{"metric":"native teacher agreement occupied IoU RS minus Rfrozen",
        "point":point,"percentile95":np.quantile(delta,[.025,.975]).tolist(),"logs":len(logs),"scenes":len(a),"replicates":10000,"seed":42,
        "scope":"Exploratory tiny-model readout comparison on 16 fixed dev scenes; not independent GT, full dev or cross-training stability"})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--artifact-root",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();run(a.artifact_root,a.output)

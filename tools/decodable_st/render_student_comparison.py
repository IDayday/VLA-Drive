"""Private fixed-scene native decoder panels, including both actual students."""
import argparse
import json
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256


def run(root,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((root/"audits/tiny_job.json").read_text());data=ForesightCurrentDataset(cfg["current_root"])
    lookup={x["token"]:i for i,x in enumerate(data.index)};rows=json.loads(Path(cfg["selection"]).read_text())[:32];index=[]
    for row in rows:
        token=row["token"];a=load_file(str(root/"evaluations/Rfrozen_tiny64_v1"/(token+".slices.safetensors")))
        b=load_file(str(root/"evaluations/RS_gradient_v1"/(token+".slices.safetensors")))
        if not np.array_equal(a["xyz"],b["xyz"]) or not np.array_equal(a["teacher_logits"],b["teacher_logits"]):
            raise ValueError("Comparison decoder target/query mismatch")
        teacher=a["teacher_logits"].sigmoid().numpy().reshape(2,64,64)
        frozen=a["student_logits"].sigmoid().numpy().reshape(2,64,64)
        lora=b["student_logits"].sigmoid().numpy().reshape(2,64,64)
        fig,axes=plt.subplots(3,4,figsize=(15,10));observation=data[lookup[token]]
        for j,im in enumerate(observation["image"]):axes[0,j].imshow(im);axes[0,j].axis("off")
        axes[0,3].axis("off");axes[0,3].text(0,1,"PRIVATE CURRENT IMAGES\nSame 64 training scenes\nBoth students: 1000 updates\nNative teacher agreement only\nGT free/unknown not verified\nNative height origin unverified",va="top")
        for i,z in enumerate((.5,1.5)):
            for j,(v,title) in enumerate([(teacher,"Native teacher"),(frozen,"Frozen Qwen student"),(lora,"Tail LoRA student"),(lora-teacher,"LoRA minus teacher")]):
                axes[i+1,j].imshow(v[i].T,origin="lower",extent=(.5,39.5,-20,20),vmin=-1 if j==3 else 0,vmax=1,cmap="coolwarm" if j==3 else "viridis")
                axes[i+1,j].set_title(f"{title}, native z={z}m")
        fig.suptitle("Fixed native decoder; sigmoid is not a calibrated safety probability")
        fig.tight_layout();dest=output/(token+".png");fig.savefig(dest,dpi=100);plt.close(fig)
        index.append({"token":token,"file":dest.name,"sha256":sha256(dest)})
    atomic_json(output/"INDEX.json",{"status":"PASS","scope":"32 fixed train scenes; private visualization, not held-out capability","panels":index})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--artifact-root",required=True);p.add_argument("--output",required=True);a=p.parse_args();run(a.artifact_root,a.output)

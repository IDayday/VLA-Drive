"""Fixed-scene native decode panels; private camera images stay in artifacts."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from safetensors.torch import load_file
from starVLA.teachers.decodable_st.vigt_adapter import load_vigt_decoder
from starVLA.model.modules.decodable_st.vigt_readout import ContextBottleneck
from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
from starVLA.teachers.decodable_st.source_registry import atomic_json
from .runtime import numerical_policy


@torch.no_grad()
def run(config,output,device="cuda",limit=32):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    numerical_policy(42,device,.30);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    decoder,_=load_vigt_decoder(config["weights"],json.loads(Path(config["config"]).read_text()));decoder.to(device)
    data=ForesightCurrentDataset(config["current_root"]);lookup={r["token"]:i for i,r in enumerate(data.index)}
    rows=json.loads(Path(config["selection"]).read_text())[:limit]
    models={}
    for grid in (16,32):
        model=ContextBottleneck(2048,decoder.contract,(grid,grid)).to(device)
        checkpoint=torch.load(Path(config["artifact_root"])/"runs"/f"compression{grid}_v1/latest.pt",weights_only=False,map_location="cpu")
        model.load_state_dict(checkpoint["model"],strict=True);models[grid]=model.eval()
    x=torch.arange(.25,40.,.5,device=device);y=torch.arange(-19.75,20.,.5,device=device)
    xx,yy=torch.meshgrid(x,y,indexing="ij");xy=torch.stack((xx,yy),-1).reshape(-1,2)
    panels=[]
    for row in rows:
        record=load_file(str(Path(config["native_context_root"])/(row["token"]+".safetensors")))
        context=record["context"].to(device)
        rebuilt={k:m(context) for k,m in models.items()}
        obs=data[lookup[row["token"]]]
        fig,axes=plt.subplots(3,4,figsize=(15,10))
        for col in range(3):axes[0,col].imshow(obs["image"][col]);axes[0,col].axis("off");axes[0,col].set_title(("F0","L0","R0")[col])
        axes[0,3].axis("off");axes[0,3].text(0,1,"PRIVATE CURRENT IMAGES\nTeacher agreement only\nNo independent free-space GT\nOriginal decoder; no ego erasure\nFixed 32 train scenes",va="top")
        for line,z in enumerate((.5,1.5),1):
            xyz=torch.cat((xy,torch.full((len(xy),1),z,device=device)),-1)[None]
            values=[]
            for c in (context,rebuilt[16],rebuilt[32]):
                logits=torch.cat([decoder(c,q) for q in xyz.split(2048,dim=1)],1)
                values.append(logits.sigmoid().reshape(len(x),len(y)).cpu().numpy())
            for col,value in enumerate(values):
                axes[line,col].imshow(value.T,origin="lower",extent=(0,40,-20,20),vmin=0,vmax=1,cmap="viridis",aspect="equal")
                axes[line,col].set_title(("Native teacher","256-query bottleneck","1024-query bottleneck")[col]+f" z={z}m")
            axes[line,3].imshow((values[2]-values[0]).T,origin="lower",extent=(0,40,-20,20),vmin=-1,vmax=1,cmap="coolwarm")
            axes[line,3].set_title("1024 minus native probability")
        fig.suptitle("Native coordinate slices; sigmoid values are not calibrated safety probabilities")
        fig.tight_layout();path=output/(row["token"]+"_space.png");fig.savefig(path,dpi=110);plt.close(fig)
        temporal=Path(config["artifact_root"])/"sgdrive_train64_native_v1"/(row["token"]+".safetensors")
        if temporal.exists():
            t=load_file(str(temporal));a=t["current_logits"].softmax(1)[0,1].amax(-1).numpy();b=t["future_logits"].softmax(1)[0,1].amax(-1).numpy()
            fig,ax=plt.subplots(1,3,figsize=(12,4))
            for axis,v,title in zip(ax,(a,a,b),("Native current","Persistence (native coordinates)","Native future (horizon unverified)")):
                axis.imshow(v.T,origin="lower",vmin=0,vmax=1,extent=(0,50,-25,25));axis.set_title(title)
            fig.suptitle("SGDrive stage1: local binary occupancy; no free-space or occupancy-flow claim")
            fig.tight_layout();fig.savefig(output/(row["token"]+"_time.png"),dpi=110);plt.close(fig)
        panels.append({"token":row["token"],"log":row["log"],"spatial":path.name,"private":True,"selection":"fixed train64 first32, before model results"})
    atomic_json(output/"INDEX.json",{"status":"PASS","scope":"actual native/learned-bottleneck decode visualization, not quality acceptance","panels":panels})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda");p.add_argument("--limit",type=int,default=32);a=p.parse_args()
    run(json.loads(Path(a.config).read_text()),a.output,a.device,a.limit)

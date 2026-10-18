"""Compare actual continuous/resumed checkpoints, retaining strict failures."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from starVLA.teachers.decodable_st.source_registry import atomic_json,sha256


def compare(a,b):
    rows=[];exact_errors=[]
    def visit(x,y,path):
        if isinstance(x,torch.Tensor):
            if not isinstance(y,torch.Tensor) or x.shape!=y.shape or x.dtype!=y.dtype:
                exact_errors.append(path+": tensor schema");return
            exact=torch.equal(x,y)
            if not exact:
                if x.is_floating_point():
                    xf=x.double().reshape(-1);yf=y.double().reshape(-1);d=xf-yf
                    rows.append({"path":path,"exact":False,"elements":x.numel(),"max_abs":d.abs().max().item(),
                        "normalized_l2":(d.norm()/xf.norm().clamp_min(1e-30)).item(),
                        "cosine":torch.nn.functional.cosine_similarity(xf,yf,dim=0,eps=1e-30).item()})
                else:exact_errors.append(path+": discrete tensor")
        elif isinstance(x,np.ndarray):
            if not np.array_equal(x,y):exact_errors.append(path+": numpy state")
        elif isinstance(x,dict):
            if set(x)!=set(y):exact_errors.append(path+": dictionary inventory");return
            for k in x:visit(x[k],y[k],path+"/"+str(k))
        elif isinstance(x,(list,tuple)):
            if type(x)!=type(y) or len(x)!=len(y):exact_errors.append(path+": sequence schema");return
            for i,(xx,yy) in enumerate(zip(x,y)):visit(xx,yy,path+"/"+str(i))
        elif x!=y:exact_errors.append(path+": scalar")
    # Extra is diagnostic metadata, not execution state. Its recorded first-step
    # gradient evidence differs between uninterrupted and resumed segments.
    for k in ("schema","identity","model","optimizer","scheduler","scaler","rng","progress","sampler"):
        visit(a[k],b[k],k)
    return {"status":"PASS" if not rows and not exact_errors else "INCONCLUSIVE",
        "strict_bitwise_equal":not rows and not exact_errors,"discrete_state_errors":exact_errors,
        "differing_float_tensors":rows,"compared_fields":["model","optimizer","scheduler","scaler","rng","progress","sampler"],
        "scope":"actual same-source FP32 four-update warmup; does not establish post-LoRA or long-run bitwise resume"}


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--continuous",required=True);p.add_argument("--resumed",required=True)
    p.add_argument("--output",required=True);a=p.parse_args();torch.set_num_threads(2)
    result=compare(torch.load(a.continuous,map_location="cpu",weights_only=False),torch.load(a.resumed,map_location="cpu",weights_only=False))
    result["artifacts"]={k:{"path":v,"sha256":sha256(v)} for k,v in [("continuous",a.continuous),("resumed",a.resumed)]}
    atomic_json(a.output,result);print(json.dumps({k:v for k,v in result.items() if k not in ("differing_float_tensors","artifacts")},indent=2))

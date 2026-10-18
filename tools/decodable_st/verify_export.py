"""Independent deployment process: teacher import and artifact reads are denied."""
import argparse
import importlib.abc
import json
from pathlib import Path
import sys
import time


class DenyTeachers(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith(("starVLA.teachers", "third_party.decodable_st", "vigt", "internvl")):
            raise ImportError("Teacher import denied in standalone deployment: "+fullname)
        return None


def run(root,device="cuda"):
    sys.meta_path.insert(0,DenyTeachers())
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    root=Path(root);record=json.loads((root/"manifest.json").read_text())
    forbidden=("/weights/vigt/","/weights/sgdrive/","/external/ViGT/","/external/SGDrive/","native_v1/","dino_targets_v1")
    def deny_reads(event,args):
        if event=="open" and args and isinstance(args[0],str) and any(s in args[0] for s in forbidden):
            raise PermissionError("Teacher/target artifact access denied: "+args[0])
    sys.addaudithook(deny_reads)
    from starVLA.model.modules.decodable_st.export_loader import load_trusted_export
    start=time.monotonic();model,record=load_trusted_export(root,device)
    probe=torch.load(root/"current_only_probe.pt",map_location="cpu",weights_only=False)
    with torch.no_grad():
        prediction=model.predict_action(probe["observation"],initial_noise=probe["noise"].to(device))
    torch.testing.assert_close(prediction.cpu(),probe["trajectory"],rtol=1e-4,atol=1e-5)
    times=[]
    for i in range(6):
        if device.startswith("cuda"):torch.cuda.synchronize()
        t=time.monotonic()
        with torch.no_grad():model.predict_action(probe["observation"],initial_noise=probe["noise"].to(device))
        if device.startswith("cuda"):torch.cuda.synchronize()
        if i:times.append(time.monotonic()-t)
    loaded=[x for x in sys.modules if x.startswith(("starVLA.teachers","third_party.decodable_st","vigt","internvl"))]
    if loaded:raise AssertionError(loaded)
    result={"status":"PASS","max_abs_trajectory_error":float((prediction.cpu()-probe["trajectory"]).abs().max()),
        "teacher_modules_loaded":loaded,"teacher_and_target_reads_denied":True,"latency_seconds":times,
        "load_and_first_predict_seconds":time.monotonic()-start-sum(times),"precision":"FP32 TF32 disabled",
        "scope":"independent export engineering, not representation/planning capability","provenance":record["provenance"]}
    (root/"DEPLOYMENT_REPORT.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--device",default="cuda");a=p.parse_args()
    run(a.root,a.device)

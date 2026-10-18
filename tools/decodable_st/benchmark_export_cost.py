"""Same device/input/noise FP32 original S0 vs exported student execution cost."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import time
import torch
from starVLA.model.modules.decodable_st.export_loader import load_trusted_export


def run(root,output,device="cuda"):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    root=Path(root);model,identity=load_trusted_export(root,device)
    probe=torch.load(root/"current_only_probe.pt",map_location="cpu",weights_only=False);obs=probe["observation"];noise=probe["noise"].to(device)
    examples=[{"image":list(ims),"lang":lang,"state":state.numpy()} for ims,lang,state in zip(obs.images,obs.navigation,obs.ego_state)]
    lm=model.baseline.qwen_vl_interface.model.model.language_model;times={"original_S0":[],"exported_student":[]}
    for repeat in range(6):
        for kind in ("original_S0","exported_student") if repeat%2==0 else ("exported_student","original_S0"):
            with (lm.disable_adapter() if kind=="original_S0" and hasattr(lm,"disable_adapter") else nullcontext()):
                torch.cuda.synchronize();start=time.monotonic()
                with torch.no_grad():
                    value=model.baseline.predict_action(examples,initial_noise=noise.clone()) if kind=="original_S0" else model.predict_action(obs,initial_noise=noise.clone())
                torch.cuda.synchronize()
                if repeat:times[kind].append(time.monotonic()-start)
    med={k:statistics.median(v) for k,v in times.items()}
    result={"status":"PASS","measurement":"alternating six windows; first discarded; one real current three-view scene",
       "precision":"FP32 TF32 disabled, original Euler10, single candidate","seconds":times,"medians":med,
       "student_over_original":med["exported_student"]/med["original_S0"],"reader_enabled":model.reader_enabled,
       "source_export_sha256":identity["weights_sha256"],"scope":"latency only, not model planning benefit or across-scene p90"}
    Path(output).write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cuda");a=p.parse_args();run(a.root,a.output,a.device)

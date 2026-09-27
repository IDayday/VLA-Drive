"""Measure the original native policy forward without appended world computation."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import torch
from tools.local_interaction_mask_v2.extract_current import load_foundation
from tools.local_interaction_mask_v2.data import current_metadata_from_training_pickle,current_example
from tools.local_interaction_mask_v2.planner_runtime import CurrentOnlyCorpus,predict_payload
from tools.local_interaction_mask_v2.train_foundation import atomic_json
from starVLA.model.modules.joint_world.local_planner import predict_local


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('foundation','public-qwen','cache','dataset','perception-checkpoint','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--limit',type=int,default=4);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    world,metadata=load_foundation(a.foundation,a.public_qwen,None,a.perception_checkpoint,'fp32')
    corpus=CurrentOnlyCorpus(a.cache);identity=corpus.manifest['identity'];spec=json.loads(Path(a.dataset).read_text())
    rows=[]
    for i in range(min(a.limit,len(corpus))):
        payload=corpus[i];token=payload['token']
        record=current_metadata_from_training_pickle(Path(spec['meta_root'])/(token+'.pkl'),token)
        example,_=current_example(Path(spec['observations'])/(token+'.npz'),record)
        reference=predict_payload(world.baseline.action_model,None,payload,identity)
        torch.cuda.synchronize();start=time.perf_counter()
        native=world.baseline.native_conditions([example])
        result=predict_local(world.baseline.action_model,None,native,{},[token])
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        condition_error=float((native-payload['native_actions'].cuda()).abs().max())
        action_error=float((result['normalized_actions']-reference['normalized_actions']).abs().max())
        row={'token':token,'native_current_images_to_DiT_seconds':elapsed,
            'condition_error':condition_error,'action_error':action_error}
        rows.append(row)
        if condition_error!=0 or action_error>1e-5:
            atomic_json(out/'failed.json',row);raise AssertionError('Native policy does not match the frozen cached A0')
    if not rows:raise ValueError('Empty native benchmark')
    atomic_json(out/'result.json',{'status':'PASS','code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'foundation_sha256':identity['foundation_sha256'],'current_identity':corpus.manifest['identity_sha256'],
        'foundation_step':metadata['step'],'rows':rows,
        'scope':'CUDA-synchronized singleton original Qwen/native action tokens and original DiT; no world Reader/current head/local graph forward. Prepared current images exclude disk decode/crop/resize; first-call effects retained.',
        'peak_loaded_wrapper_gpu_bytes':torch.cuda.max_memory_allocated(),
        'memory_scope':'Loaded wrapper includes unused world parameters, so this is not standalone A0 memory cost',
        'trainable_parameters_updated':False,'PDMS_computed':False})
    atomic_json(out/'status.json',{'status':'complete'})


if __name__=='__main__':main()

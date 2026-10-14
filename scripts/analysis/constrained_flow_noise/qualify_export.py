"""Exercise the updated original inference export without noise/cache/teacher/scorer."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, SFT_SHA256, file_sha, seed_for
from starVLA.rl.flow_grpo.data import KeyedDataset
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.analysis.constrained_flow_noise.build_condition_cache import SFT_EXPORT,VLM,DATA_ROOT,TRAIN_LIST


def prefix_hash(values):
    digest=hashlib.sha256()
    for name,value in sorted(values.items()):
        if name.startswith('action_model.'):continue
        digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(str(list(value.shape)).encode())
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--export',required=True)
    p.add_argument('--registration',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();host=require_dispatch();configure_numerics();torch.cuda.set_device(0)
    if Path(a.output).exists():raise FileExistsError(a.output)
    export=Path(a.export)
    if not (export/'COMPLETE').is_file() or file_sha(Path(SFT_EXPORT)/'pytorch_model.pt')!=SFT_SHA256:
        raise ValueError('Incomplete export or changed SFT')
    with torch.serialization.safe_globals([set]):
        original=torch.load(Path(SFT_EXPORT)/'pytorch_model.pt',map_location='cpu',weights_only=True,mmap=True)
        current=torch.load(export/'pytorch_model.pt',map_location='cpu',weights_only=True,mmap=True)
    original=original.get('module',original);current=current.get('module',current)
    if set(original)!=set(current) or any('noise_head' in name for name in current):raise ValueError('Deployment key closure changed')
    before,after=prefix_hash(original),prefix_hash(current)
    if before!=after:raise ValueError('Frozen upstream parameters changed')
    changed=sum(not torch.equal(value,current[name]) for name,value in original.items() if name.startswith('action_model.'))
    del original,current
    os.environ['BASE_VLM']=VLM;os.environ['VLM_ATTN_IMPLEMENTATION']='flash_attention_2'
    from infer import VLAAgent
    agent=VLAAgent(str(export),device='cuda',qwen_forward_mode='optimized')
    cfg=copy.deepcopy(agent.model_config);cfg.datasets.vla_data.data_root=DATA_ROOT;cfg.datasets.vla_data.datalist_path=TRAIN_LIST
    cfg.enable_image_aug=0;cfg.w_depth=0
    data=KeyedDataset(cfg);records=[]
    for token in json.loads(Path(a.registration).read_text())['calibration_tokens'][:2]:
        observation=prepare_policy_observation([data[(token,seed_for('original_ode_initial',2101,token))]])
        inputs=observation.examples()
        if any(any(key in example for key in ('action','reward','cost','GT','map')) for example in inputs):
            raise ValueError('Privileged label entered original deployment examples')
        root=torch.randn((1,8,4),device='cuda',dtype=torch.bfloat16,
               generator=torch.Generator(device='cuda').manual_seed(seed_for('original_ode_initial',2101,token)))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            prediction=agent.model.predict_action_infer_1d(inputs,initial_noise=root)
        action=np.asarray(prediction['normalized_actions'])
        if action.shape!=(1,8,4) or not np.isfinite(action).all():raise ValueError('Invalid native deployment output')
        records.append(dict(token=token,shape=list(action.shape),finite=True,input_keys=sorted(inputs[0])))
    atomic_json(Path(a.output),dict(status='PASS',host=host,export_sha256=file_sha(export/'pytorch_model.pt'),
                upstream_hash_before=before,upstream_hash_after=after,changed_action_tensors=changed,
                noise_head_absent=True,teacher_cache_search_scorer_not_used=True,inference_seed=2101,checks=records,
                actual_original_inference_entrypoint='VLAAgent.model.predict_action_infer_1d'))


if __name__=='__main__':main()

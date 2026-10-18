"""Real-image Qwen/condition/action gradient qualification; never formal weights."""
import argparse
import json
import time
from pathlib import Path
import torch
from starVLA.rl.qwen_sourceflow.contracts import ROOT, QWEN_PATH, POPULATION, ACTION_CONFIG, require_dispatch
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenEncoder, QwenObservationProcessor
from starVLA.model.modules.qwen_sourceflow.condition_provider import QwenConditionProvider
from starVLA.model.modules.qwen_sourceflow.source_flow_policy import SourceFlowPolicy
from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.model.modules.qwen_sourceflow.source_initialization import load_action_initialization
from scripts.analysis.qwen_sourceflow.prepare_data import process_log
from PIL import Image


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True)
    p.add_argument('--initialization',help='Omit ONLY for separately labelled tiny-action interface qualification')
    a=p.parse_args();host=require_dispatch();torch.set_num_threads(8);torch.manual_seed(42)
    started=time.monotonic();device='cuda';torch.cuda.set_device(0)
    pop=json.loads(POPULATION.read_text());token=pop['train_tokens'][0];log=pop['token_logs'][token]
    _,rows,_=process_log((log,[token],'trainval','/mnt/project/DriveDreamer-Policy/navsim_raw'))
    row=rows[token];obs={k:row[k] for k in ['proprio','navigation']};obs['images']={}
    for c,path in row['images'].items():
        with Image.open(path) as im:obs['images'][c]=im.convert('RGB')
    processor=QwenObservationProcessor(QWEN_PATH);batch=processor([obs]);proprio=torch.tensor([row['proprio']],device=device)
    encoder=QwenEncoder(QWEN_PATH).to(device)
    encoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if a.initialization:
        action=load_action_initialization(a.initialization,device=device)
    else:
        action=ActionDiT(hidden_dim=32,action_dim=3,ffn_dim=64,text_dim=4096,freq_dim=16,eps=1e-6,
            num_heads=2,attn_head_dim=12,num_layers=3,use_gradient_checkpointing=True).to(device)
    provider=QwenConditionProvider(action,encoder.hidden_dim,encoder.num_layers).to(device)
    policy=SourceFlowPolicy(action).to(device)
    encoder.train();provider.train();policy.train()
    target=ActionCodec.norm_odo(torch.tensor([row['requested_GT']],device=device))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        encoded=encoder(batch);condition=provider(encoded,proprio)
        loss,details=policy(condition,target)
    loss.backward();torch.cuda.synchronize()
    grad={}
    for label,module in [('qwen',encoder),('provider',provider),('action',policy)]:
        names={n:float(v.grad.float().norm()) for n,v in module.named_parameters() if v.grad is not None}
        if not names or not all(torch.isfinite(torch.tensor(v)) for v in names.values()):
            raise ValueError('Missing/nonfinite gradient '+label)
        grad[label]=dict(tensors=len(names),positive=sum(v>0 for v in names.values()),
            norm=float(torch.tensor(list(names.values())).norm()),first_names=list(names)[:6])
    if grad['qwen']['norm']==0 or grad['provider']['norm']==0 or grad['action']['norm']==0:
        raise ValueError('Action loss disconnected')
    result=dict(status='PASS',host=host,qualification_only=True,formal_initialization_eligible=False,
        action_scope='full_generic_initialized_source' if a.initialization else 'tiny_action_interface_only',
        loss=float(loss.detach()),gradients=grad,images=batch['image_grid_thw'].tolist(),
        visual_tokens=encoded['visual_mask'].sum(1).tolist(),text_tokens=encoded['text_mask'].sum(1).tolist(),
        hidden_count=len(encoded['hidden_states']),hidden_zero_is_embedding=True,layer_mapping=provider.layer_mapping,
        condition_bytes=sum(v.numel()*v.element_size() for layer in condition['visual_kv'] for v in layer.values()),
        peak_GPU_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.monotonic()-started)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

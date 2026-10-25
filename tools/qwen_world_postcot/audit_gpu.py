"""Real-image, full-width selected-Qwen and train/deploy isolation audit."""
import argparse
import gc
import json
from pathlib import Path
import pickle
import time
import torch
import yaml
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.model.modules.qwen_world_postcot.model import QwenWorldPostCoT,TRAIN_ONLY_PREFIXES
from starVLA.rl.qwen_world_postcot.data import WorldSceneData
from starVLA.rl.qwen_world_postcot.contracts import require_dispatch,file_sha

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);p.add_argument('--data',default='/mnt/project/qwen-sourceflow-native-20261018/data/TRAIN_v3')
    p.add_argument('--config',default='configs/qwen_world_postcot/il.yaml');a=p.parse_args();require_dispatch()
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0');cfg=yaml.safe_load(Path(a.config).read_text());path='/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct'
    data=WorldSceneData(a.data);tokens=data.tokens[:2];observations=[data.observation(t) for t in tokens]
    processor=QwenObservationProcessor(path);inputs=processor(observations);ego=torch.tensor([v['proprio'] for v in observations],device=device)
    started=time.monotonic();model=QwenWorldPostCoT(path,config=cfg['model']).to(device).eval()
    counts={name:sum(p.numel() for p in module.parameters()) for name,module in model.named_children()}
    counts['action_trunk']=sum(p.numel() for p in model.actor.parameters())
    counts['deployment_total']=sum(v.numel() for k,v in model.named_parameters() if not k.startswith(TRAIN_ONLY_PREFIXES))
    counts['IL_trainable']=sum(p.numel() for p in model.parameters());counts['qwen_visual']=sum(p.numel() for p in model.qwen.model.visual.parameters())
    kwargs={k:v.to(device) for k,v in inputs.items() if k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        features=model.qwen(inputs)
        official=model.qwen.model(**kwargs,use_cache=False,output_hidden_states=True,return_dict=True)
        parity={str(k):dict(equal=torch.equal(v,official.hidden_states[k]),max_abs=float((v.float()-official.hidden_states[k].float()).abs().max())) for k,v in features['selected_hidden_states'].items()}
    if not all(v['equal'] for v in parity.values()):raise AssertionError('Selected hidden-state indexing differs from official current forward')
    del official,features;gc.collect();torch.cuda.empty_cache()
    # Recompute all module parameters through the current image path; labels
    # cannot enter predict() at all. All default readouts are physically removed.
    noise=torch.randn(len(tokens),8,3,device=device);before=model.qwen.forward_calls
    original=model.predict(inputs,ego,noise)
    calls=model.qwen.forward_calls-before
    for name in ('dino_readout','future_readout','object_readout','post_head'):delattr(model,name)
    model.train_heads=False
    stripped=model.predict(inputs,ego,noise)
    error=float((original['requested_trajectory']-stripped['requested_trajectory']).abs().max())
    if error!=0 or calls!=1:raise AssertionError('Train-head removal or single-Qwen forward contract failed')
    # Temporary hooks must be gone before recomputation. Selected features
    # still carry a real gradient into Qwen from the resampler.
    model.train();model.configure_stage(False);model.qwen.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    with torch.autocast('cuda',dtype=torch.bfloat16):memory=model.encode_condition(inputs,ego);loss=memory.actor_memory.float().square().mean()
    loss.backward()
    qwen_grad=sum(float(p.grad.float().norm()) for p in model.qwen.parameters() if p.grad is not None)
    if qwen_grad<=0:raise AssertionError('Selected-memory path detached Qwen gradients')
    if not model.qwen.last_capture_metadata['hooks_removed']:raise AssertionError('Capture hooks remained installed during recomputation')
    torch.cuda.synchronize()
    result=dict(status='PASS',real_current_images=True,tokens=tokens,parameters=counts,selected_layer_parity=parity,
        single_current_qwen_calls=calls,deployment_head_removal_max_abs=error,qwen_gradient_norm_sum=qwen_grad,
        capture=model.qwen.last_capture_metadata,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        device=torch.cuda.get_device_name(),seconds=time.monotonic()-started,
        source_files={str(p):file_sha(p) for p in Path('starVLA/model/modules/qwen_world_postcot').glob('*.py')})
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()

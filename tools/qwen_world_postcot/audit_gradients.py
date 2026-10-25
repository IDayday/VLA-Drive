"""Real current images/teacher tensors/PDM candidates: full-size autograd audit."""
import argparse
import json
from pathlib import Path
import torch
import yaml
from starVLA.model.modules.qwen_world_postcot.model import QwenWorldPostCoT
from starVLA.model.modules.qwen_world_postcot.teacher_targets import TeacherCacheReader
from starVLA.model.modules.qwen_world_postcot.candidate_bank import CandidateBank
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.rl.qwen_world_postcot.data import WorldSceneData
from starVLA.rl.qwen_world_postcot.batching import BatchBuilder,to_device
from starVLA.rl.qwen_world_postcot.contracts import require_dispatch

def gradients(model):
    groups=dict(qwen_vision=0.,qwen_language=0.,memory=0.,actor_input=0.,actor_attention=0.,actor_ffn=0.,actor_output=0.,post=0.,dino=0.,future=0.,objects=0.)
    for name,p in model.named_parameters():
        if p.grad is None:continue
        val=float(p.grad.float().square().sum())
        if name.startswith('qwen.model.visual.'):key='qwen_vision'
        elif name.startswith('qwen.'):key='qwen_language'
        elif name.startswith(('scene_provider.','predictive_provider.')):key='memory'
        elif name.startswith('post_head.'):key='post'
        elif name.startswith('dino_readout.'):key='dino'
        elif name.startswith('future_readout.'):key='future'
        elif name.startswith('object_readout.'):key='objects'
        elif name.startswith('policy.actor.'):
            key='actor_output' if '.action_output.' in name else 'actor_ffn' if '.ffn.' in name else 'actor_attention' if '_attn.' in name else 'actor_input'
        else:continue
        groups[key]+=val
    return {k:v**.5 for k,v in groups.items()}

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--teacher-cache',required=True);p.add_argument('--candidate-bank',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--config',default='configs/qwen_world_postcot/il.yaml');a=p.parse_args();require_dispatch()
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=WorldSceneData(a.data);teachers=TeacherCacheReader(a.teacher_cache,expected_cohort='TRAIN',allow_building=True);bank=CandidateBank(a.candidate_bank,allowed_splits=('TRAIN',))
    common=sorted(set(teachers.records)&set(bank.manifest['scene_ids']))
    if not common:raise ValueError('No real image/teacher/PDM-candidate intersection; never substitute random labels')
    processor=QwenObservationProcessor('/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct');builder=BatchBuilder(data,processor,teachers,bank,42)
    batch=to_device(builder.build(common[:1]),'cuda')
    for k in ['candidate_trajectory','candidate_labels','dino_target','future_target']:batch[k].requires_grad_(True)
    model=QwenWorldPostCoT('/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct',config=yaml.safe_load(Path(a.config).read_text())['model']).cuda().train()
    model.qwen.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.configure_stage(False)
    results={};optimizer=torch.optim.AdamW(model.actor.parameters(),lr=1e-4,fused=True)
    # Unlock zero AdaLN gates with actual supervised updates before measuring
    # the expected FFN gradient. No model is promoted from this diagnostic.
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(batch,weights=dict(x=1.,dino=0.,future=0.,object=0.,post=1.))['loss']
        loss.backward();optimizer.step()
    del optimizer
    for objective in ['x','dino','future','object','post']:
        model.zero_grad(set_to_none=True)
        for k in ['candidate_trajectory','candidate_labels','dino_target','future_target']:batch[k].grad=None
        before=model.qwen.forward_calls
        weights={k:float(k==objective) for k in ['x','dino','future','object','post']}
        with torch.autocast('cuda',dtype=torch.bfloat16):result=model(batch,weights=weights)
        result['loss'].backward();g=gradients(model)
        leaked={k:batch[k].grad is not None for k in ['candidate_trajectory','candidate_labels','dino_target','future_target']}
        if any(leaked.values()):raise AssertionError('Detached supervision acquired gradients: '+str(leaked))
        if g['memory']<=0 or g['qwen_language']<=0:raise AssertionError('Auxiliary objective failed to train current student representations')
        if objective=='post' and (g['actor_attention']<=0 or g['actor_ffn']<=0 or g['post']<=0 or g['actor_output']!=0):raise AssertionError('Post shared-trunk/final-projection gradient contract failed: '+str(g))
        results[objective]=dict(gradients=g,detached_inputs=leaked,qwen_calls=model.qwen.forward_calls-before,loss=float(result['loss']))
        if results[objective]['qwen_calls']!=1:raise AssertionError('Auxiliary candidate replay reran Qwen')
    record=dict(status='PASS',real_current_images=True,real_official_PDM_labels=True,real_DINO_and_WA_targets=True,
        diagnostic_not_formal_checkpoint=True,scenes=common[:1],objectives=results,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)

if __name__=='__main__':main()

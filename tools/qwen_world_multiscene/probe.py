"""Full real IL70 two/four-scene Scope numerical and throughput probe.

Run only on a GPU explicitly assigned by the campaign owner. This tool neither
dispatches nor trains: zero Adam steps, zero scorer queries, no output weights.
"""
import argparse
import json
from pathlib import Path
import torch
from starVLA.rl.qwen_world_performance.batch_runtime import PerformanceRuntime
from starVLA.rl.qwen_world_performance.data_protocol import load_manifest
from starVLA.rl.qwen_world_postcot.scope_actor import ScopeRecipeActor
from starVLA.rl.qwen_world_multiscene.actor import MultiSceneScopeActor
from starVLA.rl.qwen_world_multiscene.conditions import PackedConditions
from starVLA.rl.qwen_world_multiscene.diagnostics import compare_execution
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('config','manifest','output','il-checkpoint','parent-selection','data'):p.add_argument('--'+key,required=True)
    p.add_argument('--scenes',type=int,choices=(2,4),default=2)
    p.add_argument('--tokens',nargs='+',help='Distinct legal TRAIN tokens; default first distinct manifest entries')
    p.add_argument('--warmup',type=int,default=1);p.add_argument('--iterations',type=int,default=2)
    a=p.parse_args();a.qualification=True;a.qualification_force_auxiliary=False
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.cuda.set_device(0)
    runtime=PerformanceRuntime();cfg=runtime.load_recipe(a)
    actor,_,_=runtime.make_actor(cfg,a,torch.device('cuda',0))
    #Explicit opt-in subclass, with precisely the SAME loaded parameters,
    #frozen IL actor/memory, buffers and device. No second actor/LoRA init.
    if type(actor)is not ScopeRecipeActor:raise TypeError('Unexpected source Scope actor class')
    actor.__class__=MultiSceneScopeActor;actor._loss=ResearchLoss(cfg)
    conditions=runtime.conditions;manifest,logs=load_manifest(a.manifest)
    tokens=a.tokens if a.tokens is not None else manifest['tokens'][:a.scenes]
    if len(tokens)!=a.scenes or len(set(tokens))!=a.scenes or any(t not in logs for t in tokens):raise ValueError('Distinct registered TRAIN scenes required')
    g=cfg['grpo']['sample']['group_size'];samples=[]
    #Source chains are real; advantages below are explicitly numerical fixtures,
    #not fabricated canonical feedback. A PG0 second scene tests denominator.
    with torch.no_grad():
        for i,token in enumerate(tokens):
            cond=conditions.get(token,actor.device)
            chain,times,deltas,steps=actor.sample_action_chain(cond,8,generator=torch.Generator().manual_seed(42+i))
            old=actor.action_chain_logprobs(cond,chain,times,deltas,stoch_idx=steps).detach()
            advantages=torch.linspace(-1,1,g,device=actor.device)if i%2==0 else torch.zeros(g,device=actor.device)
            samples.append(dict(token=token,log=logs[token],cond_bg=cond,chain=chain,timesteps=times,deltas=deltas,
                stoch_idx=steps,old_logp=old,advantages=advantages,adv_mask=torch.full((g,),float(i%2==0),device=actor.device)))
    packed=PackedConditions(conditions,selected_layers=actor.scene_provider.selected_layers).pack([r['cond_bg']for r in samples],actor.device)
    for _ in range(a.warmup):compare_execution(actor,samples,packed,loss=actor._loss)
    results=[]
    for i in range(a.iterations):
        result=compare_execution(actor,samples,packed,loss=actor._loss);results.append(result);atomic_json(out/f'iteration{i:02}.json',result)
    receipt=dict(schema='world_true_multiscene_GPU_probe_v1',status='COMPLETE_DIAGNOSTIC',base_source='5e5220c6059ed7e38b54b3198415fe4af3b5bf45',
        initial_model_sha256=runtime.parent['model_sha256'],trainable_closure=actor.closure,tokens=tokens,
        gradients_scope='Actual full loaded IL70 52.7M Scope; all params stored FP32/BF16 forwards',
        advantages_scope='synthetic diagnostic A/mask to test original per-scene PG+IL reductions; no canonical feedback claimed',
        formal_training=False,optimizer_steps=0,scoring_queries=0,queries='Real source action chains only, no official scorer called',
        B1_deployment='Not modified or batched; inherited original single-scene path',iterations=results,
        acceptance='No automatic deployment acceptance: report actual fresh guard, gradient discrepancies, speed and memory to campaign owner')
    atomic_json(out/'COMPLETE.json',receipt)
    print(json.dumps(dict(status=receipt['status'],output=str(out),fresh_ratio_all_pass=all(x['fresh_ratio_within_1e_4']for x in results))))


if __name__=='__main__':main()

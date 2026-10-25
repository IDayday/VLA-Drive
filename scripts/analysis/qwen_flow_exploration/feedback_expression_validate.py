"""CPU-only lifecycle/source-FM check; tiny weights and mock scores are discarded."""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile

import torch

from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.model.modules.qwen_sourceflow.source_flow_policy import SourceFlowPolicy
from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_flow_exploration.checkpoint import save_state,load_state
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol,Ledger
from starVLA.rl.qwen_flow_exploration.feedback.absorption import native_source_fm_loss
from starVLA.rl.qwen_flow_exploration.feedback.expression import (CONTRACT,read_registration,optimizer_step,evaluate,require_probe_lock)
from starVLA.rl.qwen_flow_exploration.feedback.student import EvaluationStopped,independent_fm_draw
from starVLA.rl.qwen_sourceflow.actor import NativeRecipeActor
from starVLA.rl.qwen_sourceflow.recipe import expected_recipe
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.safe_flow_improvement.contracts import seed_for
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state,restore_adapters
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--protocol',required=True)
    parser.add_argument('--registration',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU qualification must not allocate GPUs')
    torch.set_num_threads(2);torch.manual_seed(607)
    protocol=read_protocol(args.protocol);reg=read_registration(args.registration,protocol);checks=[]
    assert [row['category'] for row in reg['scenes']]==['stable_zero_repair','other_safety_repair','other_safety_repair','safe_EP_improvement']
    assert len({row['log_id'] for row in reg['scenes']})==4 and CONTRACT['logical_evaluation_outputs']==4*4*5==80
    checks.append('Same sealed TRAIN base; deterministic four distinct logs/categories; one existing scored target per scene; fixed512x4 updates/80 evaluations')
    for row in reg['scenes']:
        assert row['scene_token'] in protocol['splits']['TRAIN'] and row['split']=='TRAIN'
        assert row['teacher_reward']>=max(.9,row['candidate0_reward']+.005)
        target=torch.tensor(row['normalized_action']);physical=torch.tensor(row['planned_trajectory'])
        torch.testing.assert_close(ActionCodec.denorm_odo(target),physical,atol=1e-5,rtol=0)
    checks.append('Real qualified single teacher labels roundtrip source3D codec; no averaging or fabricated probabilities')
    architecture=dict(action_dim=3,hidden_dim=32,ffn_dim=64,text_dim=40,freq_dim=16,num_heads=2,
        attn_head_dim=12,num_layers=2,eps=1e-6,use_gradient_checkpointing=False)
    head=ActionDiT(**architecture).eval().requires_grad_(False)
    generator=torch.Generator().manual_seed(608)
    conditions={row['scene_token']:dict(context=torch.randn(1,5,40,generator=generator),
        context_mask=torch.ones(1,5,dtype=torch.bool),visual_mask=torch.ones(1,7,dtype=torch.bool),
        visual_kv=[dict(k=torch.randn(1,7,24,generator=generator),v=torch.randn(1,7,24,generator=generator)) for _ in range(2)])
        for row in reg['scenes']}
    class Tiny:
        def __init__(self):
            self.actor=NativeRecipeActor(copy.deepcopy(head),expected_recipe('R_CANONICAL'))
            self.policy=SourceFlowPolicy(self.actor.action_expert).eval();self.ledger=Ledger();self.scored=0;self.roots=[]
        def condition(self,token,split):
            assert split=='TRAIN';return conditions[token]
        def normalized_gt(self,*args):raise AssertionError('GT path entered expression probe')
        def source_fm_loss(self,condition,target,noise,times):
            self.ledger.add(model_forward_calls=1)
            return native_source_fm_loss(self.actor.action_velocity,condition,target.unsqueeze(0),noise=noise,timestep=times)
        def sample_ode(self,condition,root):
            assert root.shape==(1,8,3);self.roots.append(root.clone());self.ledger.add(model_forward_calls=10)
            return self.policy.sample_ode(condition,root)
        def score_actions(self,token,action,split,**kwargs):
            self.scored+=1;self.ledger.add(scored_candidates=1,scorer_batch_calls=1)
            # Temp fixture only, never published as a NAVSIM score.
            return [dict(status='OK',scene_token=token,seed=kwargs['seed'],raw_reward=.5,
                native_subscores=dict(score=.5),normalized_action=action[0].tolist(),
                planned_trajectory=ActionCodec.denorm_odo(action)[0].tolist(),
                latent_or_proposal_parameters=kwargs['latent_or_proposal_parameters'][0],synthetic_fixture=True)]
    def learner():
        env=Tiny();opt=torch.optim.AdamW([p for p in env.actor.parameters() if p.requires_grad],lr=5e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
        schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min((step+1)/8.,1.));return env,opt,schedule
    continuous,opt,schedule=learner();used=[]
    for update in range(2):used.append(optimizer_step(continuous,reg,opt,schedule,4*update)['LR_used'])
    with tempfile.TemporaryDirectory(prefix='expression_CPU_fixture_') as temporary:
        state=dict(phase='READY',round=2,updates=2,next_inner=0,arm='CPU_expression_fixture',seed=42,scene_cursor=8)
        saved=save_state(temporary,state,dict(batch=None,adapters=adapter_state(continuous.actor),optimizer=opt.state_dict(),scheduler=schedule.state_dict()),'CPU_FIXTURE_ONLY')
        for update in (2,3):optimizer_step(continuous,reg,opt,schedule,4*update)
        expected=tensor_digest(dict(adapters=adapter_state(continuous.actor),optimizer=opt.state_dict(),scheduler=schedule.state_dict()))
        resumed,ropt,rschedule=learner();restored,payload=load_state(saved,'CPU_FIXTURE_ONLY')
        restore_adapters(resumed.actor,payload['adapters']);ropt.load_state_dict(payload['optimizer']);rschedule.load_state_dict(payload['scheduler'])
        for update in (2,3):optimizer_step(resumed,reg,ropt,rschedule,4*update)
        assert restored==state and expected==tensor_digest(dict(adapters=adapter_state(resumed.actor),optimizer=ropt.state_dict(),scheduler=rschedule.state_dict()))
        assert {int(v['step']) for v in ropt.state.values()}=={4}
        checks.append('True tiny native attention-LoRA positive sourceFM performs four optimizer steps; READY resume reproduces every adapter/Adam/scheduler bitwise; no GT called')
        test=Path(temporary)/'evaluation';before=torch.get_rng_state().clone()
        try:evaluate(resumed,reg,test,4,stop_requested=lambda:len(list((test/'scenes').glob('*.json')))>=3)
        except EvaluationStopped:pass
        else:raise AssertionError('Stop between committed outputs was ignored')
        assert resumed.scored==3
        result=evaluate(resumed,reg,test,4)
        assert result['logical_single_outputs']==20 and resumed.scored==20
        evaluate(resumed,reg,test,4);assert resumed.scored==20
        assert torch.equal(before,torch.get_rng_state())
        first=reg['scenes'][0]['scene_token'];expected_root=torch.randn((1,8,3),generator=torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',2101,first)))
        assert torch.equal(resumed.roots[0],expected_root)
        checks.append('Actual source10step shift5 tinyODE uses original CPU/B1 scene-bound roots; interrupted20-output evaluation reuses committed records, no extra draw/score and no global RNG change')
    epsilon,timestep=independent_fm_draw(42,0,'teacher_expression_probe')
    assert not torch.equal(epsilon,expected_root) and timestep.dtype==torch.float32 and 0<=float(timestep[0])<=1000
    assert used==[5e-5/8,2*5e-5/8]
    checks.append('FM epsilon/time namespace independent of generation roots; continuous source sigma and eight-update warmup with LR_used distinguished')
    try:require_probe_lock(reg,Path(args.registration).parent/'BUDGET_LOCK_TEMPLATE.json','UNSEALED')
    except ValueError:pass
    else:raise AssertionError('Unapproved mechanism budget accepted')
    checks.append('Root-only budget/source/CPU-proof gate rejects unapproved template; CPU weights and synthetic scoring fixtures discarded')
    output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve previous CPU evidence')
    result=dict(status='PASS',qualification_only=True,synthetic_score_fixture_not_driving_evidence=True,
        registration_sha256=reg['sha256'],tiny_action_architecture=architecture,tiny_trainable_LoRA_tensors=len(ropt.state),
        full_model480_validation_reused_from_native_S1_S2_S3_and_reasserted_at_probe_start=True,
        checks=checks,source_sha256=file_sha(__file__))
    result['sha256']=fingerprint(result);atomic_json(output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()

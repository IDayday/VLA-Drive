"""Finite actual-model/scorer qualification before independent route training."""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from starVLA.rl.qwen_flow_exploration.protocol import ROOT,read_protocol,group_advantages
from starVLA.rl.qwen_flow_exploration.native import NativeEnvironment,tensor_bytes
from starVLA.rl.qwen_sourceflow.actor import NativeRecipeActor
from starVLA.rl.qwen_sourceflow.recipe import expected_recipe
from starVLA.rl.safe_flow_improvement import v1_reward
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Preserve qualification attempts')
    protocol=read_protocol();env=None;start=time.monotonic();checks=[];score_errors=[];probe_token=None
    try:
        env=NativeEnvironment(experiment_id='common_qualification',ledger_path=out.with_suffix('.ledger.json'))
        env.model.policy.action_expert.eval();torch.cuda.reset_peak_memory_stats()
        for split in ['TRAIN','DEV_SELECT']:
            for j,token in enumerate(protocol['smoke'][split]):
                condition=env.condition(token,split)
                roots=torch.randn((4,8,3),generator=torch.Generator().manual_seed(70000+j)).to(env.device)
                generated=env.sample_ode(condition,roots)
                if j==0:
                    one=torch.cat([env.model.policy.sample_ode(condition,z[None]) for z in roots])
                    env.ledger.add(model_forward_calls=40)
                    torch.testing.assert_close(generated,one,atol=1e-5,rtol=1e-5)
                    checks.append(dict(split=split,test='ODE_B1_chunk4',normalized_max_abs=float((generated-one).abs().max())))
                records=env.score_actions(token,generated,split,proposal_source='common_smoke')
                if split=='TRAIN' and probe_token is None and np.std([r['raw_reward'] for r in records])>1e-8:
                    probe_token=token
                if j<2:
                    log=records[0]['log_id'];physical=np.array([r['planned_trajectory'] for r in records])
                    singleton=env.service.score((token,log,physical[:1]))[0]
                    reversed_=env.service.score((token,log,physical[::-1].copy()))[-1]
                    official=env.service.pool.submit(v1_reward.score_scene,(token,log,physical[:1])).result()[0]
                    env.ledger.add(candidate_attempts=6,scored_candidates=6,scorer_batch_calls=3,logical_scoring_cost=6)
                    for other in [singleton,reversed_,official]:
                        error=max(abs(records[0]['native_subscores'][k]-other['metrics'][k]) for k in records[0]['native_subscores'])
                        assert error<=1e-12,('candidate-context score changed',error)
                        np.testing.assert_allclose(records[0]['simulated_trajectory'],other['simulated_states'],atol=1e-10,rtol=1e-10)
                        score_errors.append(error)
                    checks.append(dict(split=split,test='official_singleton_batch_order_and_simulated_state_isolation',max_score_abs=max(score_errors)))
                if j%8==0:print(json.dumps(dict(stage='smoke',split=split,scenes=j+1)),flush=True)
        checks.append(dict(test='smoke_real_population',TRAIN=32,DEV_SELECT=16,real_candidates=192))
        token=probe_token or protocol['smoke']['TRAIN'][0];condition=env.condition(token,'TRAIN')
        # Establish source data-label identity before any supervised update.
        physical=env.dataset('TRAIN').target(token)
        torch.testing.assert_close(env.codec.denorm_odo(env.normalized_gt(token)),physical,atol=1e-5,rtol=1e-6)
        cfg=expected_recipe('R_CANONICAL');cfg['grpo']['sample']['group_size']=16;cfg['grpo']['train']['adv_clip_max']=None
        env.actor=NativeRecipeActor(env.model.policy.action_expert,cfg);env.actor.record_sampling_terms=True
        opt=torch.optim.AdamW([p for p in env.actor.parameters() if p.requires_grad],lr=5e-5,betas=(.9,.95),weight_decay=.01)
        generator=torch.Generator(device='cpu').manual_seed(80000)
        rollout,records=env.sample_source_sde(condition,16,generator,token=token)
        adv=group_advantages(rollout['rewards']);ratios=[]
        before={n:p.detach().cpu().clone() for n,p in env.actor.named_parameters() if p.requires_grad}
        for inner in range(4):
            opt.zero_grad();loss_value=0.;error=0.
            for offset,(loss,metrics,weight) in enumerate(env.policy_chunks(rollout,adv['advantages'][0],adv['mask'][0],condition)):
                current=env.actor._probability_audit['current']['source_clamped_logprob'].mean((-1,-2))
                old=rollout['old_logp'][offset*4:offset*4+len(current)].to(env.device)
                error=max(error,float((current-old).exp().sub(1).abs().max()))
                (loss*weight).backward();loss_value+=float(loss.detach())*weight
            if inner==0:assert error<1e-4,error
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in env.actor.parameters() if p.requires_grad)
            opt.step();env.ledger.add(optimizer_updates=1);ratios.append(error)
        changed=sum(not torch.equal(before[n],p.detach().cpu()) for n,p in env.actor.named_parameters() if p.requires_grad)
        assert changed==480 and all(float(state['step'])==4 for state in opt.state.values())
        checks.append(dict(test='actual_native_source_SDE_G16_chunk4_and_four_optimizer_steps',changed_LoRA_tensors=changed,ratios=ratios))
        peak=torch.cuda.max_memory_allocated();record=dict(status='PASS',qualification_only=True,checks=checks,
            condition_bytes_max=max(env.condition_sizes),peak_GPU_bytes=peak,wall_seconds=time.monotonic()-start,
            ledger=env.ledger.snapshot(),not_a_driving_effectiveness_result=True)
        atomic_json(out,record);print(json.dumps(record),flush=True)
    except BaseException:
        atomic_json(out,dict(status='FAILED',qualification_only=True,error=traceback.format_exc(),
             completed_checks=checks,protocol_sha256=protocol['sha256'],
             ledger=env.ledger.snapshot() if env is not None else None,wall_seconds=time.monotonic()-start));raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()

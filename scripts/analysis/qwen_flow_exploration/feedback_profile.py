"""Assigned-GPU native search/FM/resampling qualification; never formal weights."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol
from starVLA.rl.qwen_flow_exploration.native import NativeEnvironment
from starVLA.rl.qwen_flow_exploration.feedback.process import NoveltyScale, process_summary
from starVLA.rl.qwen_flow_exploration.feedback.search import METHODS, run_search, discovery_budget_curve
from starVLA.rl.qwen_flow_exploration.feedback.student import adapter_digest, independent_fm_draw, frozen_digest, evaluate_student, teacher_normalized_target
from scripts.analysis.qwen_flow_exploration.feedback_search_bank import jsonable


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    p.add_argument('--train-scenes',type=int,default=8);p.add_argument('--dev-scenes',type=int,default=4)
    p.add_argument('--optimizer-updates',type=int,default=4);p.add_argument('--G',type=int,default=4)
    a=p.parse_args();protocol=read_protocol(a.protocol);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    env=NativeEnvironment(protocol,trainable='lora',scorer_workers=1,cache_scenes=a.train_scenes+a.dev_scenes,
                          ledger_path=out/'ledger.json',experiment_id='feedback_qualification_only')
    started=time.monotonic()
    try:
        train=protocol['splits']['TRAIN'][:a.train_scenes];dev=protocol['splits']['DEV_SELECT'][:a.dev_scenes]
        before_train=evaluate_student(env,train,'TRAIN',seeds=list(range(9101,9101+a.G)),namespace='feedback_profile_probe')
        before_dev=evaluate_student(env,dev,'DEV_SELECT',seeds=list(range(9101,9101+a.G)),namespace='feedback_profile_probe')
        token=train[0];cond=env.condition(token,'TRAIN')
        def iid(round_index,count):
            roots=torch.randn((count,8,3),generator=torch.Generator().manual_seed(10000+round_index))
            action=env.sample_ode(cond,roots)
            return env.score_actions(token,action,'TRAIN',proposal_source='qualification_ODE')
        initial=iid(-1,8)
        scale=NoveltyScale.fit_train([process_summary(row['feedback'])['novelty_features'] for row in initial],split='TRAIN')
        def score(requests):
            normalized=env.codec.norm_odo(torch.tensor(requests,dtype=torch.float32))
            return env.score_actions(token,normalized,'TRAIN',proposal_source='qualification_structured',requested_trajectories=requests)
        searches={}
        for method in METHODS:
            result=run_search(method,initial,initial_velocity_xy=env.observation(token,'TRAIN')['proprio'][:2],
                              sample_iid=iid,score_requests=score,seed=42,novelty_scale=scale if method=='F4_NOVELTY' else None)
            result['budget_curve']=discovery_budget_curve(result,reference_reward=initial[0]['raw_reward'])
            searches[method]=result
        # Native label/gradient probe uses a real feasible external request if
        # one exists; it is NOT promoted into formal teacher qualification.
        external=[row for row in searches['F3_FEEDBACK']['candidates'][8:]
                  if row.get('raw_reward') is not None and process_summary(row['feedback'])['feasible']
                  and float(np.linalg.norm(np.asarray(row['planned_trajectory'])[:,:2]-np.asarray(initial[0]['planned_trajectory'])[:,:2],axis=1).mean())>.01]
        external.sort(key=lambda row:-row['raw_reward'])
        probe=external[0] if external else None
        frozen_before=frozen_digest(env);adapters_before=adapter_digest(env.actor)
        params=[param for param in env.actor.parameters() if param.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=5e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
        steps=[]
        for update in range(a.optimizer_updates):
            optimizer.zero_grad();losses=[]
            for index,scene in enumerate(train[:2]):
                condition=env.condition(scene,'TRAIN');noise,times=independent_fm_draw(42,update*2+index,'GT_qualification')
                target=env.normalized_gt(scene)
                loss,_=env.source_fm_loss(condition,target,noise=noise,times=times)
                if index==0 and probe is not None:
                    noise,times=independent_fm_draw(42,update,'teacher_qualification')
                    teacher,_=env.source_fm_loss(condition,teacher_normalized_target(probe,env.codec),noise=noise,times=times)
                    loss=loss+.1*teacher
                (loss/min(2,len(train))).backward();losses.append(float(loss.detach()))
                env.ledger.add(model_backward_calls=1)
            if any(param.grad is None or not torch.isfinite(param.grad).all() for param in params):
                raise ValueError('Actual student LoRA gradient is missing/nonfinite')
            norm=float(torch.sqrt(sum(param.grad.detach().double().square().sum() for param in params)))
            optimizer.step();env.ledger.add(optimizer_updates=1)
            steps.append(dict(update=update+1,loss=float(np.mean(losses)),gradient_norm=norm))
        if adapter_digest(env.actor)==adapters_before or frozen_digest(env)!=frozen_before:
            raise ValueError('Student failed to update LoRA or changed frozen model/reference')
        env.set_behavior_identity(dict(kind='qualification_student',updates=a.optimizer_updates,LoRA_sha256=adapter_digest(env.actor)))
        after_train=evaluate_student(env,train,'TRAIN',seeds=list(range(9101,9101+a.G)),namespace='feedback_profile_probe')
        after_dev=evaluate_student(env,dev,'DEV_SELECT',seeds=list(range(9101,9101+a.G)),namespace='feedback_profile_probe')
        payload=dict(status='PASS',qualification_only=True,no_formal_initialization=True,source_protocol=protocol['sha256'],
            base_model_sha256=env.base['model_sha256'],training_scenes=a.train_scenes,DEV_SELECT_scenes=a.dev_scenes,G=a.G,
            optimizer_steps=steps,actual_LoRA_Adam_states=len(optimizer.state),all_actual_Adam_steps_match=all(float(x['step'])==a.optimizer_updates for x in optimizer.state.values()),
            frozen_model_and_reference_hash_unchanged=True,external_feasible_probe_available=probe is not None,
            external_probe_is_not_a_formal_success_label=True,not_improvement_qualification=True,
            external_probe_ADE_to_base_m=(float(np.linalg.norm(np.asarray(probe['planned_trajectory'])[:,:2]-np.asarray(initial[0]['planned_trajectory'])[:,:2],axis=1).mean()) if probe is not None else None),
            searches=jsonable(searches),
            before_TRAIN=jsonable(before_train),after_TRAIN=jsonable(after_train),before_DEV=jsonable(before_dev),after_DEV=jsonable(after_dev),
            ledger=env.ledger.snapshot(),peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.monotonic()-started)
        atomic_json(out/'qualification.json',payload)
        print(json.dumps({key:value for key,value in payload.items() if key not in ['searches','before_TRAIN','after_TRAIN','before_DEV','after_DEV']},indent=2))
    finally:env.close()


if __name__=='__main__':main()

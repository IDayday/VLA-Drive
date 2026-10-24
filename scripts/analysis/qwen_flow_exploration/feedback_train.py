"""S1/S2/S3 independent native LoRA source-FM students with exact READY resume."""
import argparse
import json
import math
from pathlib import Path
import time
import uuid
import traceback
import signal

import numpy as np
import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state,restore_adapters
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol,require_budget_lock
from starVLA.rl.qwen_flow_exploration.native import NativeEnvironment
from starVLA.rl.qwen_flow_exploration.checkpoint import save_state,load_state
from starVLA.rl.qwen_flow_exploration.feedback.student import (adapter_digest,frozen_digest,load_teacher_bank,
    choose_teacher,teacher_normalized_target,independent_fm_draw,evaluate_student,EvaluationStopped)
from scripts.analysis.qwen_flow_exploration.feedback_search_bank import jsonable


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--protocol',required=True);p.add_argument('--arm',choices=['S1','S2','S3'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--search-bank');p.add_argument('--teacher-calibration')
    p.add_argument('--updates',type=int,default=256);p.add_argument('--scene-batch',type=int,default=16);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save-every',type=int,default=32);p.add_argument('--resume');p.add_argument('--stop-after-update',type=int)
    p.add_argument('--evaluate',action='store_true');a=p.parse_args();protocol=read_protocol(a.protocol);require_budget_lock(protocol)
    stopped=[False]
    def request_stop(signum,frame):stopped[0]=True
    signal.signal(signal.SIGUSR1,request_stop);signal.signal(signal.SIGTERM,request_stop)
    if a.updates!=256 or a.scene_batch!=16:raise ValueError('Initial registered FM window is256updates/B16; use profile for qualification')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=bool(a.resume))
    teachers={token:[] for token in protocol['splits']['TRAIN']};teacher_hash=None;beta=0.;calibration_sha=None
    if a.arm!='S1':
        if not a.search_bank or not a.teacher_calibration:raise ValueError('Actual fixed TRAIN teacher bank/calibration required')
        methods=('F2_PDMS',) if a.arm=='S2' else ('F3_FEEDBACK','F4_NOVELTY')
        teachers,teacher_hash=load_teacher_bank(a.search_bank,protocol,expected_method=methods)
        calibration=json.loads(Path(a.teacher_calibration).read_text());body=dict(calibration)
        if body.pop('sha256')!=fingerprint(body) or calibration['status']!='COMPLETE' or calibration['calibrated_split']!='TRAIN' or calibration['common_protocol_sha256']!=protocol['sha256']:
            raise ValueError('Teacher scale not fixed on the same TRAIN protocol')
        expected=calibration['scalar_teacher_bank_hash' if a.arm=='S2' else 'physical_teacher_bank_hash']
        if expected!=teacher_hash:raise ValueError('Teacher bank changed after calibration')
        beta=calibration['teacher_coefficient'];calibration_sha=calibration['sha256']
    elif a.search_bank or a.teacher_calibration:raise ValueError('S1 is GT-only; do not attach search data')
    config=dict(schema='native_feedback_student_v1',common_protocol_sha256=protocol['sha256'],arm=a.arm,seed=a.seed,
        updates=a.updates,scene_batch=a.scene_batch,LR=5e-5,betas=[.9,.95],eps=1e-8,weight_decay=.01,warmup=8,
        schedule='linear8update_warmup_then_constant',teacher_bank_hash=teacher_hash,teacher_calibration=calibration_sha,
        teacher_coefficient=beta,GT_coefficient=1.,FM_repeat=1,source_loss='original_source_weighted_FM',
        fixed_scene_denominator=True,teacher_missing='zero_teacher_term_keep_GT_and_scene',source_file_sha256=file_sha(__file__))
    config['sha256']=fingerprint(config)
    if (output/'config.json').exists() and json.loads((output/'config.json').read_text())!=config:raise ValueError('Changed student scientific configuration')
    atomic_json(output/'config.json',config)
    attempt=uuid.uuid4().hex;attempt_root=output/'attempts'/attempt;attempt_root.mkdir(parents=True)
    invocation=dict(attempt=attempt,start_unix=time.time(),status='LOADING',config_sha256=config['sha256'],resume=a.resume)
    atomic_json(attempt_root/'invocation.json',invocation)
    env=NativeEnvironment(protocol,trainable='lora',scorer_workers=1,ledger_path=attempt_root/'ledger.json',experiment_id='feedback_'+a.arm)
    params=[p for p in env.actor.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=5e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/8.,1.))
    state=dict(phase='READY',round=0,updates=0,next_inner=0,arm='feedback_'+a.arm,seed=a.seed,scene_cursor=0)
    frozen_before=frozen_digest(env);teacher_exposures=0;GT_exposures=0
    if a.resume:
        state,payload=load_state(a.resume,protocol['sha256'])
        if payload['config_sha256']!=config['sha256'] or state['arm']!='feedback_'+a.arm or state['seed']!=a.seed:raise ValueError('Wrong student resume identity')
        restore_adapters(env.actor,payload['adapters']);optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
        frozen_before=payload['frozen_hash'];teacher_exposures=payload['teacher_exposures'];GT_exposures=payload['GT_exposures']
    start_update=state['updates'];old_lineage=list(state.get('metric_lineage',[]))
    invocation.update(status='RUNNING',start_update=start_update,start_scene_cursor=state['scene_cursor'])
    atomic_json(attempt_root/'invocation.json',invocation)
    tokens=protocol['splits']['TRAIN'];orders={}
    def boundary():
        state['metric_lineage']=old_lineage+([dict(attempt=attempt,first_update=start_update+1,last_update=state['updates'])] if state['updates']>start_update else [])
        return save_state(output,state,dict(batch=None,config_sha256=config['sha256'],adapters=adapter_state(env.actor),
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),teacher_exposures=teacher_exposures,
            GT_exposures=GT_exposures,frozen_hash=frozen_before,teacher_bank_hash=teacher_hash,
            independent_FM_RNG=dict(seed=a.seed,namespaces=['GT','teacher'],exposure_cursor=state['scene_cursor'])),protocol['sha256'])
    try:
        if not a.resume:boundary()
        metrics=(attempt_root/'metrics.jsonl').open('x')
        while state['updates']<a.updates:
            if stopped[0]:boundary();break
            optimizer.zero_grad();started=time.monotonic();GT_losses=[];teacher_losses=[];current_teacher=0
            for slot in range(a.scene_batch):
                exposure=state['scene_cursor']+slot;epoch,index=divmod(exposure,len(tokens))
                if epoch not in orders:orders[epoch]=torch.randperm(len(tokens),generator=torch.Generator().manual_seed(a.seed+epoch)).tolist()
                token=tokens[orders[epoch][index]];condition=env.condition(token,'TRAIN')
                noise,times=independent_fm_draw(a.seed,exposure,'GT')
                loss,_=env.source_fm_loss(condition,env.normalized_gt(token),noise=noise,times=times)
                (loss/a.scene_batch).backward();GT_losses.append(float(loss.detach()));GT_exposures+=1
                env.ledger.add(model_backward_calls=1)
                teacher=choose_teacher(teachers[token],a.seed,exposure)
                if teacher is not None:
                    noise,times=independent_fm_draw(a.seed,exposure,'teacher')
                    loss,_=env.source_fm_loss(condition,teacher_normalized_target(teacher,env.codec),noise=noise,times=times)
                    (beta*loss/a.scene_batch).backward();teacher_losses.append(float(loss.detach()));current_teacher+=1;teacher_exposures+=1
                    env.ledger.add(model_backward_calls=1)
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise ValueError('Missing/nonfinite source-FM LoRA gradient')
            gradient_norm=float(torch.sqrt(sum(p.grad.detach().double().square().sum() for p in params)))
            optimizer.step();scheduler.step();state.update(updates=state['updates']+1,round=state['round']+1,scene_cursor=state['scene_cursor']+a.scene_batch)
            env.ledger.add(optimizer_updates=1,update_seconds=time.monotonic()-started)
            row=dict(attempt=attempt,updates=state['updates'],scene_cursor=state['scene_cursor'],GT_loss=float(np.mean(GT_losses)),
                teacher_loss=float(np.mean(teacher_losses)) if teacher_losses else None,teacher_scenes=current_teacher,
                GT_exposures=GT_exposures,teacher_exposures=teacher_exposures,gradient_norm=gradient_norm,lr=optimizer.param_groups[0]['lr'],
                teacher_coefficient=beta,update_seconds=time.monotonic()-started)
            metrics.write(json.dumps(row)+'\n');metrics.flush()
            if state['updates']%a.save_every==0 or state['updates']==a.updates or state['updates']==a.stop_after_update or stopped[0]:boundary()
            if stopped[0] or (a.stop_after_update and state['updates']>=a.stop_after_update):break
        metrics.close()
        if state['updates']==a.updates and not stopped[0]:
            if frozen_digest(env)!=frozen_before:raise ValueError('FM student changed frozen model/reference')
            state['phase']='COMPLETE';checkpoint=boundary()
            env.set_behavior_identity(dict(kind='feedback_student',arm=a.arm,updates=state['updates'],LoRA_sha256=adapter_digest(env.actor)))
            result=dict(status='COMPLETE',arm=a.arm,updates=state['updates'],scene_exposures=state['scene_cursor'],GT_exposures=GT_exposures,
                teacher_exposures=teacher_exposures,teacher_scene_coverage=sum(bool(v) for v in teachers.values()),
                frozen_hash_unchanged=True,checkpoint=str(checkpoint),config_sha256=config['sha256'],ledger=env.ledger.snapshot())
            if a.evaluate:
                for split in ['TRAIN','DEV_SELECT']:
                    evaluation=evaluate_student(env,protocol['diagnostic'][split],split,seeds=protocol['evaluation_seeds'],output=output/'evaluation'/split,stop_requested=lambda:stopped[0])
                    atomic_json(output/f'evaluation_{split}.json',jsonable(evaluation))
                result['evaluation']='fresh_prior_single_ODE_each_fixed_seed_not_candidate_selection'
            atomic_json(output/'COMPLETE.json',result);print(json.dumps(result))
        invocation['status']='COMPLETE' if state['updates']==a.updates and not stopped[0] else 'STOPPED_AT_READY_BOUNDARY'
    except EvaluationStopped:
        state['phase']='READY';boundary();invocation['status']='STOPPED_AT_READY_BOUNDARY'
    except BaseException:
        invocation.update(status='FAILED',error=traceback.format_exc())
        raise
    finally:
        invocation.update(end_unix=time.time(),end_update=state['updates'],last_valid_checkpoint=json.loads((output/'latest.json').read_text()) if (output/'latest.json').exists() else None)
        atomic_json(attempt_root/'invocation.json',invocation)
        env.close()


if __name__=='__main__':main()

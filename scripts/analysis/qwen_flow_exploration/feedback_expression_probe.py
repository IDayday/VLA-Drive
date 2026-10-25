"""Register or run one bounded TRAIN teacher-expression diagnostic, not a method arm."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time
import traceback
import uuid

import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol
from starVLA.rl.qwen_flow_exploration.checkpoint import save_state,load_state
from starVLA.rl.qwen_flow_exploration.feedback.expression import (CONTRACT,prepare_registration,
    read_registration,require_probe_lock,optimizer_step,evaluate,validate_evaluation_population)
from starVLA.rl.qwen_flow_exploration.feedback.student import frozen_digest,adapter_digest,EvaluationStopped
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state,restore_adapters


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--protocol',required=True);p.add_argument('--admission-analysis',required=True)
    p.add_argument('--teacher-bank',required=True);p.add_argument('--reference-strata',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('train');p.add_argument('--protocol',required=True);p.add_argument('--registration',required=True)
    p.add_argument('--budget-lock',required=True);p.add_argument('--output',required=True);p.add_argument('--resume')
    p.add_argument('--save-every',type=int,default=64)
    args=parser.parse_args();protocol=read_protocol(args.protocol)
    if args.command=='prepare':
        path=Path(args.output)
        if path.exists():raise FileExistsError('Keep original registration; choose a new probe version')
        registration=prepare_registration(protocol,args.admission_analysis,args.teacher_bank,args.reference_strata)
        path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,registration)
        template=dict(phase='UNAPPROVED_MECHANISM_PROBE',protocol_sha256=protocol['sha256'],
            registration_sha256=registration['sha256'],contract=CONTRACT,
            execution_source_sha='ROOT_SEALS_IMMUTABLE_COMMIT_AFTER_CPU_QUALIFICATION',
            qualification_file_sha256='ROOT_BINDS_ACTUAL_qualification_cpu.json')
        template['sha256']=fingerprint(template);atomic_json(path.parent/'BUDGET_LOCK_TEMPLATE.json',template)
        print(json.dumps(dict(status='REGISTERED_NOT_DISPATCHED',registration_sha256=registration['sha256'],
            scenes=[dict(category=row['category'],teacher_reward=row['teacher_reward'],external=row['teacher_external']) for row in registration['scenes']])));return
    registration=read_registration(args.registration,protocol)
    source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    lock=require_probe_lock(registration,args.budget_lock,source_sha)
    if args.save_every!=64:raise ValueError('Probe fixed64-update save cadence; no extra runtime variants')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=bool(args.resume))
    stopped=[False]
    def request_stop(signum,frame):stopped[0]=True
    signal.signal(signal.SIGUSR1,request_stop);signal.signal(signal.SIGTERM,request_stop)
    attempt=uuid.uuid4().hex;folder=output/'attempts'/attempt;folder.mkdir(parents=True)
    invocation=dict(status='LOADING',attempt=attempt,start_unix=time.time(),registration_sha256=registration['sha256'],
        execution_source_sha=source_sha,budget_lock_sha256=lock['sha256'],resume=args.resume)
    atomic_json(folder/'invocation.json',invocation)
    from starVLA.rl.qwen_flow_exploration.native import NativeEnvironment
    env=NativeEnvironment(protocol,trainable='lora',scorer_workers=1,cache_scenes=4,
        experiment_id='TRAIN_teacher_expression_probe',ledger_path=folder/'ledger.json')
    params=[p for p in env.actor.parameters() if p.requires_grad]
    if len(params)!=480 or any(not name.startswith('action_expert.') or not name.endswith(('lora_A','lora_B'))
            for name,p in env.actor.named_parameters() if p.requires_grad):
        raise ValueError('Expression probe must use exactly the original480 native attention LoRA tensors')
    optimizer=torch.optim.AdamW(params,lr=5e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/8.,1.))
    initial_adapter_sha=adapter_digest(env.actor);frozen_before=frozen_digest(env)
    state=dict(phase='READY',round=0,updates=0,next_inner=0,arm='feedback_teacher_expression',seed=42,
        scene_cursor=0,evaluated_milestones=[])
    if args.resume:
        state,payload=load_state(args.resume,protocol['sha256'])
        if state['arm']!='feedback_teacher_expression' or payload['registration_sha256']!=registration['sha256']:
            raise ValueError('Wrong expression resume identity')
        restore_adapters(env.actor,payload['adapters']);optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
        if payload['frozen_hash']!=frozen_before:raise ValueError('Frozen base/reference differs on resume')
        initial_adapter_sha=payload['initial_adapter_sha256']
        if any(step<state['updates'] and step not in state['evaluated_milestones'] for step in CONTRACT['evaluation_updates']):
            raise ValueError('Resume has advanced past an uncompleted evaluation milestone')
    start_update=state['updates'];old_lineage=list(state.get('metric_lineage',[]))
    invocation.update(status='RUNNING',start_update=start_update);atomic_json(folder/'invocation.json',invocation)
    def boundary():
        state['metric_lineage']=old_lineage+([dict(attempt=attempt,first_update=start_update+1,last_update=state['updates'])] if state['updates']>start_update else [])
        return save_state(output,state,dict(batch=None,registration_sha256=registration['sha256'],
            budget_lock_sha256=lock['sha256'],adapters=adapter_state(env.actor),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),initial_adapter_sha256=initial_adapter_sha,frozen_hash=frozen_before,
            FM_noise_identity=dict(seed=42,namespace='feedback_source_FM_teacher_expression_probe',exposure_cursor=state['scene_cursor'])),protocol['sha256'])
    try:
        if not args.resume:boundary()
        with (folder/'metrics.jsonl').open('x') as metrics:
            while True:
                if stopped[0]:boundary();break
                update=state['updates']
                if update in CONTRACT['evaluation_updates'] and update not in state['evaluated_milestones']:
                    env.set_behavior_identity(dict(kind='TRAIN_teacher_expression_probe',updates=update,adapter_sha256=adapter_digest(env.actor)))
                    evaluate(env,registration,output/'evaluation'/f'update{update:06d}',update,stop_requested=lambda:stopped[0])
                    state['evaluated_milestones'].append(update);boundary()
                if update==512:break
                row=optimizer_step(env,registration,optimizer,scheduler,state['scene_cursor'])
                state.update(updates=update+1,round=update+1,scene_cursor=4*(update+1))
                row.update(attempt=attempt,updates=state['updates'],teacher_exposures=state['scene_cursor'])
                metrics.write(json.dumps(row)+'\n');metrics.flush()
                if state['updates']%64==0 or state['updates'] in CONTRACT['evaluation_updates'] or stopped[0]:boundary()
        if state['updates']==512 and state['evaluated_milestones']==[0,128,256,512] and not stopped[0]:
            actual_outputs=validate_evaluation_population(output/'evaluation',registration)
            if frozen_digest(env)!=frozen_before:raise ValueError('Expression probe modified frozen model/reference')
            if len(optimizer.state)!=480 or {int(value['step']) for value in optimizer.state.values()}!={512}:
                raise ValueError('Wrong actual expression Adam update counts')
            state['phase']='COMPLETE';checkpoint=boundary()
            current_ledger=env.ledger.snapshot()
            history=[json.loads(path.read_text()) for path in (output/'attempts').glob('*/ledger.json') if path.parent!=folder]
            cumulative={key:sum(row.get(key,0) for row in history+[current_ledger])
                for key in ('optimizer_updates','model_forward_calls','model_backward_calls','scored_candidates','scorer_batch_calls','qwen_prefills','update_seconds')}
            result=dict(status='COMPLETE',qualification=True,TRAIN_fit_only_not_generalization=True,
                formal_arm=False,registration_sha256=registration['sha256'],updates=512,teacher_exposures=2048,
                expected_logical_evaluation_outputs=80,actual_retained_single_outputs=actual_outputs,
                evaluated_milestones=state['evaluated_milestones'],cumulative_physical_cost_including_failed_attempts=cumulative,
                frozen_hash_unchanged=True,checkpoint=str(checkpoint),ledger_this_invocation=current_ledger)
            result['sha256']=fingerprint(result);atomic_json(output/'COMPLETE.json',result)
            invocation['status']='COMPLETE'
        else:invocation['status']='STOPPED_AT_READY_BOUNDARY'
    except EvaluationStopped:
        boundary();invocation['status']='STOPPED_AT_COMMITTED_SINGLE_OUTPUT_BOUNDARY'
    except BaseException:
        # A bad CUDA context is not asked to save RNG/model again. Last legal
        # READY stays latest; completed evaluation records remain independently sealed.
        invocation.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        invocation.update(end_unix=time.time(),end_update=state['updates'],
            last_valid_checkpoint=json.loads((output/'latest.json').read_text()) if (output/'latest.json').exists() else None)
        atomic_json(folder/'invocation.json',invocation);env.close()


if __name__=='__main__':main()

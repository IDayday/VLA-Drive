"""Bounded TRAIN-only single-teacher expressibility probe, never a method arm."""
import csv
import json
from pathlib import Path
import time
import uuid

import torch

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.safe_flow_improvement.contracts import seed_for
from .process import SAFETY
from .student import independent_fm_draw,teacher_normalized_target,adapter_digest,EvaluationStopped


CONTRACT=dict(kind='TRAIN_teacher_expression_mechanism_only',seed=42,updates=512,scene_batch=4,
    teacher_per_scene=1,teacher_exposures=2048,FM_repeat=1,GT_coefficient=0.,anchor_coefficient=0.,
    teacher_coefficient=1.,LR=5e-5,betas=[.9,.95],eps=1e-8,weight_decay=.01,
    warmup_updates=8,scheduler='linear8warmup_then_constant',effective_gradient_clip=0.,
    evaluation_updates=[0,128,256,512],evaluation_seeds=[2101,2102,2103,2104,2105],
    logical_evaluation_outputs=80,ODE_steps=10,ODE_grid='source_shift5',
    root_namespace='native_qwen_sourceflow_ODE',model_batch_per_seed=1,
    training_scope='same original native attention LoRA480 tensors; everything else frozen',
    no_GRPO_no_formal_arm_no_generalization_claim=True)


def sealed(path):
    value=json.loads(Path(path).read_text());body=dict(value)
    if body.pop('sha256')!=fingerprint(body):raise ValueError('Changed sealed probe asset: '+str(path))
    return value


def source_hashes():
    root=Path(__file__).resolve().parents[4]
    paths=['starVLA/rl/qwen_flow_exploration/feedback/expression.py',
        'scripts/analysis/qwen_flow_exploration/feedback_expression_probe.py',
        'scripts/analysis/qwen_flow_exploration/feedback_expression_validate.py',
        'starVLA/rl/qwen_flow_exploration/native.py','starVLA/rl/qwen_flow_exploration/checkpoint.py',
        'starVLA/rl/qwen_flow_exploration/feedback/student.py',
        'starVLA/model/modules/qwen_sourceflow/source_flow_policy.py',
        'starVLA/model/modules/qwen_sourceflow/action_codec.py',
        'starVLA/model/modules/qwen_sourceflow/cached_action_runner.py','starVLA/rl/qwen_sourceflow/actor.py']
    return {name:file_sha(root/name) for name in paths}


def safe(row):return all(row['native_subscores'][name]==1 for name in SAFETY)


def prepare_registration(protocol,admission,teacher_bank,reference_strata):
    from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
    admission=Path(admission);summary=sealed(admission/'summary.json')
    table=admission/'private_scene_admission.csv'
    if summary['protocol_sha256']!=protocol['sha256'] or file_sha(table)!=summary['private_scene_csv_sha256']:
        raise ValueError('Changed TRAIN admission population')
    strata=sealed(reference_strata)
    if strata['protocol_sha256']!=protocol['sha256']:raise ValueError('Other fixed reference')
    rows=[row for row in csv.DictReader(table.open()) if row['method']=='F3_FEEDBACK' and row['teacher_admitted']=='True']
    rows.sort(key=lambda row:(row['log_id'],row['scene_token']))
    zero=[row for row in rows if row['stratum']=='stable_zero' and row['teacher_is_safety_recovery']=='True']
    if len(zero)!=1:raise ValueError('Expected unique previously discovered stable-zero repair')
    chosen=[('stable_zero_repair',zero[0])];used_logs={zero[0]['log_id']}
    for row in rows:
        if row['teacher_is_safety_recovery']=='True' and row['stratum']!='stable_zero' and row['log_id'] not in used_logs:
            chosen.append(('other_safety_repair',row));used_logs.add(row['log_id'])
            if len(chosen)==3:break
    for row in rows:
        if row['base0_safe']=='True' and row['teacher_EP_only']=='True' and row['log_id'] not in used_logs:
            chosen.append(('safe_EP_improvement',row));break
    if len(chosen)!=4:raise ValueError('Four predefined different-log TRAIN categories unavailable')
    selected=[]
    for category,meta in chosen:
        token=meta['scene_token'];path=Path(teacher_bank)/'scenes'/f'{token}.json';record=sealed(path)
        if (token not in protocol['splits']['TRAIN'] or record['split']!='TRAIN' or
            record['scene_token']!=token or record['result']['method']!='F3_FEEDBACK' or
            record['common_protocol_sha256']!=protocol['sha256'] or record['base_model_sha256']!=protocol['base_model_sha256']):
            raise ValueError('Probe teacher outside same base/TRAIN bank')
        base=record['result']['candidates'][0];teachers=record['teacher_candidates']
        if category=='safe_EP_improvement':
            teachers=[row for row in teachers if safe(base) and row['native_subscores']['ego_progress']>base['native_subscores']['ego_progress']
                and all(row['native_subscores'][key]==base['native_subscores'][key]
                        for key in row['native_subscores'] if key not in ('score','pdm_score','ego_progress'))]
        if not teachers:raise ValueError('Selected category lacks a qualified fixed target')
        teacher=min(teachers,key=lambda row:(-row['raw_reward'],not row['external_proposal'],row['candidate_id'],fingerprint(row['planned_trajectory'])))
        if not safe(teacher) or teacher['raw_reward']<max(.9,base['raw_reward']+.005) or teacher.get('on_policy_logprob') is not None:
            raise ValueError('Probe label is not an existing qualified real teacher')
        if category!='safe_EP_improvement' and safe(base):raise ValueError('Repair category has no candidate0 safety violation')
        target=teacher_normalized_target(teacher,ActionCodec)
        selected.append(dict(scene_token=token,log_id=record['log_id'],split='TRAIN',category=category,
            fixed_reference_stratum=strata['by_split']['TRAIN'][token]['stratum'],
            independent_five_reference_mean=strata['by_split']['TRAIN'][token]['reference_mean'],
            candidate0_reward=base['raw_reward'],candidate0_subscores=base['native_subscores'],
            teacher_candidate_id=teacher['candidate_id'],teacher_source=teacher['proposal_source'],
            teacher_external=teacher['external_proposal'],teacher_reward=teacher['raw_reward'],
            teacher_subscores=teacher['native_subscores'],planned_trajectory=teacher['planned_trajectory'],
            normalized_action=target.tolist(),source_scene_file=str(path),source_scene_file_sha256=file_sha(path),
            teacher_record_sha256=fingerprint(teacher)))
    registration=dict(schema='feedback_teacher_expression_registration_v1',qualification=True,
        protocol_sha256=protocol['sha256'],base_model_sha256=protocol['base_model_sha256'],contract=CONTRACT,
        selection_rule='Unique admitted stable-zero repair; first two other candidate0 safety repairs sorted(log_id,token), distinct logs; then first safe EP-only improvement in another log. Highest qualified teacher score; ties external first, candidate_id, physical hash. No new policy evaluation used.',
        teacher_reference='candidate0 for repair/admission; independent five-ref only names fixed strata',
        scenes=selected,source_files=source_hashes(),reference_strata_file_sha256=file_sha(reference_strata),
        admission_analysis_sha256=summary['sha256'])
    registration['sha256']=fingerprint(registration)
    return registration


def read_registration(path,protocol,check_sources=True):
    value=sealed(path)
    if (value['protocol_sha256']!=protocol['sha256'] or value['base_model_sha256']!=protocol['base_model_sha256'] or
        value['contract']!=CONTRACT or len(value['scenes'])!=4):raise ValueError('Changed mechanism probe contract')
    tokens=[row['scene_token'] for row in value['scenes']]
    if len(set(tokens))!=4 or len({row['log_id'] for row in value['scenes']})!=4 or any(token not in protocol['splits']['TRAIN'] for token in tokens):
        raise ValueError('Probe is not four distinct-log TRAIN scenes')
    if check_sources and value['source_files']!=source_hashes():raise ValueError('Probe source changed after registration')
    return value


def require_probe_lock(registration,path,source_sha):
    lock=sealed(path);qualification=Path(path).parent/'qualification_cpu.json'
    if (lock.get('phase')!='MECHANISM_PROBE_LOCKED' or lock['registration_sha256']!=registration['sha256'] or
        lock['protocol_sha256']!=registration['protocol_sha256'] or lock['contract']!=CONTRACT or
        lock['execution_source_sha']!=source_sha or lock['qualification_file_sha256']!=file_sha(qualification)):
        raise ValueError('Root has not sealed this bounded mechanism probe/source/qualification')
    q=sealed(qualification)
    if q['status']!='PASS' or q['registration_sha256']!=registration['sha256']:raise ValueError('Wrong CPU qualification')
    return lock


def optimizer_step(env,registration,optimizer,scheduler,exposure):
    started=time.monotonic();optimizer.zero_grad();losses=[];lr_used=optimizer.param_groups[0]['lr']
    for slot,row in enumerate(registration['scenes']):
        condition=env.condition(row['scene_token'],'TRAIN')
        noise,times=independent_fm_draw(42,exposure+slot,'teacher_expression_probe')
        target=torch.tensor(row['normalized_action'],dtype=torch.float32)
        loss,_=env.source_fm_loss(condition,target,noise=noise,times=times)
        (loss/4).backward();losses.append(float(loss.detach()));env.ledger.add(model_backward_calls=1)
    params=[p for p in env.actor.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise ValueError('Missing/nonfinite expression gradient')
    grad_norm=float(torch.sqrt(sum(p.grad.detach().double().square().sum() for p in params)))
    optimizer.step();scheduler.step();elapsed=time.monotonic()-started
    env.ledger.add(optimizer_updates=1,update_seconds=elapsed)
    return dict(FM_loss=sum(losses)/4,per_scene_FM_loss=losses,gradient_norm=grad_norm,
                LR_used=lr_used,LR_next=optimizer.param_groups[0]['lr'],update_seconds=elapsed)


@torch.no_grad()
def evaluate(env,registration,output,update,stop_requested=lambda:False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity=dict(schema='TRAIN_teacher_expression_single_ODE_evaluation_v1',qualification=True,
        registration_sha256=registration['sha256'],protocol_sha256=registration['protocol_sha256'],
        base_model_sha256=registration['base_model_sha256'],updates=update,adapter_sha256=adapter_digest(env.actor),
        seeds=CONTRACT['evaluation_seeds'],tokens_sha256=fingerprint([row['scene_token'] for row in registration['scenes']]),
        root_namespace=CONTRACT['root_namespace'],model_batch=1,ODE_grid='source_shift5',ODE_steps=10,
        no_teacher_output_selection=True,TRAIN_fit_only_not_generalization=True)
    identity['sha256']=fingerprint(identity)
    if (output/'identity.json').exists() and sealed(output/'identity.json')!=identity:raise ValueError('Changed expression evaluation identity')
    atomic_json(output/'identity.json',identity);records=[];new=0
    history=[json.loads(path.read_text()) for path in (output/'invocations').glob('*.json')]
    past_inference=sum(row['physical_inference_attempts'] for row in history)
    past_scoring=sum(row['physical_scoring_attempts'] for row in history)
    invocation=dict(status='RUNNING',identity_sha256=identity['sha256'],start_unix=time.time(),
        physical_inference_attempts=0,physical_scoring_attempts=0,new_outputs=0,reused_outputs=0)
    invfile=output/'invocations'/f'{uuid.uuid4().hex}.json';invfile.parent.mkdir(exist_ok=True)
    atomic_json(invfile,invocation);started=time.monotonic()
    try:
        for row in registration['scenes']:
            token=row['scene_token'];condition=None
            for seed in CONTRACT['evaluation_seeds']:
                path=output/'scenes'/f'{token}_seed{seed}.json'
                if path.exists():
                    record=sealed(path)
                    if record['evaluation_identity']!=identity['sha256'] or record['status']!='OK':raise ValueError('Wrong cached mechanism output')
                    records.append(record);invocation['reused_outputs']+=1;continue
                if stop_requested():raise EvaluationStopped('Expression evaluation stopped at committed single-output boundary')
                if past_inference+invocation['physical_inference_attempts']>=20 or past_scoring+invocation['physical_scoring_attempts']>=20:
                    raise RuntimeError('Fixed physical20-output milestone budget exhausted; preserve failure for root review, no automatic retry')
                if condition is None:condition=env.condition(token,'TRAIN')
                root=torch.randn((1,8,3),generator=torch.Generator(device='cpu').manual_seed(seed_for(CONTRACT['root_namespace'],seed,token)),dtype=torch.float32)
                invocation['physical_inference_attempts']+=1;atomic_json(invfile,invocation)
                action=env.sample_ode(condition,root)
                invocation['physical_scoring_attempts']+=1;atomic_json(invfile,invocation)
                record=env.score_actions(token,action,'TRAIN',proposal_source='teacher_expression_single_ODE',seed=seed,
                    latent_or_proposal_parameters=[dict(root=root[0].tolist(),single_candidate=True,selection='none')])[0]
                record.update(evaluation_identity=identity['sha256'],qualification=True,TRAIN_fit_only_not_generalization=True)
                record['sha256']=fingerprint(record);path.parent.mkdir(exist_ok=True);atomic_json(path,record)
                records.append(record);new+=1;invocation['new_outputs']+=1
        invocation['status']='COMPLETE'
    except EvaluationStopped:
        invocation['status']='STOPPED_AT_COMMITTED_OUTPUT';raise
    except BaseException:
        invocation['status']='FAILED';raise
    finally:
        invocation.update(end_unix=time.time(),wall_seconds=time.monotonic()-started);atomic_json(invfile,invocation)
    if len(records)!=20:raise ValueError('Expression evaluation must preserve all4x5 outputs')
    result=dict(status='COMPLETE',identity_sha256=identity['sha256'],updates=update,logical_single_outputs=20,
        PDMS=100*sum(row['raw_reward'] for row in records)/20,new_outputs_this_invocation=new,
        cumulative_physical_inferences=past_inference+invocation['physical_inference_attempts'],
        cumulative_physical_scoring_attempts=past_scoring+invocation['physical_scoring_attempts'],
        cumulative_worker_wall_seconds=sum(row.get('wall_seconds',0.) for row in history)+invocation['wall_seconds'],
        all_outputs_retained=True,qualification=True,TRAIN_fit_only_not_generalization=True)
    result['sha256']=fingerprint(result);atomic_json(output/'COMPLETE.json',result)
    return result


def validate_evaluation_population(output,registration):
    """A completed optimizer cursor cannot hide missing old milestone records."""
    output=Path(output);total=0
    for update in CONTRACT['evaluation_updates']:
        folder=output/f'update{update:06d}';identity=sealed(folder/'identity.json');done=sealed(folder/'COMPLETE.json')
        expected={f"{row['scene_token']}_seed{seed}.json" for row in registration['scenes'] for seed in CONTRACT['evaluation_seeds']}
        if ({path.name for path in (folder/'scenes').glob('*.json')}!=expected or
            identity['registration_sha256']!=registration['sha256'] or done['identity_sha256']!=identity['sha256'] or
            done['logical_single_outputs']!=20 or done['cumulative_physical_inferences']!=20 or
            done['cumulative_physical_scoring_attempts']!=20):raise ValueError('Missing/foreign/over-budget mechanism evaluation')
        for name in expected:
            row=sealed(folder/'scenes'/name)
            if row['status']!='OK' or row['evaluation_identity']!=identity['sha256']:raise ValueError('Invalid mechanism evaluation record')
            total+=1
    if total!=80:raise ValueError('Expression probe must retain all80 actual single outputs')
    return total

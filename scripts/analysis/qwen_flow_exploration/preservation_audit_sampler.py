"""Terminal frozen SDE-vs-ODE matched-root mean audit; never a training entry."""
import argparse
import fcntl
import gc
import json
from pathlib import Path
import time
import traceback
import uuid
import numpy as np
import torch
from starVLA.rl.qwen_flow_exploration.native import NativeEnvironment
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol,require_scene
from starVLA.rl.qwen_flow_exploration.common_analysis import bootstrap,checked_seal
from starVLA.rl.qwen_flow_exploration.preservation.deployment import export_adapters,read_adapter_package
from starVLA.rl.qwen_flow_exploration.preservation.sampler_audit import (validate_budget,validate_group,validate_records,validate_ODE_roots,
    reference_rng_book,freeze_and_hash,model_hash,publish_group,load_group,mean_summary,NAMESPACE,METRICS,SAFETY)
from starVLA.rl.simwam_recipe.lora_adapter import adapter_state,restore_adapters
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def paired_mean(current,reference,logs):
    """Common log bootstrap over fixed-scene candidate means, never best-of."""
    result={}
    for key in METRICS:
        c=np.asarray([[row['native_subscores'][key] for row in group] for group in current],dtype=np.float64)
        r=np.asarray([[row['native_subscores'][key] for row in group] for group in reference],dtype=np.float64)
        result[key]=bootstrap(c.mean(1)-r.mean(1),logs,repeats=2000,seed=42,kind='score' if key=='score' else 'rate')
    c=np.asarray([[all(row['native_subscores'][key]==1 for key in SAFETY) for row in group] for group in current],dtype=np.float64)
    r=np.asarray([[all(row['native_subscores'][key]==1 for key in SAFETY) for row in group] for group in reference],dtype=np.float64)
    result['joint_safe_fraction']=bootstrap(c.mean(1)-r.mean(1),logs,repeats=2000,seed=42,kind='rate')
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--protocol',required=True);p.add_argument('--budget-lock',required=True)
    p.add_argument('--calibration',required=True);p.add_argument('--reference-pool',required=True)
    p.add_argument('--output',required=True);p.add_argument('--arm',choices=['BASE','P0','P1','P2','P3','P4','P5'],required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--checkpoint');p.add_argument('--adapter-package')
    p.add_argument('--base-audit',help='Completed BASE audit required for learned arms; exact root/budget comparison')
    p.add_argument('--scorer-workers',type=int,default=1)
    a=p.parse_args();started=time.monotonic();protocol=read_protocol(a.protocol)
    calibration=checked_seal(json.loads(Path(a.calibration).read_text()),'completed TRAIN calibration')
    budget=validate_budget(json.loads(Path(a.budget_lock).read_text()),protocol,calibration,arm=a.arm,training_seed=a.seed)
    if (calibration['status']!='COMPLETE' or calibration['protocol_sha256']!=protocol['sha256'] or
            calibration['base_model_sha256']!=protocol['base_model_sha256']):raise ValueError('Wrong initial reference calibration')
    if file_sha(a.reference_pool)!=calibration['reference_SDE_pool_sha256']:raise ValueError('Initial SDE pool changed')
    pool=torch.load(a.reference_pool,map_location='cpu',weights_only=False);tokens=protocol['splits']['TRAIN'][:128]
    if (pool['schema']!='initial_base_source_SDE_reference_pool_v1' or pool['status']!='COMPLETE' or
            pool['protocol_sha256']!=protocol['sha256'] or pool['base_model_sha256']!=protocol['base_model_sha256'] or
            pool['tokens']!=tokens or len(pool['rollouts'])!=128 or len(pool['record_groups'])!=128):
        raise ValueError('Wrong original same-kernel128/G16 population')
    for token,rollout,rows in zip(tokens,pool['rollouts'],pool['record_groups']):
        require_scene(protocol,token,'TRAIN');validate_group(rollout,rows,token,protocol)
    book=reference_rng_book(pool);base_result=None
    if a.arm=='BASE':
        if a.checkpoint or a.adapter_package or a.base_audit:raise ValueError('BASE uses the common original initial policy only')
    else:
        if bool(a.checkpoint)==bool(a.adapter_package) or not a.base_audit:
            raise ValueError('A complete terminal policy and completed matched BASE audit are required')
        base_result=checked_seal(json.loads((Path(a.base_audit)/'result.json').read_text()),'BASE sampler audit')
        seal=json.loads((Path(a.base_audit)/'COMPLETE').read_text())
        if (base_result['arm']!='BASE' or base_result['status']!='COMPLETE' or base_result['budget_sha256']!=budget['sha256'] or
                base_result['reference_pool_sha256']!=calibration['reference_SDE_pool_sha256'] or
                seal['result_file_sha256']!=file_sha(Path(a.base_audit)/'result.json')):
            raise ValueError('Wrong/incomplete matched-root BASE audit')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    metadata=None;adapters=None
    if a.arm!='BASE':
        package=Path(a.adapter_package) if a.adapter_package else export_adapters(a.checkpoint,out/'deployment',protocol,arm=a.arm,seed=a.seed)
        adapters,metadata=read_adapter_package(package,protocol,arm=a.arm,seed=a.seed)
    identity=dict(schema='terminal_read_only_matched_root_SDE_ODE_audit_v1',arm=a.arm,training_seed=a.seed,
        protocol_sha256=protocol['sha256'],base_model_sha256=protocol['base_model_sha256'],budget_sha256=budget['sha256'],
        reference_pool_sha256=calibration['reference_SDE_pool_sha256'],calibration_sha256=calibration['sha256'],
        adapter_sha256=metadata['adapter_sha256'] if metadata else pool['initial_adapter_hash'],
        trained_updates=metadata['checkpoint_updates'] if metadata else 0,
        TRAIN_tokens_sha256=fingerprint(tokens),scenes=128,G=16,K=10,chunk=4,stochastic_steps=[7,8,9],
        reference_rng_seed=42,reference_rng_namespace=NAMESPACE,RNG_constant_key='unfiltered128',
        reference_rng_book_sha256=tensor_digest(book),initial_roots='each actual SDE x0 and ODE input exactly equal saved reference x0',
        readout='all-candidate means; unmerged frozen NativeRecipeActor diagnostic; separate from primary pure-merged deployment',
        base_audit_identity=base_result['identity_sha256'] if base_result else None,
        no_optimizer_dual_loss_or_selection=True,
        source_files={str(Path(__file__).resolve()):file_sha(__file__),
            'preservation/sampler_audit.py':file_sha(Path(__file__).resolve().parents[3]/'starVLA/rl/qwen_flow_exploration/preservation/sampler_audit.py')})
    identity['sha256']=fingerprint(identity)
    if (out/'identity.json').exists():
        if json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Terminal audit identity changed')
    else:atomic_json(out/'identity.json',identity)
    if (out/'COMPLETE').exists():
        result=checked_seal(json.loads((out/'result.json').read_text()),'terminal sampler audit result')
        if (result['identity_sha256']!=identity['sha256'] or
                json.loads((out/'COMPLETE').read_text())['result_file_sha256']!=file_sha(out/'result.json')):
            raise ValueError('Changed completed audit publication')
        previous=[load_group(out/'scenes',identity,i,token,book[i],protocol) for i,token in enumerate(tokens)]
        if (mean_summary([value['records'] for value in previous])!=result['SDE_current'] or
                mean_summary([value['ODE_records'] for value in previous])!=result['ODE_current']):
            raise ValueError('Completed audit means do not match all saved candidates')
        print(json.dumps(dict(status='REUSED_COMPLETE',arm=a.arm,result=str(out/'result.json'))));return
    invocation=uuid.uuid4().hex;attempt=out/'invocations'/invocation;attempt.mkdir(parents=True)
    record=dict(status='LOADING',invocation=invocation,start_unix=time.time(),identity_sha256=identity['sha256'],
        new_SDE_candidates=0,new_ODE_candidates=0,reused_current_candidates=0,physical_SDE_generation_attempts=0,
        physical_ODE_generation_attempts=0,reference_historical_logical_candidates=2048)
    atomic_json(attempt/'invocation.json',record);env=None
    try:
        env=NativeEnvironment(protocol,trainable='lora',scorer_workers=a.scorer_workers,
                              ledger_path=attempt/'ledger.json',experiment_id=a.arm+'_terminal_read_only_sampler_audit')
        if tensor_digest(adapter_state(env.actor))!=pool['initial_adapter_hash']:raise ValueError('Changed initial native actor/kernel initialization')
        if adapters is not None:restore_adapters(env.actor,adapters)
        if tensor_digest(adapter_state(env.actor))!=identity['adapter_sha256']:raise ValueError('Actual terminal LoRA restoration failed')
        del adapters;gc.collect();frozen_hash=freeze_and_hash(env)
        env.set_behavior_identity(dict(arm=a.arm,training_seed=a.seed,adapter_sha256=identity['adapter_sha256'],
                                      read_only_sampler_audit=identity['sha256']))
        reference_calls=env.actor._reference_calls;record.update(status='RUNNING',model_loading_seconds=time.monotonic()-started)
        atomic_json(attempt/'invocation.json',record);groups=[];ODE_groups=[]
        for i,token in enumerate(tokens):
            if (out/'scenes'/token).exists():
                saved=load_group(out/'scenes',identity,i,token,book[i],protocol)
                groups.append(saved['records']);ODE_groups.append(saved['ODE_records']);record['reused_current_candidates']+=4096//128
            else:
                condition=env.condition(token);rng=torch.Generator(device='cpu');rng.set_state(book[i]['before'])
                if a.arm=='BASE':rollout,rows=pool['rollouts'][i],pool['record_groups'][i]
                else:
                    record['physical_SDE_generation_attempts']+=16;atomic_json(attempt/'invocation.json',record)
                    rollout,rows=env.sample_source_sde(condition,16,rng,token=token,proposal_source=a.arm+'_terminal_audit_source_SDE')
                    if not torch.equal(rng.get_state(),book[i]['after']):raise ValueError('Actual source SDE changed original random-number consumption')
                    record['new_SDE_candidates']+=16
                if not torch.equal(rollout['chain'][:,0],pool['rollouts'][i]['chain'][:,0]):raise ValueError('Updated SDE initial roots differ from exact calibration roots')
                validate_group(rollout,rows,token,protocol)
                roots=pool['rollouts'][i]['chain'][:,0].clone()
                record['physical_ODE_generation_attempts']+=16;atomic_json(attempt/'invocation.json',record)
                with torch.no_grad():actions=env.sample_ode(condition,roots)
                ODE_rows=env.score_actions(token,actions,'TRAIN',proposal_source=a.arm+'_terminal_audit_matched_root_ODE',
                    latent_or_proposal_parameters=[dict(root=roots[j].tolist(),candidate_index=j,single_output=False,oracle_selection=False,
                                                       audit_identity=identity['sha256']) for j in range(16)])
                record['new_ODE_candidates']+=16;validate_records(ODE_rows,token,protocol);validate_ODE_roots(ODE_rows,roots)
                publish_group(out/'scenes',identity,i,token,rollout,rows,ODE_rows,book[i]);groups.append(rows);ODE_groups.append(ODE_rows)
                del condition
            atomic_json(out/'progress.json',dict(completed_scenes=i+1,total_scenes=128,new_this_invocation=record['new_SDE_candidates']+record['new_ODE_candidates'],
                                                reused_current_candidates=record['reused_current_candidates']))
            atomic_json(attempt/'invocation.json',record)
        if model_hash(env)!=frozen_hash or any(p.grad is not None or p.requires_grad for p in env.actor.parameters()):
            raise ValueError('Read-only sampler audit changed/freed frozen parameters/buffers or produced gradients')
        if env.actor._reference_calls!=reference_calls:raise ValueError('Sampler audit consulted reference-policy forward')
        logs=[group[0]['log_id'] for group in groups]
        result=dict(status='COMPLETE',schema=identity['schema'],identity_sha256=identity['sha256'],arm=a.arm,training_seed=a.seed,
            protocol_sha256=protocol['sha256'],budget_sha256=budget['sha256'],reference_pool_sha256=calibration['reference_SDE_pool_sha256'],
            split='TRAIN',scenes=128,logs=len(set(logs)),G=16,
            SDE_initial=mean_summary(pool['record_groups']),SDE_current=mean_summary(groups),ODE_current=mean_summary(ODE_groups),
            SDE_current_minus_initial=paired_mean(groups,pool['record_groups'],logs),
            SDE_current_minus_ODE_current=paired_mean(groups,ODE_groups,logs),
            ODE_current_minus_initial_ODE=None,
            logical_current_readouts=4096,unique_initial_SDE_reference_cost=2048,
            historical_reference_cost_semantics='One shared calibration origin of2048 real scores; reused by every audit, never multiplied into new physical scorer calls',
            frozen_parameter_buffer_hash_before=frozen_hash,frozen_parameter_buffer_hash_after=frozen_hash,
            no_optimizer_step_or_lambda_update=True,unmerged_native_actor=True,
            deployment_boundary='TRAIN128/G16 unmerged sampler diagnostic, not primary DEV/Navtest pure-merged5seed single-output policy score',
            calibration_P1_matching=calibration['P1_matching']['status'],replay_coefficient_bound_hit=calibration['replay_gradient_calibration']['max_bound_hit'])
        if base_result is not None:
            base_identity=json.loads((Path(a.base_audit)/'identity.json').read_text())
            baseline=[load_group(Path(a.base_audit)/'scenes',base_identity,i,token,book[i],protocol)['ODE_records'] for i,token in enumerate(tokens)]
            result['ODE_initial']=mean_summary(baseline);result['ODE_current_minus_initial_ODE']=paired_mean(ODE_groups,baseline,logs)
        else:result['ODE_initial']=result['ODE_current']
        record.update(status='COMPLETE',end_unix=time.time(),wall_seconds=time.monotonic()-started)
        atomic_json(attempt/'invocation.json',record);env.ledger.save()
        invocations=[json.loads(path.read_text()) for path in (out/'invocations').glob('*/invocation.json')]
        result['physical_new_candidates_including_retries']=sum(v['new_SDE_candidates']+v['new_ODE_candidates'] for v in invocations)
        result['physical_generation_attempts_including_failures']=sum(v['physical_SDE_generation_attempts']+v['physical_ODE_generation_attempts'] for v in invocations)
        result['cumulative_worker_wall_seconds']=sum(v.get('wall_seconds',0) for v in invocations)
        result['current_invocation_ledger']=env.ledger.snapshot();result['GPU_semantics']='Shared worker occupancy, not exclusive compute seconds'
        result['sha256']=fingerprint(result);atomic_json(out/'result.json',result)
        atomic_json(out/'COMPLETE',dict(identity_sha256=identity['sha256'],result_file_sha256=file_sha(out/'result.json')))
        print(json.dumps(dict(status='COMPLETE',arm=a.arm,SDE=result['SDE_current']['mean_PDMS'],ODE=result['ODE_current']['mean_PDMS'])))
    except BaseException:
        record.update(status='FAILED',end_unix=time.time(),wall_seconds=time.monotonic()-started,error=traceback.format_exc())
        atomic_json(attempt/'invocation.json',record);raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()

"""Partition scenes across workers; preserve batch-one native ODE inference."""
import argparse
import fcntl
import json
from pathlib import Path
import time
import traceback
import uuid
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol,require_scene,ROOT
from starVLA.rl.qwen_flow_exploration import evaluation as shared
from starVLA.rl.qwen_flow_exploration.common_analysis import checked_seal,validate_root_record
from scripts.analysis.qwen_flow_exploration.preservation_evaluate import verified_row,METRICS
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    p.add_argument('--family',choices=['base','latent','adapters'],required=True);p.add_argument('--arm',required=True)
    p.add_argument('--package');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--root-scale',type=float,default=1.)
    p.add_argument('--split',choices=['TRAIN','DEV_SELECT','DEV_AUDIT','NAVTEST'],default='DEV_SELECT')
    p.add_argument('--full-split',action='store_true');p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
    p.add_argument('--qualification',action='store_true');p.add_argument('--limit',type=int)
    p.add_argument('--execution-proof');p.add_argument('--scorer-workers',type=int,default=1)
    a=p.parse_args();started=time.monotonic();start_unix=time.time();protocol=read_protocol(a.protocol)
    if a.root_scale not in (.7,1.,1.3) or (a.arm!='L0-S' and a.root_scale!=1.) or (a.family!='base' and a.arm=='L0-S'):
        raise ValueError('Only the preregistered fixed-root control may change the root scale')
    if a.limit is not None and (not a.qualification or a.limit<1):raise ValueError('A formal population cannot be shortened')
    if a.split=='NAVTEST' and a.qualification:raise ValueError('Qualify on TRAIN/DEV_SELECT; final NAVTEST is locked first')
    if a.split in ('DEV_AUDIT','NAVTEST') and not (ROOT/'FINAL_CONFIGURATION_LOCK.json').exists():
        raise PermissionError('Final weights and selection must be locked before audit/test')
    full=(protocol['splits'][a.split] if a.full_split or a.split not in protocol['diagnostic'] else protocol['diagnostic'][a.split])
    if a.limit:full=full[:a.limit]
    tokens=shared.partition(full,a.rank,a.world)
    for token in tokens:require_scene(protocol,token,a.split)
    proof=shared.qualified_execution(a.execution_proof,a.qualification)
    value,metadata=shared.load_package(a.family,a.package,protocol,a.arm,a.seed,a.qualification)
    identity=shared.identity(protocol,a.family,a.arm,a.seed,metadata,full,a.qualification,proof,a.root_scale)
    identity['split']=a.split;identity['sha256']=fingerprint(identity)
    if a.split in ('DEV_AUDIT','NAVTEST'):
        locked=checked_seal(json.loads((ROOT/'FINAL_CONFIGURATION_LOCK.json').read_text()),'final model selection')
        selected=dict(family=a.family,arm=a.arm,training_seed=a.seed,root_scale=a.root_scale,
                      package_identity_sha256=identity['package_identity_sha256'],deployment_contract=identity['deployment_contract'])
        if locked['protocol_sha256']!=protocol['sha256'] or fingerprint(selected) not in locked['approved_evaluation_configurations']:
            raise PermissionError('Policy is not in the fixed final evaluation plan')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    owner=(out/'owner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    execution=dict(rank=a.rank,world=a.world,full_tokens_sha256=fingerprint(full),local_tokens_sha256=fingerprint(tokens),
                   entry_sha256=file_sha(__file__),identity_sha256=identity['sha256'])
    for name,document in [('identity.json',identity),('execution.json',execution)]:
        if (out/name).exists():
            if json.loads((out/name).read_text())!=document:raise ValueError('Changed scientific or worker partition identity')
        else:atomic_json(out/name,document)
    expected={f'{t}_seed{s}.json' for t in tokens for s in protocol['evaluation_seeds']}
    def checked_rows():
        if {p.name for p in (out/'scenes').glob('*.json')}!=expected:raise ValueError('Wrong exact worker membership')
        rows=[]
        for token in tokens:
            for seed in protocol['evaluation_seeds']:
                row=verified_row(out/'scenes'/f'{token}_seed{seed}.json',identity,token,seed)
                validate_root_record(row,identity);rows.append(row)
        return rows
    if (out/'SHARD_COMPLETE.json').exists():
        checked_rows();seal=checked_seal(json.loads((out/'SHARD_COMPLETE.json').read_text()),'completed shard')
        if seal['execution']!=execution:raise ValueError('Complete shard belongs to a different partition')
        print({'status':'REUSED_COMPLETE_SHARD','rank':a.rank});return
    invocation=uuid.uuid4().hex;attempt=out/'invocations'/invocation;attempt.mkdir(parents=True)
    record=dict(status='LOADING',invocation=invocation,start_unix=start_unix,execution=execution,
        new=0,reused=0,physical_inference_attempts=0,completed_inferences=0,
        package_validation_seconds=time.monotonic()-started)
    atomic_json(attempt/'invocation.json',record);env=None
    try:
        env,policy=shared.make_runtime(protocol,a.family,value,'cuda',attempt/'ledger.json',a.arm+'_partitioned_single_ODE',a.split,a.scorer_workers)
        del value
        env.set_behavior_identity(dict(evaluation_identity=identity['sha256'],arm=a.arm,
            root_policy_sha256=identity['root_policy_sha256'],adapter_sha256=identity['adapter_sha256']))
        record.update(status='RUNNING',model_loading_seconds=time.monotonic()-started)
        for token in tokens:
            needed=[(seed,out/'scenes'/f'{token}_seed{seed}.json') for seed in protocol['evaluation_seeds']]
            condition=env.condition(token,a.split) if any(not path.exists() for _,path in needed) else None
            generated=shared.roots_for(policy,condition,token,protocol['evaluation_seeds'],identity) if condition is not None else []
            if condition is not None and policy is not None:env.ledger.add(latent_network_scene_forwards=1)
            for (seed,path),root_row in zip(needed,generated or [(s,None,None) for s,_ in needed]):
                if path.exists():
                    row=verified_row(path,identity,token,seed);validate_root_record(row,identity);record['reused']+=1;continue
                if root_row[0]!=seed:raise AssertionError('Seed order changed')
                _,root,parameters=root_row
                record['physical_inference_attempts']+=1;atomic_json(attempt/'invocation.json',record)
                action=env.sample_ode(condition,root);record['completed_inferences']+=1
                atomic_json(attempt/'invocation.json',record)
                row=env.score_actions(token,action,a.split,proposal_source=a.arm+'_partitioned_single_ODE',seed=seed,
                    latent_or_proposal_parameters=[parameters])[0]
                row['evaluation_identity']=identity['sha256'];row['sha256']=fingerprint(row)
                validate_root_record(row,identity)
                path.parent.mkdir(exist_ok=True);atomic_json(path,row);record['new']+=1
            del condition
            atomic_json(out/'progress.json',dict(new=record['new'],reused=record['reused'],expected=len(expected)))
            atomic_json(attempt/'invocation.json',record)
        rows=checked_rows();record.update(status='COMPLETE',end_unix=time.time(),wall_seconds=time.monotonic()-started)
        atomic_json(attempt/'invocation.json',record);env.ledger.save()
        result=dict(status='COMPLETE',execution=execution,scenes=len(tokens),scene_seed_records=len(rows),
            PDMS=100*sum(row['raw_reward'] for row in rows)/len(rows),single_output=True)
        result['sha256']=fingerprint(result);atomic_json(out/'SHARD_COMPLETE.json',result)
        print({'status':'COMPLETE_SHARD','rank':a.rank,'records':len(rows)})
    except BaseException:
        record.update(status='FAILED',end_unix=time.time(),wall_seconds=time.monotonic()-started,error=traceback.format_exc())
        atomic_json(attempt/'invocation.json',record);raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()

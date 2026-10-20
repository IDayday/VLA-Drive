"""Prepare two final small exports, then one shared-S0 online qualification.

No training or simulation is dispatched. The default plan phase reads metadata
only. Export is explicitly CPU-only; online execution requires a separately
admitted GPU and delegates to the existing fresh-worker qualification tool.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

from .common import artifact, digest, load_config, lock, read_json, sha256, verify_artifact, verify_identity, write_json


class WaitingInputs(Exception):
    pass


def endpoint(run, *, arm, seed, base_hash, context_pin, repair_pin):
    """Only the explicit completed 1000 endpoint, never latest/best/smoke."""
    run=Path(run).resolve()
    missing=[str(run/name) for name in ('IDENTITY.json','RESULT.json') if not (run/name).is_file()]
    if missing:raise WaitingInputs('Final endpoint metadata not yet present: '+', '.join(missing))
    identity=read_json(run/'IDENTITY.json');result=read_json(run/'RESULT.json')
    repair=identity.get('data_artifact') if arm=='A1' else identity.get('repair_data',{}).get('dataset')
    if (identity.get('schema')!='action_effect_small_training_v1' or identity.get('arm')!=arm
        or identity.get('seed')!=seed or identity.get('max_updates')!=1000
        or identity.get('base_binding_hash')!=base_hash
        or identity.get('context_artifact')!=context_pin or repair!=repair_pin):
        raise ValueError('Final endpoint arm/seed/Base/population/data/horizon mismatch: '+arm)
    runtime=identity.get('runtime',{})
    if (runtime.get('trainable_dtype')!='float32' or runtime.get('autocast')!='disabled'
        or runtime.get('deterministic') is not True or runtime.get('matmul_tf32') is not False
        or runtime.get('cudnn_tf32') is not False):
        raise ValueError('Final endpoint changed the registered FP32 execution: '+arm)
    teacher=identity.get('effect_teacher',{})
    if (not teacher or teacher.get('external_weights_loaded') is not False
        or teacher.get('base_binding_hash')!=base_hash or not teacher.get('checkpoint')):
        raise ValueError('Effect arm is missing its own trained teacher provenance: '+arm)
    if arm=='B1-effect' and identity.get('effect_protocol')!='verified_pair_v1':
        raise ValueError('Only the preregistered verified-pair B1-effect can deploy here')
    if (result.get('schema')!='action_effect_training_result_v1'
        or result.get('status')!='COMPLETE' or result.get('optimizer_step')!=1000
        or result.get('termination_reason')!='MAX_UPDATES'):
        raise WaitingInputs('Final step1000 MAX_UPDATES RESULT is not complete: '+arm)
    reported=result.get('checkpoint',{})
    # Fresh termination reports the full receipt; terminal resume reports its
    # already-resolved artifact. Both must join the same immutable sibling.
    checkpoint=verify_artifact(reported.get('checkpoint',reported)).resolve()
    if checkpoint.parent!=run:raise ValueError('Checkpoint escapes the explicit training run: '+arm)
    receipt=read_json(checkpoint.with_suffix('.COMPLETE.json'))
    reported_matches=(receipt==reported if 'checkpoint' in reported else receipt['checkpoint']==reported)
    if (not reported_matches or receipt.get('optimizer_step')!=1000
        or receipt.get('terminal') is not True or receipt.get('termination_reason')!='MAX_UPDATES'
        or result.get('run_identity')!=digest(identity) or receipt.get('run_identity')!=digest(identity)):
        raise ValueError('Final checkpoint/RESULT/receipt/training identity mismatch: '+arm)
    return {'arm':arm,'identity':artifact(run/'IDENTITY.json'),'result':artifact(run/'RESULT.json'),
        'completion':artifact(checkpoint.with_suffix('.COMPLETE.json')),'checkpoint':artifact(checkpoint),
        'run_identity':digest(identity),'optimizer_step':1000,'seed':seed,
        'effect_teacher':identity['effect_teacher']}


def plan(args):
    cfg=load_config(args.config);root=Path(cfg['artifact_root'])
    from tools.st_world_policy.base_binding import read_binding
    binding=read_binding(cfg['base_binding']) # No large Base files are opened in the planner.
    campaign=verify_identity(read_json(root/'CAMPAIGN_BINDING.json'))
    if (campaign['base_binding']!=artifact(cfg['base_binding'])
        or campaign['base_checkpoint_binding_hash']!=binding['binding_sha256']):
        raise ValueError('Campaign original S0 binding changed')
    if cfg['cohort_size']!=1024 or cfg['training']['max_updates']!=1000:
        raise ValueError('This final closure is fixed to the registered1024/1000 campaign')
    cohort=verify_identity(read_json(verify_artifact(campaign['cohort'])))
    selected=[r for r in cohort['entries'] if r['scene_id']==args.scene_id]
    if (len(selected)!=1 or cohort.get('split')!='train'
        or cohort.get('base_binding_hash')!=binding['binding_sha256']):
        raise ValueError('Use exactly one current scene from this original TRAIN cohort')
    context=artifact(root/'contexts/MANIFEST_001024.json')
    repair=artifact(root/'g0/population_001024/REPAIR_DATASET.json')
    models=root/'models/population_001024'
    paths={'A1':Path(args.a_run) if args.a_run else models/'route_a/A1'/f'seed_{args.seed}',
           'B1-effect':Path(args.b_run) if args.b_run else models/'route_b/B1-effect'/f'seed_{args.seed}'}
    endpoints={};waiting=[]
    for arm,run in paths.items():
        try:endpoints[arm]=endpoint(run,arm=arm,seed=args.seed,base_hash=binding['binding_sha256'],
                                   context_pin=context,repair_pin=repair)
        except WaitingInputs as exc:waiting.append(str(exc))
    if len(endpoints)==2 and endpoints['A1']['effect_teacher']['checkpoint']!=endpoints['B1-effect']['effect_teacher']['checkpoint']:
        raise ValueError('A1 and B1-effect must reference the same frozen own G1 teacher')
    if args.seed!=42:raise ValueError('This one-scene deployment qualification uses the primary training seed42')
    out=Path(args.output).resolve()
    exports={arm:out/'exports'/arm for arm in paths}
    commands={}
    for arm,route in (('A1','A'),('B1-effect','B')):
        checkpoint=endpoints[arm]['checkpoint']['path'] if arm in endpoints else '<WAITING_FINAL_CHECKPOINT>'
        commands[arm]=[sys.executable,'-m','tools.action_effect.export_models','--phase','export',
            '--checkpoint',checkpoint,'--route',route,'--base-binding',cfg['base_binding'],
            '--output',str(exports[arm]),'--device','cpu']
    online=[sys.executable,'-m','tools.action_effect.qualify_online_deployment',
        '--base-binding',cfg['base_binding'],'--cohort',campaign['cohort']['path'],'--scene-id',args.scene_id,
        '--a-bundle',str(exports['A1']/'BUNDLE.json'),'--b-bundle',str(exports['B1-effect']/'BUNDLE.json'),
        '--output',str(out/'online'),'--device',args.device,'--modes','S0','A','B',
        '--warmup','2','--repeats','8','--max-seconds',str(args.max_seconds)]
    for name in ('s0','a','b'):
        prediction=getattr(args,name+'_predictions',None)
        if prediction:online.extend(['--'+name+'-predictions',str(Path(prediction).resolve())])
    repo=Path(__file__).resolve().parents[2]
    closure_files=('tools/action_effect/prepare_final_online.py','tools/action_effect/export_models.py',
        'tools/action_effect/qualify_online_deployment.py','tools/action_effect/evaluate_effect_flow.py',
        'starVLA/dataloader/foresight_dataset.py','tools/foresight/checkpoints.py','tools/foresight/deployment_precision.py')
    return {'schema':'action_effect_final_online_plan_v1','status':'WAITING_INPUTS' if waiting else 'READY',
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'closure_source':{name:sha256(repo/name) for name in closure_files},
        'waiting':waiting,'config':artifact(args.config),'base_binding':artifact(cfg['base_binding']),
        'base_binding_hash':binding['binding_sha256'],'campaign':artifact(root/'CAMPAIGN_BINDING.json'),
        'cohort':campaign['cohort'],'context':context,'repair':repair,'scene_id':args.scene_id,
        'endpoints':endpoints,'output':str(out),'export_commands':commands,'online_command':online,
        'budget':{'training_updates':0,'simulator_calls':0,'CPU_only_exports':2,
            'shared_full_Base_loads_in_online_worker':1,'head_blob_loads_in_online_worker':0,
            'modes':3,'warmups_per_mode':2,'measured_per_mode':8,'poison_checks_per_mode':1,
            'total_online_predictions':33,'total_VLM_calls':33,'total_original_DiT_steps':330,
            'maximum_online_wall_seconds':args.max_seconds},
        'scope':'one real TRAIN scene engineering wiring and warm one-scene latency; no scientific population result'}


def invoke(command, *, cwd, output, cpu, timeout):
    """Separate processes free CPU export state before the single online load."""
    env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2',
         'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    if cpu:env['CUDA_VISIBLE_DEVICES']=''
    with Path(str(output)+'.stdout.log').open('w') as stdout,Path(str(output)+'.stderr.log').open('w') as stderr:
        done=subprocess.run(command,cwd=cwd,stdout=stdout,stderr=stderr,env=env,timeout=timeout)
    if done.returncode:raise RuntimeError('Closure child failed; keep logs and choose a new output namespace: '+str(output))


def validate_bundle(path, endpoint_pin, *, route, base_pin, base_hash):
    m=verify_identity(read_json(path))
    if (m.get('schema')!='action_effect_research_deployment_bundle_v1' or m.get('route')!=route
        or m.get('arm')!=endpoint_pin['arm'] or m.get('base_binding')!=base_pin
        or m.get('base_binding_hash')!=base_hash
        or m.get('training_provenance',{}).get('checkpoint')!=endpoint_pin['checkpoint']
        or m['training_provenance'].get('optimizer_step')!=1000
        or m['training_provenance'].get('run_identity')!=endpoint_pin['run_identity']
        or any(m.get(k) is not False for k in ('effect_teacher_required','simulator_required',
              'training_cache_required','GT_required','LiDAR_required','ACTIVE_updated'))):
        raise ValueError('Private export is not the exact final inference-only endpoint')
    verify_artifact(m['private_head']);verify_artifact(m['source_package'])
    # Source metadata validation only here; strict nn.Module reconstruction is
    # owned by export and by the actual fresh online worker, not a second load.
    repo=Path(__file__).resolve().parents[2]
    if not m.get('source_inventory'):raise ValueError('Empty source contract')
    for relative,want in m['source_inventory'].items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts or sha256(repo/relative)!=want:
            raise ValueError('Fresh export source changed before online qualification')
    return artifact(path)


def execute(args, prepared):
    if prepared['status']!='READY':return prepared
    out=Path(prepared['output']);repo=Path(__file__).resolve().parents[2]
    if args.phase in ('prepare','all'):
        if out.exists():raise FileExistsError('Preserve old closure evidence; choose a fresh --output')
        out.mkdir(parents=True)
        with lock(out/'PREPARE.lock'):
            write_json(out/'PLAN.json',prepared)
            for arm in ('A1','B1-effect'):
                invoke(prepared['export_commands'][arm],cwd=repo,output=out/arm,cpu=True,timeout=args.export_max_seconds)
            bundles={arm:validate_bundle(out/'exports'/arm/'BUNDLE.json',prepared['endpoints'][arm],
                route=route,base_pin=prepared['base_binding'],base_hash=prepared['base_binding_hash'])
                for arm,route in (('A1','A'),('B1-effect','B'))}
            receipt={'schema':'action_effect_final_online_preparation_v1','status':'PREPARED',
                'plan':artifact(out/'PLAN.json'),'endpoints':prepared['endpoints'],'bundles':bundles,
                'training_updates':0,'simulator_calls':0,'online_status':'NOT_RUN'}
            write_json(out/'PREPARATION.json',receipt)
    if args.phase=='prepare':return read_json(out/'PREPARATION.json')
    if args.phase in ('online','all'):
        receipt=read_json(out/'PREPARATION.json')
        if (receipt.get('schema')!='action_effect_final_online_preparation_v1' or receipt.get('status')!='PREPARED'
            or receipt.get('plan')!=artifact(out/'PLAN.json') or read_json(out/'PLAN.json')!=prepared
            or receipt.get('endpoints')!=prepared['endpoints']):
            raise ValueError('Prepared export belongs to another source/endpoint/online command')
        for arm,route in (('A1','A'),('B1-effect','B')):
            if receipt['bundles'][arm]!=validate_bundle(out/'exports'/arm/'BUNDLE.json',prepared['endpoints'][arm],
                    route=route,base_pin=prepared['base_binding'],base_hash=prepared['base_binding_hash']):
                raise ValueError('Prepared small bundle changed')
        if (out/'online').exists():raise FileExistsError('Do not rerun a completed or failed online worker in the same directory')
        with lock(out/'ONLINE_ADMISSION.lock'):
            invoke(prepared['online_command'],cwd=repo,output=out/'online_launcher',cpu=False,timeout=args.max_seconds+60)
        done=read_json(out/'online/COMPLETE.json')
        if done.get('status')!='PASS':raise ValueError('Actual online qualification did not pass')
        return {'status':'PASS','scope':prepared['scope'],'preparation':artifact(out/'PREPARATION.json'),
            'online_receipt':artifact(out/'online/COMPLETE.json'),'scientific_gain':'UNTESTED'}
    return prepared


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=('plan','prepare','online','all'),default='plan')
    p.add_argument('--a-run');p.add_argument('--b-run');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--scene-id',default='fc90fd5b50ae59fe');p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-seconds',type=int,default=1800);p.add_argument('--export-max-seconds',type=int,default=1200)
    for mode in ('s0','a','b'):p.add_argument('--'+mode+'-predictions')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    try:
        if not args.device.startswith('cuda:') or not 1<=args.max_seconds<=7200 or not 1<=args.export_max_seconds<=3600:
            raise ValueError('Explicit one-CUDA-device and bounded wall budgets required')
        prepared=plan(args)
        result=prepared if args.phase=='plan' or args.dry_run else execute(args,prepared)
        print(json.dumps(result));return 2 if result['status']=='WAITING_INPUTS' else 0
    except Exception as exc:
        print(json.dumps({'status':'FAIL','error_type':type(exc).__name__,'error':str(exc)}));return 1


if __name__=='__main__':raise SystemExit(main())

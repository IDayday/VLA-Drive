"""Prepare/compare native reference-cache qualification; never dispatch a job.

Training is exclusively the existing train --qualification entry. The optional
migration command only creates an evidence-bound draft; it never changes runs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

P = Path('/mnt/project/qwen-world-performance-rl-20261103')
SCHEMA = 'world_full_IL70_reference_cache_world8_native_equivalence_v1'


def read(path):return json.loads(Path(path).read_text())


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Evidence must not be overwritten: '+str(path))
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def verify_identity(value):
    body=dict(value);sha=body.pop('sha256',None)
    if fingerprint(body)!=sha:raise ValueError('Identity seal changed')


def make_plan(root,source,*,python=sys.executable,config=None,job_root=None):
    """Plan only. The operator allocates the same eight GPUs for each stage."""
    import yaml
    from starVLA.rl.qwen_world_performance.contracts import validate_recipe
    root=Path(root).resolve();source=Path(source).resolve();config=Path(config or P/'configs/S0_LARGE_SAFETYEP.yaml')
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if subprocess.run(['git','diff','--quiet','HEAD'],cwd=source).returncode:
        raise ValueError('Qualification must use a tracked-clean frozen source')
    if not (source/'starVLA/rl/qwen_world_research_algorithms/frozen_reference.py').is_file():
        raise ValueError('The selected source lacks the reference cache implementation')
    cfg=validate_recipe(yaml.safe_load(config.read_text()));spec=cfg['research']['performance']
    if (spec['scenes_per_collection'],spec['group_size'],spec['inner_updates'],spec['algorithm'])!=(128,16,4,'safetyep'):
        raise ValueError('Qualification requires actual B128/G16/M4 SafetyEP, not a reduced fixture')
    if cfg['seed']!=42 or cfg['world_size']!=8:raise ValueError('Same IL70/seed42/world8 required')
    if root.exists() and any((root/name).exists() for name in ('PLAN.json','resolved_recipe.yaml','original','cached')):
        raise FileExistsError('Use a new qualification output; never replace an earlier attempt')
    root.mkdir(parents=True,exist_ok=True);resolved=root/'resolved_recipe.yaml';resolved.write_bytes(config.read_bytes())
    jobs=Path(job_root or P/'jobs');pending=root/'cached/checkpoints/update000002_inner2'
    base=[str(python),'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8',
        '-m','tools.qwen_world_performance.train','--config',str(resolved),'--manifest',str(P/'data/NAVTRAIN_SCENES.json'),
        '--il-checkpoint','/mnt/project/qwen-world-rl-optimization-20261030/IL_PARENT',
        '--parent-selection','/mnt/project/qwen-world-rl-optimization-20261030/PARENT_SELECTION.json',
        '--data','/mnt/project/qwen-world-rl-optimization-20261030/data/NAVTRAIN','--qualification','--save-every','2']
    stages=[]
    for name,out,stop,enabled,resume in [('original4','original',4,False,None),('cached2','cached',2,True,None),('cached_resume4','cached',4,True,pending)]:
        command=base+['--output',str(root/out),'--stop-after-update',str(stop)]
        if enabled:command+=['--frozen-reference-cache']
        if resume:command+=['--resume',str(resume)]
        stages.append(dict(id=name,command=command,cwd=str(source),cache_enabled=enabled,
            depends_on=stages[-1]['id'] if stages else None,
            transport_result=str(jobs/('QUAL_REFERENCE_CACHE_NATIVE_'+name)/'RESULT.json')))
    return dict(schema='world_reference_cache_native_operator_plan_v1',status='PREPARED_NOT_DISPATCHED',
        root=str(root),source=str(source),source_commit=source_commit,config=str(resolved),config_sha256=file_sha(resolved),
        B=128,G=16,M=4,world=8,seed=42,parent_model_sha256=spec['parent_model_sha256'],
        initial_Adam_step=0,fresh_valid_scenes_per_path=128,PG_queries_per_path=2048,
        stages=stages,original4=str(root/'original/checkpoints/update000004_inner0'),
        original2=str(root/'original/checkpoints/update000002_inner2'),cached2=str(pending),
        cached4=str(root/'cached/checkpoints/update000004_inner0'),
        compare_command=[str(python),'-m','tools.qwen_world_performance.reference_cache_native_qualification',
            'compare','--plan',str(root/'PLAN.json'),'--output',str(root/'COMPLETE.json')],
        boundary='Native COMPLETE/files/transaction/8rank validators; no legacy reconstruction',
        admission='Operator-owned existing grant/leases/headroom only; this tool never opens SSH or dispatches')


def inspect_checkpoint(path,updates,inner,*,want_cache):
    """Load only sealed own CPU native state; run the actual strict validators."""
    import torch
    from starVLA.rl.qwen_world_performance.recovery import validate_boundary,validate_ranks
    from starVLA.rl.qwen_world_research_algorithms.frozen_reference import KEY,digest
    path=Path(path);seal=read(path/'COMPLETE');files=read(path/'files.json');identity=read(path/'identity.json')
    if (seal.get('schema'),seal.get('world'),seal.get('updates'),seal.get('next_inner'))!=(2,8,updates,inner):
        raise ValueError('Wrong native world/update/pending boundary: '+str(path))
    if file_sha(path/'files.json')!=seal['files_sha256']:raise ValueError('Native inventory seal changed')
    required={'optimizer.pt','identity.json','transaction.json'}|{f'rank{rank}.pt' for rank in range(8)}
    if not required.issubset(files):raise ValueError('Incomplete native8rank checkpoint')
    if any(file_sha(path/name)!=sha for name,sha in files.items()):raise ValueError('Native file content changed')
    verify_identity(identity)
    spec=identity['recipe']['research']['performance']
    if (spec['scenes_per_collection'],spec['group_size'],spec['inner_updates'])!=(128,16,4):raise ValueError('Wrong B/G/M')
    enabled='frozen_IL_reference_cache' in identity.get('execution_optimizations',{})
    if enabled!=want_cache:raise ValueError('Cache execution flag differs from qualification path')
    transaction=read(path/'transaction.json')
    common=torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False)
    common_hash={k:digest(common[k]) for k in ('adapters','optimizer','scheduler')}
    summaries=[];ranks={};all_tokens=[]
    for rank in range(8):
        own=torch.load(path/f'rank{rank}.pt',map_location='cpu',weights_only=False)
        if own['identity_sha256']!=identity['sha256']:raise ValueError('Foreign rank identity')
        summary=validate_boundary(own['state'],own['rollout'],identity,common['optimizer'],common['scheduler'],expected_rank=rank)
        if summary!=transaction['ranks'][rank] or summary['phase']!=seal['phase']:raise ValueError('Native transaction does not reproduce')
        summaries.append(summary);state=own['state'];stream=copy.deepcopy(state['scene_stream']);stream.pop('run_identity')
        item=dict(RNG_sha256=digest(own['rng']),scene_stream_without_own_run_label_sha256=digest(stream),
            counters={k:state[k] for k in ('updates','next_inner','collections','scene_cursor','raw_scene_exposures','valid_scene_exposures','padded_scene_exposures','new_candidates')},
            pending=own['rollout'] is not None)
        if inner:
            window=own['rollout'];leaves=window['samples']
            if any((KEY in leaf)!=want_cache for leaf in leaves):raise ValueError('Pending frozen-reference cache presence differs')
            material=[]
            for leaf in leaves:
                keep=('chain','timesteps','deltas','stoch_idx','cond_bg','old_logp','rewards','advantages','adv_mask','normalization_std','global_std','token','log','valid_scene','stream_position')
                material.append({k:leaf[k] for k in keep})
            item['pending_math_sha256']=digest(material);item['tokens']=[s['token'] for s in leaves];all_tokens+=item['tokens']
            item['cached_leaves']=sum(KEY in s for s in leaves)
        ranks[str(rank)]=item
    validate_ranks(summaries)
    if inner and len(set(all_tokens))!=128:raise ValueError('Expected128 genuine distinct TRAIN scenes')
    del common
    return dict(checkpoint=str(path),files_sha256=seal['files_sha256'],identity=identity,
        common=common_hash,ranks=ranks,strict_native_validation=True)


def journal_rows(root,rank):
    rows=[json.loads(line) for line in (Path(root)/f'metrics_rank{rank}.jsonl').read_text().splitlines() if line]
    if [r['updates'] for r in rows]!=[1,2,3,4]:raise ValueError('Expected continuous actual four-update journal')
    for r in rows:
        if r['group_size']!=16 or r['exploration']['B']!=128 or r['exploration']['M']!=4:raise ValueError('Reduced qualification journal')
        if r['inner_completed']==1 and r['ratio_max_abs_from_one']>1e-4:raise ValueError('Fresh ratio tolerance changed')
    fields=('updates','inner_completed','raw_scene_exposures','new_candidates','lr','loss','gradient_norm','optimizer_group_lrs_used','optimizer_group_lrs')
    return [{k:r[k] for k in fields} for r in rows]


def compare(plan):
    from starVLA.rl.simwam_recipe.lineage import scientific_identity
    from starVLA.rl.qwen_world_research_algorithms.frozen_reference import digest
    if file_sha(plan['config'])!=plan['config_sha256']:raise ValueError('Qualification config changed')
    transports=[]
    for stage in plan['stages']:
        result=read(stage['transport_result'])
        if result.get('returncode')!=0 or result.get('status')!='EXITED':
            raise ValueError('Every operator transport must have actual EXITED RESULT returncode0')
        if result.get('command')!=stage['command'] or Path(result.get('cwd','')).resolve()!=Path(plan['source']).resolve():
            raise ValueError('Transport RESULT does not belong to the exact planned source/command')
        if not result.get('child_pid') or not result.get('child_start_ticks'):
            raise ValueError('Actual identified native child was not observed')
        transports.append(dict(stage=stage['id'],path=stage['transport_result'],sha256=file_sha(stage['transport_result'])))
    endpoints={k:inspect_checkpoint(plan[k],2 if k.endswith('2') else 4,2 if k.endswith('2') else 0,want_cache=k.startswith('cached')) for k in ('original2','cached2','original4','cached4')}
    pairs={}
    for u in (2,4):
        a,b=endpoints[f'original{u}'],endpoints[f'cached{u}']
        if scientific_identity(a['identity'])!=scientific_identity(b['identity']):raise ValueError('Scientific identity changed')
        common={k:a['common'][k]==b['common'][k] for k in a['common']}
        ranks={str(r):a['ranks'][str(r)]==b['ranks'][str(r)] for r in range(8)}
        # Cache count is execution metadata; compare mathematical pending state.
        if u==2:
            ranks={str(r):{k:v for k,v in a['ranks'][str(r)].items() if k!='cached_leaves'}==
                          {k:v for k,v in b['ranks'][str(r)].items() if k!='cached_leaves'} for r in range(8)}
        if not all(common.values()) or not all(ranks.values()):raise ValueError('Native parameters/Adam/scheduler/RNG/data mismatch at'+str(u))
        pairs[str(u)]=dict(common_bitexact=common,all8rank_RNG_cursor_math_bitexact=ranks)
    root=Path(plan['root']);resume=[];pending=Path(plan['cached2'])
    for rank in range(8):
        candidates=[read(p) for p in (root/'cached/invocations').glob(f'*/rank{rank}.json')]
        seen=[v for v in candidates if v.get('start_update')==2 and v.get('end_update')==4 and v.get('resume_checkpoint')==str(pending)
            and v.get('resume_files_sha256')==file_sha(pending/'files.json') and not v.get('error')]
        if len(seen)!=1:raise ValueError('Missing unique actual native inner2→4 restore invocation')
        resume.append(dict(rank=rank,invocation=seen[0]['invocation'],start_update=2,end_update=4,resume_files_sha256=seen[0]['resume_files_sha256']))
        for dirname in ('original','cached'):
            result=read(root/dirname/f'result_rank{rank}.json')
            if (not result.get('qualification') or result.get('updates')!=4 or result.get('rank')!=rank
                    or result.get('frozen_SHA_before')!=result.get('frozen_SHA_after') or result.get('DDP_bucket_profile',{}).get('has_rebuilt_buckets',0)):
                raise ValueError('Missing/changed actual full-model rank qualification result')
        if digest(journal_rows(root/'original',rank))!=digest(journal_rows(root/'cached',rank)):
            raise ValueError('Actual per-step loss/gradnorm/LR journal differs')
    identities=[v['identity'] for v in endpoints.values()]
    if any(i['parameter_contract']['trainable_numel']!=52_707_840 for i in identities):raise ValueError('Not real full52.7M Scope')
    return dict(status='PASS',schema=SCHEMA,world=8,B=128,G=16,M=4,trainable_numel=52_707_840,
        strict_native_pending_restore=True,independent_IL70_Adam0=True,qualification_only=True,
        actual_native_boundaries={k:dict(checkpoint=v['checkpoint'],files_sha256=v['files_sha256']) for k,v in endpoints.items()},
        comparisons=pairs,actual_resume_invocations=resume,transport_results=transports,
        all8rank_four_update_journal_bitexact=True,scientific_identity=scientific_identity(identities[0]),
        qualified_source_files_sha256=digest(identities[0]['source_files']),
        cache_helper_sha256=file_sha(Path(__file__).resolve().parents[2]/'starVLA/rl/qwen_world_research_algorithms/frozen_reference.py'),
        limits='Real world8 qualification from fresh IL70; not new NAVTEST or a long-run speed claim. Timings/own-run labels/cache payloads are not mathematical equality fields.')


def migration_draft(old,new,checkpoint,evidence):
    from starVLA.rl.simwam_recipe.lineage import scientific_identity,verify_migration
    from starVLA.rl.qwen_world_research_algorithms.frozen_reference import digest
    verify_identity(old);verify_identity(new)
    if scientific_identity(old)!=scientific_identity(new):raise ValueError('Scientific/config/assets identity changed')
    if 'frozen_IL_reference_cache' not in new.get('execution_optimizations',{}):raise ValueError('Actual new formal identity must explicitly enable cache')
    if old['purpose']!='formal' or new['purpose']!='formal':raise ValueError('Qualification identities cannot masquerade as formal identities')
    values=[read(path) for path in evidence]
    schemas={v.get('schema') for v in values}
    needed={'world_real_full_IL70_reference_cache_singleGPU_four_inner_v1',SCHEMA}
    if any(v.get('status')!='PASS' for v in values) or not needed.issubset(schemas):raise ValueError('Actual singleGPU plus world8 native PASS evidence required')
    helper_sha=file_sha(Path(__file__).resolve().parents[2]/'starVLA/rl/qwen_world_research_algorithms/frozen_reference.py')
    if any(v.get('schema')==SCHEMA and v.get('cache_helper_sha256')!=helper_sha for v in values):raise ValueError('Equivalence helper source changed')
    if any(v.get('schema')==SCHEMA and v.get('qualified_source_files_sha256')!=digest(new['source_files']) for v in values):
        raise ValueError('Actual new formal execution source differs from world8 qualified source')
    if read(Path(checkpoint)/'identity.json')!=old:raise ValueError('Not the exact old native parent identity')
    body=dict(schema='world_frozen_reference_cache_execution_migration_draft_v1',status='DRAFT_NOT_APPLIED',
        old_identity=old,new_identity=new,parent_files_sha256=file_sha(Path(checkpoint)/'files.json'),
        evidence=[dict(path=str(Path(p).resolve()),sha256=file_sha(p)) for p in evidence],
        changes='Only explicit execution source and frozen-IL cache flag; preserve same-run full Adam/RNG/cursor/pending, original cfg and fresh-ratio tolerance',
        operator_action_required=True)
    return dict(body,sha256=fingerprint(body))


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--root',type=Path,required=True);prep.add_argument('--source',type=Path,default=Path.cwd());prep.add_argument('--config',type=Path);prep.add_argument('--job-root',type=Path)
    comp=sub.add_parser('compare');comp.add_argument('--plan',type=Path,required=True);comp.add_argument('--output',type=Path,required=True)
    mig=sub.add_parser('migration-draft')
    for name in ('old-identity','new-identity','checkpoint','output'):mig.add_argument('--'+name,type=Path,required=True)
    mig.add_argument('--evidence',type=Path,action='append',required=True)
    args=parser.parse_args()
    if args.action=='prepare':value=make_plan(args.root,args.source,config=args.config,job_root=args.job_root);path=args.root/'PLAN.json'
    elif args.action=='compare':value=compare(read(args.plan));path=args.output
    else:value=migration_draft(read(args.old_identity),read(args.new_identity),args.checkpoint,args.evidence);path=args.output
    write(path,value)
    if args.action=='migration-draft':
        from starVLA.rl.simwam_recipe.lineage import verify_migration
        verify_migration(path,read(args.old_identity),read(args.new_identity),args.checkpoint)
    print(json.dumps(dict(status=value['status'],output=str(path))))


if __name__=='__main__':main()

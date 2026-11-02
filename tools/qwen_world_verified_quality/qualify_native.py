"""Plan/compare actual S6 world8 four-inner and split-native qualification only."""
import argparse
import json
from pathlib import Path
import sys

from starVLA.rl.qwen_world_verified_quality.recipe import validate_recipe,QUALITY_PROTOCOL_SHA256


def read(path):return json.loads(Path(path).read_text())


def prepare(root,source,config):
    import yaml
    cfg=validate_recipe(yaml.safe_load(Path(config).read_text()));root=Path(root).resolve();source=Path(source).resolve()
    if root.exists():raise FileExistsError('Use a new independent qualification directory')
    root.mkdir(parents=True);resolved=root/'resolved_recipe.yaml';resolved.write_bytes(Path(config).read_bytes())
    base=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8',
        '-m','tools.qwen_world_verified_quality.train','--config',str(resolved),'--manifest','/mnt/project/qwen-world-performance-rl-20261103/data/NAVTRAIN_SCENES.json',
        '--il-checkpoint','/mnt/project/qwen-world-rl-optimization-20261030/IL_PARENT','--parent-selection','/mnt/project/qwen-world-rl-optimization-20261030/PARENT_SELECTION.json',
        '--data',cfg['research']['observation_data'],'--qualification','--save-every','2']
    stages=[]
    for name,out,stop,resume in [('continuous4','continuous',4,None),('split2','split',2,None),('resume4','split',4,root/'split/checkpoints/update000002_inner2')]:
        cmd=base+['--output',str(root/out),'--stop-after-update',str(stop)]
        if resume:cmd+=['--resume',str(resume)]
        stages.append(dict(id=name,cwd=str(source),command=cmd,depends_on=stages[-1]['id'] if stages else None,
            transport_result=str(Path('/mnt/project/qwen-world-performance-rl-20261103/jobs')/('QUAL_S6_'+name)/'RESULT.json')))
    return dict(status='PREPARED_NOT_DISPATCHED',schema='world_verified_quality_native_qualification_plan_v1',root=str(root),source=str(source),
        quality_protocol_sha256=QUALITY_PROTOCOL_SHA256,stages=stages,world=8,B=64,G=16,M=4,fresh=True,actual_start_epoch=0.,cache_default=True,
        required='All8 native results/frozen SHA/continuous journal finite; actual repair+safe-high teacher activation; exact native inner2 resume; full Adam/RNG/calibration/data equality')


def compare(plan):
    import torch
    from starVLA.rl.qwen_world_safety_exploration.retention_material import digest
    from .native_review import load_validated
    root=Path(plan['root']);parent='/mnt/project/qwen-world-rl-optimization-20261030/IL_PARENT'
    for stage in plan['stages']:
        result=read(stage['transport_result'])
        if result.get('status')!='EXITED' or result.get('returncode')!=0 or result.get('command')!=stage['command'] or result.get('cwd')!=stage['cwd']:
            raise ValueError('Every S6 qualification requires exact-source/command actual transport EXITED0')
        if not result.get('child_pid') or not result.get('child_start_ticks'):raise ValueError('No actual native child identity')
    endpoints={};metadata={};classes={'.02':0,'.05':0,'inactive':0};first=True
    for name,u,i in [('continuous',2,2),('continuous',4,0),('split',2,2),('split',4,0)]:
        cp=root/name/'checkpoints'/f'update{u:06}_inner{i}';key=f'{name}{u}'
        metadata[key]=load_validated(parent,cp,qualification=True,allow_pending=bool(i))
        common=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)
        item=dict(common={k:digest(common[k]) for k in ('adapters','optimizer','scheduler')},ranks={})
        del common
        for rank in range(8):
            own=torch.load(cp/f'rank{rank}.pt',map_location='cpu',weights_only=False);s=own['state']
            item['ranks'][str(rank)]=dict(RNG=digest(own['rng']),stream=digest(s['scene_stream']),
                calibration=digest(s['performance_auxiliary']['nominal_improvement']),cursor=s['scene_cursor'])
            if name=='split' and u==2:
                for leaf in own['rollout']['samples']:
                    p=leaf['nominal_improvement'];classes['.02' if p['enabled'] and p['active_alpha']==.02 else '.05' if p['enabled'] else 'inactive']+=1
        endpoints[key]=item
    for u in (2,4):
        if endpoints[f'continuous{u}']!=endpoints[f'split{u}']:raise ValueError('S6 full Adam/weights/scheduler/8rank RNG/stream/calibration native mismatch')
    if not classes['.02'] or not classes['.05']:raise ValueError('Bounded qualification lacks actual activation of both S6 teacher classes')
    for rank in range(8):
        rows=[]
        for name in ('continuous','split'):
            result=read(root/name/f'result_rank{rank}.json')
            if result['updates']!=4 or result['rank']!=rank or not result['qualification'] or result['frozen_SHA_before']!=result['frozen_SHA_after']:
                raise ValueError('Incomplete actual fullScope qualification result')
            journal=[json.loads(v) for v in (root/name/f'metrics_rank{rank}.jsonl').read_text().splitlines()]
            if [r['updates'] for r in journal]!=[1,2,3,4]:raise ValueError('Not actual four consecutive updates')
            rows.append([{k:r[k] for k in ('updates','loss','gradient_norm','lr','optimizer_group_lrs_used')}for r in journal])
        if digest(rows[0])!=digest(rows[1]):raise ValueError('Per-step S6 values/gradnorm/LR differ after native resume')
        inv=[read(p)for p in (root/'split/invocations').glob(f'*/rank{rank}.json')]
        if not any(v.get('start_update')==2 and v.get('end_update')==4 and v.get('resume_checkpoint')==str(root/'split/checkpoints/update000002_inner2') and not v.get('error')for v in inv):
            raise ValueError('Missing actual inner2→4 native resume')
    return dict(status='PASS',schema='world_verified_quality_actual_world8_native_qualification_v1',quality_protocol_sha256=QUALITY_PROTOCOL_SHA256,
        world=8,B=64,G=16,M=4,full_IL70=True,trainable_numel=52_707_840,actual_initial_epoch=0.,
        actual_first_collection_teacher_classes=classes,native_bitexact_at_inner2_and4=True,strict_reviews=metadata,
        qualification_only=True,not_NAVTEST_gain=True)


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--root',required=True,type=Path);p.add_argument('--source',required=True,type=Path);p.add_argument('--config',required=True,type=Path)
    p=sub.add_parser('compare');p.add_argument('--plan',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=parser.parse_args()
    if a.action=='prepare':value=prepare(a.root,a.source,a.config);out=a.root/'PLAN.json'
    else:value=compare(read(a.plan));out=a.output
    if out.exists():raise FileExistsError('Do not overwrite evidence')
    out.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=value['status'],output=str(out))))

if __name__=='__main__':main()

"""Read-only IL/RL Navtest queue audit; never publishes partial-population PDMS.

Read bounded registrations, native checkpoint seals and rank metadata only.
Resource refusals are admission outcomes, not failed policy trajectories.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default


def owner_live(owner):
    try:
        s=Path('/proc',str(owner['pid']),'stat').read_text().split()
        return s[2]not in('Z','X')and int(s[21])==owner['start_ticks']
    except (FileNotFoundError,KeyError):return False


def progress(root, world=32):
    root=Path(root)
    # IL metadata is under evaluation/, RL metadata is directly under output/.
    metadata=root/'evaluation'if(root/'evaluation').is_dir()else root
    ranks={}
    for path in metadata.glob('rank*.json'):
        if not re.fullmatch(r'rank\d+\.json',path.name):continue
        r=read(path);rank=int(path.stem[4:])
        if not 0<=rank<world:raise ValueError('Rank outside registered layout')
        ranks[rank]=r
    scenes=sum(r.get('new',0)+r.get('reused',0)for r in ranks.values())
    if scenes>12146:raise ValueError('Reported logical scenes exceed the population')
    done=sum(r.get('status')=='COMPLETE'for r in ranks.values())
    summary=read(root/'SUMMARY.json')
    if summary:
        if (summary.get('status')!='COMPLETE'or summary.get('scenes')!=12146
                or summary.get('invalid_scenes')!=0 or summary.get('seed')!=2101):
            raise ValueError('Incomplete or incompatible summary: '+str(root))
        phase='COMPLETE'
    elif done==world and scenes==12146:phase='CPU_AGGREGATION'
    elif ranks:phase='INFERENCE_OR_SCORING'
    else:phase='WAITING_FOR_ADMISSION_OR_LOADING'
    result=dict(phase=phase,logical_scenes_recorded=scenes,expected_scenes=12146,
        registered_world=world,ranks_reporting=len(ranks),ranks_complete=done,root=str(root))
    if summary:result.update(PDMS=summary['PDMS'],summary=str(root/'SUMMARY.json'))
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--il-root',type=Path,default=Path('/mnt/project/qwen-world-il-research-20261030'))
    p.add_argument('--rl-root',type=Path,default=Path('/mnt/project/qwen-world-rl-optimization-20261030'))
    p.add_argument('--four-root',type=Path,default=Path('/mnt/project/qwen-world-four-frameworks-20261029'))
    p.add_argument('--world-root',type=Path,default=Path('/mnt/project/qwen-world-postcot-20261025'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    cadence=a.il_root/'navtest_cadence_20261031';state=read(cadence/'STATE.json')
    snapshot=read(a.il_root/'reporting/report_snapshot.json');rows=[]
    for method in['ARCH_'+x for x in'ABCD']+['TIME_RESIDUAL','HUBER']:
        run=(a.four_root/'runs'/method/'IL_seed42'if method.startswith('ARCH_')else
             a.world_root/'fresh_loss_IL_20261029'/method/'runs/IL_full_seed42')
        sealed=[int(p.name[6:])for p in(run/'checkpoints').glob('update*')
                if re.fullmatch(r'update\d+',p.name)and(p/'COMPLETE').is_file()]
        jobs=[j for j in state['jobs']if j['method']==method]
        scores=[s for s in snapshot['scores']if s['method']==method]
        registered=[j['updates']for j in jobs]+[s['updates']for s in scores]
        active=[]
        for job in jobs:
            if job['status']in('RUNNING','DISPATCHING','PENDING','FAILED_REVIEW_REQUIRED'):
                q=progress(Path(job['root'])/job['label'])
                active.append(dict(updates=job['updates'],controller_status=job['status'],**q))
        latest=max(sealed,default=0);last=max(registered,default=0)
        eligible=[u for u in sealed if u%2000==0 and u>=last+2000]
        rows.append(dict(method=method,latest_sealed_update=latest,
            latest_full_result=max((s['updates']for s in scores),default=None),
            last_registered_update=last,next_periodic_threshold=(last//2000+1)*2000,
            next_unregistered_eligible=max(eligible,default=None),active=active))
    rl=[]
    for owner,arm in [('SAFETY_FRESH_PAIR','PPO_SAFETY_SUCCESS_MEMORY'),
                       ('SAFETY_FRESH_PAIR','PPO_SAFETY_FPO'),('FPO_FULLTIME','PPO_FPO_FULLTIME'),
                       ('SAFETY_FEASIBLE_SCORE','PPO_SAFETY_FEASIBLE_SCORE')]:
        base=a.rl_root/'owners'/owner;s=read(base/'STATE.json');r=s['runs'][arm]
        for step,stage in r.get('stages',{}).items():
            nav=stage.get('NAVTEST',{});spec=nav.get('spec')
            if not spec:continue
            q=progress(spec['output'],spec['world'])
            rl.append(dict(arm=arm,updates=int(step),controller_status=nav.get('status'),
                owner=owner,owner_verified_live=owner_live(read(base/'OWNER.json',{})),**q))
    result=dict(schema='read_only_NAVTEST_queue_audit_v1',generated_utc=datetime.now(timezone.utc).isoformat(),
        IL_cadence_owner_verified_live=owner_live(read(cadence/'OWNER.json',{})),IL=rows,RL=rl,
        new_inference=0,new_scores=0,training_changes=0,
        note='Only complete 12146-scene summaries expose PDMS. Active old attempt resource refusals must not be confused with current resubmissions; native policy failures are not inferred from absence of a summary.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

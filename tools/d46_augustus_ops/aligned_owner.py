"""Event-driven monitor for exactly two independently initialized aligned runs.

Adoption only: root dispatches qualified formal gangs. No blind restart, no
training journal polling, no TEST scene labels exported to any learner.
"""
import argparse,fcntl,json,os,socket,time,traceback
from pathlib import Path
from .owner import read,review_native,Owner,head
from .event_wait import wait_for_events
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import file_sha,fingerprint

ROOT=Path('/mnt/project/d46-priority-rl-20261103')
SCI='09514510afa29e7fdbce834caffd93f3825eea24'
WORKER='30ad925'  # Full SHA is also sealed into every actual launch receipt.
EXPERIMENTS={'AUG3_FULL_ALIGNED':32,'AUG4_LORA_ALIGNED':16}
TERMINAL={'COMPLETE_6E','REVIEW_REQUIRED','SAVED_FOR_PERFORMANCE_REVIEW'}


def initial(directory):
    plan=read(directory/'PLAN.json');launched=read(directory/'FORMAL_LAUNCHES.json');proofs=read(directory/'QUALIFICATION_PROOFS.json')
    if plan.get('maximum_simultaneous_routes')!=2 or proofs.get('status')!='PASS':raise ValueError('Qualified MAX2 plan required')
    rows={};hosts=set()
    for r in plan['routes']:
        exp=r['experiment'];world=EXPERIMENTS[exp];output=ROOT/'runs'/(exp+'_v2')
        if r['world']!=world or len(r['hosts'])*8!=world or hosts&set(r['hosts'])or head(r['source'])!=SCI:raise ValueError('Changed fixed aligned route/gang/source')
        hosts.update(r['hosts']);nodes=[]
        for host in r['hosts']:
            found=[v for v in launched['receipts']if v['host']==host and str(output)in v['command']]
            if len(found)!=1:raise ValueError('Missing/duplicate exact formal launch')
            v=found[0];cmd=v['command']
            if(v['source_sha']!=SCI or v['source']!=r['source']or not v['worker_source_sha'].startswith(WORKER)
                or '--resume'in cmd or '--qualification'in cmd or v['gpus']!=list(range(8))
                or cmd[cmd.index('--config')+1]!=r['config']):raise ValueError('Formal must be new own IL46/Adam0, exact qualified command')
            nodes.append(dict(host=host,job=v['job'],command=cmd,receipt=v))
        rows[exp]=dict(experiment=exp,route=r,formal_output=str(output),phase='formal',status='FORMAL_RUNNING',
            stages={'formal':dict(nodes=nodes)},proofs={'qualification':proofs[exp]},events=[])
    if set(rows)!=set(EXPERIMENTS):raise ValueError('Exactly two registered aligned algorithms required')
    return dict(schema='d46_aligned_JaT_native_owner_v2',root=str(ROOT),plan_sha256=file_sha(directory/'PLAN.json'),
        formal_launches_sha256=file_sha(directory/'FORMAL_LAUNCHES.json'),routes=rows,status='RUNNING')


def stop_own(row,reason):
    for n in row['stages']['formal']['nodes']:
        job=Path(n['job'])
        if not(job/'RESULT.json').exists():
            atomic_json(job/'PERFORMANCE_STOP_REASON.json',dict(reason=reason,at=time.time(),aggregate_only=True))
            (job/'STOP_REQUESTED').touch()


def loss_review(row,evaluation):
    """Only completed aggregate NAVTEST may select/stop a model, never data."""
    for case in evaluation.get('cases',{}).values():
        if case.get('status')!='COMPLETE'or case['binding'].get('experiment')!=row['experiment']:continue
        summary=read(case['summary'],{})
        if summary.get('status')!='COMPLETE'or summary.get('scenes')!=12146 or summary.get('invalid_scenes')!=0:continue
        score=summary['PDMS']
        # Registered severe-forgetting exception; small early changes get time.
        if score<89.4955209758546-1.0:
            return dict(reason='Complete B1 PDMS more than1 point below own IL46; preserve native and review',
                PDMS=score,checkpoint=case['binding']['checkpoint'],summary=case['summary'],NAVTEST_SELECTED=True)
    return None


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--directory',type=Path,default=ROOT/'operations/AUGUSTUS_ALIGNED_20261103')
    p.add_argument('--once',action='store_true');a=p.parse_args();d=a.directory;d.mkdir(parents=True,exist_ok=True)
    with(d/'owner.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        expected=initial(d);state=read(d/'STATE.json',expected)
        for k in('schema','plan_sha256','formal_launches_sha256'):
            if state.get(k)!=expected[k]:raise ValueError('State belongs to another sealed campaign')
        atomic_json(d/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
            host=socket.gethostname(),source=__file__,at=time.time(),maximum_routes=2))
        helper=Owner(state,d)
        while not(d/'STOP_OWNER').exists():
            paths=[d,ROOT/'operations/AUGUSTUS_CAMPAIGN_20261103']
            for row in state['routes'].values():
                paths += [Path(n['job'])for n in row['stages']['formal']['nodes']]
                out=Path(row['formal_output']);paths += [out,out/'checkpoints',out/'evaluation_requests']
                paths += sorted((out/'checkpoints').glob('update*'))[-2:]
                if row['status']in TERMINAL:continue
                try:
                    if head(row['route']['source'])!=SCI or file_sha(row['route']['config'])!=row['route']['config_sha256']:raise ValueError('Scientific source/config changed')
                    done=helper.completed_nodes(row);helper.progress(row)
                    review=loss_review(row,read(ROOT/'operations/AUGUSTUS_CAMPAIGN_20261103/EVAL_STATE.json',{}))
                    if review and not row.get('performance_review'):
                        row['performance_review']=review;stop_own(row,review)
                    if done:
                        native=row.get('latest_native',{})
                        if row.get('performance_review'):row['status']='SAVED_FOR_PERFORMANCE_REVIEW'
                        elif native.get('phase')=='READY'and native.get('valid_scene_exposures',0)>=6*103288:row['status']='COMPLETE_6E'
                        else:raise ValueError('Formal exited before registered budget without explicit performance review')
                except Exception:
                    row.update(status='REVIEW_REQUIRED',error=traceback.format_exc())
                    # An actual child failure can release only its own peers.
                    if row.get('actual_child_failure'):stop_own(row,{'actual_child_failure':row['actual_child_failure']})
            state['updated']=time.time();state['status']='REVIEW_REQUIRED'if any(r['status']=='REVIEW_REQUIRED'for r in state['routes'].values())else'RUNNING'
            atomic_json(d/'STATE.json',state)
            original=read(ROOT/'operations/AUGUSTUS_CAMPAIGN_20261103/STATE.json',{})
            combined=dict(original.get('routes',{}));combined.update(state['routes'])
            aggregate=dict(schema='d46_augustus_qualified_evaluation_routes_v2',routes=combined)
            # Only structural changes should wake the evaluation owner.
            if read(d/'EVALUATION_ROUTES.json')!=aggregate:atomic_json(d/'EVALUATION_ROUTES.json',aggregate)
            if a.once or all(r['status']in TERMINAL for r in state['routes'].values()):break
            wait_for_events(paths,1800,ignore=('STATE.json','OWNER.json','EVAL_OWNER.json','EVALUATION_ROUTES.json'))


if __name__=='__main__':main()

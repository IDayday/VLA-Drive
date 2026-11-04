"""Two-route checkpoint-event monitor and existing canonical eval engine adapter.

Never launches/resets a learner. At most two complete assessment cases share
one bounded queue. Per-scene locks and source-bound records survive preemption.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_compare.contracts import file_sha,fingerprint,allowed_hosts
from tools.d46_augustus_ops import evaluations as E
from tools.d46_augustus_ops import complete_events
from tools.d46_augustus_ops.event_wait import wait_for_events
from tools.d44_native_rl.launch import launch as original_launch

ROOT=Path('/mnt/project/d46-fpo-two-routes-20261104')
PARENT=Path('/mnt/project/d46-priority-rl-20261103')
SOURCE=Path(__file__).resolve().parents[2]
TASK='d46_fpo_evaluation_20261104'
MODEL='386a0a27bd197cb002501bcb90c343659001d2e6d4f7bad0cbd5e31a91f8477d'
TEMPORARY={'training-rl-zt2','training-vlawm-zt4','training-vlawm-zt2','training-vlawm-tsinghua-zt4'}


def read(p,default=None):return json.loads(Path(p).read_text())if Path(p).is_file()else default


def binding(route,cp):
    cp=Path(cp).resolve();ident=read(cp/'identity.json',{});seal=read(cp/'COMPLETE',{});files=read(cp/'files.json',{})
    body=dict(ident);sha=body.pop('sha256',None)
    if(cp.parent!=Path(route['output']).resolve()/'checkpoints'or fingerprint(body)!=sha
        or ident.get('parent_model_sha256')!=MODEL or ident.get('qualification')is not False
        or files.get('identity.json')!=file_sha(cp/'identity.json')or seal.get('files_sha256')!=file_sha(cp/'files.json')
        or ident['recipe']['arm']!='D3_FPO'or ident['recipe']['training_uses_navtest']is not False
        or ident['scope']['trainable_numel']!=6422528 or seal.get('phase')not in('READY','INNER_COMMITTED')
        or ident['world']!=route['world']or files.get('transaction.json')!=file_sha(cp/'transaction.json')):
        raise ValueError('Not exact own complete FPO checkpoint')
    clocks=[x['valid_scene_exposures']for x in read(cp/'transaction.json')['ranks']]
    if len(clocks)!=route['world']or len(set(clocks))!=1:raise ValueError('Native valid exposure clock differs across ranks')
    if ident['recipe']!=__import__('yaml').safe_load(Path(route['config']).read_text()):raise ValueError('Changed registered recipe')
    for name,digest in ident['source_files'].items():
        if file_sha(Path(route['source'])/name)!=digest:raise ValueError('Scientific source changed '+name)
    return dict(checkpoint=str(cp),updates=seal['updates'],identity_sha256=sha,inventory_sha256=seal['files_sha256'],
        payload_sha256=files['optimizer.pt'],parent_model_sha256=MODEL,source=route['source'],experiment=route['id'],
        policy_signature=fingerprint(dict(parent=MODEL,trainable_payload=files['optimizer.pt'],scope=ident['scope'])),
        scene_budget_clock=clocks[0],pending_inner=seal['next_inner'])


def discover(root,state,routes):
    for route in routes.values():
        output=Path(route['output']);candidates=[]
        if route['id']=='FPO_MATURE':candidates.append(output/'checkpoints/update001711_inner1')
        else:
            for u in(400,800):candidates.append(output/'checkpoints'/f'update{u:06}_inner0')
        for u in route.get('early_evaluation_updates',[]):
            candidates.extend(sorted((output/'checkpoints').glob(f'update{u:06}_inner*')))
        for p in sorted((output/'evaluation_requests').glob('epoch*.json')):
            v=read(p)
            if v['epoch']>=route.get('minimum_new_evaluation_epoch',0):candidates.append(Path(v['checkpoint']))
        for cp in candidates:
            if not(cp/'COMPLETE').is_file():continue
            b=binding(route,cp);key=route['id']+'_u'+format(b['updates'],'06d')
            if key in state['cases']:
                previous=state['cases'][key]['binding']
                if previous==b or previous['policy_signature']==b['policy_signature']:continue
                key+='_'+b['policy_signature'][:8]
                if key in state['cases']:
                    if state['cases'][key]['binding']!=b:raise ValueError('Immutable checkpoint binding changed')
                    continue
            state['cases'][key]=dict(id=key,binding=b,origins=['registered_FPO_exposure_gate'],status='PENDING',jobs={},
                next_attempt={},created=time.time(),world=32,output=str(ROOT/'evaluations'/key/'evaluation'))
    return state


def notify(state_path,case_id,expected_summary_sha,**_):
    case=read(state_path)['cases'][case_id];s=read(case['summary']);body=dict(s);sha=body.pop('sha256',None)
    if(sha!=expected_summary_sha or fingerprint(body)!=sha or s.get('status')!='COMPLETE'or s.get('scenes')!=12146
            or s.get('invalid_scenes')!=0 or s.get('physical_logs')!=44):raise ValueError('Full official score required')
    request=Path('/mnt/project/qwen-world-rl-optimization-20261030/reporting/refresh_request.json')
    atomic_json(request,dict(id=case_id+'-'+sha,at=time.time(),scientific_result=False,event='evaluation_complete',
        reason='New independently attributable FPO full B1 score',evidence=case['summary']))
    all_cases=read(state_path)['cases'];selection={}
    for experiment in('FPO_MATURE','FPO_VERIFIED'):
        completed=[(c,read(c['summary']))for c in all_cases.values()if c['status']=='COMPLETE'and c['binding']['experiment']==experiment]
        if not completed:continue
        best=max(completed,key=lambda x:x[1]['PDMS'])
        stable=min((x for x in completed if x[1]['PDMS']>=best[1]['PDMS']-.2),key=lambda x:(x[1]['paired']['new_zero'],-x[1]['PDMS']))
        selection[experiment]=dict(best_score=dict(checkpoint=best[0]['binding']['checkpoint'],PDMS=best[1]['PDMS']),
            best_stable_proxy=dict(checkpoint=stable[0]['binding']['checkpoint'],PDMS=stable[1]['PDMS']),NAVTEST_SELECTED=True)
    atomic_json(ROOT/'NEW_COMPLETE_CHECKPOINT_SELECTION.json',selection)
    return dict(status='NOTIFIED',notified_sha=sha)


def launch(host,source,job,command,gpus,**kw):
    command=list(command);command[command.index('tools.d46_augustus.evaluate')]='tools.d44_native_rl.evaluate'
    return original_launch(host,source,job,command,gpus,share_min_free_mib=24000,
        worker_source=SOURCE,task_root=PARENT,task=TASK)


def inventory(host,ops_source):
    if host in read(ROOT/'operations/PREEMPTED_HOSTS.json',{}):return []
    for route in read(ROOT/'operations/STATE.json',{}).get('routes',{}).values():
        if host in route.get('hosts',[])and route.get('live_workers',0)>0 and route.get('status')!='RUNNING':return []
    code='''import pathlib,json,subprocess,sys,socket
sys.path.insert(0,SOURCE)
from tools.qwen_world_compare.job_worker import known_owner_args
apps={}
for s in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines():
 p,u,m=s.split(',');apps.setdefault(u.strip(),[]).append((int(p),int(m)))
rows=[]
for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used','--format=csv,noheader,nounits'],text=True).splitlines():
 i,u,free,used=s.split(',');owners=[]
 for pid,mem in apps.get(u.strip(),[]):
  try:
   p=pathlib.Path('/proc')/str(pid);a=[x for x in(p/'cmdline').read_bytes().decode().split(chr(0))if x];start=int((p/'stat').read_text().split()[21]);known=known_owner_args(a)
  except(OSError,ValueError,IndexError):known=False;start=None
  owners.append(dict(pid=pid,start_ticks=start,used_MiB=mem,known=known))
 rows.append(dict(index=int(i),uuid=u.strip(),free_MiB=int(free),used_MiB=int(used),owners=owners))
print(json.dumps(dict(hostname=socket.gethostname(),rows=rows)))
'''.replace('SOURCE',repr(str(SOURCE)))
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'/usr/bin/python3','-'],input=code,text=True,capture_output=True,timeout=20)
    if p.returncode:
        if host in TEMPORARY:
            values=read(ROOT/'operations/PREEMPTED_HOSTS.json',{});values[host]=dict(at=time.time(),reason=p.stderr,
                no_automatic_reuse_after_shutdown=True);atomic_json(ROOT/'operations/PREEMPTED_HOSTS.json',values)
        raise subprocess.SubprocessError('Host unavailable: '+host+' '+p.stderr)
    value=json.loads(p.stdout)
    if value['hostname']!=host+'-worker-0':raise ValueError('SSH container identity changed')
    return value['rows']


def recover_preempted(state):
    gone=read(ROOT/'operations/PREEMPTED_HOSTS.json',{})
    for case in state['cases'].values():
        if case['status']=='COMPLETE':continue
        for rank,jobs in case['jobs'].items():
            if not jobs:continue
            last=jobs[-1];job=Path(last['job'])
            result=read(job/'RESULT.json',{})
            partial=read(Path(case['output'])/f'rank{rank}.json',{})
            native_partial=(result.get('returncode')==0 and partial.get('status')=='STOPPED_AT_SCENE_BOUNDARY')
            if last['host']not in gone or(result and not native_partial):continue
            if time.time()-last['at']<300:continue
            # A shared stop marker fences even a delayed pre-CUDA predecessor.
            (job/'STOP_REQUESTED').touch()
            lock=Path(case['output'])/f'rank{rank}.lock';lock.parent.mkdir(parents=True,exist_ok=True)
            try:
                with lock.open('a')as f:
                    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    receipt=dict(at=time.time(),previous=last,shared_rank_lock_free=True,stop_marker=str(job/'STOP_REQUESTED'),
                        original_RESULT_not_fabricated=True,successful_scene_records_retained=True)
                    atomic_json(job/'PREEMPTIBLE_FENCED_REQUEUE.json',receipt)
                    case.setdefault('preempted_attempts',[]).append(receipt);jobs.pop();case['status']='PENDING'
            except BlockingIOError:pass


def route_status(plan):
    result={};active=0
    for key,r in plan['routes'].items():
        output=Path(r['output']);launches=read(r['launch_receipts'],{}).get('receipts',[])
        if not launches:
            result[key]=dict(status='QUALIFICATION_OR_PREPARING',world=r['world']);continue
        failures=[x for x in launches if read(Path(x['job'])/'RESULT.json',{}).get('returncode')not in(None,0)]
        live=[x for x in launches if not(Path(x['job'])/'RESULT.json').exists()]
        point=read(output/'latest.json',{});native={}
        if point:
            cp=Path(point['checkpoint']);seal=read(cp/'COMPLETE',{});tx=read(cp/'transaction.json',{})
            if seal.get('files_sha256')!=file_sha(cp/'files.json'):raise ValueError('Native metadata corrupted')
            clocks=[x['valid_scene_exposures']for x in tx.get('ranks',[])]
            if len(clocks)==r['world']and len(set(clocks))==1:
                native=dict(checkpoint=str(cp),updates=seal['updates'],next_inner=seal['next_inner'],scene_exposures=clocks[0],epoch=clocks[0]/103288)
        proof_path=ROOT/'operations'/('ACTUAL_START_'+key+'.json');proof=read(proof_path,{})
        if not proof and len(live)==len(launches):
            ranks=[];identity=read(output/'identity.json',{})
            for rank in range(r['world']):
                p=output/f'metrics_rank{rank}.jsonl';last={}
                if p.is_file():
                    with p.open('rb')as f:f.seek(max(0,p.stat().st_size-16384));lines=f.read().splitlines()
                    for line in reversed(lines):
                        try:last=json.loads(line);break
                        except ValueError:continue
                ranks.append(last)
            if all(v.get('updates',0)>r.get('start_update',0)and v.get('run')==identity.get('training_run_id',identity.get('sha256'))
                    and v.get('at',0)>=min(x['at']for x in launches)for v in ranks):
                proof=dict(at=time.time(),all_ranks_committed=True,updates=min(v['updates']for v in ranks),run=identity.get('training_run_id',identity['sha256']),execution_identity=identity['sha256'])
                proof_path.parent.mkdir(parents=True,exist_ok=True)
                atomic_json(proof_path,proof)
        status='REVIEW_REQUIRED'if failures else'RUNNING'if len(live)==len(launches)and proof else'INITIALIZING'if live else'SAVED_OR_COMPLETE'
        if failures:
            for x in live:(Path(x['job'])/'STOP_REQUESTED').touch()
        evaluation=read(ROOT/'operations/EVAL_STATE.json',{})
        points=sorted((c for c in evaluation.get('cases',{}).values()if c['status']=='COMPLETE'and c['binding']['experiment']==key),
            key=lambda c:c['binding']['scene_budget_clock'])
        if points:
            scores=[read(c['summary'])['PDMS']for c in points]
            severe=scores[-1]<89.4955209758546-3
            repeated=(points[-1]['binding']['scene_budget_clock']>=103288 and len(scores)>=2
                and all(v<89.4955209758546-1 for v in scores[-2:]))
            if severe or repeated:
                for x in live:
                    atomic_json(Path(x['job'])/'PERFORMANCE_REVIEW_REASON.json',dict(at=time.time(),scores=scores,
                        reason='latest complete drop>3 points, or after1E latest two drop>1 point; aggregate only'))
                    (Path(x['job'])/'STOP_REQUESTED').touch()
                status='SAVING_FOR_PERFORMANCE_REVIEW'if live else'SAVED_FOR_PERFORMANCE_REVIEW'
        if native.get('epoch',0)>=plan.get('initial_review_epochs',3):
            for x in live:(Path(x['job'])/'STOP_REQUESTED').touch()
            status='SAVING_FOR_BUDGET_REVIEW'if live else'SAVED_FOR_BUDGET_REVIEW'
        active+=int(status=='RUNNING')
        result[key]=dict(status=status,world=r['world'],hosts=r['hosts'],native=native,live_workers=len(live),failures=[x['job']for x in failures])
    return dict(at=time.time(),actual_training=active,routes=result,maximum_formal_routes=2)


def render(plan,status,state):
    from .reporting import build_fragment, document
    fragment=build_fragment(plan,status,state)
    for name,body in(('current_fragment.html',fragment),('report.html',document(fragment))):
        p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(body);os.replace(t,p)


def retain_checkpoints(plan,state):
    policy=read(ROOT/'operations/CHECKPOINT_POLICY.json',{})
    if policy.get('recent_resumable_checkpoints')!=10:return
    from .retention import rotate
    selected=read(ROOT/'NEW_COMPLETE_CHECKPOINT_SELECTION.json',{})
    for key,route in plan['routes'].items():
        pinned=[]
        for point in selected.get(key,{}).values():
            if isinstance(point,dict)and point.get('checkpoint'):pinned.append(point['checkpoint'])
        for case in state.get('cases',{}).values():
            if case['binding']['experiment']==key and case['status']not in E.TERMINAL:
                pinned.append(case['binding']['checkpoint'])
        own=[c for c in state.get('cases',{}).values()if c['binding']['experiment']==key and c['status']=='COMPLETE']
        if own:
            best=max(own,key=lambda c:read(c['summary'])['PDMS']);pinned.append(best['binding']['checkpoint'])
        known={c['binding']['checkpoint']for c in state.get('cases',{}).values()}
        for request in(Path(route['output'])/'evaluation_requests').glob('*.json'):
            point=read(request,{}).get('checkpoint')
            if point and point not in known:pinned.append(point)
        rotate(route['output'],keep=10,pinned=pinned)


def main():
    a=argparse.ArgumentParser(__doc__);a.add_argument('--once',action='store_true');args=a.parse_args()
    os.environ.setdefault('WORLD_COMPARE_ROOT',str(PARENT));os.environ.setdefault('WORLD_COMPARE_TASK',TASK)
    op=ROOT/'operations';op.mkdir(exist_ok=True)
    E.TASK=TASK;E.SOURCE=SOURCE;E.state_path=lambda root:op/'EVAL_STATE.json';E.discover=discover
    E.il_baseline=lambda root:read(PARENT/'PARENT_CONTRACT.json')['complete_navtest']['original_evaluation']
    E.launch=launch;E.inventory=inventory;complete_events.notify=notify
    E.SPARE_GPUS={h:tuple(range(8))for h in ['training-vlawm-tsinghua-zt5','training-vlawm-zt',
        'training-vlawm-tsinghua-zt','training-vlawm-tsinghua-zt2']}
    configured=read(op/'PLAN.json',{}).get('evaluation_gpu_indices')
    if configured is not None:
        E.SPARE_GPUS={h:tuple(indices)for h,indices in configured.items()if h in E.SPARE_GPUS}
    with(op/'owner.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(op/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),source=str(SOURCE),at=time.time()))
        last_render=0;previous_results=None;previous_routes=None;last_retention=None
        while not(op/'STOP_OWNER').exists():
            plan=read(op/'PLAN.json');status=route_status(plan);atomic_json(op/'STATE.json',status)
            state=read(op/'EVAL_STATE.json',E.initial_state());recover_preempted(state)
            # Inventory failures exclude a host; they never stop healthy training.
            try:
                discover(ROOT,state,plan['routes'])
                if any(c['status']not in E.TERMINAL for c in state['cases'].values()):
                    state=E.tick(ROOT,state,plan['routes'],ops_source=SOURCE,execute=not(op/'EVALUATION_DISPATCH_HOLD').exists())
                else:atomic_json(op/'EVAL_STATE.json',state)
            except Exception as error:
                atomic_json(op/'CONTROLLER_ERROR.json',dict(at=time.time(),error=repr(error)))
            signature=tuple((k,read(Path(v['output'])/'latest.json',{}).get('checkpoint'))for k,v in plan['routes'].items())
            if signature!=last_retention:
                try:retain_checkpoints(plan,state);last_retention=signature
                except Exception as error:
                    atomic_json(op/'RETENTION_ERROR.json',dict(at=time.time(),error=repr(error)))
            results=[(c['id'],c['status'])for c in state.get('cases',{}).values()]
            results += [(str(p),p.stat().st_mtime_ns)for p in sorted((ROOT/'engineering').glob('*_v*/*/RESULT.json'))]
            route_states=[(k,v['status'])for k,v in status['routes'].items()]
            if args.once or time.time()-last_render>=1800 or results!=previous_results or route_states!=previous_routes:
                render(plan,status,state);last_render=time.time()
                if results!=previous_results:
                    atomic_json(Path('/mnt/project/qwen-world-rl-optimization-20261030/reporting/refresh_request.json'),
                        dict(id='FPO-engineering-'+str(time.time_ns()),at=time.time(),scientific_result=False,
                            event='engineering_progress',evidence=str(ROOT/'engineering/STATUS.json')))
                previous_results=results;previous_routes=route_states
            if args.once:break
            paths=[op]
            paths += [ROOT/'engineering']+[p for p in(ROOT/'engineering').glob('*_v*/*')if p.is_dir()]
            for r in plan['routes'].values():
                out=Path(r['output']);paths += [out,out/'checkpoints',out/'evaluation_requests']
            for c in state.get('cases',{}).values():
                paths += [Path(c['output'])]+[Path(v[-1]['job'])for v in c['jobs'].values()if v]
            wait_for_events(paths,60,ignore=('STATE.json','OWNER.json','EVAL_STATE.json','report.html'))


if __name__=='__main__':main()

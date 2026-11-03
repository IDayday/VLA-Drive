"""CPU-only queue checks: no SSH, scorer, CUDA, model load or worker launch."""
import copy
import json
from pathlib import Path

import pytest

from starVLA.rl.qwen_world_compare.contracts import fingerprint, file_sha
from tools.d46_augustus_ops import evaluations as q


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def route(root, exp):
    source=root/'runtime'/q.SCIENCES[exp];source.mkdir(parents=True,exist_ok=True)
    (source/'science.py').write_text('same frozen science\n')
    return dict(experiment=exp,phase='formal',status='FORMAL_RUNNING',stages={'formal':{}},
        formal_output=str(root/'runs'/(exp+q.VERSIONS[exp])),route=dict(world=32 if 'FULL'in exp else 16,source=str(source)))


def checkpoint(root, r, updates, *, qualification=False, parent=q.PARENT_SHA):
    cp=Path(r['formal_output'])/'checkpoints'/f'update{updates:06d}_inner0';cp.mkdir(parents=True,exist_ok=True)
    recipe=dict(parent='D_IL46K',experiment_id=r['experiment'],training_uses_navtest=False,
                augustus={'protocol':'fixed'},global_scenes=128 if 'FULL'in r['experiment']else 64,inner_updates=2 if 'ALIGNED'in r['experiment']else 1)
    identity=dict(schema='d44_native_rl_v1',algorithm_schema=q.SCHEMA,architecture_id='D',
        experiment_id=r['experiment'],parent_model_sha256=parent,qualification=qualification,
        recipe=recipe,world=r['route']['world'],augustus_protocol_sha256=fingerprint(recipe['augustus']),
        scope={'names':['actual']},source_files={'science.py':file_sha(Path(r['route']['source'])/'science.py')})
    identity['sha256']=fingerprint(identity);write(cp/'identity.json',identity)
    for name in ('optimizer.pt','transaction.json')+tuple(f'rank{i}.pt'for i in range(identity['world'])):
        (cp/name).write_bytes(b'CPU fixture, never a model')
    write(cp/'transaction.json',dict(ranks=[dict(valid_scene_exposures=updates*recipe['global_scenes']//recipe['inner_updates'])for _ in range(identity['world'])]))
    files={p.name:file_sha(p)for p in cp.iterdir()};write(cp/'files.json',files)
    write(cp/'COMPLETE',dict(world=identity['world'],phase='READY',next_inner=0,updates=updates,
                             files_sha256=file_sha(cp/'files.json')))
    return cp,identity


def test_exact_early_and_native_gates_reject_qualification_and_foreign_parent(tmp_path):
    root=tmp_path;r=route(root,'AUG1_FULL');cp,identity=checkpoint(root,r,100)
    state=q.initial_state();q.discover(root,state,{r['experiment']:r})
    assert list(state['cases'])==['AUG1_FULL_u000100']
    request=Path(r['formal_output'])/'evaluation_requests/epoch0.125.json'
    write(request,dict(seed=2101,checkpoint=str(cp),updates=100,source_run=identity['sha256']))
    q.discover(root,state,{r['experiment']:r})
    assert len(state['cases'])==1 and len(next(iter(state['cases'].values()))['origins'])==2
    checkpoint(root,r,200,qualification=True)
    with pytest.raises(ValueError,match='seal mismatch'):q.discover(root,state,{r['experiment']:r})
    checkpoint(root,r,200,parent='foreign')
    with pytest.raises(ValueError,match='seal mismatch'):q.discover(root,state,{r['experiment']:r})
    qualification=copy.deepcopy(r);qualification['phase']='qualification';qualification['stages']={}
    skipped=q.initial_state();q.discover(root,skipped,{r['experiment']:qualification});assert not skipped['cases']
    inspect=q.initial_state();(Path(r['formal_output'])/'checkpoints/update000200_inner0/COMPLETE').unlink()
    q.tick(root,inspect,{r['experiment']:r},execute=False)
    assert not q.state_path(root).exists() and len(inspect['cases'])==1


def test_spare_only_two_cases_two_known_workers_and_unchanged_official_command(tmp_path,monkeypatch):
    root=tmp_path;routes={e:route(root,e)for e in q.EARLY if 'ALIGNED'not in e}
    for exp,r in routes.items():
        for u in q.EARLY[exp]:checkpoint(root,r,u)
    monkeypatch.setattr(q,'allowed_hosts',lambda **kw:list(q.SPARE_GPUS)+['training-vlawm-tsinghua-zt5'])
    probes=[]
    def inventory(host,source):
        probes.append(host)
        return [dict(index=i,uuid='GPU'+str(i),free_MiB=80000,used_MiB=0,owners=[])for i in range(8)]
    monkeypatch.setattr(q,'inventory',inventory)
    launches=[]
    def launch(host,science,job,command,gpus,**kw):
        assert gpus[0]in q.SPARE_GPUS[host] and kw['share_min_free_mib']==24000
        assert kw['worker_source']==root/'ops_frozen'
        assert command[command.index('-m')+1]=='tools.d46_augustus.evaluate'
        assert command[-2:]==['--world','32'] and '--limit'not in command
        r=dict(job=str(job),host=host,gpus=gpus,command=command);launches.append(r);return r
    monkeypatch.setattr(q,'launch',launch)
    state=q.initial_state();q.tick(root,state,routes,ops_source=root/'ops_frozen')
    assert set(probes)==set(q.SPARE_GPUS)
    active={c['id']for c in state['cases'].values()if c['active_shards']}
    assert active=={'AUG1_FULL_u000100','AUG2_LORA_u000200'}  # Same early scene budget first.
    assert all(sum(r['host']==host and gpu in r['gpus']for r in launches)<=2
               for host,gpus in q.SPARE_GPUS.items()for gpu in gpus)
    # Unknown or hidden allocations cannot become shareable just because free>=24GB.
    hidden=dict(index=0,uuid='x',free_MiB=74000,used_MiB=7000,owners=[])
    unknown=dict(index=1,uuid='y',free_MiB=60000,used_MiB=20000,owners=[dict(known=False,used_MiB=20000)])
    full=dict(index=2,uuid='z',free_MiB=30000,used_MiB=50000,
              owners=[dict(known=True,used_MiB=25000),dict(known=True,used_MiB=25000)])
    assert not q.available_slots('training-vlawm-zt2',[hidden,unknown,full],{})


def test_partial_exit0_nonprechild75_and_unknown_failure_never_retry(tmp_path,monkeypatch):
    r=route(tmp_path,'AUG1_FULL');checkpoint(tmp_path,r,100)
    state=q.initial_state();q.discover(tmp_path,state,{'AUG1_FULL':r});base=next(iter(state['cases'].values()))
    for code,prechild in [(0,None),(75,False),(-9,None)]:
        case=copy.deepcopy(base);job=tmp_path/f'job{code}_{prechild}'
        write(job/'RESULT.json',dict(returncode=code,child_not_started=prechild,end_unix=0))
        case['jobs']={'0':[dict(job=str(job),host='training-rl-zt2',gpus=[1])]}
        if code==0:write(Path(case['output'])/'rank0.json',dict(status='COMPLETE',rank=0,world=32,new=12,reused=0))
        assert not q.reconcile(case) and case['status']=='FAILED_REQUIRES_REVIEW'
        queue=q.initial_state();queue['cases'][case['id']]=case
        monkeypatch.setattr(q,'allowed_hosts',lambda **kw:[])
        q.tick(tmp_path,queue,{'AUG1_FULL':r})
        assert len(case['jobs']['0'])==1 and (job/'RESULT.json').is_file()


def test_only_true_prechild75_retries_after_cooldown_and_full32_completion_calls_aggregator(tmp_path,monkeypatch):
    r=route(tmp_path,'AUG1_FULL');checkpoint(tmp_path,r,100)
    state=q.initial_state();q.discover(tmp_path,state,{'AUG1_FULL':r});case=next(iter(state['cases'].values()))
    old=tmp_path/'prechild';write(old/'RESULT.json',dict(returncode=75,child_not_started=True,end_unix=1000))
    case['jobs']={'0':[dict(job=str(old),host='training-rl-zt2',gpus=[1])]}
    monkeypatch.setattr(q,'time',type('Clock',(),{'time':staticmethod(lambda:1200.)}))
    monkeypatch.setattr(q,'allowed_hosts',lambda **kw:['training-rl-zt2'])
    monkeypatch.setattr(q,'inventory',lambda *a:[dict(index=1,uuid='u',free_MiB=80000,used_MiB=0,owners=[])])
    calls=[]
    def launch(host,source,job,command,gpus,**kw):
        calls.append(command);return dict(host=host,job=str(job),gpus=gpus)
    monkeypatch.setattr(q,'launch',launch)
    q.tick(tmp_path,state,{'AUG1_FULL':r});assert not any(c[c.index('--rank')+1]=='0'for c in calls)
    # No live jobs remain in this second independent fixture queue.
    case['jobs']={'0':[dict(job=str(old),host='training-rl-zt2',gpus=[1])]};calls.clear()
    monkeypatch.setattr(q,'time',type('Clock',(),{'time':staticmethod(lambda:1400.)}))
    q.tick(tmp_path,state,{'AUG1_FULL':r});assert calls[0][calls[0].index('--rank')+1]=='0'
    assert read_result(old)['returncode']==75 and len(case['jobs']['0'])==2
    for rank in range(32):
        job=tmp_path/f'fullrank{rank}';write(job/'RESULT.json',dict(returncode=0))
        case['jobs'][str(rank)]=[dict(job=str(job),host='training-rl-zt2',gpus=[1])]
        write(Path(case['output'])/f'rank{rank}.json',dict(status='COMPLETE',rank=rank,world=32,
              new=len(range(rank,q.POPULATION,32)),reused=0))
    assert q.reconcile(case)
    aggregated=[]
    def summary(root,node):aggregated.append(node['id']);node['status']='COMPLETE'
    monkeypatch.setattr(q,'summarize_case',summary)
    monkeypatch.setattr(q,'allowed_hosts',lambda **kw:[])
    q.tick(tmp_path,state,{'AUG1_FULL':r})
    assert aggregated==[case['id']] and state['complete']is True


def read_result(job):
    return json.loads((job/'RESULT.json').read_text())

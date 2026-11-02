"""CPU-only same-spec completion-first F/G/H evaluation adoption."""
import argparse,json,os,time
from pathlib import Path
from . import eval_owner as E
from starVLA.rl.flow_grpo.transactions import atomic_json


def self_check():
    import tempfile
    from unittest.mock import patch
    from types import SimpleNamespace
    from . import native_priority
    with tempfile.TemporaryDirectory(prefix='same-spec-completion-')as temp:
        root=Path(temp);(root/'resources').mkdir()
        def entry(arm,u,n):
            return dict(label=arm+'_u'+str(u),arm=arm,updates=u,cohort='NAVTEST',world=32,evaluation=str(root/arm/str(u)),status='RUNNING',ranks={str(i):dict(status='COMPLETE')for i in range(n)})
        near=entry('REVIVAL_G16',20000,28);far=entry('REVIVAL_G16',48000,16);ended=entry('REVIVAL_SATURATED_G8',42000,31)
        entries=[far,ended,near];runs=[dict(id=a,parent_checkpoint='/same_IL70')for a in ('REVIVAL_G16','REVIVAL_SATURATED_G8')]
        state=dict(evaluations={x['label']:x for x in entries},runs={})
        for x in entries:state['runs'].setdefault(x['arm'],dict(gates={}))['gates'][str(x['updates'])]=dict(checkpoint=dict(path='/same_CP'),export=dict(status='COMPLETE',path='/same_export'))
        calls=[]
        def launch(run,node,command,family,route):
            node.update(job='/MOCK_ONLY/'+family,status='STARTING');calls.append(dict(arm=run['id'],command=command,family=family))
        owner=SimpleNamespace(state=state,registry=dict(runs=runs,evaluation=dict(data_paths=dict(NAVTEST='/same_data'),manifests=dict(NAVTEST='/same_manifest'))),claims_path=root/'resources/EVALUATION_CLAIMS.json',max_shards=32,max_workers=64,minimum=32768,stop_new=lambda:False,allow_new=lambda run:True,launch_job=launch)
        sample=dict(gpus=[dict(index=i,uuid='CPU_MOCK_'+str(i),free_MiB=80000)for i in range(4)],owners=[])
        with patch.object(E,'ROOT',root),patch.object(E,'usable_gpu',return_value=True),patch.object(native_priority,'filter_eval_routes',side_effect=lambda r,s:r):
            E.EvaluationOwner.dispatch_ranks(owner,[('CPU_ONLY_MOCK',sample)])
        assert len(calls)==4 and all(c['arm']=='REVIVAL_G16'and'/20000/'not in c['family']and'_u20000/'in c['family']for c in calls)
        assert [c['command'][c['command'].index('--rank')+1]for c in calls]==['28','29','30','31']
        assert all(c['command'][c['command'].index('--world')+1]=='32'and c['command'][c['command'].index('--seed')+1]=='2101'for c in calls)
        assert len(json.loads(owner.claims_path.read_text()))==4
        retention_old=entry('IMPROVEMENT_VERIFIED_RETENTION_G8',10000,11);retention_new=entry('IMPROVEMENT_VERIFIED_RETENTION_G8',12000,26)
        assert sorted([retention_new,retention_old],key=E.completion_priority)[0]is retention_old
        print(json.dumps(dict(status='PASS_CPU_ONLY_NO_DISPATCH',proof='28/32 currentG16 completes ranks28..31 before16/32 laterG16 or31/32 endedroute; claims persisted eachlaunch; allworld32/seed2101 commands unchanged',dispatches=0)))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--study',choices=['F','G','H']);p.add_argument('--registry');p.add_argument('--interval',type=float,default=30);p.add_argument('--self-check',action='store_true');a=p.parse_args()
    if a.self_check:self_check();return
    if a.study=='H':
        from .improvement_campaign import evaluation_class
        owner=evaluation_class()()
    elif a.study in ('F','G'):
        from .standard_evaluation import StandardEvaluationOwner
        owner=StandardEvaluationOwner(a.registry,max_shards=32,max_workers=32 if a.study=='G'else 64,minimum=32768)
    else:p.error('ExplicitF/G/H andregistry required')
    if Path(a.registry).resolve()!=owner.registry_path:p.error('Foreignregistry')
    path=Path(__file__).resolve();receipt=dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),source=str(path),source_sha256=E.file_sha(path),scheduling_source=str(Path(E.__file__).resolve()),scheduling_source_sha256=E.file_sha(E.__file__),started_unix=time.time(),registry=str(owner.registry_path),task=E.TASK,root=str(E.ROOT),adoption='same-speccompletion-first; allscientificsources/shardworlds/scenecaches unchanged')
    atomic_json(owner.directory/'OWNER.json',receipt)
    owner.state['operational_scheduler']=dict(schema='world_same_spec_completion_first_v1',priority='G16:completed-first/admissioncap32; Retention:oldest10kthen12/14k; allmissingranks before nextpoint',source_sha256=E.file_sha(E.__file__),no_training_or_gpu_worker_signals=True)
    owner.flush();owner.run(interval=a.interval)

if __name__=='__main__':main()

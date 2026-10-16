"""Finite endpoint lock and final single-ODE evaluation of the registered recipe.

The prepare command only verifies existing input/cache assets. Serve waits for
both actual 20k endpoints and all DEV stages, then evaluates the two endpoints
and SFT. It never selects a checkpoint from Navtest or restarts training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from starVLA.rl.simwam_recipe.contracts import ROOT,POPULATION,SFT_EXPORT,SFT_SHA256,file_sha,fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.analysis.simwam_recipe.controller_runtime import require_controller_host,require_worker_host,probe_routes,preflight_analysis_compatibility
from scripts.analysis.simwam_recipe.registered_followup import add_arguments,selected_variant,require_registration,matched_control

DATA=Path('/mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1/dataset')
DATALIST=Path('/mnt/project/DriveDreamer-Policy/test_meta.json')
CACHE=Path('/mnt/project/DriveDreamer-Policy-full-navtrain/runs/full_navtrain_epoch1/metric_cache_navtest_v1')
ROUTES={'R_SOURCE':['training-vlawm-tsinghua-zt'],
        'R_CANONICAL':['training-vlawm-tsinghua-zt2'],
        'SFT':['training-vlawm-tsinghua-zt3','training-vlawm-tsinghua-zt4']}


def allowed(host):
    return require_worker_host(host)


def prepare(output):
    pop=json.loads(Path(POPULATION).read_text());tokens=json.loads(DATALIST.read_text())
    if len(tokens)!=12146 or len(set(tokens))!=12146 or set(tokens)!=set(pop['navtest_tokens']):
        raise ValueError('Navtest input membership differs from fixed population')
    def check(token):
        log=pop['token_logs'][token]
        meta=DATA/'meta/test'/(token+'.pkl');cache=CACHE/log/'unknown'/token/'metric_cache.pkl'
        if not meta.is_file() or not cache.is_file():raise FileNotFoundError((str(meta),str(cache)))
        return (meta.stat().st_size,cache.stat().st_size)
    with ThreadPoolExecutor(max_workers=8) as pool:sizes=list(pool.map(check,tokens))
    if any(min(s)<=0 for s in sizes):raise ValueError('Empty input/cache')
    value=dict(status='EXISTING_ASSET_MEMBERSHIP_READY_NO_NEW_INFERENCE',population_identity=pop['identity_sha256'],
        scenes=len(tokens),logs=len({pop['token_logs'][t] for t in tokens}),tokens_sha256=fingerprint(pop['navtest_tokens']),
        data_root=str(DATA),navtest_datalist=str(DATALIST),navtest_datalist_sha256=file_sha(DATALIST),v1_cache=str(CACHE),
        metadata_bytes=sum(s[0] for s in sizes),metric_cache_bytes=sum(s[1] for s in sizes),new_candidates=0,
        limits='Existing assets and exact membership checked; individual payload/score validity remains enforced by the actual evaluation worker.',
        navtest_scores_read_for_selection=False)
    value['sha256']=fingerprint(value);atomic_json(output,value);return value


def make_lock(snapshot,queue,root=Path(ROOT),names=('R_SOURCE','R_CANONICAL'),include_sft=True):
    pop=json.loads(Path(POPULATION).read_text());assets=prepare(queue/'assets.json')
    models={}
    for name in names:
        run=root/'runs'/(name+'_seed42_v1');identity=json.loads((run/'identity.json').read_text())
        for rank in range(8):
            result=json.loads((run/f'result_rank{rank}.json').read_text())
            if result['status']!='COMPLETE' or result['updates']!=20000 or result['collections']!=5000 or result['new_candidates']!=320000:
                raise ValueError('Registered formal endpoint is incomplete')
        ck=run/'checkpoints/update020000_inner0';complete=json.loads((ck/'COMPLETE').read_text())
        if complete['updates']!=20000 or complete['next_inner']!=0 or file_sha(ck/'files.json')!=complete['files_sha256']:
            raise ValueError('Endpoint recovery identity changed')
        export=root/'exports'/(name+'_update020000');seal=json.loads((export/'COMPLETE').read_text())
        if seal['run_identity']!=identity['sha256'] or seal['updates']!=20000 or not seal['frozen_upstream_identical']:
            raise ValueError('Wrong endpoint export')
        sha=file_sha(export/'pytorch_model.pt')
        if sha!=seal['weight_sha256']:raise ValueError('Changed endpoint weight')
        models[name]=dict(export=str(export),weight_sha256=sha,run_identity=identity['sha256'],checkpoint=str(ck),
            checkpoint_files_sha256=complete['files_sha256'],updates=20000,config_sha256=file_sha(export/'config.yaml'))
    if file_sha(Path(SFT_EXPORT)/'pytorch_model.pt')!=SFT_SHA256:raise ValueError('SFT changed')
    if include_sft:models['SFT']=dict(export=SFT_EXPORT,weight_sha256=SFT_SHA256,updates=0,config_sha256=file_sha(Path(SFT_EXPORT)/'config.yaml'))
    value=dict(status='LOCKED_BEFORE_FINAL_NAVTEST',created_utc=datetime.now(timezone.utc).isoformat(),
        selection_rule='Registered fixed update20000 for named variants; no score-based selection',
        formal_budget_complete=True,models=models,weight_sha256=[v['weight_sha256'] for v in models.values()],
        population_identity=pop['identity_sha256'],sampler_sha256=file_sha(snapshot/'scripts/analysis/simwam_recipe/evaluate_policy.py'),
        inference_seeds=[2101],training_seeds=[42],modes=['E_NATIVE','E_SHIFT5'],
        data_root=assets['data_root'],navtest_datalist=assets['navtest_datalist'],v1_cache=assets['v1_cache'],
        asset_identity=assets['sha256'],historically_observed_Navtest_not_fully_blind=True)
    value['sha256']=fingerprint(value);atomic_json(queue/'LOCK.json',value);return value


def inventory(host):
    allowed(host)
    code="""import json,os,subprocess,time
def gpu():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return [dict(index=int(a),uuid=b.strip(),free_MiB=int(c),utilization=int(d)) for a,b,c,d in (line.split(',') for line in text.splitlines())]
first=gpu();time.sleep(2);second=gpu()
print(json.dumps(dict(first=first,second=second,CPU_affinity=len(os.sched_getaffinity(0)),load=os.getloadavg())))
"""
    return json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,
        shlex.join([sys.executable,'-c',code])],text=True))


def dispatch(snapshot,queue,selection,inventories,root=Path(ROOT),routes=ROUTES):
    for name,hosts in routes.items():
        model=selection['models'][name];shards=8*len(hosts);label=name+'_final20000' if name!='SFT' else 'SFT_final_matched'
        output=root/'evaluations'/(label+'_NAVTEST')
        for hi,host in enumerate(hosts):
            allowed(host)
            if len(inventories[host]['second'])!=8 or any(min(a['free_MiB'],b['free_MiB'])<14000 for a,b in zip(inventories[host]['first'],inventories[host]['second'])):
                raise RuntimeError('Insufficient measured free memory; preserve all owners: '+host)
            for gpu in range(8):
                rank=hi*8+gpu;record=queue/(label+f'_rank{rank}.json')
                payload=dict(snapshot=str(snapshot),output=str(output),export=model['export'],weight_sha256=model['weight_sha256'],
                    final_lock=str(queue/'LOCK.json'),rank=rank,shards=shards,GPU=gpu,host=host,python=sys.executable,receipt=str(record),
                    log=str(queue/(label+f'_rank{rank}.log')),uuid=inventories[host]['second'][gpu]['uuid'])
                code="""import json,os,subprocess,time
from pathlib import Path
p=json.loads(PAYLOAD);receipt=Path(p['receipt']);complete=Path(p['output'])/('rank'+str(p['rank'])+'.json')
if complete.exists() and json.loads(complete.read_text())['status']=='ASSIGNED_COMPLETE':
 print('COMPLETE',p['rank']);raise SystemExit(0)
if receipt.exists():
 old=json.loads(receipt.read_text());proc=Path('/proc')/str(old['pid'])
 if proc.exists() and (proc/'stat').read_text().split()[21]==old['start_ticks']:
  if p['output'] not in (proc/'cmdline').read_bytes().decode(errors='replace'):raise RuntimeError('PID identity mismatch')
  print('RUNNING',p['rank'],old['pid']);raise SystemExit(0)
 raise RuntimeError('Prior worker exited without completion; inspect failure, no automatic restart')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(p['GPU']),LOCAL_RANK='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1')
cmd=[p['python'],'-m','scripts.analysis.simwam_recipe.evaluate_policy','--export',p['export'],'--export-sha256',p['weight_sha256'],'--output',p['output'],'--cohort','NAVTEST','--shard',str(p['rank']),'--shards',str(p['shards']),'--final-lock',p['final_lock']]
with open(p['log'],'ab',buffering=0) as log:
 child=subprocess.Popen(cmd,cwd=p['snapshot'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
ticks=(Path('/proc')/str(child.pid)/'stat').read_text().split()[21]
receipt.write_text(json.dumps(dict(p,pid=child.pid,start_ticks=ticks,command=cmd,started=time.time()),indent=2)+'\\n')
print('STARTED',p['rank'],child.pid)
""".replace('PAYLOAD',repr(json.dumps(payload)))
                subprocess.run(['ssh','-o','BatchMode=yes',host,shlex.join([sys.executable,'-c',code])],check=True)


def audit_workers(queue):
    """Recognize only our recorded PID/start identities; never signal a worker."""
    by_host={}
    for path in sorted(queue.glob('*_rank*.json')):
        receipt=json.loads(path.read_text());by_host.setdefault(receipt['host'],[]).append(receipt)
    checks=[]
    for host,receipts in by_host.items():
        code="""import json
from pathlib import Path
rows=json.loads(PAYLOAD);result=[]
for r in rows:
 p=Path('/proc')/str(r['pid']);live=False
 try:live=p.exists() and (p/'stat').read_text().split()[21]==r['start_ticks'] and r['output'] in (p/'cmdline').read_bytes().decode(errors='replace')
 except FileNotFoundError:pass
 f=Path(r['output'])/('rank'+str(r['rank'])+'.json');state=json.loads(f.read_text()) if f.exists() else {}
 result.append(dict(host=r['host'],rank=r['rank'],output=r['output'],live=live,status=state.get('status'),pid=r['pid']))
print(json.dumps(result))
""".replace('PAYLOAD',repr(json.dumps(receipts)))
        checks+=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,
            shlex.join([sys.executable,'-c',code])],text=True))
    atomic_json(queue/'worker_status.json',dict(time=time.time(),workers=checks))
    for row in checks:
        if not row['live'] and row['status']!='ASSIGNED_COMPLETE':
            raise RuntimeError('Owned evaluation worker exited without a complete denominator; inspect its log: '+str(row))
    return checks


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('command',choices=['prepare','serve']);p.add_argument('--snapshot')
    p.add_argument('--output');p.add_argument('--once',action='store_true')
    p.add_argument('--administrative-local',action='store_true');add_arguments(p)
    p.add_argument('--source-compatibility')
    p.add_argument('--followup-hosts','--x1-hosts',dest='followup_hosts',nargs='+',default=['training-vlawm-tsinghua-zt3','training-vlawm-tsinghua-zt4'])
    a=p.parse_args();require_controller_host(a.administrative_local)
    variant=selected_variant(a)
    root=require_registration(variant) if variant else Path(ROOT)
    if len(a.followup_hosts)!=len(set(a.followup_hosts)):raise ValueError('Duplicate follow-up hosts')
    routes={variant:a.followup_hosts} if variant else ROUTES
    names=(variant,) if variant else ('R_SOURCE','R_CANONICAL')
    queue=root/'final_navtest_queue';queue.mkdir(exist_ok=True)
    if a.command=='prepare':
        result=prepare(Path(a.output) if a.output else queue/'assets.json');print(json.dumps(result));return
    if not a.snapshot:raise ValueError('Frozen evaluation source is required')
    snapshot=Path(a.snapshot);guard=(queue/'controller.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    atomic_json(queue/'transport_preflight.json',dict(time=time.time(),pid=os.getpid(),administrative_local=a.administrative_local,
        routes=probe_routes([h for hosts in routes.values() for h in hosts],sys.executable),
        analysis_compatibility=preflight_analysis_compatibility(snapshot,a.source_compatibility)))
    last_probe=0
    try:
        while True:
            require_controller_host(a.administrative_local);stages=root/'milestone_queue/progress.json'
            stage=json.loads(stages.read_text()) if stages.exists() else {}
            results=[root/'runs'/(name+'_seed42_v1')/f'result_rank{rank}.json'
                     for name in names for rank in range(8)]
            if stage.get('status')!='COMPLETE' or not all(f.exists() for f in results):
                atomic_json(queue/'progress.json',dict(status='WAITING_FOR_REGISTERED_ENDPOINTS_AND_DEV_STAGES',pid=os.getpid(),time=time.time(),Navtest_opened=False))
                if a.once:return
                time.sleep(20);continue
            lock=queue/'LOCK.json'
            if lock.exists():
                selection=json.loads(lock.read_text());copy=dict(selection);sha=copy.pop('sha256')
                if fingerprint(copy)!=sha:raise ValueError('Final lock modified')
            else:selection=make_lock(snapshot,queue,root,names,include_sft=not variant)
            if not (queue/'DISPATCHED.json').exists():
                hosts=sorted({h for hs in routes.values() for h in hs})
                with ThreadPoolExecutor(max_workers=4) as pool:inventories=dict(zip(hosts,pool.map(inventory,hosts)))
                uuids=[g['uuid'] for info in inventories.values() for g in info['second']]
                if len(uuids)!=len(set(uuids)):raise ValueError('Hosts alias the same physical GPUs')
                atomic_json(queue/'inventory.json',dict(time=time.time(),hosts=inventories,distinct_GPUs=len(uuids),no_evictions=True))
                dispatch(snapshot,queue,selection,inventories,root,routes)
                atomic_json(queue/'DISPATCHED.json',dict(lock=selection['sha256'],routes=routes,time=time.time()))
            if time.monotonic()-last_probe>=60:
                audit_workers(queue);last_probe=time.monotonic()
            complete=[]
            for name,hosts in routes.items():
                label=name+'_final20000' if name!='SFT' else 'SFT_final_matched'
                output=root/'evaluations'/(label+'_NAVTEST')
                files=[output/f'rank{rank}.json' for rank in range(8*len(hosts))]
                if all(f.exists() and json.loads(f.read_text())['status']=='ASSIGNED_COMPLETE' for f in files):complete.append(name)
            atomic_json(queue/'progress.json',dict(status='FINAL_SCORING' if len(complete)<len(routes) else 'ANALYZING',
                completed_models=complete,Navtest_opened=True,pid=os.getpid(),time=time.time()))
            if len(complete)==len(routes):
                workers=audit_workers(queue)
                if any(w['live'] for w in workers):
                    if a.once:return
                    time.sleep(5);continue
                for name in (names if variant else ['SFT','R_SOURCE','R_CANONICAL']):
                    label=name+'_final20000' if name!='SFT' else 'SFT_final_matched'
                    output=root/'evaluations'/(label+'_NAVTEST');summary=output/'summary.json'
                    if summary.exists():continue
                    cmd=[sys.executable,'-m','scripts.analysis.simwam_recipe.analyze_evaluation','--evaluation',str(output),'--output',str(summary)]
                    if name!='SFT':cmd+=['--reference',str(Path(ROOT)/'evaluations/SFT_final_matched_NAVTEST')]
                    if a.source_compatibility:cmd+=['--source-compatibility',a.source_compatibility]
                    subprocess.run(cmd,cwd=snapshot,check=True)
                if variant:
                    output=root/'evaluations'/(variant+'_final20000_NAVTEST')
                    control_name,control_root=matched_control(variant)
                    control=output/('versus_'+control_name+'.json')
                    if not control.exists():
                        command=[sys.executable,'-m','scripts.analysis.simwam_recipe.analyze_evaluation',
                            '--evaluation',str(output),'--reference',str(control_root/'evaluations'/(control_name+'_final20000_NAVTEST')),
                            '--output',str(control)]
                        if a.source_compatibility:command+=['--source-compatibility',a.source_compatibility]
                        subprocess.run(command,cwd=snapshot,check=True)
                atomic_json(queue/'COMPLETE.json',dict(status='REGISTERED_MODELS_COMPLETE_PAIRED_NAVTEST',scenes_per_model=12146,
                    modes=['E_NATIVE','E_SHIFT5'],inference_seeds=[2101],lock=selection['sha256'],time=time.time()))
                atomic_json(queue/'progress.json',dict(status='COMPLETE',completed_models=complete,
                    Navtest_opened=True,all_owned_workers_exited=True,pid=os.getpid(),time=time.time()))
                return
            if a.once:return
            time.sleep(20)
    except Exception:
        atomic_json(queue/'failure.json',dict(error=traceback.format_exc(),pid=os.getpid(),time=time.time()))
        raise


if __name__=='__main__':main()

"""Bounded user-pinned A/B/C/D Test observations using the native evaluator."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import traceback

from tools.qwen_world_postcot.user_interim_owner import Request, read, write, fingerprint, sha
from starVLA.rl.qwen_world_compare import resources
from starVLA.rl.qwen_world_compare.interim_navtest import SCHEMA

LOCK=threading.RLock()
RESERVED={}


def inventory():
    samples=[]
    from starVLA.rl.qwen_world_compare.contracts import allowed_hosts
    hosts=allowed_hosts()
    def get(h):
        try:return h,resources.probe(h),None
        except Exception as e:return h,None,str(e)
    errors={}
    for _ in range(2):
        with ThreadPoolExecutor(max_workers=max(1,len(hosts)))as pool:rows=list(pool.map(get,hosts))
        samples.append({h:r for h,r,e in rows if e is None});errors.update({h:e for h,r,e in rows if e})
    routes=[];seen=set()
    for h,b in samples[1].items():
        if h not in samples[0]:continue
        a=samples[0][h]
        if [g['uuid']for g in a['gpus']]!=[g['uuid']for g in b['gpus']]:raise ValueError('Changed physical GPU mapping')
        for x,y in zip(a['gpus'],b['gpus']):
            owners=[o for s in (a,b)for o in s['owners']if o['uuid']==y['uuid']]
            if (y['uuid']in seen or min(x['free_MiB'],y['free_MiB'])<32768
                    or any(not o['visible']for o in owners)
                    or resources.gpu_activity_issue(x,a['owners'])or resources.gpu_activity_issue(y,b['owners'])):continue
            seen.add(y['uuid']);routes.append(dict(host=h,gpu=y['index'],uuid=y['uuid'],free_MiB=min(x['free_MiB'],y['free_MiB'])))
        b['available_memory_bytes']=b['MemAvailable_bytes']
    return routes,dict(first=samples[0],second=samples[1],errors=errors)


class InterimRequest(Request):
    def __init__(self,root,pin):
        self.resources=resources;self.root=root;self.pin=pin;self.source=Path(__file__).resolve().parents[2]
        self.source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.source,text=True).strip()
        self.export=root/'export';self.output=root/'evaluation';self.reg=root/'registration.json'
        self.proof=root/'native_export_parity.json';self.jobs={};self.attempts={};self.world=8

    def done(self,name):
        if name not in self.jobs:return None
        path=Path(self.jobs[name]['receipt_dir'])/'RESULT.json'
        if not path.exists():return None
        result=read(path)
        with LOCK:
            for u,v in list(RESERVED.items()):
                if v==(str(self.root),name):RESERVED.pop(u)
        return result.get('returncode')==0 and result.get('status')=='EXITED'

    def route(self,routes,inv,*,parity=False):
        with LOCK:
            from starVLA.rl.qwen_world_compare.contracts import authorization
            grant,_=authorization()
            # Finite temporary hosts use the idle primary slot. The worker's
            # native deadline guard then owns checkpoint/exit before expiry.
            routes=[r for r in routes if r['host']not in grant.get('temporary_hosts',[])or
                    not any(o['uuid']==r['uuid']for o in inv['second'][r['host']]['owners'])]
            if not parity:
                if sum(v[1]!='parity'for v in RESERVED.values())>=16:return None
                campaign=read(self.root.parent/'PINNED.json')
                # Keep one sufficiently large device for the short native/export
                # checks, then the two legacy-family checks/rollouts. Other
                # GPUs remain fully available to the registered full workers.
                if not Path(campaign['legacy_queue_complete']).exists():
                    routes=[r for r in routes if r['uuid']!=campaign['parity_gpu_reserved_uuid']]
            route=super().route([r for r in routes if r['uuid']not in RESERVED],inv,parity=parity)
            if route:RESERVED[route['uuid']]=(str(self.root),None)
            return route

    def submit(self,name,module,args,route=None):
        number=self.attempts.get(name,0)+1;self.attempts[name]=number
        directory=self.root/'jobs'/f'{name}_attempt{number:03d}';directory.mkdir(parents=True,exist_ok=False)
        host=route['host']if route else socket.gethostname().removesuffix('-worker-0')
        module='tools.qwen_world_compare.export'if name=='export'else'tools.qwen_world_compare.evaluate'
        command=[resources.MODEL_PYTHON,'-u','-m',module,*args]
        spec=dict(host=host,command=command,source=str(self.source),source_sha=self.source_sha,
                  gpu_uuids=[route['uuid']]if route else[],receipt_dir=str(directory))
        self.jobs[name]=spec;write(directory/'spec.json',spec)
        invocation=['/usr/bin/python3','-u','-m','tools.qwen_world_compare.job_worker',
            '--receipt-root',str(directory),'--command-json',json.dumps(command),'--cwd',str(self.source)]
        if route:
            with LOCK:RESERVED[route['uuid']]=(str(self.root),name)
            guard=49152 if name=='parity' else 32768
            invocation+=['--gpus',str(route['gpu']),'--expected-gpu-uuids-json',json.dumps([route['uuid']]),'--minimum-free-mib',str(guard)]
            from starVLA.rl.qwen_world_compare.contracts import authorization
            grant,_=authorization()
            if route['host']in grant.get('temporary_hosts',[]):invocation+=['--lease-slot','primary']
            else:invocation+=['--share-min-free-mib',str(guard)]
        code='''import subprocess,os,json
from pathlib import Path
s=__SPEC__;d=Path(s['directory']);f=open(d/'transport.log','ab',buffering=0)
p=subprocess.Popen(s['invocation'],cwd=s['source'],env=dict(os.environ,PYTHONPATH=s['source']),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,start_ticks=int((Path('/proc')/str(p.pid)/'stat').read_text().split()[21]))))'''.replace('__SPEC__',repr(dict(invocation=invocation,directory=str(directory),source=str(self.source))))
        write(directory/'operator.json',json.loads(resources.remote(host,['/usr/bin/python3','-c',code])))

    def register(self):
        meta=read(self.export/'export.json');parent=meta['IL_parent']
        if (meta['architecture_id']!=self.pin['architecture_id']or parent['seal']['updates']!=self.pin['updates']
                or parent['seal']['files_sha256']!=self.pin['files_sha256']or parent['path']!=self.pin['checkpoint']):
            raise ValueError('Export is not the pinned architecture/native checkpoint')
        data=Path('/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST')
        protocol=read('/mnt/project/qwen-world-postcot-20261025/evaluations/DEV_IL_update035000/identity.json')['protocol']
        row=dict(schema=SCHEMA,user_instruction='评测最近的ckpts',architecture_id=meta['architecture_id'],
            model_sha256=meta['model_sha256'],native_checkpoint=parent['path'],native_model_sha256=parent['model_sha256'],
            native_files_sha256=parent['seal']['files_sha256'],updates=parent['seal']['updates'],
            export=str(self.export),export_sha256=sha(self.export/'export.json'),data_sha256=sha(data/'manifest.json'),
            protocol_sha256=protocol['sha256'],seed=2101,count=12146,inference='single_candidate_cleanx_K10_shift5',
            Test_observed=True,used_for_parent_selection=False,used_for_actor_update=False)
        row['sha256']=fingerprint(row);write(self.reg,row)

    def run(self):
        self.submit('export','', ['--checkpoint',self.pin['checkpoint'],'--output',str(self.export)])
        while not all(self.done('rank'+str(i))is True for i in range(self.world)):
            for name in list(self.jobs):
                if self.done(name)is False:
                    result=read(Path(self.jobs[name]['receipt_dir'])/'RESULT.json')
                    if result.get('status')=='REFUSED_RESOURCE'and result.get('child_not_started')and self.attempts[name]<4:
                        self.jobs.pop(name);continue
                    raise RuntimeError('Worker failed; evidence retained: '+str(self.jobs[name]['receipt_dir']))
            if self.done('export')is True:
                if not self.reg.exists():self.register()
                parity=(self.proof.exists()and read(self.proof).get('status')=='PASS'and self.done('parity')is True)
                if parity:
                    campaign=read(self.root.parent/'PINNED.json');proofs=[]
                    for label in campaign['models']:
                        d=self.root.parent/label;proof=d/'native_export_parity.json'
                        exits=list((d/'jobs').glob('parity_attempt*/RESULT.json'))
                        if proof.exists()and read(proof).get('status')=='PASS'and any(read(p).get('status')=='EXITED'and read(p).get('returncode')==0 for p in exits):proofs.append(label)
                    if len(proofs)==len(campaign['models']):
                        with LOCK:
                            marker=self.root.parent/'ALL_NATIVE_PARITY_PASS.json'
                            if not marker.exists():write(marker,dict(status='PASS',models=proofs,all_parity_workers_exited=True))
                for rank in range(self.world)if parity else[0]:
                    name='rank'+str(rank)if parity else'parity'
                    if name in self.jobs:continue
                    routes,inv=inventory();write(self.root/'inventory_latest.json',inv)
                    route=self.route(routes,inv,parity=not parity)
                    if route is None:break
                    args=['--export',str(self.export),'--output',str(self.output),'--cohort','NAVTEST',
                          '--data','/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST','--rank',str(rank),
                          '--world',str(self.world),'--seed','2101','--user-interim-registration',str(self.reg)]
                    if rank==0:args+=['--native-parity-output',str(self.proof)]
                    if not parity:args+=['--parity-only']
                    self.submit(name,'',args,route)
            time.sleep(20)
        with(self.root/'summary.log').open('x')as log:
            subprocess.run([resources.MODEL_PYTHON,'-m','tools.qwen_world_compare.evaluate','--summarize',
                '--evaluation',str(self.output),'--output',str(self.root/'SUMMARY.json')],cwd=self.source,stdout=log,stderr=subprocess.STDOUT,check=True)
        summary=read(self.root/'SUMMARY.json')
        if summary['scenes']!=12146 or summary['invalid_scenes']or summary['qualification']:raise ValueError('Incomplete full Test')
        write(self.root/'COMPLETE.json',dict(status='COMPLETE',PDMS=summary['PDMS'],scenes=12146,all_owned_workers_exited=True))
        return summary


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True);p.add_argument('--report-dir',required=True);a=p.parse_args()
    root=Path(a.root);lock=(root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if(root/'owner.json').exists():raise FileExistsError('Owner already registered')
    write(root/'owner.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),at=time.time()))
    requests={}
    for label,pin in read(root/'PINNED.json')['models'].items():
        d=root/label;d.mkdir(exist_ok=False);requests[label]=InterimRequest(d,pin)
    completed={};failed={}
    with ThreadPoolExecutor(max_workers=min(6,len(requests)))as pool:
        futures={pool.submit(v.run):k for k,v in requests.items()}
        for f in as_completed(futures):
            label=futures[f]
            try:completed[label]=f.result()
            except Exception:failed[label]=traceback.format_exc()
            write(root/'status.json',dict(completed=completed,failed=failed,at=time.time()))
    if failed:write(root/'failure.json',dict(failed=failed));return
    report=Path(a.report_dir).resolve();report.mkdir(parents=True,exist_ok=True)
    write(report/'results.json',completed)
    (report/'README.md').write_text('# User-requested early four-framework Navtest\n\nFull12146, canonicalv1.1, single clean-x K10shift5, seed2101. Budgets differ: no architecture ranking or parent selection.\n\n| Model | PDMS |\n|---|---:|\n'+'\n'.join(f'| {k} | {v["PDMS"]:.6f} |'for k,v in completed.items())+'\n')
    source=Path(__file__).resolve().parents[2];prefix=str(report.relative_to(source))
    subprocess.run(['git','add','--',prefix],cwd=source,check=True);subprocess.run(['git','commit','-m','Publish full early four-framework Navtest observations'],cwd=source,check=True)
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=source,text=True).strip()
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=source,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=source,text=True).split()[0]
    if head!=remote:raise ValueError('Remote source SHA mismatch')
    write(root/'COMPLETE.json',dict(status='COMPLETE',models=list(completed),remote_sha=remote,all_owned_workers_exited=True))


if __name__=='__main__':main()

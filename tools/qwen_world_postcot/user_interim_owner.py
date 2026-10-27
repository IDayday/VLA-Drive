"""Finite IL37k Navtest request; reuses native export/predict/scoring and workers."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fingerprint(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,path)


class Request:
    def __init__(self,root):
        from starVLA.rl.qwen_sourceflow import resources
        self.resources=resources;self.root=root;self.source=Path(__file__).resolve().parents[2]
        self.source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.source,text=True).strip()
        self.pin=read(root/'PINNED.json');self.export=root/'export';self.output=root/'evaluation';self.jobs={}
        self.reg=root/'registration.json';self.proof=root/'native_export_parity.json'

    def done(self,name):
        path=self.root/'jobs'/name/'result.json'
        if not path.exists():return None
        row=read(path);return row.get('status')=='PROCESS_COMPLETE' and row.get('exit_code')==0

    def active(self):return [j for name,j in self.jobs.items() if self.done(name) is None]

    def submit(self,name,module,args,route=None):
        if module not in ['tools.qwen_world_postcot.export','tools.qwen_world_postcot.evaluate']:
            raise PermissionError('This owner cannot start training or another module')
        directory=self.root/'jobs'/name;directory.mkdir(parents=True,exist_ok=False)
        spec=dict(source=str(self.source),source_sha=self.source_sha,
            host=route['host'] if route else socket.gethostname().removesuffix('-worker-0'),
            gpus=[route['gpu']] if route else [],gpu_uuids=[route['uuid']] if route else [],
            min_free_MiB=49152 if name=='rank0' else 32768 if route else 0,
            world_shared_lease=bool(route),
            module=module,command=[self.resources.MODEL_PYTHON,'-u','-m',module,*args],receipt_dir=str(directory))
        write(directory/'spec.json',spec);self.jobs[name]=spec
        write(directory/'operator.json',self.resources.launch(directory/'spec.json',str(self.source)))

    def register(self):
        meta=read(self.export/'export.json');seal=read(self.export/'COMPLETE')
        if seal['export_sha256']!=sha(self.export/'export.json') or meta['IL_parent']['seal']['updates']!=37000:
            raise ValueError('Unsealed/mismatched export')
        data=Path('/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST')
        manifest=read(data/'manifest.json');complete=read(data/'COMPLETE')
        if complete['manifest_sha256']!=sha(data/'manifest.json') or len(manifest['tokens'])!=12146:
            raise ValueError('Changed full input population')
        population=Path('/mnt/project/flow-safe-improvement-20261013/population/manifest.json')
        if manifest['tokens']!=read(population)['navtest_tokens']:raise ValueError('Wrong Navtest membership/order')
        protocol=read('/mnt/project/qwen-world-postcot-20261025/evaluations/DEV_IL_update035000/identity.json')['protocol']
        registration=dict(schema='world_postcot_user_interim_NAVTEST_v1',checkpoint=self.pin['checkpoint'],
            checkpoint_files_sha256=self.pin['files_sha256'],native_model_sha256=self.pin['native_model_sha256'],
            model_sha256=meta['model_sha256'],updates=37000,output=str(self.output),seed=2101,count=12146,
            data_sha256=complete['manifest_sha256'],token_sha256=fingerprint(manifest['tokens']),
            population_sha256=sha(population),protocol_sha256=protocol['sha256'],prediction_type='x0',
            inference='single_candidate_cleanx_K10_shift5',used_for_parent_selection=False,
            used_for_training_or_tuning=False,user_instruction=self.pin['user_instruction'],
            standing_instruction=self.pin['standing_instruction'],source_sha=self.source_sha)
        registration['sha256']=fingerprint(registration);write(self.reg,registration)

    def route(self,routes,inventory,*,parity=False):
        active=self.active();used={u for j in active for u in j['gpu_uuids']}
        for r in sorted(routes,key=lambda x:-x['free_MiB']):
            if r['uuid'] in used or r['free_MiB']<(49152 if parity else 32768):continue
            host=inventory['second'][r['host']]
            if host['available_memory_bytes']<128*1024**3:continue
            loading=0
            for name,j in self.jobs.items():
                if j['host']!=r['host'] or self.done(name) is not None:continue
                status=self.output/(name+'.json')
                if not status.exists() or read(status).get('new',0)<1:loading+=1
            if loading>=2:continue
            lease_path=Path('/mnt/project/runtime_gpu_shared_workers')/(r['uuid']+'.lock')
            lease_path.parent.mkdir(exist_ok=True)
            with lease_path.open('a') as lease:
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
            # Positively identify current cohosts; do not treat invisible/unknown
            # owners as idle or evict any process.
            program=r'''import json,subprocess
from pathlib import Path
uuid=__UUID__;rows=[]
for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines():
 p,u=line.split(',')
 if u.strip()!=uuid:continue
 path=Path('/proc')/p.strip();args=[v for v in (path/'cmdline').read_bytes().decode().split('\0')if v]
 prefixes=['/mnt/project/qwen-','/mnt/project/VLA-Drive-qwen','/mnt/project/VLA-AD/experiments/recogdrive','/mnt/project/recogdrive']
 known=any(a.startswith(prefix) for a in args for prefix in prefixes)
 rows.append(dict(pid=int(p),start_ticks=int((path/'stat').read_text().split()[21]),known_project=known))
print(json.dumps(rows))'''.replace('__UUID__',repr(r['uuid']))
            try:owners=json.loads(self.resources.remote(r['host'],['/usr/bin/python3','-c',program]))
            except Exception:continue
            if any(not o['known_project'] for o in owners):continue
            write(self.root/'admission'/f'{r["uuid"]}.json',dict(at=time.time(),route=r,owners=owners,no_eviction=True))
            return r
        return None

    def run(self):
        self.submit('export','tools.qwen_world_postcot.export',['--checkpoint',self.pin['checkpoint'],'--output',str(self.export)])
        while not all(self.done('rank'+str(i)) is True for i in range(8)):
            if any(self.done(name) is False for name in self.jobs):
                while self.active(): time.sleep(15)
                raise RuntimeError('Native worker failed; other owned workers drained; no automatic retry')
            if self.done('export') is True:
                if not self.reg.exists():self.register()
                parity_ready=self.proof.exists() and read(self.proof)['status']=='PASS'
                ranks=range(8) if parity_ready else [0]
                for rank in ranks:
                    name='rank'+str(rank)
                    if name in self.jobs:continue
                    routes,inventory=self.resources.available(min_free_MiB=32768)
                    write(self.root/'inventory_latest.json',inventory)
                    route=self.route(routes,inventory,parity=rank==0)
                    if route is None:break
                    args=['--export',str(self.export),'--output',str(self.output),'--cohort','NAVTEST',
                          '--data','/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST',
                          '--seed','2101','--rank',str(rank),'--world','8','--user-interim-registration',str(self.reg)]
                    if rank==0:args+=['--native-parity-output',str(self.proof)]
                    self.submit(name,'tools.qwen_world_postcot.evaluate',args,route)
            time.sleep(15)  # Necessary compute admission/completion orchestration.
        summary=self.root/'SUMMARY.json'
        with (self.root/'summary.log').open('x') as log:
            subprocess.run([self.resources.MODEL_PYTHON,'-m','tools.qwen_world_postcot.summarize',
                '--evaluation',str(self.output),'--output',str(summary)],cwd=self.source,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        s=read(summary)
        if s['status']!='COMPLETE' or s['scenes']!=12146 or s['physical_logs']!=136:
            raise ValueError('Not a complete valid Navtest population')
        write(self.root/'COMPLETE.json',dict(status='COMPLETE',PDMS=s['PDMS'],scenes=12146,logs=136,
            all_owned_workers_exited=True,registration_sha256=read(self.reg)['sha256'],at=time.time()))
        self.publish(s)

    def publish(self,s):
        report=self.source/'reports/qwen_world_IL37_navtest_20261027';report.mkdir(exist_ok=True,parents=True)
        write(report/'summary.json',s);write(report/'registration.json',read(self.reg));write(report/'native_parity.json',read(self.proof))
        (report/'README.md').write_text('# Qwen-World/PostCoT IL37k Navtest\n\n'
            f'Full12146 scenes/136 logs, canonicalv1.1, single clean-x K10 shift5, seed2101. **PDMS {s["PDMS"]:.6f}**.\n\n'
            'This is user-requested interim Test observation, not completed100k IL, RL gain, DEV selection or RL-parent promotion. '
            'The current-only native predict and independent official scorer are unchanged. Native/export actual prediction parity passed. '
            'Original training/controller and all other real workloads continue unchanged. No auxiliary teacher, candidate bank, scorer or GT selects the deployed output.\n\n'
            'Exact protocol, component means and complete-scene checks are in summary.json; registration binds native/export and data/source hashes. '
            'Owned worker timing, loading, forward, scoring and native-parity costs are retained under the asset-root job/rank receipts.\n')
        def git(*args):return subprocess.check_output(['git',*args],cwd=self.source,text=True)
        if git('rev-parse','HEAD').strip()!=self.source_sha:raise ValueError('Source changed before publication')
        prefix=str(report.relative_to(self.source))
        for entry in git('status','--porcelain=v1','-z','--untracked-files=all').split('\0'):
            if entry and not entry[3:].startswith(prefix+'/'):raise ValueError('Unrelated report change')
        subprocess.run(['git','add','--',prefix],cwd=self.source,check=True)
        subprocess.run(['git','commit','-m','Record full user-requested World PostCoT IL37k Navtest'],cwd=self.source,check=True)
        branch=git('branch','--show-current').strip();commit=git('rev-parse','HEAD').strip()
        subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=self.source,check=True)
        remote=git('ls-remote','origin','refs/heads/'+branch).split()[0]
        if remote!=commit:raise ValueError('Remote SHA mismatch')
        write(self.root/'PUBLICATION.json',dict(status='PUBLISHED',commit=commit,remote_SHA=remote,at=time.time()))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    lock=(root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if(root/'owner.json').exists():raise FileExistsError('Never duplicate an existing owner')
    request=Request(root);write(root/'owner.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),at=time.time(),source_sha=request.source_sha))
    try:request.run()
    except BaseException:
        write(root/'failure.json',dict(at=time.time(),error=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':main()

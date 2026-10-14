"""One bounded node-local tmpfs copy of immutable spatial records, shared by heads.

Files are copied byte for byte and SHA256-verified. The shared project dataset
and scientific identities are untouched. Never modifies system page-cache or
Lustre settings; stops admission at the host RAM safety reserve.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
from iqe.contracts import require
from iqe.io import atomic_json,digest,file_hash,lock,read_json
from .context_reader import version


def stage_root(config):
    base=read_json(config.root/'BASE_BINDING.json')['checkpoint_hash']
    return Path('/dev/shm')/('iqe-study-spatial-'+base[:16]+'-'+config.identity[:16])


def available_bytes():
    values=dict(s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemAvailable'].split()[0])*1024


def copy_verified(source,target,expected_checksum):
    before=version(source)
    if target.exists():
        require(file_hash(target)==expected_checksum,'existing temporary context corrupted')
        return
    target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd,name=tempfile.mkstemp(prefix='.'+target.name+'.',dir=target.parent)
    try:
        h=hashlib.sha256()
        with os.fdopen(fd,'wb') as output,open(source,'rb') as stream:
            while chunk:=stream.read(8*1024**2):h.update(chunk);output.write(chunk)
            output.flush();os.fsync(output.fileno())
        require(version(source)==before and h.hexdigest()==expected_checksum,'source changed while staging context')
        require(file_hash(name)==expected_checksum,'temporary context copy changed bytes')
        os.replace(name,target)
    finally:
        if Path(name).exists():Path(name).unlink()


def build(config_path,workers=8,max_gib=224,min_available_gib=180,rate_mib=512):
    from .config import load_config
    from .binding import scenes_by_role
    from .resource_guard import check
    from .context import spatial_key
    require(1<=workers<=8 and 0<max_gib<=224 and min_available_gib>=128 and 0<rate_mib<=512,
            'RAM stage resource limits')
    config=load_config(config_path);check(config,'stage-immutable-spatial-RAM')
    root=stage_root(config);root.mkdir(parents=True,exist_ok=True,mode=0o700)
    require(any(row.split()[1]=='/dev/shm' and row.split()[2]=='tmpfs' for row in Path('/proc/mounts').read_text().splitlines()),
            'RAM staging must be actual tmpfs')
    out=config.root/'runtime/spatial_RAM_stage'/socket.gethostname();started=time.monotonic()
    with lock(root/'BUILD.lock'):
        scenes=scenes_by_role(config)['scorer_fit']
        size=sum((config.root/'spatial_context'/(s.scene_id+'.pt')).stat().st_size for s in scenes)
        require(size<=max_gib*1024**3,'fixed D_S exceeds RAM stage budget')
        free=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
        existing=sum(p.stat().st_size for p in root.glob('*.pt'))
        require(free>=size-existing+1024**3,'insufficient tmpfs capacity')
        cancelled=threading.Event();limit=threading.Lock();schedule=[time.monotonic()]
        def one(scene):
            if cancelled.is_set():return None
            source=config.root/'spatial_context'/(scene.scene_id+'.pt');target=root/source.name
            record=read_json(str(source)+'.COMPLETE.json')
            require(record['key']==spatial_key(config,scene),'RAM stage observation/source identity')
            if not target.exists() and available_bytes()<min_available_gib*1024**3:
                cancelled.set();return None
            source_version=version(source);receipt_version=version(str(source)+'.COMPLETE.json')
            copy_verified(source,target,record['checksum'])
            atomic_json(str(target)+'.COMPLETE.json',record,immutable=True)
            with limit:
                now=time.monotonic();at=max(now,schedule[0]);schedule[0]=at+source.stat().st_size/(rate_mib*1024**2)
            if at>now:time.sleep(at-now)
            return {'scene_id':scene.scene_id,'checksum':record['checksum'],'source_version':list(source_version),
                    'source_receipt_version':list(receipt_version),'bytes':source.stat().st_size}
        records=[]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for at in range(0,len(scenes),64):
                if cancelled.is_set():break
                records.extend(x for x in executor.map(one,scenes[at:at+64]) if x is not None)
                atomic_json(out/'STATUS.json',{'status':'COPYING_VERIFIED_BYTES','pid':os.getpid(),'root':str(root),
                     'records':len(records),'requested':len(scenes),'bytes':sum(x['bytes'] for x in records),
                     'MemAvailable':available_bytes(),'safety_reserve_GiB':min_available_gib,'seconds':time.monotonic()-started})
        status='PARTIAL_MEMORY_GUARD' if cancelled.is_set() else 'COMPLETE'
        result={'schema':'readonly_spatial_tmpfs_v1','status':status,'host':socket.gethostname(),'root':str(root),
                'config_identity':config.identity,'Base_binding_hash':file_hash(config.root/'BASE_BINDING.json'),
                'source_scene_hash':digest([s.scene_id for s in scenes]),'requested':len(scenes),
                'records':{x['scene_id']:x for x in records},'total_bytes':sum(x['bytes'] for x in records),
                'max_GiB':max_gib,'safety_reserve_GiB':min_available_gib,'source_role':'scorer_fit',
                'numeric_or_sample_change':False,'persistent_storage_bytes_added':0}
        if status=='COMPLETE':atomic_json(root/'MANIFEST.json',result,immutable=True)
        atomic_json(out/f'RESULT_{os.getpid()}.json',result,immutable=True)
        atomic_json(out/'STATUS.json',{k:v for k,v in result.items() if k!='records'}|{'records':len(records),'pid':os.getpid(),'seconds':time.monotonic()-started})
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--max-gib',type=int,default=224)
    p.add_argument('--min-available-gib',type=int,default=180);p.add_argument('--rate-mib',type=int,default=512)
    a=p.parse_args();r=build(a.config,a.workers,a.max_gib,a.min_available_gib,a.rate_mib)
    print({k:v for k,v in r.items() if k!='records'},flush=True)


if __name__=='__main__':main()

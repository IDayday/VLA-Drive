"""Bounded read-only page-cache warming of existing verified FP32 contexts.

No tensor rewrite, new cache, sampler consumption, model or GPU call. This only
brings original immutable file pages onto the training host before random use.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import socket
import threading
import time
from iqe.contracts import require
from iqe.io import atomic_json,file_hash,read_json,lock
from .context_reader import version


def run(config_path,workers=8,mib_per_second=512,min_available_gib=128,kind='spatial'):
    from .config import load_config
    from .resource_guard import check
    from .binding import scenes_by_role
    from .context import spatial_key
    config=load_config(config_path);check(config,'warm-immutable-spatial-pages')
    require(1<=workers<=8 and 1<=mib_per_second<=512 and min_available_gib>=128,'bounded page warm allocation')
    require(kind in ('spatial','joint'),'unknown immutable context kind')
    if kind=='joint':
        from .data import ReusedContext
        from iqe.internal_baseline.features import cache_key
        cache=ReusedContext(config,capacity=0).cache
        path_for=cache.path
        key_for=lambda scene:cache_key(cache.config,scene)
    else:
        path_for=lambda scene:config.root/'spatial_context'/(scene.scene_id+'.pt')
        key_for=lambda scene:spatial_key(config,scene)
    out=config.root/('runtime/'+kind+'_page_warm')/socket.gethostname()
    with lock(out/'RUN.lock'):
        roles=scenes_by_role(config);scenes=roles['scorer_fit']+roles['stage_val']
        total=sum(path_for(s).stat().st_size for s in scenes)
        require(total<=(64 if kind=='joint' else 256)*1024**3,'page warm scope exceeds bounded role bytes')
        deadline=time.monotonic()+3600;rate=mib_per_second*1024**2
        limiter=threading.Lock();schedule=[time.monotonic()];cancel=threading.Event()
        def available():
            rows=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
            return int(rows['MemAvailable'].strip().split()[0])*1024
        def throttle(size):
            with limiter:
                now=time.monotonic();start=max(now,schedule[0]);schedule[0]=start+size/rate
            if start>now:time.sleep(start-now)
        def one(scene):
            if cancel.is_set():return None
            require(time.monotonic()<deadline,'one-hour page warm budget elapsed')
            if available()<min_available_gib*1024**3:
                cancel.set();return None
            path=path_for(scene);before=version(path)
            receipt=read_json(str(path)+'.COMPLETE.json')
            require(receipt['key']==key_for(scene),'page warm context identity changed')
            h=hashlib.sha256();count=0
            with open(path,'rb') as stream:
                while chunk:=stream.read(8*1024**2):
                    h.update(chunk);count+=len(chunk);throttle(len(chunk))
            require(version(path)==before and h.hexdigest()==receipt['checksum'],'page warm detects corruption')
            return {'scene_id':scene.scene_id,'bytes':count,'checksum':h.hexdigest()}
        started=time.monotonic();results=[]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for at in range(0,len(scenes),64):
                if cancel.is_set():break
                results.extend(x for x in executor.map(one,scenes[at:at+64]) if x is not None)
                atomic_json(out/'STATUS.json',{'status':'WARMING','pid':os.getpid(),'host':socket.gethostname(),
                   'records':len(results),'total_records':len(scenes),'bytes_read':sum(x['bytes'] for x in results),
                   'seconds':time.monotonic()-started,'MemAvailable':available(),'rate_limit_MiB_s':mib_per_second})
        result={'status':'MEMORY_GUARD_STOPPED' if cancel.is_set() else 'COMPLETE','records':results,
                'total_records':len(scenes),'bytes_read':sum(x['bytes'] for x in results),
                'seconds':time.monotonic()-started,'config_identity':config.identity,
                'Base_binding_hash':file_hash(config.root/'BASE_BINDING.json'),
                'rate_limit_MiB_s':mib_per_second,'GPU_calls':0,'sampler_consumed':0,'new_tensor_files':0,'context_kind':kind}
        atomic_json(out/f'RESULT_{os.getpid()}.json',result,immutable=True)
        atomic_json(out/'STATUS.json',{k:v for k,v in result.items() if k!='records'}|{'pid':os.getpid()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--mib-per-second',type=int,default=512)
    p.add_argument('--min-available-gib',type=int,default=128)
    p.add_argument('--kind',choices=('spatial','joint'),default='spatial')
    a=p.parse_args();run(a.config,a.workers,a.mib_per_second,a.min_available_gib,a.kind)


if __name__=='__main__':main()

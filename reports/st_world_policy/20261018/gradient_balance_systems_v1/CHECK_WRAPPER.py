"""Author tests: actual tiny CPU children, never real gradient/model execution."""
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import tempfile
import threading
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gradient_wrapper',HERE/'supervise_gradient_once.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def main():
    started=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);results=[]
    with tempfile.TemporaryDirectory(prefix='gradient-wrapper-fixture-') as folder:
        root=Path(folder)
        for name,code,options in [
            ('normal',"print('fixture complete')",{}),
            ('native_exit',"import os;os._exit(23)",{}),
            ('monitor_error','import time;time.sleep(10)',{'sample':lambda *a:(_ for _ in ()).throw(FileNotFoundError('fixture monitor race'))}),
            ('timeout','import time;time.sleep(10)',{'wall':40.25}),
            ('rss_limit','import time;time.sleep(10)',{'rss_limit':1}),
            ('stdout_limit',"import sys,time;sys.stdout.write('x'*(3<<20));sys.stdout.flush();time.sleep(10)",{}),
            ('sigterm','import time;time.sleep(10)',{'_signal':signal.SIGTERM}),
            ('sigint','import time;time.sleep(10)',{'_signal':signal.SIGINT}),
        ]:
            attempt=root/name;attempt.mkdir();timer=None
            if '_signal' in options:
                sig=options.pop('_signal');timer=threading.Timer(.3,lambda:os.kill(os.getpid(),sig));timer.start()
            try:r=m.supervise([sys.executable,'-c',code],attempt=attempt,output=attempt/'output',env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),**options)
            finally:
                if timer:timer.cancel();timer.join()
            assert r['live_owned_remaining']==0,r
            assert (r['status']=='EXITED')==(name=='normal'),r
            if name=='native_exit':assert r['exit_code']==23,r
            if name=='timeout':assert r['reason']=='WALL_LIMIT',r
            if name=='rss_limit':assert r['reason']=='RSS_LIMIT',r
            if name=='stdout_limit':assert r['reason']=='STDOUT_LIMIT',r
            if name.startswith('sig'):assert r['reason'].startswith('SUPERVISOR_SIGNAL_'),r
            child=json.loads((attempt/'CHILD.json').read_text())
            assert not Path('/proc/'+str(child['pid'])).exists(),child
            results.append({'case':name,**r})
        # Fast zero-exit children still need final nominal accounting.
        attempt=root/'fast_overread';attempt.mkdir()
        counts={'hash_nominal_bytes':10,'torch_load_nominal_bytes':0,'safetensors_nominal_bytes':0}
        code='import json;open('+repr(str(attempt/'COUNTERS.json'))+',"w").write('+repr(json.dumps(counts))+')'
        r=m.supervise([sys.executable,'-c',code],attempt=attempt,output=attempt/'output',env=os.environ.copy(),read_limit=1)
        assert r['reason']=='READ_LIMIT' and r['live_owned_remaining']==0,r
        results.append({'case':'fast_overread',**r})
        attempt=root/'fast_rchar';attempt.mkdir()
        data=root/'read_fixture.bin';data.write_bytes(b'x'*(1<<20))
        code='p='+repr(str(data))+'\nfor _ in range(4):\n with open(p,"rb") as f: f.read()\n'
        r=m.supervise([sys.executable,'-c',code],attempt=attempt,output=attempt/'output',env=os.environ.copy())
        assert r['status']=='EXITED' and r['observed_rchar_bytes']>=4<<20,r
        results.append({'case':'fast_exit_preserves_rchar',**r})
        # Precharge persists before invocation and rejects without calling loader.
        p=root/'small.bin';p.write_bytes(b'0123456789');admit=m.ReadAdmission(root/'counts.json',limit=10);calls=[]
        def fake_load(path,**kwargs):
            assert json.loads((root/'counts.json').read_text())['torch_load_nominal_bytes']==10
            calls.append(str(path));return 'same result'
        wrapped=admit.wrap(fake_load,'torch_load_nominal_bytes')
        assert wrapped(p,mmap=True)=='same result'
        try:wrapped(p,mmap=True)
        except RuntimeError:pass
        else:raise AssertionError('overbudget mmap admitted')
        assert calls==[str(p)] and admit.total()==10
        results.append({'case':'mmap_preadmission','passed':True,'calls':len(calls),'admitted_bytes':admit.total()})
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    record={'status':'AUTHOR_FIXTURES_PASS_NOT_INDEPENDENT_APPROVAL','results':results,
        'wall_seconds':time.monotonic()-started,'child_cpu_seconds':usage.ru_utime+usage.ru_stime-before.ru_utime-before.ru_stime,
        'parent_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'child_peak_rss_bytes':usage.ru_maxrss*1024,'real_model_or_cache_reads':0,'GPU_used':False,'new_scores':0}
    with (HERE/'WRAPPER_AUTHOR_TESTS_REVISION2.json').open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps({k:v for k,v in record.items() if k!='results'}))

if __name__=='__main__':main()

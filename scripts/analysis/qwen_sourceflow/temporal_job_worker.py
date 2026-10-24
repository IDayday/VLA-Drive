"""Exact-PID owned temporal IL invocation, fixed GPU UUIDs and live task scope."""
import fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
spec=json.loads(Path(sys.argv[1]).read_text());root=Path(spec['receipt_dir']);root.mkdir(parents=True,exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,spec['gpus']))
lock=(root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (root/'process.json').exists(),'Explicit new attempt required'
S=Path(spec['source']);sys.path.insert(0,str(S))
from starVLA.rl.qwen_sourceflow.temporal_contracts import require_temporal_dispatch
from starVLA.rl.flow_grpo.transactions import atomic_json
start=time.time();record=dict(status='PREPARING',start_unix=start,spec=str(Path(sys.argv[1]).resolve()))
try:
    host=require_temporal_dispatch();assert host==spec['host']
    module=spec['command'][spec['command'].index('-m')+1]
    permitted={'torch.distributed.run','scripts.analysis.qwen_sourceflow.validate_temporal_deployment'}
    if module not in permitted:raise ValueError('Unregistered temporal job module')
    if module=='torch.distributed.run' and 'scripts.analysis.qwen_sourceflow.train_temporal_il' not in spec['command']:
        raise ValueError('Unregistered distributed temporal task')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()==spec['source_sha']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=S,text=True).strip()
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpus=[(u.strip(),int(f),int(v)) for u,f,v in (r.split(',') for r in rows.splitlines())]
    assert [gpus[i][0] for i in spec['gpus']]==spec['gpu_uuids']
    assert all(gpus[i][1]>=spec['min_free_MiB'] and gpus[i][2]<=10 for i in spec['gpus']), 'Dedicated idle devices required'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,spec['gpus'])),PYTHONPATH=str(S),
        OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    with (root/'worker.log').open('x') as log:
        child=subprocess.Popen(spec['command'],cwd=S,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record.update(status='RUNNING',pid=child.pid,start_ticks=int((Path('/proc')/str(child.pid)/'stat').read_text().split()[21]),host=host,
            source=str(S),source_sha=spec['source_sha'],gpus=spec['gpus'],gpu_uuids=spec['gpu_uuids'],command=spec['command'])
        atomic_json(root/'process.json',record);stopping=False
        while True:
            try:code=child.wait(timeout=30);break
            except subprocess.TimeoutExpired:pass
            try:require_temporal_dispatch();withdrawn=False
            except PermissionError:withdrawn=True
            if not stopping and (withdrawn or (root/'STOP_AFTER_BOUNDARY').exists()):
                if module!='torch.distributed.run':
                    child.terminate();stopping=True;continue
                signaled=[]
                for p in Path('/proc').iterdir():
                    if not p.name.isdigit():continue
                    try:
                        if os.getpgid(int(p.name))!=child.pid:continue
                        argv=(p/'cmdline').read_bytes().decode().split('\0')
                        if '-m' not in argv or argv[argv.index('-m')+1]!='scripts.analysis.qwen_sourceflow.train_temporal_il':continue
                        fd=os.pidfd_open(int(p.name))
                        try:signal.pidfd_send_signal(fd,signal.SIGUSR1)
                        finally:os.close(fd)
                        signaled.append(int(p.name))
                    except (FileNotFoundError,ProcessLookupError):pass
                atomic_json(root/'stop_request.json',dict(at=time.time(),pids=signaled,withdrawn=withdrawn));stopping=True
    record.update(status='PROCESS_COMPLETE' if code==0 else 'FAILED',exit_code=code,end_unix=time.time(),wall_seconds=time.time()-start)
except BaseException:
    record.update(status='FAILED',error=traceback.format_exc(),end_unix=time.time(),wall_seconds=time.time()-start);raise
finally:atomic_json(root/'result.json',record)

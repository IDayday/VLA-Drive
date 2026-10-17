"""Two exact CPU-only index builds; one invocation per stage, no auto retry."""
import argparse,hashlib,json,os,selectors,signal,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mnt/project/st-world-policy-20261016')
SOURCE=Path('/mnt/project/VLA-Drive-st-metadata-qualified-runtime-v1-20261017')
PREP=ROOT/'audits/indexed_composition_preparation_v1/PREPARED.json'
sys.path.insert(0,str(SOURCE))
from tools.st_world_policy import metadata_storage as m
from tools.st_world_policy import cache_storage_budget as b
MIB=1<<20
STREAM_LIMIT=64<<10
CONTROL_WRITE_LIMIT=MIB

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    if Path(p).stat().st_size>MIB:raise ValueError('control exceeds1MiB')
    return json.loads(Path(p).read_bytes())
def publish(guard,path,value):
    # Bound serialized bytes, including JSON escaping and atomic-file overhead,
    # before creating even the temporary file. The same lock covers census/write.
    required=b.json_write_bound(value)
    if required>CONTROL_WRITE_LIMIT:raise ValueError('serialized control exceeds reserved bound')
    with m.metadata_writing(**guard) as ctx:
        if path.exists():raise ValueError('one-shot artifact exists; explicit diagnostic required')
        before=ctx.check(before_commit=True,path=path)
        if before['charged_disk_bytes']+required>min(ctx.limits['max_disk_bytes'],ctx.budget._total(ctx.claim)):
            raise ValueError('control atomic write exceeds remaining reservation')
        b._atomic(path,value);ctx.check(before_commit=True)
def main():
    start=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('request_index','fit_index'),required=True)
    for k in ('review','review-sha256','checkpoint-receipt','checkpoint-sha256'):p.add_argument('--'+k,required=True)
    a=p.parse_args();prep=load(PREP)
    if socket.gethostname()!='training-vla-zt-worker-0':raise ValueError('host differs')
    if 'training-vla-zt' not in load('/mnt/project/server_dispatch_policy.json')['allowed_hosts']:raise ValueError('host withdrawn')
    if subprocess.check_output(['git','rev-parse','--short=8','HEAD'],cwd=SOURCE,text=True).strip()!='1162b955':raise ValueError('runtime SHA differs')
    if subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE).strip():raise ValueError('runtime dirty')
    review,_=m.pinned_json(a.review,a.review_sha256)
    if (review.get('status')!='PASS_BOUNDED_INDEX_FOLLOWUP' or review.get('owner_role')!='IndependentSystems'
       or review.get('prepared_sha256')!=sha(PREP) or review.get('controller_sha256')!=sha(__file__)
       or not time.time()+185<=review['valid_until_unix']):raise ValueError('independent exact unexpired review missing')
    m._checkpoint_reserve(a.checkpoint_receipt,a.checkpoint_sha256,now=time.time()+185)
    item=prep['stages'][a.stage];reg,_=m.pinned_json(item['registration'],item['registration_sha256'])
    out=Path(item['output_root'])
    if out!=Path(reg['output_root']):raise ValueError('output differs')
    guard=dict(policy_path=prep['policy'],policy_sha256=prep['policy_sha256'],
      registration_path=item['registration'],registration_sha256=item['registration_sha256'],
      checkpoint_receipt=a.checkpoint_receipt,checkpoint_sha256=a.checkpoint_sha256)
    started=out/'CONTROL_BUILD_STARTED.json'; result_path=out/'CONTROL_BUILD_RESULT.json'
    command=[sys.executable,'-m','tools.st_world_policy.metadata_storage','run']
    for k,v in guard.items():command+=['--'+k.replace('_','-'),str(v)]
    record=dict(schema='st_bounded_index_followup_invocation_v1',stage=a.stage,command=command,
      controller_sha256=sha(__file__),prepared_sha256=sha(PREP),cpu_affinity=[0,1],gpu_visible=False,
      maximum_seconds=180,max_aggregate_rss_bytes=2<<30,max_stream_bytes=STREAM_LIMIT,
      max_serialized_control_write_bytes=CONTROL_WRITE_LIMIT,automatic_retry=False,
      started_unix=time.time())
    publish(guard,started,record)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1',
      'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2'}
    child=subprocess.Popen(['taskset','-c','0,1',*command],cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,
      stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    sel=selectors.DefaultSelector();buffers={}
    for label,stream in [('stdout',child.stdout),('stderr',child.stderr)]:
        os.set_blocking(stream.fileno(),False);sel.register(stream,selectors.EVENT_READ,label);buffers[label]=bytearray()
    peak=0;failure=None
    try:
        while sel.get_map() or child.poll() is None:
            peak=max(peak,m.process_tree_rss(os.getpid()))
            if time.monotonic()-start>175:raise RuntimeError('finite invocation deadline')
            if peak>2<<30:raise RuntimeError('aggregate RSS exceeded')
            for key,_ in sel.select(.1):
                data=os.read(key.fd,65536)
                if not data:sel.unregister(key.fileobj);continue
                buffer=buffers[key.data]
                if len(buffer)+len(data)>STREAM_LIMIT:raise RuntimeError('bounded output exceeded')
                buffer.extend(data)
        child.wait(timeout=1)
    except Exception as exc:
        failure=str(exc)
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=1)
    finally:
        sel.close();child.stdout.close();child.stderr.close()
    raw=bytes(buffers['stdout']).decode('utf8','replace')
    try:result=json.loads(raw,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    except Exception:result=None
    good=(child.returncode==0 and failure is None and isinstance(result,dict)
      and result.get('status')=='GUARDED_METADATA_RETURN')
    if good:
        artifact=result.get('artifact')
        good=(isinstance(artifact,dict) and all(type(artifact.get(k)) is int and artifact[k]==0
          for k in ('image_files_opened','tensor_files_opened','tensor_disk_bytes')))
        if not good:failure='malformed or nonzero metadata-only result'
    record.update(status='PASS' if good else 'FAILED',returncode=child.returncode,failure=failure,
      invocation_seconds=time.monotonic()-start,peak_aggregate_rss_bytes=peak,stdout_json=result,
      stderr=bytes(buffers['stderr']).decode('utf8','replace'),stdout_unparsed=raw if result is None else None,
      finished_unix=time.time())
    publish(guard,result_path,record)
    print(json.dumps({'status':record['status'],'result':str(result_path),'sha256':sha(result_path),
      'seconds':record['invocation_seconds'],'peak_rss_bytes':peak}))
    return 0 if good else 2
if __name__=='__main__':raise SystemExit(main())

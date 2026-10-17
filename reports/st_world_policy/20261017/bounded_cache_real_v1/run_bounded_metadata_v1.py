"""Lead finite metadata invocations only; no retries, teacher or training."""
import time
START=time.monotonic()
import argparse, hashlib, json, os, selectors, signal, socket, subprocess, sys
from pathlib import Path
ROOT=Path('/mnt/project/st-world-policy-20261016')
SOURCE=Path('/mnt/project/VLA-Drive-st-metadata-qualified-runtime-v1-20261017')
PREP=ROOT/'audits/metadata_migration_preparation_v1/PREPARED_V6.json'
sys.path.insert(0,str(SOURCE))
from tools.st_world_policy import metadata_storage as m
from tools.st_world_policy import cache_storage_budget as b
MIB=1<<20

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    raw=Path(path).read_bytes()
    if len(raw)>MIB:raise ValueError('control too large')
    return json.loads(raw)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['future_build','future_reference','current_index'],required=True)
    for name in ['checkpoint-receipt','checkpoint-sha256','review','review-sha256']:
        p.add_argument('--'+name,required=True)
    a=p.parse_args(); prep=load(PREP)
    assert socket.gethostname()=='training-vla-zt-worker-0'
    policy=load('/mnt/project/server_dispatch_policy.json')
    assert 'training-vla-zt' in policy['allowed_hosts']
    assert subprocess.check_output(['git','rev-parse','--short=8','HEAD'],cwd=SOURCE,text=True).strip()=='1162b955'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE).strip()
    review,_=m.pinned_json(a.review,a.review_sha256)
    assert review['status']=='PASS_BOUNDED_METADATA_ADMISSION'
    assert review['prepared_sha256']==sha(PREP) and review['controller_sha256']==sha(__file__)
    assert time.time()<review['valid_until_unix']
    prefix='index' if a.stage=='current_index' else 'future'
    regpath=prep[prefix+'_storage_registration']; regsha=prep[prefix+'_storage_registration_sha256']
    reg,_=m.pinned_json(regpath,regsha); out=Path(reg['output_root'])
    guard=dict(policy_path=prep['new_policy'],policy_sha256=prep['new_policy_sha256'],
       registration_path=regpath,registration_sha256=regsha,
       checkpoint_receipt=a.checkpoint_receipt,checkpoint_sha256=a.checkpoint_sha256)
    upper=a.stage.upper(); started_path=out/('CONTROL_'+upper+'_STARTED.json')
    result_path=out/('CONTROL_'+upper+'_RESULT.json')
    assert not started_path.exists() and not result_path.exists(), 'No automatic retry: retain charged prior invocation'
    previous_seconds=0.; previous_bytes=0
    if a.stage=='future_reference':
        earlier=load(out/'CONTROL_FUTURE_BUILD_RESULT.json')
        assert earlier['status']=='PASS' and earlier['returncode']==0
        previous_seconds=earlier['invocation_seconds']; previous_bytes=earlier['charged_source_bytes']
        assert previous_seconds<=450 and previous_bytes<=4*(1<<30)
        command=[sys.executable,'-m','tools.st_world_policy.compare_future_metadata',
          '--plan',prep['future_plan'],'--plan-sha256',prep['future_plan_sha256'],
          '--registration',prep['future_preparation_registration'],
          '--registration-sha256',prep['future_preparation_registration_sha256'],
          '--complete-sha256',sha(out/'COMPLETE.json')]
    else:
        command=[sys.executable,'-m','tools.st_world_policy.metadata_storage','run']
        for k,v in guard.items():command+=['--'+k.replace('_','-'),str(v)]
    cap=1200 if a.stage=='current_index' else 450
    def publish(path,value):
        with m.metadata_writing(**guard) as ctx:
            # The same campaign lock covers both checking and publishing the
            # one-shot marker. A competing controller must fail before Popen.
            if path.exists():
                raise ValueError('Immutable invocation marker already exists: '+str(path))
            ctx.check(before_commit=True,path=path)
            b._atomic(path,value)
            ctx.check(before_commit=True)
    start_record={'schema':'st_finite_metadata_invocation_v1','stage':a.stage,'command':command,
       'controller_sha256':sha(__file__),'prepared_sha256':sha(PREP),'registered_unix':time.time(),
       'maximum_seconds':cap,'source_read_reservation':None if a.stage=='current_index' else 4*(1<<30),
       'prior_seconds':previous_seconds,'prior_charged_source_bytes':previous_bytes,
       'automatic_retry':False,'cpu_affinity':[0,1],'gpu_visible':False,'max_aggregate_rss_bytes':4*(1<<30)}
    publish(started_path,start_record)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1',
       'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2'}
    command=['taskset','-c','0,1',*command]
    child=subprocess.Popen(command,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,
       stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    sel=selectors.DefaultSelector(); buffers={}
    for label,stream in [('stdout',child.stdout),('stderr',child.stderr)]:
        os.set_blocking(stream.fileno(),False);sel.register(stream,selectors.EVENT_READ,label);buffers[label]=bytearray()
    peak=0; failure=None
    try:
        while sel.get_map() or child.poll() is None:
            elapsed=time.monotonic()-START; rss=m.process_tree_rss(os.getpid());peak=max(peak,rss)
            if elapsed>cap-5:raise RuntimeError('finite invocation deadline')
            if rss>4*(1<<30):raise RuntimeError('aggregate parent/child RSS exceeded')
            for key,_ in sel.select(.25):
                data=os.read(key.fd,65536)
                if not data:sel.unregister(key.fileobj);continue
                buffer=buffers[key.data]
                if len(buffer)+len(data)>MIB:raise RuntimeError('bounded stdout/stderr exceeded')
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
    raw=bytes(buffers['stdout']).decode('utf8','replace'); stderr=bytes(buffers['stderr']).decode('utf8','replace')
    try:result=json.loads(raw)
    except Exception:result=None
    good=child.returncode==0 and failure is None and isinstance(result,dict)
    expected='REFERENCE_MATCH' if a.stage=='future_reference' else 'GUARDED_METADATA_RETURN'
    good=good and result.get('status')==expected
    if a.stage=='current_index':source_bytes=None
    elif good:
        source_bytes=(result['reference_invocation_cost'] if a.stage=='future_reference' else result['artifact']['cost'])['source_bytes_read']
    else:source_bytes=4*(1<<30)  # Unknown failed work consumes its full reserved half.
    elapsed=time.monotonic()-START
    if a.stage!='current_index':
        good=good and elapsed+previous_seconds<=900 and source_bytes+previous_bytes<=8*(1<<30)
    record={**start_record,'status':'PASS' if good else 'FAILED','returncode':child.returncode,
       'failure':failure,'invocation_seconds':elapsed,'charged_source_bytes':source_bytes,
       'peak_aggregate_rss_bytes':peak,'stdout_json':result,'stderr':stderr,
       'stdout_unparsed':raw if result is None else None,'finished_unix':time.time()}
    publish(result_path,record)
    print(json.dumps({'status':record['status'],'result':str(result_path),'sha256':sha(result_path),
      'seconds':elapsed,'peak_rss':peak,'charged_source_bytes':source_bytes}))
    return 0 if good else 2

if __name__=='__main__':raise SystemExit(main())

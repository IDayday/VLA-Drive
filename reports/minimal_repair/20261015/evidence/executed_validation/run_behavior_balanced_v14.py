"""Finite6 balanced common-endpoint analyses, using existing verified geometry code."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
w=Path('/mnt/project/VLA-Drive-minimal-repair-analysis-v11-20261015');sys.path.insert(0,str(w))
from tools.decision_local.common import atomic_json,check_host,digest,read,source_sha
r=Path('/mnt/project/minimal-repair-artifacts/20261015');reg=r/'BALANCED_V14_BEHAVIOR_REGISTRY.json'
info=read(reg);reg_hash=digest(reg);resource=check_host();host=resource['host'];start=time.monotonic()
assert host in info['allowed_hosts'] and source_sha(w)==info['analysis_source_sha']
state=Path(info['runtime']);(state/'claims').mkdir(parents=True,exist_ok=True);(state/'results').mkdir(exist_ok=True)
features=Path('/dev/shm/decision-local-minimal-repair-v5-20261015/anchor_dev_full_v1')
labels=Path('/dev/shm/decision-local-repair-labels-v12-20261015/targets_dev1696_v5')
assert read(features/'COMPLETE.json')['identity']==read(features/'identity.json')['identity']
assert read(labels/'COMPLETE.json')['identity']==read(labels/'identity.json')['identity']
while time.monotonic()-start<14400-650:
    check_host();assert digest(reg)==reg_hash
    ready=[j for j in info['jobs'] if not (state/'claims'/j['id']).exists() and all(Path(x).exists() for x in j['requires'])]
    if not ready:
        if all((state/'claims'/j['id']).exists() for j in info['jobs']):break
        time.sleep(10);continue
    j=ready[0];claim=state/'claims'/j['id']
    try:claim.mkdir()
    except FileExistsError:continue
    argv=[sys.executable,'-u',str(w/'tools/decision_local/evaluate_minimal_repair_behavior.py'),
        '--predictions',j['predictions'],'--scores',j['scores'],'--labels',str(labels),'--features',str(features),
        '--anchor','/mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json',
        '--output',j['output'],'--workers',str(j['workers'])]
    log=state/(j['id']+'.log');started=time.time()
    with log.open('x') as stream:
        child=subprocess.Popen(argv,cwd=w,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(claim/'LAUNCH.json',dict(host=host,pid=child.pid,argv=argv,started_unix=started,max_seconds=j['max_seconds'],analysis_source_sha=source_sha(w),registry_sha256=reg_hash))
        timed_out=False
        try:code=child.wait(timeout=j['max_seconds'])
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM);code=child.wait(timeout=30)
    atomic_json(state/'results'/(j['id']+'.json'),dict(returncode=code,complete=code==0 and (Path(j['output'])/'COMPLETE.json').exists(),
        timed_out=timed_out,host=host,elapsed_seconds=time.time()-started,no_automatic_retry=True))
atomic_json(state/(host+'_'+str(os.getpid())+'_FINISHED.json'),dict(elapsed_seconds=time.monotonic()-start,pending=[j['id'] for j in info['jobs'] if not (state/'claims'/j['id']).exists()]))

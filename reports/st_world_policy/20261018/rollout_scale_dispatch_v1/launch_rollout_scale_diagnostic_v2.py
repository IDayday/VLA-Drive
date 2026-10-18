"""Lead-only single-candidate SMOKE/PROFILE; adapted from the reviewed Flow launcher.

prepare writes metadata only. launch requires exact independent-review admission.
No automatic follow-up, formal training, legacy-control rerun or resume is started.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import sys
import time
import importlib.util

ROOT = Path('/mnt/project/st-world-policy-20261016')
REPO = Path('/mnt/project/VLA-Drive-st-rollout-scale-runtime-v2-20261018')
SHA = '751fcfa2b41549ae16f7b3204aef8dd18b912016'
HOST = 'training-vlawm-tsinghua-zt5-worker-0'
PYTHON = '/root/miniconda3/envs/ddp/bin/python'
VISIBLE = '0,1,2,3'
CPUSET = '0-15'
sys.path.insert(0, str(REPO))
from starVLA.dataloader.st_world_targets import atomic_json
from tools.st_world_policy.check_formal_approval import file_hash, source_fingerprint, verify_registration
from tools.st_world_policy.derive_flow_diagnostic_manifest import derive_manifest
from tools.st_world_policy.flow_training_config import validate_config, target_contract
from tools.st_world_policy.training_config import environment_record


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    return {'path':str(Path(path).resolve()),'sha256':file_hash(path)}


def checked_artifact(value):
    if artifact(value['path']) != value:
        raise ValueError('Admission artifact changed: '+value['path'])
    return read(value['path'])


def fresh_host():
    policy = read('/mnt/project/server_dispatch_policy.json')
    authorization = ROOT/'resources/AUTHORIZATION_20261016T203442.json'
    assert socket.gethostname() == HOST and HOST in policy['allowed_aliases']
    assert HOST.removesuffix('-worker-0') in read(authorization)['allowed_hosts']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip() == SHA
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip()
    assert set(range(16)).issubset(os.sched_getaffinity(0))
    rows = [r.split(',') for r in subprocess.check_output(['nvidia-smi','--id='+VISIBLE,
        '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).splitlines()]
    assert len(rows)==4 and all(int(r[4])<64 and int(r[5])<5 for r in rows), 'Real/unknown jobs protected'
    selected={row[1].strip() for row in rows}
    applications=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True)
    assert not any(line.split(',')[0].strip() in selected for line in applications.splitlines()), 'Real/unknown compute process protected'
    mem = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    assert int(mem['MemAvailable'].split()[0])*1024 >= 512*2**30
    # Keep the existing 1 TiB shared reserve separate from this 96 GiB study.
    assert shutil.disk_usage(ROOT).free >= (1024+96)*2**30
    assert Path('/mnt/navsim/maps').is_dir()
    return rows, authorization


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','launch'])
    p.add_argument('--stage',choices=['PROFILE'],required=True)
    p.add_argument('--admission')
    a=p.parse_args()
    smoke=a.stage=='SMOKE'
    out=ROOT/('smoke/scene_center_rollout_std_v1_4update' if smoke else
              'profiles/scene_center_rollout_std_v1_fulltrain240')
    rows,authorization=fresh_host()
    seconds,updates,calls=(900,4,640) if smoke else (7200,240,31744)
    if a.command=='prepare':
        assert not a.admission and not out.exists(), 'Immutable diagnostic namespace already exists'
        parent=ROOT/'formal/consequence/registrations/C1_fulltrain_seed42_official_4096_v1'
        assert file_hash(parent/'CONFIG.json')=='f1c4cc851d2e055cc7e2cb10c08e032574c424e4aa8b8f27d46007774a6bba0e'
        cfg=read(parent/'CONFIG.json')
        manifest=derive_manifest(ROOT/'audits/flow_policy/full_train_contexts_v2/TRAINING_MANIFEST.json',
            'b810cafbdaa42afd8554c08be443779e7d66edce56292c0c30940e728c0ed656',
            scope=a.stage,max_scenes=32 if smoke else None)
        cfg['execution'].update(scope=a.stage,sample_limit=len(manifest['entries']),stop_after=updates)
        cfg['fpo']['advantage']='scene_center_rollout_std_v1'
        cfg['runtime'].update(output_root=str(out/'runs'),max_seconds=seconds,
            save_every=2 if smoke else 240,profile_mode='none' if smoke else 'timing')
        cfg['runtime']['official_execution']['adaptation_reason']='One explicit C1 rollout advantage scale; original G32/micro4/world4/4096 clock/warm100, original S0 initialization; bounded diagnostic only.'
        cfg['feedback']['max_calls']=calls
        validate_config(cfg)
        out.mkdir(parents=True,exist_ok=False)
        for name,value in [('CONFIG.json',cfg),('TRAINING_MANIFEST.json',manifest),
                           ('TARGET_CONTRACT.json',target_contract(cfg)),('ENVIRONMENT.json',environment_record())]:
            atomic_json(out/name,value,immutable=True)
        lease={'schema':'st_training_resource_lease_v1','hostnames':[HOST],'world_size':4,
            'cuda_visible_devices_by_host':{HOST:VISIBLE},'gpu_uuids_by_host':{HOST:[r[1].strip() for r in rows]},
            'valid_from_unix':time.time()-1,'valid_until_unix':time.time()+10800,
            'max_invocation_seconds':seconds,'authorization':str(authorization),
            'purpose':a.stage+' unique rollout scale; no formal initialization; 16 CPUs and four global scorers'}
        atomic_json(out/'RESOURCE_LEASE.json',lease,immutable=True)
        reg=read(parent/'REGISTRATION.json')
        reg.update(run_id='C1_rollout_std_'+a.stage.lower()+'_v1',source_sha=SHA,
                   source_fingerprint=source_fingerprint(REPO),execution=cfg['execution'],weights_usable_for_formal_init=False)
        reg['implementer_agents']=list(dict.fromkeys(reg['implementer_agents']+['/root/rollout_scale_implementation']))
        for key,name in [('config','CONFIG.json'),('data_manifest','TRAINING_MANIFEST.json'),
                         ('target_contract','TARGET_CONTRACT.json'),('environment','ENVIRONMENT.json'),
                         ('resource_lease','RESOURCE_LEASE.json')]:
            reg['artifacts'][key]=artifact(out/name)
        atomic_json(out/'REGISTRATION.json',reg,immutable=True)
        verify_registration(out/'REGISTRATION.json',repo=REPO)
        atomic_json(out/'BUDGET.json',{'scope':a.stage,'max_new_updates':updates,'max_real_score_calls':calls,
            'scene_exposures':updates*32,'unique_population':len(manifest['entries']),
            'invocation_seconds':seconds,'new_output_reserve_bytes':96*2**30,'existing_shared_reserve_bytes':1024*2**30,
            'RAM_sampled_stop_bytes':256*2**30,'GPU_reserved_stop_bytes_per_rank':48*2**30,'max_study_native_directories':6,
            'aggregate_smoke_and_profile_reservation':True,
            'parent_is_original_S0':True,'weights_usable_for_formal_init':False,'automatic_retry':False,
            'original_LR_horizon':4096,'launcher':artifact(__file__)},immutable=True)
        print(json.dumps({'status':'METADATA_PREPARED','scope':a.stage,'registration':artifact(out/'REGISTRATION.json')}))
        return
    assert a.admission, 'Exact independent-review admission is required'
    admission=read(a.admission)
    assert admission['status']=='APPROVED_BOUNDED_DIAGNOSTIC' and admission['formal_training_allowed'] is False
    assert admission['source_sha']==SHA and admission['scope']==a.stage
    assert admission['registration']==artifact(out/'REGISTRATION.json') and admission['launcher']==artifact(__file__)
    reg=checked_artifact(admission['registration'])
    binding={key:admission[key] for key in ('source_sha','registration','launcher')}
    for role in ('correctness','systems'):
        review=checked_artifact(admission[role])
        assert str(review['status']).startswith(('PASS','APPROVE'))
        assert review['binding']==binding, 'Independent '+role+' review belongs to another run'
    if not smoke:
        completed=checked_artifact(admission['smoke_completion'])
        assert completed['status']=='PASS_REAL_FOUR_RANK_SMOKE'
        assert completed['source_sha']==SHA and completed['new_optimizer_updates']==4
        assert completed['valid_official_feedback']==512 and completed['finite_losses_and_gradients']
        assert completed['frozen_encoder_unchanged'] and completed['world_size']==4
    monitor_path=ROOT/'resources/monitor_rollout_scale_owned_run_v2.py'
    assert file_hash(monitor_path)=='b24a26a27e79cd71919ae2e769061e5b6b033d5993c9fde8260f0065098b2ce2'
    assert artifact(monitor_path)==admission['monitor']
    specification=importlib.util.spec_from_file_location('owned_diagnostic_monitor',monitor_path)
    monitor=importlib.util.module_from_spec(specification);specification.loader.exec_module(monitor)
    script=ROOT/'resources/sample_profile_process_tree_v2.py'
    if not smoke:
        assert file_hash(script)=='2b20a741b8b7187844326419dc0d8c6e32d5eb94b332cdcd76ef9cd8f175fb21'
    cfg=checked_artifact(reg['artifacts']['config'])
    validate_config(cfg)
    verify_registration(out/'REGISTRATION.json',repo=REPO)
    assert cfg['execution']['scope']==a.stage and cfg['execution']['stop_after']==updates
    assert cfg['fpo']['advantage']=='scene_center_rollout_std_v1'
    lease=checked_artifact(reg['artifacts']['resource_lease'])
    assert [r[1].strip() for r in rows]==lease['gpu_uuids_by_host'][HOST]
    assert time.time()+seconds+480<lease['valid_until_unix']
    run=Path(cfg['runtime']['output_root'])/reg['run_id']
    assert not run.exists() and not (out/'LAUNCH.json').exists(), 'Duplicate launch/implicit resume rejected'
    locks=[]
    for path in [out/'DISPATCH.lock',*[Path(f'/tmp/foresight_gpu_{i}.lock') for i in range(4)]]:
        lock=path.open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    fresh_host()  # Recheck after taking the cooperative allocation locks.
    cmd=['timeout','--signal=TERM','--kill-after=180s',str(seconds+300)+'s','taskset','-c',CPUSET,
        PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4',
        '-m','tools.st_world_policy.train_flow_policy','train','--registration',str(out/'REGISTRATION.json')]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':VISIBLE,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8',
         'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'}
    env.pop('FLASH_ATTENTION_DETERMINISTIC',None)
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with (out/'TRAIN.log').open('xb') as stream:
        child=subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            start_new_session=True,pass_fds=tuple(x.fileno() for x in locks))
    atomic_json(out/'LAUNCH.json',{'host':HOST,'pid':child.pid,'started_unix':time.time(),'argv':cmd,
        'cwd':str(REPO),'source_sha':SHA,'scope':a.stage,'max_new_updates':updates,'output':str(run),
        'admission':artifact(a.admission),'dispatch_policy_sha256':file_hash('/mnt/project/server_dispatch_policy.json'),
        'authorization_sha256':file_hash(authorization),'fresh_gpus':rows,'free_bytes':shutil.disk_usage(ROOT).free,
        'automatic_followup':False,'formal':False},immutable=True)
    atomic_json(out/'EXECUTION_BINDING.json',{'preparation_budget':artifact(out/'BUDGET.json'),
        'final_launcher':artifact(__file__),'monitor':artifact(monitor_path),
        'admission':artifact(a.admission),'source_sha':SHA,
        'change':'Preserve immutable SMOKE controls; PROFILE sums both stages output/native directories and safe-stops this run on any unmapped selected-GPU process. Exact reviews bind this launcher.',
        'memory_limit_semantics':'5-second sampled owned RSS and driver GPU use, not a hard instantaneous quota',
        'lease_semantics':'10800-second lease; runtime budget and outer timeout remain 900/1200 SMOKE or 7200/7500 PROFILE'},immutable=True)
    print(json.dumps({'status':'BOUNDED_DIAGNOSTIC_LAUNCHED','pid':child.pid,'output':str(out)}),flush=True)
    observer=None
    if not smoke:
        observer_cmd=['timeout','--signal=TERM','--kill-after=5s','7410s','taskset','-c','15',
            'prlimit','--fsize=268435456','--as=1073741824','--','/usr/bin/python3',
            str(script),'--profile-root',str(out)]
        with (out/'PROCESS_OBSERVER.log').open('xb') as stream:
            observer=subprocess.Popen(observer_cmd,cwd=REPO,env={**env,'CUDA_VISIBLE_DEVICES':''},
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(out/'PROCESS_OBSERVER_LAUNCH.json',{'pid':observer.pid,'argv':observer_cmd,
            'script':artifact(script),'bounded_seconds':7410,'output_limit_bytes':268435456,
            'scope':'live subtree sampling; no signals to training or other jobs'},immutable=True)
    monitored=monitor.wait_owned(child,out,run,[r[1].strip() for r in rows])
    code=monitored['exit_code']
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    atomic_json(out/'EXIT.json',{'exit_code':code,'ended_unix':time.time(),
        'waited_child_cpu_seconds':after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        'child_maxrss_high_water_bytes':after.ru_maxrss*1024,
        'resource_monitor':monitored,
        'scope':'waited child accounting; not wall time times workers'},immutable=True)
    if observer:
        observer.wait(timeout=15)
    raise SystemExit(code if code else (2 if monitored['stop_request'] else 0))


if __name__=='__main__':
    main()

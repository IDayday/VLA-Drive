#!/usr/bin/env python3
"""One-command preflight, smoke and bounded 11-arm pipeline; no cloud submission."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
from contracts import candidate_budget, device_ids, read, sha, validate_prepared, validate_training

STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def runtime_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted([*ROOT.glob('*.py'), *(ROOT/'vendor').rglob('*.py')]) if '__pycache__' not in p.parts}


def select_devices(value):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError('No usable accelerator in this execution environment; no training was launched')
    if value == 'auto':
        chosen = []
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            if free/total >= .95:
                chosen.append(str(i))
        if not chosen:
            raise RuntimeError('No idle visible accelerator (>=95% free memory); use allocated devices explicitly')
    else:
        chosen = device_ids(value)
    if max(map(int, chosen)) >= torch.cuda.device_count():
        raise ValueError('Requested device is not visible in this environment')
    # A parent-side mask changes ordinal semantics again in scheduler children.
    if os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
        raise ValueError('Launch entry.py without a CUDA_VISIBLE_DEVICES remap; use --devices')
    return chosen


def configure_environment(out):
    for key, value in {'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1',
                       'TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
                       'NO_ALBUMENTATIONS_UPDATE':'1','OMP_NUM_THREADS':'1',
                       'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}.items():
        os.environ[key] = value
    for key, name in [('HF_HOME','hf'),('TORCH_HOME','torch'),('XDG_CACHE_HOME','xdg'),
                      ('TRITON_CACHE_DIR','triton'),('NUMBA_CACHE_DIR','numba'),
                      ('MPLCONFIGDIR','mpl'),('TMPDIR','tmp')]:
        p = out/'cache'/name
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p)


def report(a, cfg, check_assets=False, check_device=False):
    errors = []
    result = {'input_protocol_version': cfg.get('input_protocol_version'),
              'parameter_scope_version': cfg.get('parameter_scope_version'),
              'candidate_budget': candidate_budget(cfg), 'errors': errors}
    try:
        validate_training(cfg, formal=True)
    except (ValueError, KeyError) as exc:
        errors.append(str(exc))
    try:
        p = validate_prepared(a.prepared, cfg, complete=True)
        result['prepared'] = {k: p.get(k) for k in ['status','scenes','matched_F','missing_F','strict_F','exception_F']}
    except (ValueError, KeyError, OSError) as exc:
        errors.append(str(exc))
    if check_assets:
        from asset_adapter import config, validate
        try:
            result['assets'] = validate(config(a.assets), full_hash=a.deep, deep=a.deep)
            errors.extend(result['assets']['errors'])
        except Exception as exc:
            errors.append('Asset validation: '+str(exc))
    if check_device:
        try:
            result['devices'] = select_devices(a.devices)
        except (ValueError, RuntimeError) as exc:
            errors.append(str(exc))
    result['status'] = 'READY' if not errors else 'NOT_READY'
    result['device_checked'] = check_device
    result['real_model_validated'] = False
    return result


def remaining(deadline, now=None):
    return max(0., deadline - (time.time() if now is None else now))


def child(command, log, deadline, state_path, state, env=None):
    if STOP or remaining(deadline) <= 0:
        raise TimeoutError('Global pipeline deadline reached before launch')
    with Path(log).open('x') as stream:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        state['active_pid'] = proc.pid
        state['active_start_ticks'] = process_ticks(proc.pid)
        state['active_command'] = command
        save(state_path, state)
        try:
            last = 0.
            sent = False
            while proc.poll() is None:
                if (STOP or remaining(deadline) <= 0) and not sent:
                    proc.send_signal(signal.SIGTERM)
                    sent = True
                if time.monotonic()-last >= 60:
                    state.update(remaining_s=remaining(deadline), updated_utc=datetime.now(timezone.utc).isoformat())
                    save(state_path, state)
                    last = time.monotonic()
                time.sleep(1)
            return proc.returncode
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                proc.wait()
            state.pop('active_pid', None)
            state.pop('active_start_ticks', None)
            state.pop('active_command', None)
            save(state_path, state)


def live_previous(root):
    for process_file in [Path(root)/'PIPELINE.json']:
        if not process_file.exists():
            continue
        state = read(process_file)
        for pid, ticks in [(state.get('pid'),state.get('pid_start_ticks')),*[(r['pid'],r.get('start_ticks')) for r in state.get('parallel_children',[])],(state.get('active_pid'),state.get('active_start_ticks'))]:
            if pid is None:
                continue
            cmdline = Path('/proc')/str(pid)/'cmdline'
            if cmdline.exists() and ((ticks is not None and ticks==process_ticks(pid)) or str(Path(root).resolve()).encode() in cmdline.read_bytes()):
                raise ValueError('Previous pipeline still has a live task process')


def process_ticks(pid):
    try:
        return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
    except (FileNotFoundError, IndexError):
        return None


def parallel_smokes(base, devices, out, deadline, state_path, state, cfg):
    """Exercise the actual target concurrency before formal multi-device phases."""
    active=[];peak_parallel=0
    try:
        for i,device in enumerate(devices):
            if STOP or remaining(deadline)<=0:raise TimeoutError('Parallel validation deadline reached')
            arm=['A','E_VLM','C_DiT'][i%3]
            folder=out/f'parallel_smoke_device{device}'
            command=[sys.executable,'-u',str(ROOT/'run.py'),*base,'--mode','smoke','--arm',arm,
                     '--seed',str(cfg['training_seeds'][0]),'--out',str(folder),
                     '--smoke-updates','1','--smoke-wall-seconds',str(max(1,min(1800,int(remaining(deadline)))))]
            log=(out/f'parallel_smoke_device{device}.log').open('x')
            try:
                proc=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':device},stdout=log,stderr=subprocess.STDOUT)
            except BaseException:
                log.close();raise
            active.append({'process':proc,'log':log,'folder':folder,'device':device,'arm':arm})
            peak_parallel=max(peak_parallel,sum(x['process'].poll() is None for x in active))
            state['parallel_children']=[{'pid':x['process'].pid,'start_ticks':process_ticks(x['process'].pid)} for x in active]
            save(state_path,state)
            time.sleep(1)
        last=0.
        while any(x['process'].poll() is None for x in active):
            failed=any(x['process'].poll() not in [None,0] for x in active)
            if STOP or remaining(deadline)<=0 or failed:
                raise RuntimeError('Parallel validation stopped, failed, or exhausted the global deadline')
            if time.monotonic()-last>=60:
                state.update(remaining_s=remaining(deadline),updated_utc=datetime.now(timezone.utc).isoformat());save(state_path,state);last=time.monotonic()
            time.sleep(1)
        records=[]
        for x in active:
            r=read(x['folder']/'RESULT.json')
            if x['process'].returncode!=0 or r.get('status')!='COMPLETE' or not r.get('frozen_parameters_buffers_exact') or not r.get('reference_exact'):
                raise ValueError('Parallel smoke acceptance failed')
            records.append({'device':x['device'],'arm':x['arm'],'output':str(x['folder'])})
        if peak_parallel!=len(devices):raise ValueError('Target concurrent process count was not observed')
        save(out/'MULTICARD_ACCEPTANCE.json',{'status':'PASS','devices':devices,'peak_parallel':peak_parallel,'records':records,'scope':'target concurrent independent single-card replicas; not DDP'})
    finally:
        for x in active:
            if x['process'].poll() is None:x['process'].send_signal(signal.SIGTERM)
        for x in active:x['process'].wait();x['log'].close()
        state.pop('parallel_children',None)
        save(state_path,state)


def pipeline(a, cfg, preflight, out):
    total = cfg.get('total_wall_seconds')
    if type(total) not in [int, float] or not 0 < total <= 86400:
        raise ValueError('Explicit total_wall_seconds in (0, 86400] required')
    ident = {'training': cfg, 'assets': sha(a.assets), 'prepared': sha(Path(a.prepared)/'PREPARATION.json'), 'code': runtime_hashes()}
    state = {'status':'RUNNING','pid':os.getpid(),'pid_start_ticks':process_ticks(os.getpid()),'identity':ident,'phases':{},'smokes':{},
             'deadline_unix':time.time()+total,'started_utc':datetime.now(timezone.utc).isoformat()}
    previous = None
    if a.resume_from:
        live_previous(a.resume_from)
        previous = read(Path(a.resume_from)/'PIPELINE.json')
        if previous['identity'] != ident:
            raise ValueError('Pipeline recovery identity changed')
        state.update(deadline_unix=previous['deadline_unix'], started_utc=previous['started_utc'], parent_pipeline=str(Path(a.resume_from).resolve()))
    deadline = state['deadline_unix']
    state_path = out/'PIPELINE.json'
    save(state_path, state)
    base = ['--assets', str(Path(a.assets).resolve()), '--training', str(Path(a.training).resolve()), '--prepared', str(Path(a.prepared).resolve())]
    try:
        if remaining(deadline) <= 0:
            raise TimeoutError('Original global deadline already exhausted; recovery does not reset it')
        # Always validate current runtime and all three scopes before formal phases.
        for arm in ['A','E_VLM','C_DiT']:
            if previous and arm in previous.get('smokes', {}):
                folder = Path(previous['smokes'][arm])
                r = read(folder/'RESULT.json')
                if r['status'] != 'COMPLETE' or r.get('mode') != 'smoke':
                    raise ValueError('Previous smoke is incomplete')
            else:
                folder = out/('smoke_'+arm)
                command = [sys.executable,'-u',str(ROOT/'run.py'),*base,'--mode','smoke','--arm',arm,
                           '--seed',str(cfg['training_seeds'][0]),'--out',str(folder),
                           '--smoke-updates','1','--smoke-wall-seconds',str(max(1,min(1800,int(remaining(deadline)))))]
                env = {**os.environ,'CUDA_VISIBLE_DEVICES':preflight['devices'][0]}
                if child(command,out/(arm+'.smoke.log'),deadline,state_path,state,env) != 0:
                    raise RuntimeError('Real-model smoke failed: '+arm)
                r = read(folder/'RESULT.json')
                if r['status'] != 'COMPLETE' or not r.get('frozen_parameters_buffers_exact') or not r.get('reference_exact'):
                    raise ValueError('Real-model smoke failed state checks')
            state['smokes'][arm] = str(folder)
            save(state_path, state)
        if previous and previous.get('input_audit'):
            audit=Path(previous['input_audit'])
            if read(audit/'INPUT_ACCEPTANCE.json').get('scenes')!=1405 or read(audit/'RESULT.json').get('status')!='COMPLETE':raise ValueError('Prior input audit incomplete')
        else:
            audit=out/'inputs'
            command=[sys.executable,'-u',str(ROOT/'run.py'),*base,'--mode','inputs','--arm','F','--seed',str(cfg['training_seeds'][0]),'--out',str(audit)]
            if child(command,out/'inputs.log',deadline,state_path,state,{**os.environ,'CUDA_VISIBLE_DEVICES':preflight['devices'][0]})!=0:raise RuntimeError('Full multimodal input audit failed')
        state['input_audit']=str(audit)
        save(state_path,state)
        # Verify a real joint optimizer continuation, including both module groups.
        if previous and previous.get('resume_check'):
            resume_check=Path(previous['resume_check'])
            if read(resume_check).get('status')!='PASS':raise ValueError('Prior real resume check failed')
        else:
            checkpoints={}
            for name,parent in [('continuous',None),('resumed',read(Path(state['smokes']['A'])/'RESULT.json')['checkpoint'])]:
                folder=out/('resume_'+name)
                command=[sys.executable,'-u',str(ROOT/'run.py'),*base,'--mode','smoke','--arm','A',
                         '--seed',str(cfg['training_seeds'][0]),'--out',str(folder),'--smoke-updates','2',
                         '--smoke-wall-seconds',str(max(1,min(1800,int(remaining(deadline)))))]
                if parent:command+=['--resume',parent]
                if child(command,out/('resume_'+name+'.log'),deadline,state_path,state,{**os.environ,'CUDA_VISIBLE_DEVICES':preflight['devices'][0]})!=0:raise RuntimeError('Real continuation smoke failed: '+name)
                checkpoints[name]=read(folder/'RESULT.json')['checkpoint']
            resume_check=out/'RESUME_ACCEPTANCE.json'
            command=[sys.executable,str(ROOT/'verify_resume.py'),'--continuous',checkpoints['continuous'],
                     '--resumed',checkpoints['resumed'],'--out',str(resume_check)]
            if child(command,out/'verify_resume.log',deadline,state_path,state)!=0:raise RuntimeError('Real model/optimizer/RNG continuation differs')
        state['resume_check']=str(resume_check)
        save(state_path,state)
        if len(preflight['devices'])>1:
            parallel_smokes(base,preflight['devices'],out,deadline,state_path,state,cfg)
        save(out/'ENGINEERING_ACCEPTANCE.json', {'status':'PASS','scope':'three scopes and all multimodal C/F layouts',
              'smokes':state['smokes'],'input_audit':str(audit),'resume_check':str(resume_check),'code':ident['code'],'multicard_accepted':len(preflight['devices'])>1})
        if a.action == 'smoke':
            state.update(status='SMOKE_COMPLETE',remaining_s=remaining(deadline))
            save(state_path,state)
            return 0
        for phase in ['initial','train','final']:
            old_folder = previous.get('phases',{}).get(phase) if previous else None
            if old_folder and (Path(old_folder)/'RESULT.json').exists() and read(Path(old_folder)/'RESULT.json')['status']=='COMPLETE':
                from schedule import accepted_job,jobs
                receipt=read(Path(old_folder)/'RESULT.json')
                expected={(j['arm'],j['seed']) for j in jobs(cfg,phase)}
                if receipt['phase']!=phase or len(receipt['jobs'])!=len(expected) or {(j['job']['arm'],j['job']['seed']) for j in receipt['jobs']}!=expected:
                    raise ValueError('Recovered phase job coverage differs')
                for item in receipt['jobs']:accepted_job(item['output'],item['job'],phase,cfg)
                state['phases'][phase]=old_folder
                save(state_path,state)
                continue
            folder = out/phase
            state['phases'][phase] = str(folder)
            state['stage'] = phase
            save(state_path,state)
            command = [sys.executable,'-u',str(ROOT/'schedule.py'),*base,'--phase',phase,
                       '--devices',','.join(preflight['devices']),'--out',str(folder),
                       '--scheduler-wall-seconds',str(remaining(deadline))]
            if phase != 'initial':
                command += ['--previous',state['phases']['initial' if phase=='train' else 'train']]
            if phase == 'train' and old_folder and (Path(old_folder)/'TRAINING.json').exists():
                command += ['--recover-from',old_folder]
            if child(command,out/(phase+'.scheduler.log'),deadline,state_path,state) != 0:
                raise RuntimeError('Phase incomplete: '+phase)
            if read(folder/'RESULT.json')['status'] != 'COMPLETE':
                raise ValueError('Phase did not complete: '+phase)
        state['stage']='analysis'
        command=[sys.executable,str(ROOT/'analyze.py'),'--initial',state['phases']['initial'],
                 '--final',state['phases']['final'],'--prepared',str(Path(a.prepared).resolve()),'--out',str(out/'analysis.json')]
        if child(command,out/'analysis.log',deadline,state_path,state) != 0:
            raise RuntimeError('Paired analysis failed')
        state.update(status='COMPLETE',remaining_s=remaining(deadline))
        save(state_path,state)
        return 0
    except BaseException as exc:
        state.update(status='INCOMPLETE',error_type=type(exc).__name__,error=str(exc),remaining_s=remaining(deadline))
        save(state_path,state)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','check','smoke','run'])
    p.add_argument('--assets',default=str(ROOT/'assets.local.json'))
    p.add_argument('--training',default=str(ROOT/'TRAINING.confirmed.json'))
    p.add_argument('--prepared',default=str(ROOT/'prepared_v4'))
    p.add_argument('--out',help='New output directory; default runs/<action>_<UTC>_<pid>')
    p.add_argument('--devices',default='auto')
    p.add_argument('--deep',action='store_true',help='Hash checkpoint and all raw assets; verify images')
    p.add_argument('--cpu-only',action='store_true',help='check only; never claim accelerator readiness')
    p.add_argument('--resume-from')
    a=p.parse_args()
    if a.cpu_only and a.action not in ['check','plan']:
        p.error('--cpu-only is valid only for check/plan')
    out=Path(a.out).resolve() if a.out else ROOT/'runs'/f'{a.action}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}_{os.getpid()}'
    out.mkdir(parents=True,exist_ok=False)
    configure_environment(out)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        cfg=read(a.training)
        result=report(a,cfg,check_assets=a.action!='plan',check_device=a.action!='plan' and not a.cpu_only)
        if result['status']=='READY' and a.action in ['plan','check'] and not result['device_checked']:
            result['status']='PLAN_READY' if a.action=='plan' else 'CPU_READY'
        save(out/'PREFLIGHT.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        if a.action in ['plan','check']:
            return 0 if result['status'] in ['READY','PLAN_READY','CPU_READY'] else 2
        if result['status']!='READY':
            return 2
        return pipeline(a,cfg,result,out)
    except BaseException as exc:
        save(out/'ERROR.json',{'type':type(exc).__name__,'message':str(exc)})
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())

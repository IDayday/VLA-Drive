"""Controller state contracts and owned synthetic-child expiry, no real grant edits."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from tools.qwen_world_postcot import campaign as campaign
from tools.qwen_world_postcot.job_worker import deadline_tick


def test_selection_requires_complete_window_and_uses_dev_without_gain_gate():
    rows=[dict(updates=step,checkpoint=f'/unit/{step}',model_sha256=str(step),
               PDMS=50.,qualification=False,cohort='DEV') for step in campaign.IL_MILESTONES]
    value=campaign.select_checkpoint(rows,kind='IL',completed_updates=100000)
    assert value['selected']['updates']==100000
    assert value['Navtest_used'] is False
    with pytest.raises(ValueError,match='window'):
        campaign.select_checkpoint(rows[:-1],kind='IL',completed_updates=100000)
    rows[0]['PDMS']=float('nan')
    with pytest.raises(ValueError,match='valid'):
        campaign.select_checkpoint(rows,kind='IL',completed_updates=100000)


def test_no_formal_dispatch_without_actual_engineering_acceptance(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    monkeypatch.setattr(campaign,'accepted',lambda source:(False,'WAITING_ENGINEERING_ACCEPTANCE'))
    monkeypatch.setattr(campaign,'ensure_training',lambda *args,**kwargs:pytest.fail('Unexpected formal dispatch'))
    result=campaign.tick('/unit/source',dict(source='/unit/source',jobs={}))
    assert result['stage']=='WAITING_ENGINEERING_ACCEPTANCE'


def test_idle_permanent_precedes_idle_temporary_and_known_sharing(monkeypatch):
    monkeypatch.setattr(campaign,'authorization',lambda:({'permanent_hosts':['permanent','shared']},{}))
    monkeypatch.setattr(campaign,'allowed_hosts',lambda **kwargs:['temporary','shared','permanent'])
    monkeypatch.setattr(campaign.time,'sleep',lambda value:None)
    def sample(host):
        return dict(worker_python_exists=True,gpus=[dict(index=i,uuid=f'{host}{i}',total_MiB=81920,free_MiB=80000,utilization=0)for i in range(8)],
            owners=[dict(visible=True,output='/mnt/project/qwen-sourceflow-own/run')]if host=='shared'else[])
    monkeypatch.setattr(campaign,'probe',sample)
    assert campaign.choose_training_host(50000)['host']=='permanent'
    assert campaign.choose_training_host(50000,excluded_uuids=['permanent0'])['host']=='temporary'


def test_intent_adoption_and_verified_exit_prevent_duplicate_migration(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    root=tmp_path/'jobs/owned';root.mkdir(parents=True)
    job=dict(name='formal_IL',role='training',receipt=str(root),host='unit',uuids=['owned_gpu'],
             gpus=[0],command=['unit'],status='RUNNING',dispatch_unix=time.time())
    (root/'CAMPAIGN_INTENT.json').write_text(json.dumps(dict(source='/unit/source',invocation='owned',job=job)))
    (root/'RESULT.json').write_text(json.dumps(dict(returncode=0,status='EXITED',host='unit',command=['unit'],
        cwd='/unit/source',gpus=[0],gpu_uuids=['owned_gpu'])))
    state=dict(source='/unit/source',jobs={})
    monkeypatch.setattr(campaign,'owned_exit_verified',lambda record:False)
    campaign.poll_jobs(state)
    assert state['jobs']['owned']['status']=='WAITING_OWNED_EXIT'
    assert campaign.claimed_uuids(state)=={'owned_gpu'}
    monkeypatch.setattr(campaign,'owned_exit_verified',lambda record:True)
    campaign.poll_jobs(state)
    assert state['jobs']['owned']['status']=='COMPLETE'
    assert not campaign.claimed_uuids(state)


def test_formal_rl_initialization_never_uses_qualification_weights(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'ROOT',tmp_path);monkeypatch.setattr(campaign,'RL_RUN',tmp_path/'formal')
    monkeypatch.setattr(campaign,'choose_training_host',lambda *args,**kwargs:dict(host='unit',gpus=list(range(8)),uuids=[],share=None))
    commands=[]
    monkeypatch.setattr(campaign,'dispatch',lambda state,source,name,command,route,**kwargs:commands.append(command))
    campaign.ensure_training(dict(jobs={}),'/unit/source',dict(minimum_free_MiB=64000),'RL',
        dict(selected={'checkpoint':'/unit/frozen_real_IL'}))
    assert commands[0][commands[0].index('--il-checkpoint')+1]=='/unit/frozen_real_IL'
    assert '--resume' not in commands[0] and '--qualification' not in commands[0]
    assert 'RL_from_selected_full_IL' not in ' '.join(commands[0])


def child_state(child,command):
    return dict(temporary=True,guard_unix=100.,deadline_unix=1000.,child_pid=child.pid,
        child_start_ticks=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]),command=command)


def test_absolute_expiry_kills_only_owned_hung_child_and_records_nonlossless(tmp_path):
    command=[sys.executable,'-u','-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print("READY",flush=True);time.sleep(60)']
    child=subprocess.Popen(command,stdout=subprocess.PIPE,text=True,start_new_session=True)
    try:
        assert child.stdout.readline().strip()=='READY'
        state=child_state(child,command);stop=tmp_path/'STOP'
        deadline_tick(state,child,tmp_path,stop,now=101.)
        assert stop.exists() and child.poll() is None and not state.get('forced_expiry')
        deadline_tick(state,child,tmp_path,stop,now=910.)
        assert state['forced_expiry']
        assert child.poll() is None
        deadline_tick(state,child,tmp_path,stop,now=970.)
        assert child.wait(timeout=3)==-signal.SIGKILL
        receipt=json.loads((tmp_path/'FORCED_EXPIRY.json').read_text())
        assert 'FORCED_EXIT_REVIEW_REQUIRED' in receipt['recovery_contract']
        assert 'uncommitted work may be lost' in receipt['recovery_contract']
        assert receipt['expiry_SIGKILL']['leader_identity_verified']
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()


def test_expiry_refuses_changed_pid_start_identity(tmp_path):
    command=[sys.executable,'-c','import time;time.sleep(60)']
    child=subprocess.Popen(command,start_new_session=True)
    try:
        state=child_state(child,command);state['child_start_ticks']+=1
        with pytest.raises(RuntimeError,match='identity changed'):
            deadline_tick(state,child,tmp_path,tmp_path/'STOP',now=910.)
        assert child.poll() is None
    finally:
        os.killpg(child.pid,signal.SIGKILL);child.wait()


def test_guard_signals_real_owned_synthetic_rank_handler_not_launcher(tmp_path):
    module=tmp_path/'tools/qwen_world_postcot';module.mkdir(parents=True)
    (tmp_path/'tools/__init__.py').write_text('');(module/'__init__.py').write_text('')
    marker=tmp_path/'USR1_RECEIVED'
    (module/'train_il.py').write_text('import signal,time\nfrom pathlib import Path\n'
        +'signal.signal(signal.SIGUSR1,lambda *_:Path('+repr(str(marker))+').write_text("received"))\n'
        +'print("READY",flush=True)\ntime.sleep(60)\n')
    command=[sys.executable,'-u','-m','tools.qwen_world_postcot.train_il','--output',str(tmp_path/'synthetic_run')]
    child=subprocess.Popen(command,cwd=tmp_path,env=dict(os.environ,PYTHONPATH=str(tmp_path)),
        stdout=subprocess.PIPE,text=True,start_new_session=True)
    try:
        assert child.stdout.readline().strip()=='READY'
        state=child_state(child,command)
        deadline_tick(state,child,tmp_path,tmp_path/'STOP',now=101.)
        for _ in range(100):
            # write_text creates the file before the handler writes its body;
            # wait for the acknowledged content, not merely its directory entry.
            if marker.exists() and marker.read_text()=='received':break
            time.sleep(.01)
        assert marker.read_text()=='received'
        assert state['native_usr1_signals'][0]['pid']==child.pid
        assert child.poll() is None
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()


def test_expiry_covers_torchrun_style_separate_rank_session(tmp_path):
    module=tmp_path/'tools/qwen_world_postcot';module.mkdir(parents=True)
    (tmp_path/'tools/__init__.py').write_text('');(module/'__init__.py').write_text('')
    marker=tmp_path/'RANK_USR1';output=str(tmp_path/'synthetic_out')
    (module/'train_il.py').write_text('import signal,time\nfrom pathlib import Path\n'
        +'signal.signal(signal.SIGUSR1,lambda *_:Path('+repr(str(marker))+').write_text("received"))\n'
        +'signal.signal(signal.SIGTERM,signal.SIG_IGN)\nprint("RANK_READY",flush=True)\ntime.sleep(60)\n')
    rank_command=[sys.executable,'-u','-m','tools.qwen_world_postcot.train_il','--output',output]
    code='import subprocess,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);p=subprocess.Popen('+repr(rank_command)+',start_new_session=True);print(p.pid,flush=True);time.sleep(60)'
    command=[sys.executable,'-u','-c',code,'--output',output]
    child=subprocess.Popen(command,cwd=tmp_path,env=dict(os.environ,PYTHONPATH=str(tmp_path)),
        stdout=subprocess.PIPE,text=True,start_new_session=True)
    rank_pid=None
    try:
        rank_pid=int(child.stdout.readline().strip());assert child.stdout.readline().strip()=='RANK_READY'
        state=child_state(child,command);stop=tmp_path/'STOP'
        deadline_tick(state,child,tmp_path,stop,now=101.)
        for _ in range(100):
            if marker.exists():break
            time.sleep(.01)
        assert marker.read_text()=='received'
        rank=state['native_usr1_signals'][0]
        assert rank['pid']==rank_pid and rank['pgid']==rank_pid and rank_pid!=child.pid
        deadline_tick(state,child,tmp_path,stop,now=910.)
        assert child.poll() is None
        deadline_tick(state,child,tmp_path,stop,now=970.)
        assert child.wait(timeout=3)==-signal.SIGKILL
        from tools.qwen_world_postcot.job_worker import verified_process
        for _ in range(100):
            if not verified_process(rank_pid,rank['start_ticks'],rank_pid):break
            time.sleep(.01)
        assert not verified_process(rank_pid,rank['start_ticks'],rank_pid)
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if rank_pid is not None:
            try:os.killpg(rank_pid,signal.SIGKILL)
            except ProcessLookupError:pass


def test_explicit_resource_refusal_has_no_child_and_bounded_retry(tmp_path,monkeypatch):
    from tools.qwen_world_postcot import job_worker as worker
    root=tmp_path/'worker';command=[sys.executable,'-c','raise SystemExit(0)']
    monkeypatch.setattr(sys,'argv',['worker','--receipt-root',str(root),'--gpus','0',
        '--expected-gpu-uuids-json','["owned_gpu"]','--command-json',json.dumps(command),'--cwd',str(tmp_path)])
    monkeypatch.setattr(worker.socket,'gethostname',lambda:'unit-worker-0')
    monkeypatch.setattr(worker,'require_dispatch',lambda **kwargs:'unit')
    monkeypatch.setattr(worker,'authorization',lambda:({'permanent_hosts':['unit']},{}))
    def refused(*_):raise worker.ResourceRefused('GPU lease became busy')
    monkeypatch.setattr(worker,'_gpu_admission',refused)
    monkeypatch.setattr(worker.subprocess,'Popen',lambda *args,**kwargs:pytest.fail('Refusal must occur before any child exists'))
    assert worker.main()==75
    result=json.loads((root/'RESULT.json').read_text())
    assert result['status']=='REFUSED_RESOURCE'and result['child_not_started']is True
    assert 'child_pid'not in result and result['requested_gpu_uuids']==['owned_gpu']
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    state=dict(source=str(tmp_path),jobs={'owned':dict(name='formal_IL',role='training',receipt=str(root),
        host='unit',uuids=['owned_gpu'],gpus=[0],command=command,status='RUNNING')})
    monkeypatch.setattr(campaign,'owned_exit_verified',lambda result:pytest.fail('No child existed to verify'))
    campaign.poll_jobs(state)
    assert state['jobs']['owned']['status']=='REFUSED_RESOURCE'
    assert not campaign.claimed_uuids(state)
    assert state['resource_refusals']['formal_IL']==1
    assert time.time()<state['resource_retry_after']['formal_IL']<=time.time()+301


def test_unknown_admission_failure_is_never_reclassified_retryable(tmp_path,monkeypatch):
    from tools.qwen_world_postcot import job_worker as worker
    root=tmp_path/'worker'
    monkeypatch.setattr(sys,'argv',['worker','--receipt-root',str(root),'--command-json','["unit"]','--cwd',str(tmp_path)])
    monkeypatch.setattr(worker,'require_dispatch',lambda **kwargs:'unit')
    monkeypatch.setattr(worker,'authorization',lambda:({'permanent_hosts':['unit']},{}))
    def unknown(*_):raise ValueError('Unidentified initialization bug')
    monkeypatch.setattr(worker,'_gpu_admission',unknown)
    with pytest.raises(ValueError,match='Unidentified'):
        worker.main()
    assert not(root/'RESULT.json').exists()and not(root/'REFUSED_RESOURCE.json').exists()


def test_sampling_variation_is_inference_only_and_keeps_all_three_results_required():
    value=campaign.sampling_summary({2101:{'PDMS':80.},2102:{'PDMS':81.},2103:{'PDMS':82.}})
    assert value['mean_PDMS']==81 and value['population_std_PDMS']==pytest.approx((2/3)**.5)
    assert value['independent_training_seeds']==1
    with pytest.raises(ValueError):campaign.sampling_summary({2101:{'PDMS':80.},2102:{'PDMS':81.}})


def test_controller_variation_wires_seed_role_and_exact_dev_binding(tmp_path,monkeypatch):
    monkeypatch.setattr(campaign,'ROOT',tmp_path)
    route=dict(host='unit',gpus=[0],uuids=['gpu'],share=None)
    monkeypatch.setattr(campaign,'choose_eval_routes',lambda *args,**kwargs:[route])
    commands=[]
    def dispatch(state,source,name,command,route,**kwargs):
        commands.append(command);state['jobs'][name]={'status':'RUNNING'};return name
    monkeypatch.setattr(campaign,'dispatch',dispatch)
    state={'jobs':{}}
    campaign.ensure_evaluation(state,'/unit/source','NAVTEST_seed2102_IL',
        {'path':'/unit/export','model_sha256':'model'},'NAVTEST',{},
        dev_binding='/unit/DEV_seed2101',seed=2102,seed_variation=True)
    command=commands[0]
    assert command[command.index('--seed')+1]=='2102'
    assert '--post-selection-seed-variation'in command
    assert command[command.index('--dev-evaluation')+1]=='/unit/DEV_seed2101'


def test_audited_summary_reuse_requires_unchanged_all_scene_versions_and_seal(tmp_path):
    root=tmp_path/'eval';(root/'scenes').mkdir(parents=True)
    (root/'identity.json').write_text('{}');(root/'scenes/a.json').write_text('{"value":0}')
    before=campaign._scene_file_version(root)
    value=dict(status='COMPLETE',PDMS=50.);value['sha256']=campaign.fingerprint(value)
    (root/'SUMMARY.json').write_text(json.dumps(value))
    assert campaign._sealed_summary(root/'SUMMARY.json')==value
    assert campaign._scene_file_version(root)==before
    # Same-size mutation still changes immutable version and forces a full audit.
    (root/'scenes/a.json').write_text('{"value":1}')
    assert campaign._scene_file_version(root)!=before
    value['PDMS']=60.;(root/'SUMMARY.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='summary'):
        campaign._sealed_summary(root/'SUMMARY.json')


def test_invisible_gpu_activity_and_unaccounted_memory_are_not_idle_or_safe_sharing():
    from starVLA.rl.qwen_world_postcot.resources import gpu_activity_issue
    idle=dict(uuid='physical',total_MiB=81920,free_MiB=81150,utilization=0)
    assert gpu_activity_issue(idle,[])is None
    assert gpu_activity_issue(dict(idle,utilization=100),[])=='GPU active without a reported owner'
    assert 'unexplained'in gpu_activity_issue(dict(idle,free_MiB=62000),[])
    assert 'Missing actual'in gpu_activity_issue(dict(uuid='physical',free_MiB=80000,utilization=0),[])
    owner=[dict(uuid='physical',used_MiB=19000)]
    assert gpu_activity_issue(dict(idle,free_MiB=62000,utilization=100),owner)is None
    assert 'exceeds reported owners'in gpu_activity_issue(dict(idle,free_MiB=50000,utilization=100),owner)


def test_inventory_and_full_training_host_skip_invisible_active_nodes(monkeypatch):
    from starVLA.rl.qwen_world_postcot import resources
    def sample(host):
        return dict(worker_python_exists=True,owners=[],gpus=[dict(index=i,name='A800',uuid=f'{host}{i}',
            total_MiB=81920,free_MiB=62000 if host=='unreported_allocation' else 81150,
            utilization=100 if host=='invisible_active' else 0)for i in range(8)])
    hosts=['invisible_active','unreported_allocation','idle']
    monkeypatch.setattr(resources,'allowed_hosts',lambda:hosts)
    monkeypatch.setattr(resources,'probe',sample);monkeypatch.setattr(resources.time,'sleep',lambda *_:None)
    routes,_=resources.inventory(minimum_free_MiB=50000)
    assert len(routes)==8 and {route['host']for route in routes}=={'idle'}
    monkeypatch.setattr(campaign,'authorization',lambda:({'permanent_hosts':hosts},{}))
    monkeypatch.setattr(campaign,'allowed_hosts',lambda **kwargs:hosts)
    monkeypatch.setattr(campaign,'probe',sample)
    assert campaign.choose_training_host(50000)['host']=='idle'


def test_underlock_reprobe_catches_invisible_gpu_acquisition_and_keeps_known_sharing(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from tools.qwen_world_postcot import job_worker as worker
    idle=dict(index=0,uuid='physical',total_MiB=81920,free_MiB=81150,utilization=0)
    busy=dict(idle,free_MiB=62000,utilization=100)
    monkeypatch.setattr(worker,'PRIMARY_LEASE_ROOT',tmp_path/'exclusive')
    monkeypatch.setattr(worker,'SHARED_LEASE_ROOT',tmp_path/'shared')
    samples=iter([({0:idle},[]),({0:busy},[])])
    monkeypatch.setattr(worker,'_admission_sample',lambda:next(samples))
    args=SimpleNamespace(share_min_free_mib=None,minimum_free_mib=50000)
    locks=[]
    try:
        with pytest.raises(worker.ResourceRefused,match='without a reported owner under admission lock'):
            worker._gpu_admission(args,'permanent',{'permanent_hosts':['permanent']},[0],['physical'],{},locks)
        assert len(locks)==1  # refusal happened after acquiring, not only scheduler sampling
    finally:
        for handle in locks:handle.close()
    owner=dict(pid=100,uuid='physical',used_MiB=19000)
    monkeypatch.setattr(worker,'_admission_sample',lambda:({0:busy},[owner]))
    verified=[]
    monkeypatch.setattr(worker,'_known_owners',lambda rows,*_:verified.append(rows))
    args.share_min_free_mib=50000;locks=[]
    try:
        worker._gpu_admission(args,'permanent',{'permanent_hosts':['permanent']},[0],['physical'],{},locks)
        assert len(verified)==2 and len(locks)==1
    finally:
        for handle in locks:handle.close()

"""Inactive administrative startup option; CPU metadata and known-entry checks."""
import json
from pathlib import Path
import pytest
from tools.d46_priority_rl import owner
from tools.qwen_world_compare import job_worker as worker
from starVLA.rl.qwen_world_compare.contracts import file_sha,fingerprint


def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))

@pytest.fixture
def bound(monkeypatch,tmp_path):
    monkeypatch.setattr(owner,'ROOT',tmp_path)
    source=tmp_path/'SCI';wrapper=source/'tools/d46_priority_rl/train_control.py';wrapper.parent.mkdir(parents=True);wrapper.write_text('# CPU wrapper fixture')
    math=source/'math.py';math.write_text('# original mathematical source')
    output=tmp_path/'runs/D46_D2';cp=output/'checkpoints/update001614_inner0';cp.mkdir(parents=True)
    config=tmp_path/'cfg.yaml';config.write_text('{}')
    identity=dict(world=8,qualification=False,sha256='own_identity',source_files={'math.py':file_sha(math)},recipe={'arm':'D2'})
    put(output/'identity.json',identity);put(cp/'identity.json',identity);put(cp/'files.json',{})
    put(cp/'COMPLETE',dict(world=8,updates=1614,files_sha256=file_sha(cp/'files.json')))
    p=dict(schema='d46_world8_control_startup_execution_profile_v1',protocol=dict(schema='d46_world8_CPU_control_timeout600_v1',world=8,backend='gloo',timeout_seconds=600,original_timeout_seconds=90,changes='CPU startup/control wait only',policy_optimizer_RNG_identity_unchanged=True),
        science_source=str(source),wrapper_file='tools/d46_priority_rl/train_control.py',wrapper_sha256=file_sha(wrapper),parent_model_sha256=owner.MODEL_SHA,
        arm='D2',output=str(output),identity_sha256=identity['sha256'],config=str(config),config_sha256=file_sha(config),source_files_sha256=fingerprint(identity['source_files']),anchor_updates=1614)
    p['sha256']=fingerprint(p);profile=tmp_path/'control.json';put(profile,p)
    run=dict(arm='D2',output=str(output),source='original_science',startup_control=dict(science_source=str(source),source_sha='frozen',profile=str(profile),profile_sha256=file_sha(profile)))
    monkeypatch.setattr(owner.subprocess,'check_output',lambda *a,**kw:'frozen')
    monkeypatch.setattr(worker,'known_D46_output',lambda *a:True)
    args=['python','-m','tools.d46_priority_rl.train_control','--config',str(config),'--output',str(output),'--resume',str(cp),'--control-profile',str(profile),'--control-profile-sha256',file_sha(profile)]
    return run,config,cp,p,profile,args


def test_explicit_profile_only_original_default_and_qualification_unchanged(bound):
    run,cfg,cp,p,profile,args=bound
    result=owner.startup_control(run,str(cfg),str(cp))
    assert result['entry']=='tools.d46_priority_rl.train_control'and result['source']==run['startup_control']['science_source']
    assert result['args']==args[-4:]
    assert owner.startup_control(run,str(cfg),str(cp),True)is None
    legacy=dict(run);legacy.pop('startup_control')
    assert owner.startup_control(legacy,str(cfg),str(cp))is None
    assert worker.known_owner_args(args)


def test_profile_source_or_timeout_tampering_is_rejected(bound):
    run,cfg,cp,p,profile,args=bound
    (Path(p['science_source'])/'math.py').write_text('# altered math')
    with pytest.raises(ValueError,match='science differs'):owner.startup_control(run,str(cfg),str(cp))
    p['protocol']['timeout_seconds']=900;p.pop('sha256');p['sha256']=fingerprint(p);put(profile,p)
    bad=list(args);bad[-1]=file_sha(profile)
    assert not worker.known_owner_args(bad)


def test_owned_rank_list_includes_only_explicit_wrapper_entry(tmp_path):
    import subprocess,sys,time
    output=str(tmp_path/'owned')
    # A harmless CPU child with the exact argv shape exercises real /proc
    # ancestry/startticks. It does not import or execute the training module.
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)','-m','tools.d46_priority_rl.train_control','--output',output])
    try:
        start=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21])
        state=dict(command=['python','-m','tools.d46_priority_rl.train_control','--output',output],child_pid=child.pid,child_start_ticks=start)
        matches=[r for r in worker.task_ranks(state)if r['pid']==child.pid]
        assert len(matches)==1 and matches[0]['entry']=='tools.d46_priority_rl.train_control'
        state['child_start_ticks']+=1
        assert not any(r['pid']==child.pid for r in worker.task_ranks(state))
    finally:
        child.terminate();child.wait(timeout=5)

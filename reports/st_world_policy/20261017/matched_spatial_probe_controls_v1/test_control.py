import ast
import copy
import importlib.util
import os
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('matched_control',HERE/'control.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

@pytest.mark.parametrize('arm,tokens',[('SP1',144),('SP2',144),('SP3',288)])
def test_fixed_bindings(arm,tokens):
    p,row,s=c.controls(arm,supervisor=True)
    assert p.TAG==row['checkpoint']['tag'] and s.TAG==p.TAG
    assert str(s.JOB)==row['proposed_job_root'] and row['memory_tokens']==tokens
    assert s.ROLE_READ_LIMIT==256<<30 and s.FIT_READ_LIMIT==2<<30
    assert c.module('observed_cli').Counters.snapshot.__code__.co_filename==str(c.UP/'observed_cli.py')

@pytest.mark.parametrize('arm',['SP0','SP4','../../SP1','S-P1','custom'])
def test_no_unfixed_arm(arm):
    with pytest.raises(ValueError):c.controls(arm)

def test_exit_patch_is_only_three_lines():
    source=c.raw(c.UP/'supervise_once.py',c.PINS['supervise_once.py']).decode()
    patched=c.patch_exit_race(source)
    ast.parse(patched)
    assert patched.replace("                    active = owner.discover()  # recheck after WNOWAIT, as reviewed 4b2f monitor\n                    if active:\n                        reason = reason or 'OWNED_DESCENDANTS_AFTER_LEADER_EXIT'", "                    reason = reason or 'OWNED_DESCENDANTS_AFTER_LEADER_EXIT'")==source
    with pytest.raises(ValueError):c.patch_exit_race(patched)

@pytest.mark.parametrize('field',['manual_retry','previous_stage_result','previous_artifact_reconciliation','storage_compatibility'])
def test_no_legacy_exception(field):
    _,_,s=c.controls('SP1',supervisor=True)
    with pytest.raises(ValueError):s.check_manual_fields({field:{}},'train_plan')

def test_nonobserver_command_rejected():
    _,_,s=c.controls('SP2',supervisor=True)
    with pytest.raises(ValueError):s.supervise([sys.executable,'-c','print(1)'])

@pytest.mark.parametrize('field,value',[('matched_arm','SP2'),('automatic_retry',True),('reviewer','/root/st_indexprep_boundary_fix'),('matched_controller',{})])
def test_changed_identity_rejected(field,value):
    _,row,s=c.controls('SP1',supervisor=True)
    a={'matched_arm':'SP1','automatic_retry':False,'reviewer':'independent','matched_controller':c.self_pin()}
    a[field]=value
    with pytest.raises(ValueError):s.validate(a,{},'dev_plan',0)


def test_real_normal_exit_and_real_orphan_still_fails(tmp_path):
    _,_,s=c.controls('SP1',supervisor=True)
    # Call the original bounded function; no run/admission or model is exercised.
    args=dict(cwd=tmp_path,env=dict(os.environ),cpus=[2,3],descriptors=(),wall=3,rss=1<<30,rss_reader=lambda _:0)
    normal=tmp_path/'normal';normal.mkdir()
    with s.lifetime_scope() as stopped:
        result=s._supervise([sys.executable,'-c','print("fixture")'],attempt=normal,popen=s.subprocess.Popen,stopped=stopped,**args)
    assert result['exit_code']==0 and result['limit_reason'] is None
    orphan=tmp_path/'orphan';orphan.mkdir()
    with s.lifetime_scope() as stopped:
        result=s._supervise([sys.executable,'-c','import os,time; p=os.fork(); os._exit(0) if p else time.sleep(15)'],attempt=orphan,popen=s.subprocess.Popen,stopped=stopped,**args)
    assert result['limit_reason']=='OWNED_DESCENDANTS_AFTER_LEADER_EXIT'
    assert result['process_cleanup']['live_owned_remaining']==0


def test_stale_active_leader_snapshot_is_rechecked(tmp_path,monkeypatch):
    _,_,s=c.controls('SP3',supervisor=True)
    original=s.OwnedSession.discover
    injected=[]
    def stale_once(owner):
        active=original(owner)
        if not active and owner.leader_done() and not injected:
            injected.append(True)
            return [owner.anchor]  # exactly the pre-exit discovery/post-exit WNOWAIT race
        return active
    monkeypatch.setattr(s.OwnedSession,'discover',stale_once)
    with s.lifetime_scope() as stopped:
        result=s._supervise([sys.executable,'-c','print("race fixture")'],cwd=tmp_path,env=dict(os.environ),
            cpus=[2,3],descriptors=(),attempt=tmp_path,wall=3,rss=1<<30,rss_reader=lambda _:0,
            popen=s.subprocess.Popen,stopped=stopped)
    assert injected and result['exit_code']==0 and result['limit_reason'] is None
    assert result['process_cleanup']['live_owned_remaining']==0

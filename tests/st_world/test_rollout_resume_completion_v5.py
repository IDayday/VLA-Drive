"""One real CPU lifecycle regression; no model/checkpoint/GPU/scorer use."""
import importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path

FILE=Path(__file__).resolve().parents[2]/'resources/st_rollout_resume_completion_v5.py'
SPEC=importlib.util.spec_from_file_location('completion_v5',FILE)
c=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(c)


def wait(predicate):
    deadline=time.monotonic()+8
    while time.monotonic()<deadline:
        result=predicate()
        if result:return result
        time.sleep(.02)
    raise AssertionError('Owned CPU fixture did not reach its bounded boundary')


def test_normal_launcher_exit_drains_proven_cpu_orphan_before_completion(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path/'completion');v=c.legacy()
    monkeypatch.setattr(v,'ROOT',c.ROOT)
    reg=c.ROOT/'interrupted/REGISTRATION.json';reg.parent.mkdir(parents=True)
    reg.write_text(json.dumps({'explicit_fixture':'no model loaded'}))
    observer=c.ROOT/'interrupted/observer';observer.mkdir()
    for rank in range(4):
        (observer/f'INITIAL_OBSERVER_COMPLETE_rank{rank}.json').write_text(json.dumps({
            'rank':rank,'observer_sha256':c.OLD_SHA,'runtime_sha':v.SOURCE,'completed':2,'registration':v.pin(reg)}))
    validated=[]
    def validate(arm,step,updates):
        validated.append((arm,step,updates));return {'explicit_fixture':'valid completed2 receipt'}
    monkeypatch.setattr(v,'validate_result',validate)
    pidfile=tmp_path/'child';release=tmp_path/'release'
    childcode=f"import os,time;from pathlib import Path;Path({str(pidfile)!r}).write_text(str(os.getpid()));time.sleep(60)"
    parentcode=f"import subprocess,sys,time;from pathlib import Path;subprocess.Popen([sys.executable,'-c',{childcode!r}],start_new_session=True);\nwhile not Path({str(release)!r}).exists():time.sleep(.01)"
    parent=subprocess.Popen([sys.executable,'-c',parentcode],start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    foreign=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    job={'child':parent,'owned':{},'start_ticks':None};stop=v.stop_smoke_owned;drained=[]
    def drain(jobs):
        assert validated==[('interrupted',2,2)]
        drained.append(True);return stop(jobs)
    monkeypatch.setattr(v,'stop_smoke_owned',drain)
    try:
        wait(pidfile.exists);child=int(pidfile.read_text())
        assert child in v.owned_processes(job) # Proof while parent is alive.
        release.touch();wait(lambda:parent.poll()==0)
        assert v.process_record(child)['state']!='Z' # Normal root exited; CPU child remains.
        result=c.finish_successful_job(job,'initial')
        assert result['explicit_fixture']=='valid completed2 receipt'
        assert drained==[True] and job['cleanup_complete']
        assert not any(row['state']!='Z' for row in v.owned_processes(job).values())
        assert foreign.poll() is None # Unrelated jobs are never signalled.
    finally:
        stop([job])
        if foreign.poll() is None:foreign.terminate()
        foreign.wait(timeout=5)

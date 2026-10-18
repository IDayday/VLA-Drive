import copy,importlib.util,itertools,json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest import mock
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('T3control',HERE/'controller.py');w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
c=w.controller()
class Control(unittest.TestCase):
    def test_fixed_real_inputs_command_and_schedule(self):
        i=json.loads((HERE/'INPUTS.json').read_text());cfg=c.read(i['pins']['config']['path']);c.validate_recipe(cfg)
        self.assertIn('--nproc_per_node=1',c.command());self.assertIn('1920s',c.command())
        self.assertEqual(c.registered_config(cfg),cfg);self.assertEqual(c.SOURCE,w.b.SOURCE)
        self.assertEqual(c.storage_requirement(0),(1024+2703)*(1<<30))
        with self.assertRaises(ValueError):c.command(True)
    def test_shared_gpu_free_and_uuid(self):
        with mock.patch.object(w,'protected_gpu_processes',return_value=[]),mock.patch.object(c,'EXPECTED_PROTECTED_PROCESSES',[],create=True):
            for text in ['GPU-other,81920,8000,73920,100',w.b.GPU_UUID+',81920,13000,68920,100']:
                with mock.patch.object(c.subprocess,'check_output',return_value=text),self.assertRaises(RuntimeError):c.gpu_record(True)
            with mock.patch.object(c.subprocess,'check_output',return_value=w.b.GPU_UUID+',81920,8336,73584,100'):
                actual=c.gpu_record(True)[0];self.assertTrue(actual['shared']);self.assertFalse(actual['exclusive_performance_claim'])
            with mock.patch.object(w,'protected_gpu_processes',return_value=[{'pid':123}]),mock.patch.object(c.subprocess,'check_output',return_value=w.b.GPU_UUID+',81920,8336,73584,100'),self.assertRaises(RuntimeError):c.gpu_record(True)
    def test_source_transform_rejects_unreviewed_donor(self):
        with self.assertRaises(ValueError):w.adapted_source('unreviewed')
    def test_aggregate_hold_not_double_reserved(self):
        a=dict(schema='st_T3_profile4_resource_admission_v1',status='ADMITTED',host=c.HOST,gpu='0',gpu_uuid=w.b.GPU_UUID,
            shared_gpu=True,minimum_free_gpu_bytes=68<<30,exclusive_performance_claim=False,protected_external_processes=[],
            source_sha=c.SOURCE,run_id=c.RUN_ID,output_root=str(c.OUT),controller_sha256='pin',inputs_sha256='pin',
            checkpoint_reservation_bytes=c.RESERVE,other_commitments_bytes=c.OTHER_COMMITMENTS,free_floor_bytes=c.FREE_FLOOR,
            max_invocation_seconds=1800,outer_timeout_seconds=1920,cleanup_seconds=180,rss_limit_bytes=c.RSS_LIMIT,
            rss_interval_seconds=1.,storage_interval_seconds=10.,logical_read_limit_bytes=w.READ_CAP,
            valid_from_unix=0,valid_until_unix=4000,allowed_actions=['prepare','run'],cpu_affinity=[2,3],
            checkpoint_commitments={'path':'/fixture/CP','sha256':'pin'},
            checkpoint_hold_assignment={'path':str(w.ASSIGNMENT),'sha256':'pin'})
        cp=dict(remaining_bytes=2703<<30,free_floor_bytes=c.FREE_FLOOR,valid_until_unix=4000)
        assignment=dict(parent_checkpoint_commitments=a['checkpoint_commitments'],source_slot='additional_conservative_unissued_GiB',assigned_run_id=c.RUN_ID,
            output_root=str(c.OUT),inputs_sha256='pin',config={'path':'/fixture/config','sha256':'pin'},
            runtime={'path':str(w.b.RUNTIME),'source_sha':c.SOURCE},assigned_bytes=80<<30,aggregate_bytes=2703<<30,
            other_bytes=2623<<30,additional_bytes=0,no_reassignment=True,no_parallel_reuse=True,valid_until_unix=4000)
        def read(p):
            if str(p)=='/fixture/CP':return cp
            if Path(p)==w.ASSIGNMENT:return assignment
            if Path(p)==c.INPUT:return {'pins':{'config':assignment['config']}}
            return a
        with mock.patch.object(c,'digest',return_value='pin'),mock.patch.object(c,'read',side_effect=read),mock.patch.object(c.time,'time',return_value=1000):
            c.admission('/fixture/a','pin','prepare')
            for key,value in [('assigned_run_id','other'),('additional_bytes',80<<30),('other_bytes',2703<<30)]:
                prev=assignment[key];assignment[key]=value
                with self.assertRaises(RuntimeError):c.admission('/fixture/a','pin','run')
                assignment[key]=prev
    def test_original_monitor_read_limit_uses_original_cleanup(self):
        class Owner:
            verified=1
            def __init__(self):self.calls=0;self.finishes=0
            def discover(self):self.calls+=1;return [1] if self.calls<=2 else []
            def leader_done(self):return False
            def finish(self,*,terminate,deadline):self.finishes+=1;self.terminate=terminate;self.deadline=deadline;self.discover();return 0
            def close(self):self.closed=True
        class Guard:
            high=w.READ_CAP+1
            def __init__(self,*args):pass
            def sample(self):return self.high
        with tempfile.TemporaryDirectory() as d:
            owner=Owner();ticks=itertools.count(100)
            with mock.patch.object(c,'OUT',Path(d)),mock.patch.object(c,'BOOT_IO',{'rchar':10}),mock.patch.object(c,'IO_LEGACY',None,create=True),mock.patch.object(c,'OwnedReadGuard',Guard),mock.patch.object(c,'proc_io',return_value={'rchar':10}),mock.patch.object(c,'storage',return_value={}),mock.patch.object(c,'rank_telemetry',return_value={}),mock.patch.object(c.time,'monotonic',side_effect=lambda:next(ticks)):
                _,r=c.monitor_training(None,owner,{'signal':None},lambda _:100,Path(d)/'run','test',started=100,absolute_end=250)
            self.assertEqual(r['reason'],'LOGICAL_READ_LIMIT');self.assertEqual(owner.finishes,1);self.assertTrue(owner.terminate);self.assertTrue(owner.closed)
            self.assertEqual(owner.deadline,250)
    def test_leaf_order_recycled_pid_and_unreadable_counter(self):
        from types import SimpleNamespace
        root=dict(pid=101,start_ticks='10');leaf=dict(pid=102,start_ticks='11');parent=dict(pid=100,start_ticks='1')
        same=lambda a,b:a['pid']==b['pid'] and a['start_ticks']==b['start_ticks']
        owner=SimpleNamespace(anchor=root,same=same,evidence=[dict(child=root,parent=parent),dict(child=leaf,parent=root)])
        calls=[];current={101:root,102:leaf}
        legacy=SimpleNamespace(proc_identity=lambda pid:current[pid])
        def read(proxy,lib):calls.append(proxy.anchor['pid']);return dict(rchar=10)
        guard=w.OwnedReadGuard(owner,legacy,read,dict(rchar=0));guard.sample();self.assertEqual(calls,[102,101])
        calls.clear();current[102]=dict(pid=102,start_ticks='99');guard.sample();self.assertEqual(calls,[101])
        guard.read_owned=mock.Mock(side_effect=PermissionError('unreadable'))
        with self.assertRaises(PermissionError):guard.sample()
    def test_real_cross_sid_read_and_reap(self):
        # Only our new stdlib fixture descendants; no training, GPU or original data.
        mon,_=c.monitor_helpers()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);payload=root/'small.bin';payload.write_bytes(b'x'*(1<<20))
            ready=root/'ready';go=root/'go';done=root/'done'
            leaf="import os,time;from pathlib import Path;os.setsid();Path(%r).touch();\nwhile not Path(%r).exists():time.sleep(.01)\nfor _ in range(8):Path(%r).read_bytes()\nPath(%r).touch();time.sleep(.5)"%(str(ready),str(go),str(payload),str(done))
            program="import subprocess,sys,time;subprocess.run([sys.executable,'-c',%r],check=True);time.sleep(.5)"%leaf
            with mon.lifetime_scope():
                child=subprocess.Popen([sys.executable,'-c',program],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
                owner=mon.OwnedSession(child);end=time.monotonic()+10;finished=False
                try:
                    while not ready.exists() and time.monotonic()<end:owner.discover();time.sleep(.01)
                    self.assertTrue(ready.exists());owner.discover()
                    guard=w.OwnedReadGuard(owner,mon,c.READ_OWNED,dict(rchar=w.BOOT_IO['rchar']))
                    before=guard.sample();go.touch()
                    while not done.exists() and time.monotonic()<end:owner.discover();time.sleep(.01)
                    self.assertTrue(done.exists());mid=guard.sample();self.assertGreaterEqual(mid-before,8<<20)
                    while owner.discover() and time.monotonic()<end:guard.sample();time.sleep(.02)
                    after=guard.sample();self.assertGreaterEqual(after,mid)
                    self.assertEqual(owner.finish(terminate=False,deadline=end),0);finished=True
                    self.assertGreaterEqual(owner.verified,2)
                finally:
                    if not finished:owner.finish(terminate=True,deadline=time.monotonic()+2)
                    owner.close()
if __name__=='__main__':unittest.main(verbosity=2)

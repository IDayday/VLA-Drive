"""Independent narrow matched-arm control tests. Only synthetic owned children."""
import hashlib,importlib.util,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
SOURCE=Path('/mnt/project/VLA-Drive-st-matched-probe-controls-20261017/reports/st_world_policy/20261017/matched_spatial_probe_controls_v1/control.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='c5c8521379b36c979bf8743f12c2a2baf80df5ca5d21941ba2fe9d453540c38c'
spec=importlib.util.spec_from_file_location('independent_matched_control',SOURCE)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class Tests(unittest.TestCase):
    def test_all_three_fixed_bindings_keep_original_resource_limits(self):
        seen=[]
        for arm,tokens in [('SP1',144),('SP2',144),('SP3',288)]:
            p,row,s=c.controls(arm,supervisor=True)
            self.assertEqual(p.TAG,s.TAG);self.assertEqual(str(s.JOB),row['proposed_job_root'])
            self.assertEqual(row['memory_tokens'],tokens)
            self.assertEqual(s.LIMITS,{'plan':(600,64<<30),'extract':(960,64<<30),'fit':(1260,2<<30)})
            self.assertEqual(s.ROLE_READ_LIMIT,256<<30);self.assertEqual(s.FIT_READ_LIMIT,2<<30)
            seen.append(str(s.JOB))
        self.assertEqual(len(set(seen)),3)

    def test_patch_only_changes_exit_snapshot_and_refuses_second_application(self):
        original=c.raw(c.UP/'supervise_once.py',c.PINS['supervise_once.py']).decode()
        modified=c.patch_exit_race(original)
        new="                    active = owner.discover()  # recheck after WNOWAIT, as reviewed 4b2f monitor\n                    if active:\n                        reason = reason or 'OWNED_DESCENDANTS_AFTER_LEADER_EXIT'"
        self.assertEqual(modified.replace(new,"                    reason = reason or 'OWNED_DESCENDANTS_AFTER_LEADER_EXIT'"),original)
        with self.assertRaises(ValueError):c.patch_exit_race(modified)

    def test_old_retry_exceptions_cannot_authorize_new_arm(self):
        _,_,s=c.controls('SP1',supervisor=True)
        for k in ['manual_retry','previous_stage_result','previous_artifact_reconciliation','storage_compatibility']:
            with self.assertRaises(ValueError):s.check_manual_fields({k:{}},'train_plan')
        with self.assertRaises(ValueError):s.supervise([sys.executable,'-c','pass'])
        for name in ['SP0','SP4','../SP1']:
            with self.assertRaises(ValueError):c.controls(name)

    def test_actual_small_dev_plan_validator_and_author_rejection(self):
        _,_,s=c.controls('SP1',supervisor=True)
        root=SOURCE.parent/'prepared/SP1'
        a=json.loads((root/'dev_plan_ADMISSION_TEMPLATE.json').read_text());p=json.loads((root/'PREPARATION.json').read_text())
        a.update(status='APPROVED',reviewer='/root/st_owned_descendants_review_v2',hostname=s.socket.gethostname(),cpu_affinity=[8,9],valid_until_unix=time.time()+3600)
        self.assertEqual(s.validate(a,p,'dev_plan',time.time()),(600,64<<30))
        for field,bad in [('matched_arm','SP2'),('reviewer','/root/st_indexprep_boundary_fix'),('automatic_retry',True),('gpu',{'index':0,'uuid':'GPU-test'})]:
            b=dict(a);b[field]=bad
            with self.assertRaises(ValueError):s.validate(b,p,'dev_plan',time.time())

    def test_plan_does_not_read_prior_results_or_gpu_and_cumulative_extract_charge(self):
        _,_,s=c.controls('SP1',supervisor=True)
        with patch.object(s,'read',side_effect=AssertionError('no prerequisite allowed')):
            _,account,prev=s.stage_inputs({},'dev_plan')
        self.assertIsNone(prev);self.assertEqual(account['remaining_read_bytes'],256<<30)
        a={'preparation':{'path':'fixture','sha256':'a'*64}}
        prior={'schema':'st_SP0_probe_stage_result_v1','status':'COMPLETE','stage':'dev_plan','preparation':a['preparation'],'resources':{'counters':{'hash_logical_bytes_admitted':70,'torch_load_nominal_bytes':30,'proc_self_io_delta':{'rchar':80}}}}
        with patch.object(s,'read',return_value=(prior,{'path':'fixture_result','sha256':'b'*64})):
            _,account,_=s.stage_inputs(a,'dev_extract')
        self.assertEqual(account['prior_read_bytes'],100);self.assertEqual(account['remaining_read_bytes'],(256<<30)-100)

    def test_real_owned_normal_exit_stale_snapshot_and_true_orphan(self):
        # Own synthetic sessions only; original native/data functions never run.
        _,_,s=c.controls('SP1',supervisor=True)
        original=s.OwnedSession.discover;injected=[]
        def stale_once(owner):
            active=original(owner)
            if not active and owner.leader_done() and not injected:
                injected.append(True);return [owner.anchor]
            return active
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            normal=root/'normal';normal.mkdir()
            args=dict(cwd=root,env=dict(os.environ),cpus=[8,9],descriptors=(),wall=3,rss=1<<30,rss_reader=lambda _:0,popen=s.subprocess.Popen)
            with patch.object(s.OwnedSession,'discover',stale_once),s.lifetime_scope() as stopped:
                good=s._supervise([sys.executable,'-c','print("independent exit fixture")'],attempt=normal,stopped=stopped,**args)
            self.assertTrue(injected);self.assertIsNone(good['limit_reason']);self.assertEqual(good['exit_code'],0)
            orphan=root/'orphan';orphan.mkdir()
            with s.lifetime_scope() as stopped:
                bad=s._supervise([sys.executable,'-c','import os,time; n=os.fork(); os._exit(0) if n else time.sleep(10)'],attempt=orphan,stopped=stopped,**args)
            self.assertEqual(bad['limit_reason'],'OWNED_DESCENDANTS_AFTER_LEADER_EXIT')
            self.assertEqual(bad['process_cleanup']['live_owned_remaining'],0)

if __name__=='__main__':unittest.main(verbosity=2)

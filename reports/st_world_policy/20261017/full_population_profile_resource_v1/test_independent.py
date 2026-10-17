"""Independent resource-boundary tests; no actual data, GPU, helper signals or dispatch."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path('/mnt/project/VLA-Drive-st-full-profile-launch-handoff-20261017/reports/st_world_policy/20261017/full_population_profile_launch_v1/controller.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='bd802d47f187e494c2dde72ab8c04c670cb529d2b32b187ca06dfad3f3cc6bd1'
spec=importlib.util.spec_from_file_location('independent_profile_candidate',SOURCE)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class Owner:
    verified=2
    def __init__(self,snapshots,done=True,cleanup_error=False):
        self.snapshots=list(snapshots);self.done=done;self.cleanup_error=cleanup_error
        self.events=[];self.finish_args=None
    def discover(self):
        self.events.append('discover')
        return self.snapshots.pop(0) if self.snapshots else []
    def leader_done(self):self.events.append('leader_done');return self.done
    def finish(self,**kw):
        self.events.append('finish');self.finish_args=kw;self.discover()
        if self.cleanup_error:raise RuntimeError('unwitnessed adopted child')
        return 0
    def close(self):self.events.append('close')

class ResourceTests(unittest.TestCase):
    def run_monitor(self,owner,rss=1,end=1900,now=100,storage_error=False,signal=None):
        with tempfile.TemporaryDirectory() as td:
            self.addCleanup(lambda:None)
            with patch.object(c,'OUT',Path(td)),patch.object(c.time,'monotonic',return_value=now),\
                 patch.object(c,'proc_io',side_effect=[{'rchar':100},{'rchar':400}]),\
                 patch.object(c,'rank_telemetry',return_value={'0':{'rchar':700},'1':{'rchar':900}}),\
                 patch.object(c,'storage',side_effect=RuntimeError('floor') if storage_error else None,return_value={'fixture':True}):
                return c.monitor_training(None,owner,{'signal':signal},lambda pid:rss,Path(td)/'run','independent',started=100,absolute_end=end)

    def test_normal_exit_between_snapshots_is_not_orphan(self):
        o=Owner([[101],[],[]]);rc,r=self.run_monitor(o)
        self.assertEqual(o.events[:3],['discover','leader_done','discover'])
        self.assertEqual(rc,0);self.assertIsNone(r['reason'])
        self.assertFalse(o.finish_args['terminate']);self.assertEqual(o.events.count('finish'),1)

    def test_live_orphan_after_rediscovery_remains_failure(self):
        o=Owner([[101],[102],[]]);rc,r=self.run_monitor(o)
        self.assertEqual(rc,0);self.assertEqual(r['reason'],'OWNED_DESCENDANTS_AFTER_LEADER_EXIT')
        self.assertTrue(o.finish_args['terminate']);self.assertEqual(r['live_owned_remaining'],0)

    def test_rss_failure_cannot_be_cleared_by_successful_native_save(self):
        o=Owner([[101],[]],done=False);rc,r=self.run_monitor(o,rss=c.RSS_LIMIT+1)
        self.assertEqual(rc,0);self.assertEqual(r['reason'],'RSS_LIMIT');self.assertTrue(o.finish_args['terminate'])

    def test_cleanup_uses_original180_and_is_clipped_to_absolute_end(self):
        for end,want in [(1900,280),(230,230)]:
            with self.subTest(end=end):
                o=Owner([[101],[]],done=False);self.run_monitor(o,signal=15,end=end)
                self.assertEqual(o.finish_args['deadline'],want)

    def test_unknown_cleanup_not_reported_zero_live(self):
        o=Owner([[],[]],cleanup_error=True);rc,r=self.run_monitor(o)
        self.assertIsNone(rc);self.assertIsNone(r['live_owned_remaining'])
        self.assertEqual(r['reason'],'OWNERSHIP_CLEANUP_FAILURE');self.assertEqual(o.events[-1],'close')

    def test_storage_error_is_permanent_and_io_is_not_double_added(self):
        o=Owner([[101],[]],done=False);_,r=self.run_monitor(o,storage_error=True)
        self.assertEqual(r['reason'],'MONITOR_ERROR');self.assertTrue(o.finish_args['terminate'])
        self.assertEqual(r['proc_supervisor_after_wait_delta']['rchar'],300)
        self.assertIn('not added again',r['io_scope'])

    def test_only_output_namespace_changes_and_no_metadata_checkpoint(self):
        cfg={'runtime':{'output_root':str(c.FACTORY/'configured/runs'),'max_seconds':1500},'data':{'identity':'immutable'},'optimizer':{'warmup':1000}}
        snapshot=copy.deepcopy(cfg);moved=c.registered_config(cfg)
        self.assertEqual(cfg,snapshot);self.assertEqual(moved['runtime']['output_root'],str(c.OUT/'runs'))
        self.assertFalse(Path(moved['runtime']['output_root']).is_relative_to(c.FACTORY))
        moved['runtime']['output_root']=cfg['runtime']['output_root'];self.assertEqual(moved,cfg)

    def test_partial_second_slot_prevents_third_and_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            run=Path(td);cp=run/'checkpoints';cp.mkdir();(cp/'complete1').mkdir()
            self.assertEqual(c.require_resume_slot(run),['complete1'])
            (cp/'partial2').mkdir()
            with self.assertRaises(RuntimeError):c.require_resume_slot(run)
            (cp/'partial2').rmdir();(cp/'alias').symlink_to(cp/'complete1')
            with self.assertRaises(RuntimeError):c.require_resume_slot(run)

    def test_cumulative_budget_and_new80_preserve_old2303_and_floor(self):
        self.assertEqual(c.storage_requirement(0),(80+2303+1024)*(1<<30))
        with self.assertRaises(RuntimeError):c.storage_requirement(c.RESERVE+1)
        c.validate_resume_state({'status':'PAUSED','source_sha':c.SOURCE,'completed':32,'exposure':64})
        for step,exposure in [(64,128),(32,128),(0,0)]:
            with self.assertRaises(RuntimeError):c.validate_resume_state({'status':'PAUSED','source_sha':c.SOURCE,'completed':step,'exposure':exposure})
        self.assertEqual(c.command()[:4],['timeout','--signal=TERM','--kill-after=180s','1620s'])
        self.assertIn('--nproc_per_node=2',c.command())

    def test_unsigned_template_refused_without_source_or_runtime_execution(self):
        template=SOURCE.with_name('RESOURCE_ADMISSION_TEMPLATE.json')
        # Real template and real admission parser; rejection precedes CP receipt read.
        with self.assertRaisesRegex(RuntimeError,'does not bind'):
            c.admission(str(template),hashlib.sha256(template.read_bytes()).hexdigest(),'prepare')

if __name__=='__main__':unittest.main(verbosity=2)

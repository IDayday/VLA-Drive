import errno
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.il_report.events import ILStateEvents
from tools.il_report.rl_events_reference import OwnerStateEvents
from tools.il_report.render import atomic


class EventCapacityTest(unittest.TestCase):
    def test_watch_quota_does_not_kill_report_and_request_is_still_observed(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);(root/'reporting').mkdir();control=root/'control';control.mkdir()
            with patch.object(OwnerStateEvents,'watch',side_effect=OSError(errno.ENOSPC,'inotify_add_watch')):
                monitor=ILStateEvents(root,atomic,[control])
            try:
                self.assertTrue(monitor.degraded)
                atomic(root/'reporting/refresh_request.json',dict(id='new_complete_score',scientific_result=False))
                self.assertEqual(monitor.wait(.01)[0]['kind'],'report_maintenance_request')
                (control/'metrics_rank0.jsonl').write_text('{"updates": 1}\n')
                monitor.next_metadata_check=0
                self.assertEqual(monitor.wait(.01),[])
            finally:monitor.close()

    def test_no_inotify_instance_preserves_timer_and_wakeup(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);(root/'reporting').mkdir()
            with patch.object(OwnerStateEvents,'__init__',side_effect=OSError(errno.EMFILE,'inotify_init1')):
                monitor=ILStateEvents(root,atomic)
            try:
                self.assertIsNone(monitor.fd)
                self.assertEqual(monitor.wait(.01),[])
                monitor.wake();self.assertEqual(monitor.wait(.01),[])
            finally:monitor.close()

    def test_watch_bound_does_not_expand_with_historical_results(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);(root/'reporting').mkdir();dirs=[root/str(i)for i in range(20)]
            for p in dirs:p.mkdir()
            with patch.object(ILStateEvents,'MAX_WATCHES',4):monitor=ILStateEvents(root,atomic,dirs)
            try:
                self.assertLessEqual(len(monitor.directories),4)
                self.assertTrue(monitor.monitor_status()['degraded'])
                self.assertEqual(monitor.monitor_status()['metadata_fallback_seconds'],60)
            finally:monitor.close()


if __name__=='__main__':unittest.main()

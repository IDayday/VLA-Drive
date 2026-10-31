from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.il_report import resource_review as review
from tools.il_report.events import ILStateEvents


def inventory():
    gpu=dict(index=0,uuid='GPU-test',total_MiB=81920,free_MiB=81000,utilization=0)
    host=dict(gpus=[gpu],owners=[],load1=1,CPU_affinity=128,
              MemAvailable_bytes=256*1024**3,worker_python_exists=True)
    return dict(samples=[{'host':deepcopy(host)},{'host':deepcopy(host)}],leases={},errors={})


class ResourceReviewTests(unittest.TestCase):
    def test_free_capacity_backfills_existing_queue_without_moving_native_or_changing_limit(self):
        result=review.decide(inventory(),dict(jobs=[dict(status='PENDING')]))
        self.assertTrue(result['wake_existing_queue'])
        self.assertEqual(result['native_migrations'],[])
        self.assertEqual(result['existing_eval_model_limit'],2)

    def test_known_running_worker_is_not_idle_even_at_zero_utilization(self):
        value=inventory()
        for sample in value['samples']:
            sample['host']['owners']=[dict(uuid='GPU-test',visible=True,used_MiB=920,output='existing-eval')]
        self.assertFalse(review.decide(value,dict(jobs=[]))['verified_idle_physical_gpus'])

    def test_unexplained_memory_hidden_owner_lease_and_cpu_pressure_block_backfill(self):
        for kind in ('unexplained','hidden','lease','cpu','ram'):
            with self.subTest(kind=kind):
                value=inventory()
                for sample in value['samples']:
                    host=sample['host']
                    if kind=='unexplained':host['gpus'][0]['free_MiB']=65000
                    if kind=='hidden':host['owners']=[dict(uuid='GPU-test',visible=False,used_MiB=920)]
                    if kind=='cpu':host['load1']=140
                    if kind=='ram':host['MemAvailable_bytes']=64*1024**3
                if kind=='lease':value['leases']={'GPU-test':dict(primary=True)}
                self.assertFalse(review.decide(value,dict(jobs=[dict(status='PENDING')]))['wake_existing_queue'])

    def test_aliases_cannot_count_same_physical_gpu_twice(self):
        value=inventory()
        for sample in value['samples']:sample['alias']=deepcopy(sample['host'])
        self.assertEqual(review.decide(value,dict(jobs=[]))['verified_idle_physical_gpus'],['GPU-test'])

    def test_no_pending_job_does_not_wake_or_invent_training(self):
        result=review.decide(inventory(),dict(jobs=[dict(status='COMPLETE')]))
        self.assertFalse(result['wake_existing_queue'])
        self.assertEqual(result['native_migrations'],[])

    def test_task_authorization_change_invalidates_cached_authority(self):
        self.assertNotEqual(review.permission_signature({'task_authorizations':{'A':'old'}}),
                            review.permission_signature({'task_authorizations':{'A':'new'}}))

    def test_held_lease_probe_does_not_modify_or_create_files(self):
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'lease';self.assertFalse(review.lease_held(path));self.assertFalse(path.exists())
            path.write_text('original lease');self.assertFalse(review.lease_held(path))
            self.assertEqual(path.read_text(),'original lease')

    def test_no_signal_to_absent_cpu_owner(self):
        with patch.object(review,'alive',return_value=False),patch.object(review.os,'kill')as kill:
            self.assertEqual(review.wake_cadence({}),'NO_LIVE_QUEUE_OWNER');kill.assert_not_called()

    def test_own_html_is_not_a_resource_event(self):
        self.assertEqual(ILStateEvents.signatures(None,Path('/mnt/project/qwen-world-il-research-20261030/report.html')),{})


if __name__=='__main__':unittest.main()

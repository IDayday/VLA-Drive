import unittest
from unittest.mock import patch,Mock
from types import SimpleNamespace
from pathlib import Path
from tools.qwen_world_compare.job_worker import known_owner_args,admission_with_wait,ResourceRefused


class OwnerAdmission(unittest.TestCase):
    def test_existing_legacy_owner(self):
        self.assertTrue(known_owner_args(['python','-m','tools.x','--data','/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST']))

    def test_actual_registered_safety_owners(self):
        for module,flag in [('train','--il-checkpoint'),('evaluate','--parent-checkpoint')]:
            args=['python','-m','tools.qwen_world_safety_exploration.'+module,flag,
                '/mnt/project/qwen-world-rl-optimization-20261030/IL_PARENT','--output',
                '/mnt/project/qwen-world-safety-exploration-20261031/revival_training_20261101/runs/x']
            self.assertTrue(known_owner_args(args))
            self.assertFalse(known_owner_args([x.replace('/IL_PARENT','/OTHER')for x in args]))
            self.assertFalse(known_owner_args([x.replace('tools.qwen_world_safety_exploration.','tools.unrelated.')for x in args]))

    def test_path_prefix_alone_does_not_admit(self):
        self.assertFalse(known_owner_args(['python','/mnt/project/qwen-world-safety-exploration-20261031/not_registered.py']))
        self.assertFalse(known_owner_args(['python','-m']))
        self.assertFalse(known_owner_args([]))

    def test_bounded_wait_releases_partial_locks(self):
        stream=Mock();locks=[stream];mapping={0:'old'}
        with patch('tools.qwen_world_compare.job_worker.require_dispatch'),patch('tools.qwen_world_compare.job_worker.authorization',return_value=({'x':1},{})),patch('tools.qwen_world_compare.job_worker._gpu_admission',side_effect=[ResourceRefused('busy'),None])as admit,patch('tools.qwen_world_compare.job_worker.write'),patch('tools.qwen_world_compare.job_worker.time.sleep'):
            actual=admission_with_wait(SimpleNamespace(admission_wait_seconds=1),'host',[],None,mapping,locks,{},Path('/unused'))
        self.assertEqual(actual,{'x':1});stream.close.assert_called_once();self.assertEqual(locks,[]);self.assertEqual(mapping,{});self.assertEqual(admit.call_count,2)

    def test_zero_wait_and_revocation_never_retry(self):
        with patch('tools.qwen_world_compare.job_worker.require_dispatch'),patch('tools.qwen_world_compare.job_worker.authorization',return_value=({},{})),patch('tools.qwen_world_compare.job_worker._gpu_admission',side_effect=ResourceRefused('busy'))as admit:
            with self.assertRaises(ResourceRefused):admission_with_wait(SimpleNamespace(admission_wait_seconds=0),'host',[],None,{},[],{},Path('/unused'))
            self.assertEqual(admit.call_count,1)
        with patch('tools.qwen_world_compare.job_worker.require_dispatch',side_effect=PermissionError('withdrawn')),patch('tools.qwen_world_compare.job_worker._gpu_admission')as admit:
            with self.assertRaises(ResourceRefused):admission_with_wait(SimpleNamespace(admission_wait_seconds=100),'host',[],None,{},[],{},Path('/unused'))
            admit.assert_not_called()


if __name__=='__main__':unittest.main()

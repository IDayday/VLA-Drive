import unittest
from tools.qwen_world_compare.job_worker import known_owner_args


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


if __name__=='__main__':unittest.main()

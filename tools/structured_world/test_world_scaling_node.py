"""CPU tests of exact worker arguments and reservation rejection before CUDA."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.structured_world import run_world_scaling_node as node


def manifest():
    return {'schema': 'structured_world_scaling_windows_v1', 'source_root': str(node.ROOT),
        'source_sha': '0'*40, 'python': '/qualified/python',
        'hosts': [{'host': 'node'+str(i), 'address': '10.0.0.'+str(i+1),
                   'gpu_uuids': ['GPU-'+str(i*8+j) for j in range(8)],
                   'qualification': {'path': '/proof', 'sha256': 'proof-hash'}} for i in range(2)],
        'configs': {'G1': 'navsim_G1_FULL_UNIFORM_seed42.yaml'},
        'assets': {k: '/asset/'+k for k in
            ('cache', 'geometry', 'geometry-identity', 'dino-root', 'dino-identity', 'image-root')},
        'jobs': {'G1_16_1': {'world_size': 16, 'updates': 220, 'group': 'G1',
            'mode': 'io_preserving_v1', 'run': '/probe/G1_16_1', 'port': 29661}}}


class NodeTests(unittest.TestCase):
    def test_one_and_two_node_commands_use_same_trainer_and_global32(self):
        m = manifest()
        for world in (8, 16):
            job = {**m['jobs']['G1_16_1'], 'world_size': world}
            for rank in range(world // 8):
                command = node.command(m, job, rank)
                self.assertIn('--nnodes='+str(world // 8), command)
                self.assertIn('--nproc-per-node=8', command)
                self.assertIn('--node-rank='+str(rank), command)
                self.assertEqual(command[command.index('--micro-batch')+1], str(32 // world))
                self.assertEqual(command[command.index('--debug-updates')+1], '220')
                self.assertIn('--world-scaling-profile', command)
                self.assertIn('--benchmark-no-end-save', command)
                self.assertNotIn('--resume', command)
        for key, value in (('world_size', 4), ('mode', 'precision_fast'), ('updates', 500)):
            job = {**m['jobs']['G1_16_1'], key: value}
            with self.assertRaises(ValueError): node.command(m, job, 0)

    def test_plan_never_queries_GPU_or_launches(self):
        with patch.object(node.bench, 'read', return_value=manifest()), \
             patch.object(node.bench, 'readiness', side_effect=AssertionError('GPU query')), \
             patch.object(node.bench, 'run_job', side_effect=AssertionError('worker started')), \
             patch.object(sys, 'argv', ['launch', '--manifest', '/mock', '--job', 'G1_16_1', '--node-rank', '1']):
            node.main()

    def test_reservation_rejects_before_readiness_or_child(self):
        m = manifest()
        with patch.object(node.bench, 'read', side_effect=[m, {'status': 'ENVIRONMENT_AND_OPERATOR_QUALIFIED_NOT_MODEL_ACCEPTANCE'}]), \
             patch.object(node.bench, 'digest', return_value='proof-hash'), \
             patch.object(node.bench, 'immutable_source'), \
             patch.object(node.socket, 'gethostname', return_value='node0-worker-0'), \
             patch.object(node.bench, 'authorize', side_effect=ValueError('reserved for IQE')), \
             patch.object(node.bench, 'readiness', side_effect=AssertionError('GPU query')), \
             patch.object(node.bench, 'run_job', side_effect=AssertionError('worker started')), \
             patch.object(sys, 'argv', ['launch', '--manifest', '/mock', '--job', 'G1_16_1', '--node-rank', '0', '--execute']):
            with self.assertRaisesRegex(ValueError, 'reserved for IQE'): node.main()


if __name__ == '__main__':
    unittest.main(verbosity=2)

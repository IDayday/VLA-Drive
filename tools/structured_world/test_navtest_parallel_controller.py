import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_navsim_navtest as worker
from navtest_parallel_controller import accepted_receipt, PreparationReservations


class PreparationTest(unittest.TestCase):
    def test_cpu_budget_accounts_for_other_active_banks(self):
        reservations = PreparationReservations()
        probe = lambda: dict(workers=32, effective_capacity_CPUs=64, reserved_host_CPUs=16)
        self.assertEqual(reservations.score_budget('A', probe, RuntimeError)['workers'], 32)
        self.assertEqual(reservations.score_budget('B', probe, RuntimeError)['workers'], 16)
        with self.assertRaises(RuntimeError):
            reservations.score_budget('C', probe, RuntimeError)
        reservations.release('A')
        self.assertEqual(reservations.score_budget('C', probe, RuntimeError)['workers'], 32)

    def test_preparing_banks_cannot_admit_same_gpu(self):
        reservations = PreparationReservations()
        routes = [dict(uuid='GPU-a'), dict(uuid='GPU-b')]
        self.assertEqual(reservations.select('A', lambda: routes[:1], RuntimeError), routes[:1])
        self.assertEqual(reservations.select('B', lambda: routes, RuntimeError), routes[1:])
        with self.assertRaises(RuntimeError):
            reservations.select('C', lambda: routes, RuntimeError)
        reservations.release('A')
        self.assertEqual(reservations.select('C', lambda: routes, RuntimeError), routes[:1])

    def test_all_current_attempt_workers_must_allocate_before_sharing(self):
        reservations = PreparationReservations()
        routes = [dict(uuid='GPU-a'), dict(uuid='GPU-b')]
        reservations.select('A', lambda: routes, RuntimeError)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = root/'jobs/A/attempts/current/shards'
            shards.mkdir(parents=True)
            def report(index, status, done):
                (shards/f'shard_{index:03d}.json').write_text(json.dumps(dict(
                    actual_CUDA_UUID=routes[index]['uuid'], status=status, scenes_done=done)))
            # Even complete historical reports cannot release an unlaunched bank.
            report(0, 'COMPLETE', 4); report(1, 'COMPLETE', 4)
            reservations.release_ready(root)
            self.assertIn('A', reservations.pending)
            reservations.launched('A', 'current')
            report(1, 'LOADING', 0)
            reservations.release_ready(root)
            self.assertIn('A', reservations.pending)
            report(1, 'INFERENCE', 0)
            reservations.release_ready(root)
            self.assertIn('A', reservations.pending)
            report(1, 'INFERENCE', 1)
            reservations.release_ready(root)
            self.assertNotIn('A', reservations.pending)


class OwnershipTest(unittest.TestCase):
    def test_registered_live_evaluation_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job=root/'jobs/G1';part=job/'attempts/a/shards'
            part.mkdir(parents=True)
            tool=root/'run_navsim_navtest.py';tool.write_text('# immutable test worker\n')
            request=dict(key='G1',run='/native/run',tag='latest_0080000',training_identity='train')
            reg=dict(request=request,frozen_evaluation_source=worker.SHA)
            (job/'REGISTRATION.json').write_text(json.dumps(reg))
            (part.parent/'ATTEMPT.json').write_text(json.dumps(dict(routes=[dict(host='host',uuid='GPU-test')])))
            report=dict(pid=31,proc_start_ticks='77',host='host-worker-0',visible_gpu='GPU-test',
                        actual_CUDA_UUID='GPU-test',attempt='a',shard=0,status='INFERENCE',
                        routing_tool_sha256=worker.digest(tool))
            (part/'shard_000.json').write_text(json.dumps(report))
            argv=['python',str(tool),'--worker','--output',str(job),'--attempt','a','--shard','0',
                  '--run',request['run'],'--tag',request['tag']]
            with patch.object(worker,'DEFAULT_OUTPUT',root):
                self.assertEqual(worker.registered_evaluation_owner(argv,31,'77','GPU-test','host')['request_key'],'G1')
                self.assertIsNone(worker.registered_evaluation_owner(argv,31,'78','GPU-test','host'))
                self.assertIsNone(worker.registered_evaluation_owner(argv,32,'77','GPU-test','host'))
                self.assertIsNone(worker.registered_evaluation_owner(argv,31,'77','GPU-other','host'))
                self.assertIsNone(worker.registered_evaluation_owner(argv,31,'77','GPU-test','foreign'))
                tool.write_text('# changed worker\n')
                self.assertIsNone(worker.registered_evaluation_owner(argv,31,'77','GPU-test','host'))

    def test_completed_receipt_requires_exact_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'completed').mkdir()
            req=dict(key='G1',updates=80000,training_identity='a')
            self.assertFalse(accepted_receipt(root,req))
            path=root/'completed/G1.json'
            path.write_text(json.dumps(dict(valid=True,request=req)))
            self.assertTrue(accepted_receipt(root,req))
            with self.assertRaises(ValueError):accepted_receipt(root,dict(req,updates=70000))
            path.write_text(json.dumps(dict(valid=False,request=req)))
            with self.assertRaises(ValueError):accepted_receipt(root,req)


if __name__=='__main__':unittest.main()

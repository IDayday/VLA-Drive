import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_navsim_navtest as worker
from navtest_parallel_controller import accepted_receipt


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

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from tools.qwen_world_safety_exploration.resource_migration import requeue_clean_resource_pause,requeue_clean_scene_pause


class CleanResourceRequeue(unittest.TestCase):
    def test_only_bound_clean_scene_pause_requeues(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);command=['python','-m','tools.qwen_world_safety_exploration.evaluate','--output',str(root),'--rank','2','--world','32']
            (root/'rank2.json').write_text(json.dumps(dict(status='STOPPED_AT_SCENE_BOUNDARY',rank=2,world=32,host='withdrawn',identity='sealed-evaluator-identity')))
            value=dict(host='withdrawn',command=command,returncode=0,status='EXITED')
            node=dict(job='job1',host='withdrawn',command=command,attempt=1)
            self.assertTrue(requeue_clean_scene_pause(node,value,['white']));self.assertNotIn('job',node);self.assertEqual(node['command'],command);self.assertEqual(node['attempt'],1)
            for host,code in [('white',0),('withdrawn',1)]:
                self.assertFalse(requeue_clean_scene_pause(dict(job='job1',host=host,command=command),dict(value,host=host,returncode=code),['white']))
            (root/'rank2.json').write_text(json.dumps(dict(status='RUNNING',rank=2,world=32,host='withdrawn',identity='sealed-evaluator-identity')))
            self.assertFalse(requeue_clean_scene_pause(dict(job='job1',host='withdrawn',command=command),value,['white']))

    def test_only_clean_withdrawn_native_boundaries_requeue(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);checkpoint=root/'checkpoints/update123_inner2';checkpoint.mkdir(parents=True)
            (checkpoint/'state.pt').write_bytes(b'optimizer/RNG/pending-chain unchanged')
            inv=json.dumps({'state.pt':hashlib.sha256((checkpoint/'state.pt').read_bytes()).hexdigest()}).encode();(checkpoint/'files.json').write_bytes(inv)
            seal=dict(world=8,updates=123,phase='INNER_COMMITTED',next_inner=2,files_sha256=hashlib.sha256(inv).hexdigest())
            (checkpoint/'COMPLETE').write_text(json.dumps(seal))
            for rank in range(8):(root/f'result_rank{rank}.json').write_text(json.dumps(dict(status='STOPPED_AT_SAVED_BOUNDARY',checkpoint=str(checkpoint),**seal)))
            run=dict(root=str(root));latest=dict(updates=123,checkpoint=str(checkpoint));node=dict(job='job1',host='withdrawn',attempt=1)
            before=(checkpoint/'state.pt').read_bytes()
            self.assertTrue(requeue_clean_resource_pause(node,run,latest,dict(returncode=0),['white']))
            self.assertNotIn('job',node);self.assertEqual(node['attempt'],1);self.assertEqual(node['resource_migrations'][0]['next_inner'],2);self.assertEqual((checkpoint/'state.pt').read_bytes(),before)
            for host,code in [('white',0),('withdrawn',1)]:
                n=dict(job='job1',host=host)
                self.assertFalse(requeue_clean_resource_pause(n,run,latest,dict(returncode=code),['white']));self.assertIn('job',n)
            (root/'result_rank7.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'eight matching'):requeue_clean_resource_pause(dict(job='job1',host='withdrawn'),run,latest,dict(returncode=0),['white'])


if __name__=='__main__':unittest.main()

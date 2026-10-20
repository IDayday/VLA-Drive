import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from run import validate_plan, snapshot, restore
from schedule import jobs, recover_queue
from analyze import cluster_ci
import numpy as np
import torch,random
from test_engine import Toy,CFG,update
import engine
ROOT=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
    def setUp(self):
        # Synthetic preparation identity exercises real hash validation; no scene assets.
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.prepared = Path(self.tmp.name)
        payload = b'{"synthetic": true}\n'
        (self.prepared/'PROMPTS.json').write_bytes(payload)
        (self.prepared/'PREPARATION.json').write_text(json.dumps({
            'status': 'INCOMPLETE_F_COVERAGE',
            'files': {'PROMPTS.json': hashlib.sha256(payload).hexdigest()}}))
    def test_real_checkpoint_contract_on_cpu(self):
        torch.manual_seed(22);m=Toy();ref=copy.deepcopy(m).requires_grad_(False);engine.scope(m,'joint');opt=engine.optimizer(m,CFG);update(m,ref,opt,0)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'one.pt';ident={'protocol':'test'};snapshot(p,m,opt,ident,'A',3,1,12.,None);expected_random=(random.random(),float(np.random.rand()),float(torch.rand(())))
            update(m,ref,opt,1);expected_hash=engine.state_hash(m,'all');other=Toy();engine.scope(other,'joint');oo=engine.optimizer(other,CFG);count,elapsed=restore(p,other,oo,ident,'A',3);self.assertEqual(count,1);self.assertGreaterEqual(elapsed,12.)
            self.assertEqual(expected_random,(random.random(),float(np.random.rand()),float(torch.rand(()))));update(other,ref,oo,1);self.assertEqual(engine.state_hash(other,'all'),expected_hash)
            with self.assertRaises(ValueError):restore(p,other,oo,{'protocol':'changed'},'A',3)

    def test_unlocked_formal_rejected(self):
        cfg=json.loads((ROOT/'TRAINING.template.json').read_text())
        with self.assertRaises(ValueError):validate_plan(cfg,self.prepared,'train')
        validate_plan(cfg,self.prepared,'smoke')
    def test_missing_f_not_silently_dropped(self):
        cfg=json.loads((ROOT/'TRAINING.template.json').read_text());cfg.update(formal_protocol_locked=True,epochs=1,training_seeds=[1],wall_seconds=100)
        with self.assertRaisesRegex(ValueError,'Full F coverage'):validate_plan(cfg,self.prepared,'train')
    def test_independent_jobs_not_ddp(self):
        cfg={'training_seeds':[1,2,3]};items=jobs(cfg,'train');self.assertEqual(len(items),33);self.assertEqual(len({(i['arm'],i['seed']) for i in items}),33);self.assertEqual(len(jobs(cfg,'initial')),5)
    def test_queue_reuses_completed_and_resumes_partial(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg={'epochs':1};ident={'code':'x'}
            for name,value in [('TRAINING.json',cfg),('PHASE_IDENTITY.json',ident),('QUEUE.json',{'phase':'train'})]:(root/name).write_text(json.dumps(value))
            complete=root/'A_seed1';complete.mkdir();(complete/'RESULT.json').write_text(json.dumps({'status':'COMPLETE','completed_updates':316}))
            ck=root/'C_seed1/checkpoints';ck.mkdir(parents=True);(ck/'update_000004.pt').write_bytes(b'fake checkpoint for queue test');(ck/'update_000004.json').write_text(json.dumps({'completed_updates':4}))
            pending=[{'arm':'A','seed':1},{'arm':'C','seed':1},{'arm':'E','seed':1}];queue,reused=recover_queue(root,pending,cfg,ident);self.assertEqual(len(reused),1);self.assertEqual(len(queue),2);self.assertTrue(queue[0]['resume'].endswith('update_000004.pt'));self.assertNotIn('resume',queue[1])

    def test_cluster_bootstrap_uses_drives(self):
        result=cluster_ci(np.array([1.,1.,1.,1.]),['a','a','a','b'],reps=20);self.assertEqual(result['cluster_count'],2);self.assertEqual(result['ci95'],[1.,1.])
if __name__=='__main__':unittest.main()

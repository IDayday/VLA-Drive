"""Regression checks for actual input gates, queue receipts and zero-step resume."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import contracts
import entry
import run
from cf_contract import validate_cf_batches
from prepare import choose_f
from schedule import accepted_job
from test_engine import Toy

ROOT=Path(__file__).resolve().parent


class Tests(unittest.TestCase):
    def prepared_fixture(self,root):
        root.mkdir()
        train=[f'train_{i}' for i in range(1264)];test=[f'test_{i}' for i in range(141)]
        payload={'text':'synthetic','sha256':hashlib.sha256(b'synthetic').hexdigest(),'token_count':1}
        scenes=[{'token':t,'log':f'drive_{t}_0_1','arms':{a:dict(payload) for a in ['A','B','C','F','E']}} for t in train+test]
        donors=[{'token':r['token'],'donor_token':train[(i+1)%1264] if i<1264 else train[0],'token_delta':0} for i,r in enumerate(scenes)]
        files={'PROMPTS.json':{'scenes':scenes},'DONORS.json':{'assignments':donors},'SPLITS.json':{'train':train,'test':test,'monitor':train[:64]}}
        for name,value in files.items():(root/name).write_text(json.dumps(value))
        receipt={'status':'PASS','matched_F':1405,'missing_F':0,'files':{name:contracts.sha(root/name) for name in files}}
        (root/'PREPARATION.json').write_text(json.dumps(receipt))
        return root

    def cfg(self):
        c=contracts.read(ROOT/'TRAINING.template.json')
        c.update(epochs=3,training_seeds=[11,12,13],wall_seconds=86400,formal_protocol_locked=True)
        return c

    def test_budget_includes_each_replica_monitor(self):
        r=contracts.candidate_budget(self.cfg())
        self.assertEqual(r['training_replicas'],33)
        self.assertEqual(r['updates_per_replica'],948)
        self.assertEqual(r['monitor_candidates'],67584)
        self.assertEqual(r['total_candidates'],1197264)

    def test_reject_nonfinite_and_bool_and_ignored_worker_count(self):
        for name,value in [('epochs',True),('wall_seconds',float('nan')),('vlm_lr',float('inf')),
                           ('training_seeds',[True]),('training_seeds',[1,1]),
                           ('score_workers_per_replica',2),('parameter_scope_version','legacy')]:
            c=self.cfg();c[name]=value
            with self.subTest(name=name,value=value),self.assertRaises(ValueError):contracts.validate_training(c)

    def test_f_strict_priority_and_explicit_exceptions(self):
        strict=(100.,'strict',{'token_delta':0})
        relaxed=(-100.,'relaxed',{'token_delta':1})
        policy={'max_token_delta':2,'exception_tokens':['x']}
        self.assertEqual(choose_f([relaxed,strict],'x',policy),strict)
        self.assertIsNone(choose_f([relaxed],'y',policy))
        self.assertEqual(choose_f([relaxed],'x',policy),relaxed)
        self.assertIsNone(choose_f([(-1.,'bad',{'token_delta':3})],'x',policy))

    def test_cf_position_must_match_recorded_delta(self):
        class Model:
            _special_token_ids={'history':[10],'action':list(range(11,19))}
            def _find_token_positions(self,ids,values):
                return torch.stack([(ids==v).nonzero()[:,1] for v in values],dim=1)
        m=Model()
        c={'input_ids':torch.tensor([[1,2,*range(10,19)]])}
        f={'input_ids':torch.tensor([[1,2,3,*range(10,19)]])}
        self.assertFalse(validate_cf_batches(m,c,f,{'token_delta':1},2)['strict'])
        with self.assertRaises(ValueError):validate_cf_batches(m,c,f,{'token_delta':1},0)
        with self.assertRaises(ValueError):validate_cf_batches(m,c,f,{'token_delta':2},2)
        bad={'input_ids':torch.tensor([[1,10,2,3,*range(11,19)]])}
        with self.assertRaises(ValueError):validate_cf_batches(m,c,bad,{'token_delta':1},2)

    def test_device_aliases_and_deadline_cannot_reset(self):
        self.assertEqual(contracts.device_ids('0,2'),['0','2'])
        for s in ['0,00','0,0','0, 1','', '-1']:
            with self.assertRaises(ValueError):contracts.device_ids(s)
        self.assertEqual(entry.remaining(100,now=99),1)
        self.assertEqual(entry.remaining(100,now=120),0)

    def test_resume_comparison_includes_optimizer_and_rng(self):
        from verify_resume import compare
        import engine
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model=Toy();engine.scope(model,'joint');opt=engine.optimizer(model,self.cfg())
            a=root/'a.pt';b=root/'b.pt';c=root/'c.pt'
            run.snapshot(a,model,opt,{'test':'resume'},'A',1,2,1.,None)
            run.snapshot(b,model,opt,{'test':'resume'},'A',1,2,10.,str(a))
            self.assertEqual(compare(a,b)['status'],'PASS')
            opt.param_groups[0]['lr']*=2
            run.snapshot(c,model,opt,{'test':'resume'},'A',1,2,20.,str(a))
            self.assertEqual(compare(a,c)['different_fields'],['optimizer'])

    def test_complete_result_alone_cannot_forge_phase_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d)
            (r/'RESULT.json').write_text(json.dumps({'status':'COMPLETE','mode':'train','arm':'A','seed':1,'completed_updates':948}))
            with self.assertRaises(KeyError):accepted_job(r,{'arm':'A','seed':1},'train',self.cfg())

    def test_prepared_pass_cannot_hide_missing_assignment(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.prepared_fixture(Path(d)/'prepared')
            donors=contracts.read(root/'DONORS.json');donors['assignments'].pop()
            (root/'DONORS.json').write_text(json.dumps(donors))
            receipt=contracts.read(root/'PREPARATION.json');receipt['files']['DONORS.json']=contracts.sha(root/'DONORS.json')
            (root/'PREPARATION.json').write_text(json.dumps(receipt))
            cfg=self.cfg()
            with self.assertRaises(ValueError):contracts.validate_prepared(root,cfg)

    def pipeline_fixture(self, root):
        cfg=self.cfg();cfg['total_wall_seconds']=60
        assets=root/'assets.json';assets.write_text('{}')
        training=root/'training.json';training.write_text(json.dumps(cfg))
        prepared=self.prepared_fixture(root/'prepared')
        return SimpleNamespace(assets=str(assets),training=str(training),prepared=str(prepared),resume_from=None,action='run'),cfg

    def fake_child(self, calls):
        def execute(command, log, deadline, state_path, state, env=None):
            self.assertGreater(entry.remaining(deadline),0)
            folder=Path(command[command.index('--out')+1])
            if '--mode' in command:
                mode=command[command.index('--mode')+1];folder.mkdir()
                result={'status':'COMPLETE','mode':mode,'frozen_parameters_buffers_exact':True,'reference_exact':True,'checkpoint':str(folder/'model.pt')}
                entry.save(folder/'RESULT.json',result)
                if mode=='inputs':entry.save(folder/'INPUT_ACCEPTANCE.json',{'status':'PASS','scenes':1405})
                calls.append(mode)
            elif '--phase' in command:
                phase=command[command.index('--phase')+1];folder.mkdir()
                entry.save(folder/'RESULT.json',{'status':'COMPLETE','phase':phase})
                self.assertLessEqual(float(command[command.index('--scheduler-wall-seconds')+1]),60)
                calls.append(phase)
            elif any(str(x).endswith('verify_resume.py') for x in command):
                entry.save(folder,{'status':'PASS'});calls.append('verify_resume')
            else:
                entry.save(folder,{'status':'COMPLETE'});calls.append('analysis')
            return 0
        return execute

    def test_pipeline_stages_are_ordered_and_share_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out';out.mkdir();a,cfg=self.pipeline_fixture(root);calls=[]
            with patch.object(entry,'child',side_effect=self.fake_child(calls)):
                self.assertEqual(entry.pipeline(a,cfg,{'devices':['0']},out),0)
            self.assertEqual(calls,['smoke','smoke','smoke','inputs','smoke','smoke','verify_resume','initial','train','final','analysis'])
            self.assertEqual(contracts.read(out/'PIPELINE.json')['status'],'COMPLETE')

    def test_pipeline_failure_never_starts_formal_phases(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out';out.mkdir();a,cfg=self.pipeline_fixture(root)
            with patch.object(entry,'child',return_value=2) as launch:
                with self.assertRaises(RuntimeError):entry.pipeline(a,cfg,{'devices':['0']},out)
            self.assertEqual(launch.call_count,1)
            state=contracts.read(out/'PIPELINE.json')
            self.assertEqual(state['phases'],{})
            self.assertEqual(state['status'],'INCOMPLETE')

    def test_pipeline_resume_does_not_reset_global_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/'old';old.mkdir();a,cfg=self.pipeline_fixture(root);a.action='smoke'
            with patch.object(entry,'child',side_effect=self.fake_child([])):
                entry.pipeline(a,cfg,{'devices':['0']},old)
            state=contracts.read(old/'PIPELINE.json');state.update(deadline_unix=0,pid=999999,pid_start_ticks=None)
            entry.save(old/'PIPELINE.json',state)
            a.action='run';a.resume_from=str(old);out=root/'resumed';out.mkdir()
            with patch.object(entry,'child') as launch:
                with self.assertRaises(TimeoutError):entry.pipeline(a,cfg,{'devices':['0']},out)
                launch.assert_not_called()

    def test_zero_update_resume_finishes_pending_monitor_once(self):
        # Run the real orchestration path on a small CPU model; no real data/GPU.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);prepared=root/'prepared';prepared.mkdir();out=root/'out'
            tokens=['t0','t1','t2','t3']
            for name,value in [('SPLITS.json',{'train':tokens,'test':[],'monitor':['t0']}),
                               ('PROMPTS.json',{'scenes':[{'token':t,'arms':{'A':{'text':'go'}}} for t in tokens]}),
                               ('DONORS.json',{'assignments':[]}),('PREPARATION.json',{})]:
                (prepared/name).write_text(json.dumps(value))
            cfg=self.cfg();cfg.update(epochs=1,training_seeds=[1]);cfgpath=root/'training.json';cfgpath.write_text(json.dumps(cfg))
            cp=root/'prior.pt';cp.with_suffix('.json').write_text(json.dumps({'pending_monitor':'monitor_initial'}))
            metrics=dict(PDMS=0.,NC=1.,DAC=1.,TTC=1.,EP=0.,CF=1.,DDC=1.)
            data=[{'token':t,'log':t+'_a_b','input_trajectory':np.zeros((8,3)).tolist(),'source_paths':{'metric':'none'},'base_score':{'metrics':metrics}} for t in tokens]
            calls=[]
            def score(*args):
                calls.append(1)
                if len(calls)>1:run.STOP=True
                return {'metrics':metrics}
            original_noise=run.engine.make_noise
            a=SimpleNamespace(out=str(out),assets='unused',training=str(cfgpath),prepared=str(prepared),
                              mode='train',arm='A',seed=1,resume=str(cp))
            with patch.object(run,'config',return_value={}),patch.object(run,'validate_plan'),\
                 patch.object(run,'validate',return_value={'status':'PASS'}),\
                 patch.object(run,'identity',return_value={'code':{}}),\
                 patch.object(run,'rows',side_effect=lambda c,s:data if s=='train' else []),\
                 patch.object(run,'load_model',return_value=Toy()),\
                 patch.object(run,'restore',return_value=(0,0.)),\
                 patch.object(run,'scene',return_value={'state':None,'instruction':'go'}),\
                 patch.object(run,'load_images',return_value=[]),\
                 patch.object(run.core,'build_inputs',return_value={}),\
                 patch.object(run.core,'context',side_effect=lambda m,b,s:m.condition(torch.ones(5))),\
                 patch.object(run.core,'velocity',side_effect=lambda m,c,x,s:m.velocity(c,x,s)),\
                 patch.object(run.engine,'make_noise',side_effect=lambda device,seed:original_noise('cpu',seed)),\
                 patch.object(run,'decode',return_value=np.zeros((8,3))),\
                 patch.object(run,'score_one',side_effect=score):
                run.STOP=False
                try:result=run.main(a)
                finally:run.STOP=False
            self.assertEqual(result,2)
            self.assertEqual(contracts.read(out/'RESULT.json')['status'],'INTERRUPTED')
            self.assertEqual(contracts.read(out/'evaluation/monitor_initial/ACCEPTANCE.json')['groups'],1)
            self.assertFalse((out/'ERROR.json').exists())


if __name__=='__main__':
    unittest.main()

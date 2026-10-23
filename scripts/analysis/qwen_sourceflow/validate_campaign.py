"""CPU qualification for campaign ownership, admission and incomplete results."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
from scripts.analysis.qwen_sourceflow import campaign as module
from starVLA.rl.flow_grpo.transactions import atomic_json


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=dict(path=str(Path.cwd()),sha='test')
        self.queue=module.Campaign(self.root,self.root/'control',self.source,self.source)
        self.require=patch.object(module,'require_dispatch',return_value='allowed');self.require.start()

    def tearDown(self):
        self.require.stop();self.temp.cleanup()

    def parent(self):
        atomic_json(self.queue.selection,dict(selected=dict(checkpoint='parent')))
        return dict(path='parent',files={'model.pt':'model'})

    def test_waiting_IL_never_dispatches(self):
        with patch.object(module,'launch') as launch:
            self.assertEqual(self.queue.tick()['status'],'WAITING_FOR_FULL_IL_AND_FIVE_DEV')
            launch.assert_not_called()

    def test_evaluation_source_is_separate_from_training(self):
        evaluation=dict(path='/qualified-evaluation',sha='eval')
        self.queue.evaluation=evaluation
        data=self.root/'data/NAVTEST';data.mkdir(parents=True);(data/'manifest.json').write_text('{}')
        routes=[dict(host='allowed',gpu=0,uuid='GPU-eval')]
        with patch.object(module,'available',return_value=(routes,{})),patch.object(self.queue,'submit') as submit:
            self.queue.ensure_evaluation('intermediate',dict(path='parent'),'NAVTEST')
            self.assertEqual(submit.call_args.args[-1],evaluation)
            self.assertEqual(self.queue.training,self.source)

    def test_default_evaluation_source_preserves_old_execution(self):
        self.assertEqual(self.queue.evaluation,self.source)

    def test_completion_during_dead_owner_probe(self):
        receipt=self.root/'receipt';receipt.mkdir()
        atomic_json(receipt/'operator.json',dict(operator_pid=1,start_ticks=1))
        atomic_json(self.queue.task_path('eval'),dict(name='eval',workers=[dict(receipt_dir=str(receipt))]))
        def probe(*_):
            atomic_json(receipt/'result.json',dict(status='PROCESS_COMPLETE',exit_code=0));return False
        with patch.object(module,'operator_alive',side_effect=probe):
            self.assertTrue(self.queue.done('eval'))

    def test_unfinished_qualification_blocks_formal(self):
        parent=self.parent()
        with patch.object(module,'inspect_il',return_value=parent),patch.object(module,'validate_selection'),\
             patch.object(self.queue,'ensure_training',return_value=False) as training:
            self.assertEqual(self.queue.tick()['status'],'SELECTED_PARENT_QUALIFICATION')
            self.assertEqual([call.args[0] for call in training.call_args_list],
                ['canonical_selected_IL_profile16','source_selected_IL_profile16'])

    def test_registration_changes_are_rejected(self):
        parent=self.parent()
        with patch.object(module,'inspect_il',return_value=parent),patch.object(module,'validate_selection'),\
             patch.object(self.queue,'ensure_training',return_value=False):
            self.queue.tick();path=self.queue.control/'registration.json';record=module.read(path)
            record['training_seed']=43;atomic_json(path,record)
            with self.assertRaisesRegex(ValueError,'identity changed'):self.queue.tick()

    def test_formal_in_progress_does_not_start_RL_navtest(self):
        parent=self.parent()
        def training(*args,**kwargs):return kwargs['qualification']
        with patch.object(module,'inspect_il',return_value=parent),patch.object(module,'validate_selection'),\
             patch.object(self.queue,'ensure_training',side_effect=training),\
             patch.object(self.queue,'qualification',return_value={}),\
             patch.object(self.queue,'ensure_export',return_value=None),\
             patch.object(self.queue,'ensure_evaluation',return_value=None) as evaluation:
            self.assertEqual(self.queue.tick()['status'],'FORMAL_RL_AND_DEV')
            self.assertEqual([c.args[0] for c in evaluation.call_args_list],['selected_IL_DEV','selected_IL_NAVTEST'])

    def test_zero_exit_is_not_a_formal_endpoint(self):
        run=self.root/'run';run.mkdir()
        for rank in range(8):atomic_json(run/f'result_rank{rank}.json',dict(qualification=False,
            updates=4,collections=1,raw_scene_exposures=8,new_candidates=64,next_inner=0,phase='READY',status='STOPPED_AT_SAVED_BOUNDARY'))
        with self.assertRaisesRegex(ValueError,'Incomplete'):module.formal_endpoint(run)

    def test_actual_Adam_count_is_required_for_completion(self):
        import torch
        run=self.root/'run';checkpoint=run/'checkpoints/final';checkpoint.mkdir(parents=True)
        identity=dict(data_stream_identity='fixed source population',parameter_contract=dict(trainable_names=['adapter']))
        atomic_json(run/'identity.json',identity);atomic_json(checkpoint/'identity.json',identity)
        row=dict(qualification=False,updates=50000,collections=12500,raw_scene_exposures=100000,
            new_candidates=800000,next_inner=0,phase='COMPLETE',status='COMPLETE',checkpoint=str(checkpoint))
        for rank in range(8):atomic_json(run/f'result_rank{rank}.json',row)
        torch.save(dict(optimizer=dict(state={0:dict(step=torch.tensor(49999))}),scheduler=dict(last_epoch=50000)),checkpoint/'optimizer.pt')
        files={name:module.file_sha(checkpoint/name) for name in ('optimizer.pt','identity.json')}
        atomic_json(checkpoint/'files.json',files)
        atomic_json(checkpoint/'COMPLETE',dict(schema=2,updates=50000,phase='COMPLETE',world=8,files_sha256=module.file_sha(checkpoint/'files.json')))
        with self.assertRaisesRegex(ValueError,'Actual Adam counts'):module.formal_endpoint(run)

    def test_partial_dispatch_preserves_every_intent(self):
        routes=[dict(host='allowed',gpu=i,uuid=f'GPU-{i}') for i in range(2)]
        with patch.object(module,'allowed_hosts',return_value=['allowed']),\
             patch.object(module,'launch',side_effect=[dict(operator_pid=1,start_ticks=1),RuntimeError('SSH failure')]):
            with self.assertRaisesRegex(RuntimeError,'SSH failure'):
                self.queue.submit('eval','scripts.analysis.qwen_sourceflow.evaluate_policy',[],routes,self.source)
        record=module.read(self.queue.task_path('eval'));self.assertEqual(len(record['workers']),2)
        with patch.object(module,'operator_alive',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'partially dispatched'):self.queue.done('eval')

    def test_loading_process_reserves_UUID_before_GPU_allocation(self):
        routes=[dict(host='allowed',gpu=0,uuid='GPU-pinned')]
        with patch.object(module,'allowed_hosts',return_value=['allowed']),\
             patch.object(module,'launch',return_value=dict(operator_pid=1,start_ticks=1)):
            self.queue.submit('eval','scripts.analysis.qwen_sourceflow.evaluate_policy',[],routes,self.source)
        with patch.object(module,'operator_alive',return_value=True):
            self.assertEqual(self.queue.reserved(),{'GPU-pinned'})

    def test_failure_does_not_launch_retry(self):
        receipt=self.root/'receipt';receipt.mkdir();atomic_json(receipt/'result.json',dict(status='FAILED'))
        atomic_json(self.queue.task_path('train'),dict(name='train',workers=[dict(receipt_dir=str(receipt))]))
        with patch.object(module,'available') as available:
            with self.assertRaisesRegex(RuntimeError,'No automatic retry'):
                self.queue.ensure_training('train','canonical',{},'output',qualification=False)
            available.assert_not_called()

    def test_merged_weights_wait_for_completed_ODE_parity(self):
        name='canonical_update005000_export';out=self.root/'exports/canonical_update005000';out.mkdir(parents=True)
        (out/'COMPLETE').write_text('{}');atomic_json(self.queue.task_path(name),dict(name=name,workers=[]))
        parity=name+'_parity';atomic_json(self.queue.task_path(parity),dict(name=parity,workers=[]))
        report=self.queue.control/'parity'/(name+'.json');report.parent.mkdir()
        atomic_json(report,dict(status='PASS',bundle=str(out)))
        with patch.object(self.queue,'done',side_effect=lambda task:task!=parity):
            self.assertIsNone(self.queue.ensure_export('canonical',5000,self.root/'run'))
        with patch.object(self.queue,'done',return_value=True):
            self.assertEqual(self.queue.ensure_export('canonical',5000,self.root/'run'),out)

    def test_withdrawn_host_is_rejected_before_dispatch(self):
        with patch.object(module,'allowed_hosts',return_value=[]),patch.object(module,'launch') as launch:
            with self.assertRaises(PermissionError):
                self.queue.submit('eval','scripts.analysis.qwen_sourceflow.evaluate_policy',[],
                    [dict(host='withdrawn',gpu=0,uuid='GPU0')],self.source)
            launch.assert_not_called();self.assertFalse(self.queue.task_path('eval').exists())

    def test_scene_hash_counts_fresh_rollouts_once(self):
        run=self.root/'run';run.mkdir()
        content=''.join(json.dumps(dict(updates=u,collection=(u-1)//4+1,
            inner_completed=(u-1)%4+1,token=f'scene{(u-1)//4}'))+'\n' for u in range(1,50001))
        for rank in range(8):(run/f'metrics_rank{rank}.jsonl').write_text(content)
        before=module.scene_stream_hash(run);self.assertEqual(len(set(before)),1)
        (run/'metrics_rank3.jsonl').write_text(content.replace('scene12499','different'))
        after=module.scene_stream_hash(run)
        self.assertEqual([i for i in range(8) if before[i]!=after[i]],[3])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CampaignTests))
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),scope='CPU scheduling tests; no actor/sampler/loss changes or training effects claimed')
    atomic_json(a.output,report)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()

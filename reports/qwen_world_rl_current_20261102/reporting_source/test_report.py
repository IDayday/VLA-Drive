"""Guard the report's scientific completion boundary and append-only inputs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import hashlib

spec = importlib.util.spec_from_file_location('report', Path(__file__).with_name('render_report.py'))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ReportEvidenceTests(unittest.TestCase):
    def test_partial_and_wrong_denominator_never_publish_score(self):
        for row in [dict(status='RUNNING',scenes=12000,expected_scenes=12146,PDMS=95),
                    dict(status='COMPLETE',scenes=12000,expected_scenes=12146,PDMS=95),
                    dict(status='COMPLETE',scenes=12146.5,expected_scenes=12146.5,PDMS=95),
                    dict(status='COMPLETE',scenes=12146,expected_scenes=12146,PDMS=95,invalid_scenes=1),
                    dict(status='COMPLETE',PDMS=95)]:
            result=report.normalize_result(dict(cohort='NAVTEST',**row),[],[])
            self.assertFalse(result['publishable_complete'])
            self.assertIsNone(result['display_PDMS'])

    def test_complete_population_publishes_original_score(self):
        result=report.normalize_result(dict(status='COMPLETE',scenes=12146,cohort='NAVTEST',PDMS=90.123456),[],[])
        self.assertTrue(result['publishable_complete'])
        self.assertEqual(result['display_PDMS'],90.123456)

    def test_journal_tail_ignores_incomplete_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'journal.jsonl'
            p.write_text('{"updates":1}\n{"updates":2}\n{"updates":')
            self.assertEqual([r['updates'] for r in report.read_tail(p)],[1,2])

    def test_evidence_text_escaped_and_atomic_write_complete(self):
        self.assertEqual(report.esc('<script>"'), '&lt;script&gt;&quot;')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'report.html'
            report.atomic_text(p,'<html>one</html>')
            report.atomic_text(p,'<html>two</html>')
            self.assertEqual(p.read_text(),'<html>two</html>')
            self.assertEqual([x.name for x in Path(tmp).iterdir()],['report.html'])

    def test_registration_state_and_resource_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            runroot=root/'runs'/'CONTROL';runroot.mkdir(parents=True)
            (runroot/'metrics_rank0.jsonl').write_text(json.dumps(dict(updates=4,collection=1,new_candidates=64,
                raw_scene_exposures=8,gradient_norm=.1,metrics={'clipfrac':.05},exploration={'canonical_random_mean':.8}))+'\n')
            (root/'RUNS.json').write_text(json.dumps({'runs':[dict(id='CONTROL',algorithm='PPO',root=str(runroot),world_size=1)]}))
            (root/'STATE.json').write_text(json.dumps({'runs':{'CONTROL':{'status':'TRAINING'}},'results':[
                dict(arm='CONTROL',updates=4,cohort='DEV_FIXED',status='COMPLETE',scenes=100,expected_scenes=100,PDMS=90.123456),
                dict(arm='CONTROL',updates=8,cohort='NAVTEST',status='RUNNING',scenes=100,expected_scenes=12146,PDMS=99.999999)]}))
            (root/'resources').mkdir()
            (root/'resources'/'inventory_initial.json').write_text(json.dumps({'host':{'at':1,'gpus':[dict(index=0,uuid='GPU-TEST',free_MiB=100)],'owners':[]}}))
            snapshot=report.load_snapshot(root)
            self.assertEqual(snapshot['runs'][0]['updates'],4)
            self.assertEqual(snapshot['runs'][0]['status'],'TRAINING')
            self.assertEqual(len(snapshot['resources']),1)
            document=report.render(root,snapshot,[])
            self.assertIn('90.123456',document)
            self.assertNotIn('99.999999',document)

    def test_missing_ranks_do_not_claim_global_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'metrics_rank0.jsonl').write_text('{"updates":10}\n')
            snapshot=report.journal_snapshot(dict(id='PARTIAL',root=str(root),world_size=8),[],[])
            self.assertIsNone(snapshot['updates'])
            self.assertEqual(snapshot['observed_rank_min_update'],10)

    def test_live_runs_registry_overrides_old_state_and_worker_result_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runroot=root/'actual_run';runroot.mkdir()
            (runroot/'metrics_rank0.jsonl').write_text('{"updates":4,"new_candidates":64}\n')
            job=root/'job';job.mkdir()
            (job/'RUNNING.json').write_text(json.dumps(dict(status='RUNNING',worker_pid=123,gpu_uuids=['GPU-X'])))
            (job/'RESULT.json').write_text(json.dumps(dict(status='EXITED',returncode=0)))
            owner=root/'owner.json'
            (root/'STATE.json').write_text(json.dumps(dict(status='REGISTERED',runs=[dict(id='LIVE',status='REGISTERED',root=str(root/'wrong_old_root'),host=None)])))
            (root/'RUNS.json').write_text(json.dumps(dict(runs=[dict(id='LIVE',status='DISPATCHED',root=str(runroot),host='newhost',job=str(job),owner_state=str(owner),world_size=1)])))
            owner.write_text(json.dumps(dict(status='STAGE_COMPLETE',current_job=str(job))))
            snapshot=report.load_snapshot(root);run=snapshot['runs'][0]
            self.assertEqual(run['root'],str(runroot))
            self.assertEqual(run['host'],'newhost')
            self.assertEqual(run['registry_status'],'DISPATCHED')
            self.assertEqual(run['updates'],4)
            self.assertEqual(run['status'],'STAGE_COMPLETE')
            self.assertEqual(run['worker_receipt_status'],'EXITED')
            self.assertTrue(run['worker_terminal_receipt'])
            self.assertEqual(run['worker_returncode'],0)
            self.assertFalse(run.get('training_complete',False))

    def test_withdrawal_excludes_old_gpu_inventory_from_current_authorized_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);resources=root/'resources';resources.mkdir()
            inventory={f'host{h}':dict(gpus=[dict(index=i,uuid=f'GPU-{h}-{i}') for i in range(8)],owners=[]) for h in range(8)}
            (resources/'inventory_initial.json').write_text(json.dumps(inventory))
            (resources/'TSINGHUA_ZT2_WITHDRAWAL_20261030.json').write_text(json.dumps(dict(withdrawn_host='host7',remaining_hosts=[f'host{h}' for h in range(7)],new_dispatch_allowed=False)))
            snapshot=report.load_snapshot(root)
            self.assertEqual(len(snapshot['resources']),64)
            self.assertEqual(len(snapshot['active_resources']),56)
            self.assertNotIn('host7',{r['host'] for r in snapshot['active_resources']})

    def test_campaign_nested_summary_is_read_with_empty_root_state_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);evaluation=root/'evaluation';evaluation.mkdir()
            source=dict(status='COMPLETE',cohort='TRAIN_DIAGNOSTIC',scenes=1024,expected_scenes=1024,
                invalid_scenes=0,PDMS=93.646684,IL_updates=70000,RL_updates=None,updates=70000,
                token_sha256='tokens1024',evaluation=str(evaluation))
            source['sha256']=hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            summary=evaluation/'SUMMARY.json';summary.write_text(json.dumps(source))
            campaign=root/'owners'/'CAMPAIGN';campaign.mkdir(parents=True)
            (campaign/'STATE.json').write_text(json.dumps(dict(current_gate=1000,results=[
                dict(label='IL70_TRAIN_001024',cohort='TRAIN_DIAGNOSTIC',summary=dict(source,evidence=str(summary)))])))
            (root/'STATE.json').write_text(json.dumps(dict(results=[])))
            snapshot=report.load_snapshot(root)
            self.assertEqual(len(snapshot['results']),1)
            result=snapshot['results'][0]
            self.assertTrue(result['publishable_complete'])
            self.assertEqual(result['arm'],'IL70_BASELINE')
            self.assertEqual(result['rl_updates_for_plot'],0)
            self.assertEqual(result['display_PDMS'],93.646684)

    def test_nested_summary_evaluation_directory_resolves_real_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluation=Path(tmp)
            source=dict(status='COMPLETE',cohort='TRAIN_DIAGNOSTIC',scenes=10,expected_scenes=10,
                        PDMS=90.5,RL_updates=1000,IL_updates=70000,evaluation=str(evaluation))
            (evaluation/'SUMMARY.json').write_text(json.dumps(source))
            result=report.normalize_result(dict(label='PPO_FULL_u001000_TRAIN_DIAGNOSTIC',summary=source),[],[])
            self.assertTrue(result['publishable_complete'])
            self.assertEqual(result['arm'],'PPO_FULL')
            self.assertEqual(result['rl_updates_for_plot'],1000)

    def test_campaign_eval_spec_shows_partial_count_without_scalar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);evaluation=root/'evaluation';(evaluation/'scenes').mkdir(parents=True)
            (evaluation/'scenes'/'scene1.json').write_text('{}')
            manifest=root/'panel.json';manifest.write_text(json.dumps(dict(count=10,tokens=[])))
            spec=dict(label='PPO_FULL_u001000_TRAIN_DIAGNOSTIC',cohort='TRAIN_DIAGNOSTIC',
                      output=str(evaluation),manifest=str(manifest))
            entries=report.evaluation_entries({'stages':{'1000':{'TRAIN_DIAGNOSTIC':{'spec':spec}}}})
            self.assertEqual(len(entries),1)
            result=report.normalize_result(entries[0],[],[])
            self.assertEqual(result['scenes'],1)
            self.assertEqual(result['expected_scenes'],10)
            self.assertFalse(result['publishable_complete'])
            self.assertIsNone(result['display_PDMS'])

    def test_changed_summary_seal_cannot_publish_complete_scalar(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'SUMMARY.json'
            path.write_text(json.dumps(dict(status='COMPLETE',cohort='TRAIN_DIAGNOSTIC',scenes=10,
                expected_scenes=10,PDMS=99.9,sha256='bad')))
            warnings=[]
            result=report.normalize_result(dict(evidence=str(path)),warnings,[])
            self.assertFalse(result['publishable_complete'])
            self.assertIsNone(result['display_PDMS'])
            self.assertTrue(warnings)

    def test_unreadable_summary_does_not_fall_back_to_advertised_complete_scalar(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'SUMMARY.json';path.write_text('{incomplete')
            result=report.normalize_result(dict(evidence=str(path),status='COMPLETE',cohort='NAVTEST',
                scenes=12146,expected_scenes=12146,PDMS=99.9),[],[])
            self.assertFalse(result['publishable_complete'])
            self.assertIsNone(result['display_PDMS'])


if __name__=='__main__':unittest.main()

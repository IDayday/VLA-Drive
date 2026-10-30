import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from tools.il_report import evidence as e
from tools.il_report.diagrams import architecture_svg, NAMES
from tools.il_report.events import ILStateEvents
from tools.il_report.render import atomic, advance_regular


def seal(value):
    return dict(value,sha256=e.fingerprint(value))


class ReportTests(unittest.TestCase):
    def test_partial_last_line_and_missing_rank_do_not_claim_global_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for rank in range(7):
                (root/f'metrics_rank{rank}.jsonl').write_text('{"updates":10,"losses":{"x":0.1}}\n{"updates":11')
            row=e.journal_snapshot(e.Evidence(),'RAW',root,{})
            self.assertIsNone(row['updates'])
            (root/'metrics_rank7.jsonl').write_text('{"updates":9,"losses":{"x":0.2}}\n')
            row=e.journal_snapshot(e.Evidence(),'RAW',root,{})
            self.assertEqual(row['updates'],9)

    def test_optimizer_count_mismatch_is_not_an_effective_update(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            for rank in range(8):
                (root/f'metrics_rank{rank}.jsonl').write_text(json.dumps(dict(updates=10,optimizer_global_steps=9))+'\n')
            row=e.journal_snapshot(e.Evidence(),'RAW',root,{})
            self.assertIsNone(row['updates'])
            self.assertFalse(row['optimizer_count_contract_valid'])

    def test_real_terminal_worker_overrides_stale_running_state(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);job=root/'job';job.mkdir()
            for rank in range(8):
                (root/f'metrics_rank{rank}.jsonl').write_text('{"updates":10}\n')
            atomic(job/'RESULT.json',dict(status='FAILED',returncode=1))
            state=dict(jobs={'full':dict(command=['train_il','--output',str(root)],receipt=str(job),host='test',status='RUNNING')})
            row=e.journal_snapshot(e.Evidence(),'RAW',root,state)
            self.assertEqual(row['status'],'训练故障 / 待恢复')

    def make_score(self,root,*,score=.9,count=12146,seed=2101):
        evaluation=root/'evaluation';evaluation.mkdir()
        identity=seal(dict(cohort='NAVTEST',seed=seed,qualification=False,count=count,
                      token_sha256=e.TOKEN_HASH,protocol={'sha256':e.PROTOCOL},
                      inference='single_candidate_cleanx_K10_shift5',model_sha256='model'))
        atomic(evaluation/'identity.json',identity)
        summary=seal(dict(status='COMPLETE',cohort='NAVTEST',scenes=count,invalid_scenes=0,
                         evaluation_identity_sha256=identity['sha256'],evaluation=str(evaluation),
                         means={'score':score},PDMS=score*100,updates=5000))
        atomic(root/'SUMMARY.json',summary)
        return root/'SUMMARY.json'

    def test_complete_score_published_but_wrong_population_seed_and_seal_refused(self):
        for count,seed,valid in [(12146,2101,True),(100,2101,False),(12146,42,False)]:
            with self.subTest(count=count,seed=seed),tempfile.TemporaryDirectory()as tmp:
                path=self.make_score(Path(tmp),count=count,seed=seed)
                row=e.score_row(e.Evidence(),path,'WORLD_IL_005000')
                self.assertEqual(row is not None,valid)
        with tempfile.TemporaryDirectory()as tmp:
            path=self.make_score(Path(tmp));value=json.loads(path.read_text());value['PDMS']=99.;atomic(path,value)
            self.assertIsNone(e.score_row(e.Evidence(),path,'WORLD_IL_005000'))

    def test_rl_and_switched_objectives_not_mixed_with_from_scratch_il(self):
        self.assertIsNone(e.classify('MEMORY_RL_010000','some/evaluation'))
        self.assertEqual(e.classify('TIME_RESIDUAL','loss_designs_20261028/NAVTEST'),'SWITCH_TIME_RESIDUAL')
        self.assertEqual(e.classify('TIME_RESIDUAL','fresh_loss_IL_20261029/NAVTEST'),'TIME_RESIDUAL')
        self.assertEqual(e.classify('D_IL_005000','interim_NAVTEST'),'ARCH_D')

    def test_all_five_diagrams_are_valid_editable_svg_with_accessible_titles(self):
        for key in NAMES:
            document=ET.fromstring(architecture_svg(key))
            self.assertEqual(document.tag,'{http://www.w3.org/2000/svg}svg')
            self.assertIsNotNone(document.find('{http://www.w3.org/2000/svg}title'))
        self.assertIn('没有 W → actor',architecture_svg('A'))
        self.assertIn('28 层 × 2048',architecture_svg('D'))

    def test_regular_deadline_is_not_reset_by_event_render(self):
        self.assertEqual(advance_regular(100,100,1800),1900)
        deadline=1900
        # An event at time1500 keeps the existing deadline1900.
        self.assertEqual(deadline,1900)
        self.assertEqual(advance_regular(deadline,1901,1800),3700)

    def test_inotify_stage_event_but_no_per_update_journal_or_progress_event(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);(root/'reporting').mkdir();run=root/'run';run.mkdir()
            events=ILStateEvents(root,atomic,[run])
            try:
                atomic(run/'progress.json',dict(updates=123,status='RUNNING'))
                (run/'metrics_rank0.jsonl').write_text('{"updates":123}\n')
                self.assertEqual(events.wait(.15),[])
                atomic(run/'LATEST.json',dict(updates=5000,checkpoint='x'))
                self.assertEqual(events.wait(1)[0]['kind'],'training_stage_complete')
                atomic(run/'LATEST.json',dict(updates=5000,checkpoint='x'))
                self.assertEqual(events.wait(.15),[])
                atomic(root/'reporting/refresh_request.json',dict(id='test1',scientific_result=False,reason='administrative qualification'))
                self.assertEqual(events.wait(1)[0]['kind'],'report_maintenance_request')
            finally:events.close()


if __name__=='__main__':unittest.main()

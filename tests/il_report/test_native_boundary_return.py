import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tools.il_report.navtest_burst import rank_completion, review_resource_boundary, summary_log


def sealed(path, value):
    body = dict(value)
    body['sha256'] = hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    path.write_text(json.dumps(body));return body


class NativeBoundaryTest(unittest.TestCase):
    def fixture(self, directory):
        root=Path(directory);output=root/'evaluation';output.mkdir();export=root/'export';export.mkdir()
        receipt=root/'jobs/rank0_attempt001';receipt.mkdir(parents=True)
        identity=sealed(output/'identity.json',dict(cohort='NAVTEST',qualification=False,seed=2101,
            count=12146,tokens=[str(i)for i in range(12146)],model_sha256='model',
            protocol=dict(sha256='canonical'),inference='single_candidate_cleanx_K10_shift5'))
        reg=root/'registration.json';sealed(reg,dict(model_sha256='model',protocol_sha256='canonical'))
        command=['python','-m','tools.qwen_world_compare.evaluate','--output',str(output),
            '--export',str(export),'--rank','0','--world','32','--seed','2101','--cohort','NAVTEST']
        spec=dict(receipt_dir=str(receipt),command=command,source='/sealed/scientist')
        (receipt/'spec.json').write_text(json.dumps(spec));(receipt/'STOP_REQUESTED').touch()
        result=dict(status='EXITED',returncode=0,forced_expiry=False,expiry_guard_requested=True,
            migration_required=True,temporary=True,host='allowed',cwd='/sealed/scientist',
            command=command,guard_unix=100,stop_requested_unix=100,end_unix=120)
        (receipt/'RESULT.json').write_text(json.dumps(result))
        row=dict(rank=0,world=32,status='STOPPED_AT_SCENE_BOUNDARY',identity=identity['sha256'],
                 host='allowed',new=214,reused=1,end_unix=119)
        (output/'rank0.json').write_text(json.dumps(row))
        return SimpleNamespace(output=output,export=export,reg=reg,source=Path('/sealed/scientist'),
                               world=32,jobs={'rank0':spec}),row,result

    def test_exit0_partial_shard_is_not_complete_and_can_requeue_with_exact_evidence(self):
        with tempfile.TemporaryDirectory()as d:
            request,row,result=self.fixture(d);receipt=Path(request.jobs['rank0']['receipt_dir'])
            old_result=(receipt/'RESULT.json').read_bytes();old_rank=(request.output/'rank0.json').read_bytes()
            self.assertFalse(rank_completion(request,'rank0'))
            self.assertTrue(review_resource_boundary(request,'rank0'))
            proof=json.loads((receipt/'OPERATOR_REQUEUE.json').read_text())
            self.assertEqual(proof['completed_scenes_to_reuse'],215)
            self.assertEqual(proof['remaining_scenes'],165)
            self.assertEqual((receipt/'RESULT.json').read_bytes(),old_result)
            self.assertEqual((request.output/'rank0.json').read_bytes(),old_rank)
            self.assertFalse((request.output/'COMPLETE').exists())

    def test_actual_complete_rank_requires_full_count_identity_and_world(self):
        for change,expected in [({},True),({'new':379},False),({'world':16},False),({'identity':'other'},False)]:
            with self.subTest(change=change),tempfile.TemporaryDirectory()as d:
                request,row,_=self.fixture(d);row.update(status='COMPLETE',new=380,reused=0);row.update(change)
                (request.output/'rank0.json').write_text(json.dumps(row))
                self.assertEqual(rank_completion(request,'rank0'),expected)

    def test_evaluator_can_observe_guard_before_worker_stop_file_timestamp(self):
        with tempfile.TemporaryDirectory()as d:
            request,row,result=self.fixture(d);result['stop_requested_unix']=119.5
            receipt=Path(request.jobs['rank0']['receipt_dir']);(receipt/'RESULT.json').write_text(json.dumps(result))
            self.assertLess(row['end_unix'],result['stop_requested_unix'])
            self.assertTrue(review_resource_boundary(request,'rank0'))

    def test_failed_forced_and_arbitrary_stops_never_automatically_retry(self):
        changes=[dict(returncode=1),dict(forced_expiry=True),dict(migration_required=False),
                 dict(expiry_guard_requested=False),dict(temporary=False),dict(cwd='/other/source'),
                 dict(host='other'),dict(stop_requested_unix=121)]
        for change in changes:
            with self.subTest(change=change),tempfile.TemporaryDirectory()as d:
                request,_,result=self.fixture(d);result.update(change)
                receipt=Path(request.jobs['rank0']['receipt_dir']);(receipt/'RESULT.json').write_text(json.dumps(result))
                self.assertFalse(review_resource_boundary(request,'rank0'))
                self.assertFalse((receipt/'OPERATOR_REQUEUE.json').exists())

    def test_changed_rank_or_old_identity_is_not_requeued(self):
        for change in [dict(rank=1),dict(world=16),dict(identity='other'),dict(status='FAILED')]:
            with self.subTest(change=change),tempfile.TemporaryDirectory()as d:
                request,row,_=self.fixture(d);row.update(change);(request.output/'rank0.json').write_text(json.dumps(row))
                self.assertFalse(review_resource_boundary(request,'rank0'))

    def test_parity_job_is_not_treated_as_a_full_rank(self):
        with tempfile.TemporaryDirectory()as d:
            request,_,_=self.fixture(d)
            self.assertFalse(review_resource_boundary(request,'parity'))

    def test_failed_summary_log_is_preserved_on_resume(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);(root/'summary.log').write_text('original failed summary evidence')
            with summary_log(root)as stream:stream.write('new legitimate full summary')
            self.assertEqual((root/'summary.log').read_text(),'original failed summary evidence')
            self.assertEqual(len(list(root.glob('summary_resume_*.log'))),1)


if __name__=='__main__':unittest.main()

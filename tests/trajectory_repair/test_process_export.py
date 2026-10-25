import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np

from scripts.analysis.trajectory_repair import export_saved_process as exporter
from starVLA.rl.qwen_sourceflow.contracts import fingerprint


class ProcessExportTransactionTest(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name);self.out=self.root/'output'
        exporter.prepare_attempt(self.out,'attempt')
        self.f=dict(status='OK',protocol_sha256='canonical',requested_trajectory=np.zeros((8,3)).tolist(),
                    simulated_states=np.zeros((41,11)).tolist(),non_drivable_body_by_time=np.zeros(41).tolist(),
                    metrics=dict(no_at_fault_collisions=1.,drivable_area_compliance=1.,time_to_collision_within_bound=1.,
                                 ego_progress=1.,comfort=1.,driving_direction_compliance=1.,score=1.))
        base=dict(candidate_id='base00',parent_candidate_id=None,external_proposal=False,feedback=self.f)
        children=[dict(candidate_id=str(i),parent_candidate_id='base00',external_proposal=True,
                       raw_reward=1.,parameters=[0,1,0,0],feedback=self.f) for i in range(24)]
        x=dict(status='COMPLETE',split='TRAIN',scene_token='train',log_id='log',
               base_model_sha256='base',common_protocol_sha256='protocol',result=dict(candidates=[base]+children))
        x['sha256']=fingerprint(x);self.source=self.root/'train.json';self.source.write_text(json.dumps(x))
        self.previous=exporter.SPEC
        exporter.SPEC=dict(output=str(self.out),attempt='attempt',identity='identity',
                           base_model_sha256='base',common_protocol_sha256='protocol',scorer={'protocol_sha256':'canonical'})
        self.addCleanup(setattr,exporter,'SPEC',self.previous)

    def test_complete_scene_publishes_only_after_all25_comparisons(self):
        with mock.patch.object(exporter.official_events,'score_scene_with_events',return_value=[self.f]) as score:
            value=exporter.export_one(str(self.source))
        self.assertEqual(score.call_count,25)
        self.assertEqual(value['candidate_scoring_calls_completed'],25)
        self.assertEqual(json.loads((self.out/'scenes/train.json').read_text())['records'],25)
        with mock.patch.object(exporter.official_events,'score_scene_with_events') as score:
            reused=exporter.export_one(str(self.source))
        score.assert_not_called();self.assertEqual(reused['status'],'REUSED')

    def test_partial_failure_retains_cost_and_does_not_publish_scene(self):
        with mock.patch.object(exporter.official_events,'score_scene_with_events',
                               side_effect=[[self.f],RuntimeError('injected scoring failure')]):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                exporter.export_one(str(self.source))
        self.assertFalse((self.out/'scenes/train.json').exists())
        self.assertFalse((self.out/'scenes/train.pt').exists())
        failed=json.loads((self.out/'attempts/attempt/train.json').read_text())
        self.assertEqual(failed['status'],'FAILED')
        self.assertEqual(failed['candidate_scoring_calls_started'],2)
        self.assertEqual(failed['candidate_scoring_calls_completed'],1)


if __name__=='__main__':unittest.main()

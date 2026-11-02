import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.il_report import navtest_cadence as cadence


class CadenceTests(unittest.TestCase):
    def test_queue_completion_request_is_recognized_by_report_event_reader(self):
        from tools.il_report.rl_events_reference import OwnerStateEvents
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);old=base/'old';four=base/'four';control=base/'control'
            for p in [old,four,control]:p.mkdir()
            with patch.object(cadence,'OLD',old),patch.object(cadence,'FOUR',four),\
                    patch.object(cadence,'ROOT',control),patch.object(cadence,'REPORT',base),\
                    patch.object(cadence,'runs',return_value={}),\
                    patch.object(cadence,'registered_points',return_value=set()):
                cadence.tick(dict(jobs=[]),max_concurrent_models=0)
            reader=object.__new__(OwnerStateEvents);reader.reporting=base/'reporting'
            events=reader.signatures(reader.reporting/'refresh_request.json')
            self.assertEqual([x['kind']for x in events.values()],['report_maintenance_request'])

    def test_pending_older_abcd_point_does_not_lose_next_retained_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);old=base/'old';four=base/'four';control=base/'control'
            for p in [old,four,control]:p.mkdir()
            run=base/'run';checkpoint=run/'checkpoints/update004000';checkpoint.mkdir(parents=True)
            (checkpoint/'COMPLETE').write_text('{}')
            state=dict(jobs=[dict(method='ARCH_A',updates=2000,status='PENDING',root=str(base/'older'))])
            def enqueue(method,path,step):
                return dict(method=method,updates=step,status='PENDING',root=str(base/'newer'))
            with patch.object(cadence,'OLD',old),patch.object(cadence,'FOUR',four),\
                    patch.object(cadence,'ROOT',control),patch.object(cadence,'REPORT',base),\
                    patch.object(cadence,'runs',return_value={'ARCH_A':run}),\
                    patch.object(cadence,'registered_points',return_value={('ARCH_A',2000)}),\
                    patch.object(cadence,'enqueue',side_effect=enqueue) as protect,\
                    patch.object(cadence.shutil,'disk_usage',return_value=type('Space',(),{'free':10**13})()):
                cadence.tick(state,max_concurrent_models=0)
            protect.assert_called_once_with('ARCH_A',checkpoint,4000)
            self.assertEqual([j['updates']for j in state['jobs']],[2000,4000])

    def test_at_least_two_thousand_after_existing_odd_checkpoint(self):
        self.assertEqual(cadence.choose_step([14000, 16000], [13000]), 16000)
        self.assertIsNone(cadence.choose_step([8000], [7000]))
        self.assertIsNone(cadence.choose_step([14000], [13000]))

    def test_initial_current_available_checkpoint_without_unbounded_backfill(self):
        self.assertEqual(cadence.choose_step([52000, 54000, 56000], [50000]), 52000)
        self.assertEqual(cadence.choose_step([52000, 54000, 56000], []), 56000)

    def test_future_milestone_and_duplicate_are_not_requeued(self):
        self.assertIsNone(cadence.choose_step([18000, 20000], [18000, 20000]))
        self.assertEqual(cadence.choose_step([20000, 22000], [20000]), 22000)

    def test_native_pin_survives_original_rolling_retention(self):
        with tempfile.TemporaryDirectory() as d:
            original = Path(d)/'native'; original.mkdir()
            payload = original/'model.pt'; payload.write_bytes(b'not actual model data')
            (original/'identity.json').write_text('{}')
            files = {'model.pt': hashlib.sha256(payload.read_bytes()).hexdigest()}
            (original/'files.json').write_text(json.dumps(files))
            (original/'COMPLETE').write_text(json.dumps(dict(updates=2000,
                files_sha256=hashlib.sha256((original/'files.json').read_bytes()).hexdigest())))
            destination = Path(d)/'pinned'
            cadence.pin_checkpoint(original, destination)
            payload.unlink()
            self.assertEqual((destination/'model.pt').read_bytes(), b'not actual model data')

    def test_existing_nested_full_owner_and_ad_hoc_points_are_detected(self):
        with tempfile.TemporaryDirectory() as d:
            old, four = Path(d)/'old', Path(d)/'four'; old.mkdir(); four.mkdir()
            a = four/'interim_NAVTEST'; a.mkdir()
            (a/'PINNED.json').write_text(json.dumps(dict(models={'A_IL_013000': dict(updates=13000)})))
            b = old/'fresh_loss_IL_20261029/TIME_RESIDUAL/NAVTEST/update020000'
            b.mkdir(parents=True)
            (b/'PINNED.json').write_text(json.dumps(dict(models={'TIME_RESIDUAL_IL_020000': dict(updates=20000)})))
            with patch.object(cadence, 'OLD', old), patch.object(cadence, 'FOUR', four):
                points = cadence.registered_points()
            self.assertIn(('ARCH_A', 13000), points)
            self.assertIn(('TIME_RESIDUAL', 20000), points)

    def test_32_way_shards_cover_each_scene_once(self):
        tokens = list(range(12146))
        for workers in (8, 16, 32):
            shards = [tokens[rank::workers] for rank in range(workers)]
            self.assertEqual(sorted(x for shard in shards for x in shard), tokens)
            self.assertEqual(len(set(x for shard in shards for x in shard)), 12146)


if __name__ == '__main__':
    unittest.main()

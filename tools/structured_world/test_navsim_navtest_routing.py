"""CPU-only failure/recovery tests on real filesystem locks and publication code.

Synthetic payloads are never model predictions or benchmark scores. Tests do not
access GPUs, SSH, live queues, training processes or historical assets.
"""
from concurrent.futures import ThreadPoolExecutor
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
import shutil
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).with_name('run_navsim_navtest.py')
spec = importlib.util.spec_from_file_location('navtest_routing_under_test', SCRIPT)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='navtest-routing-test-')
        self.root = Path(self.temp.name)

    def tearDown(self): self.temp.cleanup()

    def test_scene_pacing_is_only_completed_scene_wall_time(self):
        self.assertEqual(module.inference_idle_seconds(2., 1.), 0.)
        self.assertEqual(module.inference_idle_seconds(2., .2), 8.)
        self.assertEqual(module.inference_idle_seconds(0., .2), 0.)
        for duty in (0, -.1, 1.1, .01, float('nan'), float('inf')):
            with self.assertRaises(ValueError): module.inference_idle_seconds(2., duty)
        for elapsed in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError): module.inference_idle_seconds(elapsed, .2)

    def test_live_pacing_override_and_idle_default(self):
        policy = self.root/'scheduling.json'
        module.atomic(policy, {'owned_training_inference_duty_cycle': .2,
                              'owned_training_duty_by_host': {'host-a': .1}})
        args = SimpleNamespace(cohost_duty_cycle=.2, scheduling_policy=policy)
        with patch.object(module.socket, 'gethostname', return_value='host-a-worker-0'):
            self.assertEqual(module.scheduling_duty_cycle(args), .1)
            module.atomic(policy, {'owned_training_inference_duty_cycle': .05})
            self.assertEqual(module.scheduling_duty_cycle(args), .05)
        self.assertEqual(module.scheduling_duty_cycle(SimpleNamespace()), 1.)

    def record(self, root, token, identity='fixture', lease=None, owner=None):
        target = root/'predictions'/(token+'.npz'); target.parent.mkdir(parents=True,exist_ok=True)
        temp = target.with_suffix('.tmp'); temp.write_bytes(('synthetic-'+token).encode())
        record = root/'records'/(token+'.json'); record.parent.mkdir(parents=True,exist_ok=True)
        module.publish_owned(target,temp,record,dict(token=token,identity=identity,sha256=module.digest(temp)),lease,owner)

    def test_five_to_two_workers_and_global_coverage(self):
        root=self.root/'job'; rows=[dict(token=str(i).zfill(6),log='log'+str(i%136)) for i in range(12146)]
        module.atomic(root/'identity.json',dict(identity='fixture'))
        previous=module.new_attempt(root,[dict(uuid=str(i)) for i in range(5)])
        # Three old workers completed chunks, one wrote a partial chunk then
        # crashed, and another never published. All reports remain immutable.
        for first in (0,32,64):
            for i in range(first,first+32): self.record(root,rows[i]['token'])
            module.atomic(root/'work_leases'/(str(first).zfill(6)+'.json'),dict(first=first,end=first+32,status='COMPLETE',owner='old'+str(first),heartbeat_unix=time.time()))
        self.record(root,rows[96]['token'])
        old_lease=root/'work_leases/000096.json'
        module.atomic(old_lease,dict(first=96,end=128,status='ACTIVE',owner='dead-generation',heartbeat_unix=time.time()))
        for i in range(5):
            module.atomic(root/'attempts'/previous/'shards'/('shard_'+str(i).zfill(3)+'.json'),
                dict(status='COMPLETE' if i<3 else 'FAILED',prediction_identity='fixture',scenes_done=32 if i<3 else 1 if i==3 else 0,inference_seconds=10))
        recovered=module.reconcile_infrastructure_leases(root,rows)
        self.assertEqual(recovered['verified_prior_scenes'],97)
        self.assertEqual(recovered['fenced_leases'],['000096.json'])
        self.assertNotEqual(json.loads(old_lease.read_text())['owner'],'dead-generation')
        # The stale generation cannot overwrite a newly fenced prediction.
        temp=root/'stale.tmp';temp.write_bytes(b'bad')
        with self.assertRaisesRegex(RuntimeError,'Lost lease'):
            module.publish_owned(root/'predictions/000096.npz',temp,root/'records/000096.json',{},old_lease,'dead-generation')
        attempt=module.new_attempt(root,[dict(uuid=str(i)) for i in range(2)])
        def resumed(i):
            report=dict(shard=i,attempt=attempt,scenes_done=0,status='COMPLETE',prediction_identity='fixture',inference_seconds=20)
            for ordinal,lease,owner in module.leased_indices(root,rows,report):
                token=rows[ordinal]['token']
                if not (root/'records'/(token+'.json')).exists(): self.record(root,token,lease=lease,owner=owner)
                report['scenes_done']+=1
            module.atomic(root/'attempts'/attempt/'shards'/('shard_'+str(i).zfill(3)+'.json'),report)
            return report['scenes_done']
        with ThreadPoolExecutor(max_workers=2) as pool: counts=list(pool.map(resumed,range(2)))
        self.assertEqual(sum(counts),12146-96)
        self.assertNotEqual(sum(counts),12146)
        module.validate_record_union(root,[r['token'] for r in rows],'fixture')
        index=self.root/'index.json';index.write_text(json.dumps(rows))
        old_index=module.INDEX
        try:
            module.INDEX=index;module.finish_predictions(root,2,attempt)
        finally:module.INDEX=old_index
        final=json.loads((root/'COMPLETE.json').read_text())
        self.assertEqual(final['scenes'],12146);self.assertEqual(final['attempt_reports'],7)
        self.assertAlmostEqual(final['GPU_hours'],(5*10+2*20)/3600)
        self.assertEqual(len(list((root/'attempts'/previous/'shards').glob('*.json'))),5)
        # Hash-corrupted successes and incomplete COMPLETE leases are scientific errors.
        (root/'predictions/000000.npz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'Prior prediction invalid'):module.reconcile_infrastructure_leases(root,rows)

    def test_invalid_complete_is_never_queue_success(self):
        root=self.root/'score';req=dict(training_identity='train',tag='milestone_0050000',checkpoint_files_sha256={},updates=50000)
        module.atomic(root/'identity.json',dict(identity='pred',training_identity='train',checkpoint='milestone_0050000',
            checkpoint_files_sha256={},updates=50000,evaluation_source_sha=module.SHA,sampling_seed=42))
        module.atomic(root/'COMPLETE.json',dict(identity='pred',scenes=12146,full_population=True))
        module.atomic(root/'scores/identity.json',dict(identity='score',prediction_identity='pred'))
        rows=[dict(token=str(i).zfill(6),log='log'+str(i%136)) for i in range(12146)]
        index=self.root/'score-index.json';index.write_text(json.dumps(rows));old_index=module.INDEX;module.INDEX=index
        self.addCleanup(setattr,module,'INDEX',old_index)
        for row in rows:self.record(root,row['token'],identity='pred')
        summaries={}
        for variant in ('q0','q_final'):
            csv=root/'scores'/variant/'scenes.csv';csv.parent.mkdir(parents=True);csv.write_text('synthetic\n')
            summaries[variant]=dict(identity='score',scenes=12146,failed=0,valid=True,scene_csv_sha256=module.digest(csv),
                metrics={k:1. for k in module.METRICS},zero_fraction=0.)
            module.atomic(csv.parent/'summary.json',summaries[variant])
        module.atomic(root/'scores/COMPLETE.json',dict(identity='score',**summaries))
        with self.assertRaisesRegex(ValueError,'CSV population'):module.valid_score_completion(root,req)
        for variant in ('q0','q_final'):
            path=root/'scores'/variant/'scenes.csv'
            with path.open('w') as stream:
                writer=__import__('csv').DictWriter(stream,['token','log','status','variant',*module.METRICS]);writer.writeheader()
                for row in rows:writer.writerow(dict(row,status='ok',variant=variant,**{k:1. for k in module.METRICS}))
            summaries[variant]['scene_csv_sha256']=module.digest(path)
            module.atomic(path.parent/'summary.json',summaries[variant])
        module.atomic(root/'scores/COMPLETE.json',dict(identity='score',**summaries))
        self.assertTrue(module.valid_score_completion(root,req))
        summaries['q_final']['metrics']['score']=.5
        module.atomic(root/'scores/q_final/summary.json',summaries['q_final'])
        module.atomic(root/'scores/COMPLETE.json',dict(identity='score',**summaries))
        with self.assertRaisesRegex(ValueError,'mean differs'):module.valid_score_completion(root,req)
        summaries['q_final']['metrics']['score']=1.
        summaries['q_final'].update(failed=1,valid=False)
        module.atomic(root/'scores/q_final/summary.json',summaries['q_final'])
        module.atomic(root/'scores/COMPLETE.json',dict(identity='score',**summaries))
        with self.assertRaisesRegex(ValueError,'Invalid completed'):module.valid_score_completion(root,req)
        # A foreign score identity cannot certify success even with zero failures.
        summaries['q_final'].update(failed=0,valid=True)
        module.atomic(root/'scores/q_final/summary.json',summaries['q_final'])
        module.atomic(root/'scores/COMPLETE.json',dict(identity='foreign',**summaries))
        with self.assertRaisesRegex(ValueError,'identity'):module.valid_score_completion(root,req)

    def test_completed_lease_requires_all_hash_bound_records(self):
        root=self.root/'incomplete';module.atomic(root/'identity.json',dict(identity='fixture'))
        rows=[dict(token='one'),dict(token='two')];self.record(root,'one')
        module.atomic(root/'work_leases/000000.json',dict(first=0,end=2,status='COMPLETE',owner='old',heartbeat_unix=0))
        with self.assertRaisesRegex(ValueError,'Completed lease lacks'):module.reconcile_infrastructure_leases(root,rows)

    def test_real_frozen_four_scene_payloads_and_hash_corruption(self):
        reference=module.DEFAULT_OUTPUT/'qualification/G3_EVENT_LOCAL_0050000/FROZEN_REFERENCE'
        if not (reference/'COMPLETE.json').exists():self.skipTest('Read-only native four-scene reference unavailable')
        root=self.root/'native_payloads';identity=json.loads((reference/'identity.json').read_text())['identity']
        records=list((reference/'records').glob('*.json'));self.assertEqual(len(records),4)
        (root/'records').mkdir(parents=True);(root/'predictions').mkdir()
        for p in records:
            shutil.copyfile(p,root/'records'/p.name)
            shutil.copyfile(reference/'predictions'/(p.stem+'.npz'),root/'predictions'/(p.stem+'.npz'))
        module.validate_record_union(root,[p.stem for p in records],identity)
        payload=root/'predictions'/(records[0].stem+'.npz')
        with payload.open('ab') as stream:stream.write(b'infrastructure cannot hide scientific corruption')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            module.validate_record_union(root,[p.stem for p in records],identity)


if __name__=='__main__':unittest.main(verbosity=2)

"""Finite metadata fixtures only; no real W tensor access, fit or dispatch."""
import time
START=time.monotonic()
with open('/proc/self/io') as f:BOOT={k:int(v) for k,v in (line.split(':') for line in f)}
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).absolute().parent
sys.path.insert(0,str(HERE))
import train_artifact_receipt as r
import supervise_fit as s

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
PINS=json.loads((HERE/'INPUT_PINS.json').read_text())
REVIEW=PINS['review']
assert sha(REVIEW['path'])==REVIEW['sha256']
REVIEW_DATA=json.loads(Path(REVIEW['path']).read_text())
# Read only the small pinned JSON controls. Never open/stat the tensor payloads.
DOCUMENTS={REVIEW['path']:(REVIEW_DATA,REVIEW)}
for path,pin in REVIEW_DATA['input_pins_unchanged'].items():
    assert path.endswith('.json')
    raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==pin['sha256']
    DOCUMENTS[path]=(json.loads(raw),{'path':path,'sha256':pin['sha256']})
raw=Path(s.FIT_PIN['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==s.FIT_PIN['sha256']
DOCUMENTS[s.FIT_PIN['path']]=(json.loads(raw),s.FIT_PIN)


class Consumer(unittest.TestCase):
    def setUp(self):
        self.docs=copy.deepcopy(DOCUMENTS)
        self.versions={x['payload_path']:copy.deepcopy(x['file_version']) for x in REVIEW_DATA['records']}
        self.patch=patch.object(r,'file_version',side_effect=lambda path:self.versions[str(path)])
        self.patch.start();self.addCleanup(self.patch.stop)
        self.failed={'path':str(r.FAILED_PATH),'sha256':r.FAILED_SHA}
        self.prep=self.docs[r.PREPARATION_PIN['path']][0]
        self.receipt=r.create_receipt(REVIEW,self.failed,r.PREPARATION_PIN,r.TRAIN_ROOT,self.read)
        self.receipt_pin={'path':'/tmp/st_train64_metadata_fixture/ACCEPTED.json','sha256':'1'*64}
        self.docs[self.receipt_pin['path']]=(self.receipt,self.receipt_pin)
        self.ad={'stage':'fit','preparation':r.PREPARATION_PIN,s.KEY:self.receipt_pin,
                 s.SOURCES:{n:sha(HERE/n) for n in ('supervise_fit.py','train_artifact_receipt.py')}}
        self.calls=[]
        self.original=SimpleNamespace(
            validate=lambda *args:self.calls.append('validate') or (1260,2<<30),
            stage_inputs=lambda *args:self.calls.append('original_stage_inputs') or 'UNCHANGED',
            check_manual_fields=lambda *args:self.calls.append('manual_fields'),
            read=self.read,FIT_READ_LIMIT=2<<30,JOB=r.JOB)
        self.monitor=object();self.original.supervise=self.monitor
        self.consumer=s.install(self.original)

    def read(self,path,expected=None):
        value,pin=self.docs[str(path)]
        if expected is not None and pin['sha256']!=expected:raise ValueError('fixture pin mismatch')
        return copy.deepcopy(value),dict(pin)

    def validate(self):return self.consumer.validate(self.ad,self.prep,'fit',0)
    def regenerate(self):return r.create_receipt(REVIEW,self.failed,r.PREPARATION_PIN,r.TRAIN_ROOT,self.read)

    def test_fixed_train64_keeps_failed_status_and_single_attempt(self):
        self.assertEqual(self.receipt['status'],'ARTIFACTS_ACCEPTED_SUPERVISOR_FAILED')
        self.assertEqual(self.receipt['supervisor_status'],'STOPPED_NO_AUTO_RETRY')
        self.assertEqual(self.receipt['eligible_consumer'],'fit')
        self.assertEqual(len(self.receipt['payload_snapshots']),64)
        self.assertEqual(self.receipt['upstream_accounting']['attempt_count_preserved'],1)
        self.assertFalse(self.receipt['launch_approved'])

    def test_original_validate_monitor_and_fit_limits_preserved(self):
        self.assertEqual(self.validate(),(1260,2<<30))
        self.assertEqual(self.calls,['validate'])
        self.assertIs(self.consumer.supervise,self.monitor)
        path,account,pin=self.consumer.stage_inputs(self.ad,'fit')
        self.assertEqual(path,r.JOB/'stages/fit');self.assertEqual(pin,self.failed)
        self.assertEqual(account['remaining_read_bytes'],2<<30)
        self.assertEqual(account['prior_read_bytes'],0)
        self.assertEqual(account['upstream_supervisor_status_preserved'],'STOPPED_NO_AUTO_RETRY')

    def test_absent_new_field_uses_original_stage_inputs(self):
        self.assertEqual(self.consumer.stage_inputs({},'train_plan'),'UNCHANGED')
        self.assertEqual(self.calls,['original_stage_inputs'])

    def test_other_stage_and_retry_keys_rejected(self):
        for stage in ('train_plan','train_extract','dev_extract'):
            with self.subTest(stage=stage),self.assertRaises(ValueError):
                self.consumer.validate(self.ad,self.prep,stage,0)
        for key in ('manual_retry','previous_stage_result','previous_artifact_reconciliation','storage_compatibility'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.consumer.validate({**self.ad,key:{}},self.prep,'fit',0)

    def test_missing_changed_or_null_new_source_pin_rejected(self):
        for sources in ({},{'supervise_fit.py':'0'*64,'train_artifact_receipt.py':self.ad[s.SOURCES]['train_artifact_receipt.py']},
                        {'supervise_fit.py':None,'train_artifact_receipt.py':self.ad[s.SOURCES]['train_artifact_receipt.py']}):
            with self.subTest(sources=sources),self.assertRaises((ValueError,TypeError)):
                self.consumer.validate({**self.ad,s.SOURCES:sources},self.prep,'fit',0)

    def test_changed_fit_registration_rejected(self):
        prep=copy.deepcopy(self.prep);prep['fit_registration']['sha256']='0'*64
        with self.assertRaises(ValueError):self.consumer.validate(self.ad,prep,'fit',0)

    def test_wrong_failed_hash_path_root_or_preparation_rejected(self):
        cases=[({'path':self.failed['path'],'sha256':'0'*64},r.PREPARATION_PIN,r.TRAIN_ROOT),
               ({'path':'/tmp/copied.json','sha256':r.FAILED_SHA},r.PREPARATION_PIN,r.TRAIN_ROOT),
               (self.failed,r.PREPARATION_PIN,r.TRAIN_ROOT.parent/'dev'),
               (self.failed,{'path':r.PREPARATION_PIN['path'],'sha256':'0'*64},r.TRAIN_ROOT)]
        for failed,prep,root in cases:
            with self.subTest(root=root,failed=failed),self.assertRaises(ValueError):
                r.create_receipt(REVIEW,failed,prep,root,self.read)

    def test_wrong_review_rejected(self):
        with self.assertRaises(ValueError):
            r.create_receipt({**REVIEW,'sha256':'0'*64},self.failed,r.PREPARATION_PIN,r.TRAIN_ROOT,self.read)

    def test_failed_compute_cleanup_or_limits_cannot_reconcile(self):
        original=copy.deepcopy(self.docs[str(r.FAILED_PATH)][0])
        changes=[('exit_code',1),('limit_reason','WALL_LIMIT'),('first_monitor_error','bad')]
        for key,value in changes:
            failed=copy.deepcopy(original);failed['resources'][key]=value
            self.docs[str(r.FAILED_PATH)]=(failed,self.failed)
            with self.subTest(key=key),self.assertRaises(ValueError):self.regenerate()
        failed=copy.deepcopy(original);failed['resources']['process_cleanup']['live_owned_remaining']=1
        self.docs[str(r.FAILED_PATH)]=(failed,self.failed)
        with self.assertRaises(ValueError):self.regenerate()

    def test_no_artificial_complete_or_other_stage(self):
        failed=self.docs[str(r.FAILED_PATH)][0]
        failed['status']='COMPLETE'
        with self.assertRaises(ValueError):self.regenerate()
        failed['status']='STOPPED_NO_AUTO_RETRY';failed['stage']='dev_extract'
        with self.assertRaises(ValueError):self.regenerate()

    def test_population_shape_and_record_missing_rejected(self):
        review=self.docs[REVIEW['path']][0];review['scenes']=16
        with self.assertRaises(ValueError):self.regenerate()
        review['scenes']=64;review['records'][0]['shape']=[144,256]
        with self.assertRaises(ValueError):self.regenerate()
        review['records'][0]['shape']=[144,2048];review['records'].pop()
        with self.assertRaises(ValueError):self.regenerate()

    def test_nonfinite_and_wrong_mask_rejected(self):
        record=self.docs[REVIEW['path']][0]['records'][0];record['finite']=False
        with self.assertRaises(ValueError):self.regenerate()
        record['finite']=True;record['mask_all_true']=False
        with self.assertRaises(ValueError):self.regenerate()

    def test_payload_version_change_since_review_rejected(self):
        self.versions[next(iter(self.versions))]['mtime_ns']+=1
        with self.assertRaises(ValueError):self.regenerate()

    def test_resource_accounting_change_rejected(self):
        self.docs[REVIEW['path']][0]['extract_charged_read_bytes']+=1
        with self.assertRaises(ValueError):self.regenerate()

    def test_forged_receipt_or_wrong_consumer_rejected(self):
        self.receipt['eligible_consumer']='train_plan'
        self.receipt['identity']=r.identity({k:v for k,v in self.receipt.items() if k!='identity'})
        with self.assertRaises(ValueError):self.consumer.stage_inputs(self.ad,'fit')

    def test_original_dev_and_supervisor_source_unchanged(self):
        for name in ('original_dev_validator','original_supervisor'):
            pin=PINS[name];self.assertEqual(sha(HERE.parents[3]/pin['path']),pin['sha256'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Consumer))
    with open('/proc/self/io') as f:final={k:int(v) for k,v in (line.split(':') for line in f)}
    print('COST_JSON='+json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        wall_seconds=time.monotonic()-START,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        logical_read_once_bytes=final['rchar']-BOOT['rchar'],kernel_read_bytes=final['read_bytes']-BOOT['read_bytes'],
        cpu_affinity=sorted(os.sched_getaffinity(0)),fixture_disk_written_bytes=0,
        tensor_payload_read=False,checkpoint_read=False,ledger_access=False,actual_fit=False),sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)

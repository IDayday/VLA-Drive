"""Exact TRAIN64 consumer checks using small metadata fixtures, never W bytes."""
import copy,hashlib,importlib.util,json,socket,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'w_train_reconciliation_v1'
spec=importlib.util.spec_from_file_location('owner_consumer_fixtures',SOURCE/'test_consumer.py');owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
r,s=owner.r,owner.s
class Independent(unittest.TestCase):
 def setUp(self):
  self.fixture=owner.Consumer();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
 def test_all_payload_stat_fields_fail_closed(self):
  f=self.fixture;key=next(iter(f.versions));original=copy.deepcopy(f.versions[key])
  for field in ('device','inode','bytes','mtime_ns','ctime_ns','nlink'):
   with self.subTest(field=field):
    f.versions[key]=copy.deepcopy(original);f.versions[key][field]+=1
    with self.assertRaises(ValueError):f.regenerate()
 def test_changed_receipt_pin_and_payload_hash_rejected(self):
  f=self.fixture;record=f.docs[owner.REVIEW['path']][0]['records'][0]
  record['receipt']['sha256']='a'*64
  with self.assertRaises(ValueError):f.regenerate()
  record['receipt']['sha256']=owner.REVIEW_DATA['records'][0]['receipt']['sha256'];record['payload_sha256']='b'*64
  with self.assertRaises(ValueError):f.regenerate()
 def test_row_reorder_and_duplicate_not_accepted(self):
  f=self.fixture;signature=f.docs[str(r.TRAIN_ROOT/'identity.json')][0];rows=signature['rows'];rows[0],rows[1]=rows[1],rows[0]
  with self.assertRaises(ValueError):f.regenerate()
  rows[0],rows[1]=rows[1],rows[0];rows[-1]=copy.deepcopy(rows[0])
  with self.assertRaises(ValueError):f.regenerate()
 def test_original_admission_validator_still_runs_and_rejects(self):
  f=self.fixture;original=s.load_original();supervise=original.supervise;main=original.main
  original.read=f.read;original= s.install(original)
  helpers={n:owner.sha(Path(original.__file__).with_name(n)) for n in original.HELPERS}
  admission={**f.ad,'schema':'st_SP0_probe_stage_admission_v1','status':'APPROVED','reviewer':'independent-fixture-only','hostname':socket.gethostname(),'valid_until_unix':time.time()+2000,'runtime':f.prep['runtime'],'cpu_affinity':[6,7],'helpers':helpers,'gpu':None,'storage_policy':None}
  self.assertEqual(original.validate(admission,f.prep,'fit',time.time()),(1260,2<<30));self.assertIs(original.supervise,supervise);self.assertIs(original.main,main)
  for key,value in [('valid_until_unix',0),('gpu',{'index':0}),('cpu_affinity',[6]),('helpers',{})]:
   with self.subTest(key=key),self.assertRaises(ValueError):original.validate({**admission,key:value},f.prep,'fit',time.time())
  output,account,previous=original.stage_inputs(admission,'fit');self.assertEqual(output,r.JOB/'stages/fit');self.assertEqual(account['remaining_read_bytes'],2<<30);self.assertEqual(previous,f.failed)
 def test_wrong_fit_role_and_absent_reconciliation_do_not_bypass(self):
  f=self.fixture
  with self.assertRaises(ValueError):f.consumer.validate({**f.ad,'stage':'train_extract'},f.prep,'fit',0)
  no=dict(f.ad);no.pop(s.KEY)
  with self.assertRaises(ValueError):f.consumer.validate(no,f.prep,'fit',0)
 def test_receipt_creation_does_not_mutate_failed_or_grant_start(self):
  f=self.fixture;before=copy.deepcopy(f.docs);receipt=f.regenerate();self.assertEqual(before,f.docs)
  for key in ('launch_approved','automatic_retry','new_resource_authorization','scientific_result','payload_content_revalidated_here'):self.assertIs(receipt[key],False)
  self.assertEqual(receipt['upstream_accounting']['additional_extraction_attempts'],0)
 def test_exact_runtime_protocol_and_source_boundaries(self):
  self.assertEqual(owner.sha(SOURCE/'train_artifact_receipt.py'),'782f29a8aad0518c2e688e030ded9994f9feab54660d5d5773462c6e8a04edd5')
  self.assertEqual(owner.sha(SOURCE/'supervise_fit.py'),'2e630f3b1db30cdade58d960982d9f4aa5d8344518ce688e8fb1c2b96a643c06')
  self.assertEqual(owner.sha(HERE.parent/'spatial_probe_actual_preparation_v1/observed_cli.py'),'c1af891b440663d75ca3b75c1227f5a7d4247d62319a566214848e0a5550f643')
  fit=owner.DOCUMENTS[s.FIT_PIN['path']][0];protocol=Path(fit['protocol']['path']);self.assertEqual(owner.sha(protocol),fit['protocol']['sha256']);self.assertEqual(json.loads(protocol.read_text())['recipe']['updates'],300)
  self.assertEqual(fit['device'],'cpu');self.assertNotIn('torch',sys.modules)

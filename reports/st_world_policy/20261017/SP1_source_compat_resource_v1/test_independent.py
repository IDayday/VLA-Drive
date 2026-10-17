"""Small-control compatibility/resource difference only; no real payload/migration."""
import copy,hashlib,importlib.util,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path('/mnt/project/VLA-Drive-st-w-storage-source-compat-20261017/reports/st_world_policy/20261017/w_storage_source_compat_v1')
spec=importlib.util.spec_from_file_location('review_exact_compat',HERE/'compat.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
PIN={'path':str(HERE/'COMPATIBILITY.json'),'sha256':'bd899455f1b3af2efe41570bd21a0e1b59109c671d38339022ccb1323f24609f'}
PROOF=c.verify(PIN)  # real small git/source proof once; no native run identity
with patch.object(c,'verify',return_value=PROOF):INSTALL=c.install(PIN)
w=sys.modules[c.TARGET]
OLD=Path('/mnt/project/st-world-policy-20261016/audits/full_future_index_preparation_v1/finalized_phase_v1/PROPOSED_POLICY.json')
POLICY=json.loads(OLD.read_text())
CLAIM=json.loads(Path(__file__).resolve().parents[1].joinpath('SP1_dev_typed_migration_resource_v1/PROPOSED_SP1_DEV_CLAIM_NOT_ADMITTED.json').read_text())
POLICY['claims'].append(CLAIM);POLICY['frozen_w_package']['members'][CLAIM['root']]=CLAIM
SIGNATURES={row['signature_path']:json.loads(Path(row['signature_path']).read_text()) for row in POLICY['frozen_w_package']['members'].values()}

def validate(sigs=SIGNATURES,policy=POLICY):
    def deep_stub(s):
        row=next(x for x in policy['claims'] if x.get('cache_identity')==s['identity'])
        return s['binding'],{k:row[k] for k in ('output_bytes','staging_bytes','metadata_bytes')}
    with patch.object(w,'_pin',side_effect=lambda pin:sigs[pin['path']]),patch.object(w,'validate_signature',side_effect=deep_stub):
        return w.validate_package(policy,{r['root']:r for r in policy['claims']})

class Test(unittest.TestCase):
    def test_actual_shared_guard_mixed_sources_now_accepts_and_preserves_documents(self):
        before=copy.deepcopy(SIGNATURES);self.assertEqual(validate(),138426288);self.assertEqual(SIGNATURES,before)
    def test_arbitrary_third_runtime_still_rejected(self):
        s=copy.deepcopy(SIGNATURES);s[CLAIM['signature_path']]['binding']['source_sha']='0'*40
        with self.assertRaisesRegex(ValueError,'Unapproved'):validate(s)
    def test_base_or_environment_change_still_rejected(self):
        for field in ('base','environment'):
            s=copy.deepcopy(SIGNATURES);v=s[CLAIM['signature_path']]
            if field=='base':v['checkpoint']['base_binding']='0'*64
            else:v['binding']['environment']={'changed':True}
            with self.assertRaisesRegex(Exception,'share original Base'):validate(s)
    def test_same_arm_endpoint_source_is_not_canonicalized(self):
        s=copy.deepcopy(SIGNATURES)
        train=next(k for k,v in s.items() if v['binding']['arm']=='S-P0' and v['binding']['observations']['split']=='train')
        s[train]['binding']['source_sha']=c.NEW
        with self.assertRaisesRegex(Exception,'same arm checkpoint/runtime'):validate(s)
    def test_only_common_expression_is_patched(self):
        original=(c.ROOTS[c.NEW]/'tools/st_world_policy/spatial_w_storage.py').read_text()
        new=c.patched_source(original)
        self.assertEqual(new.replace("_verified_common_compute_source(binding['source_sha'])","binding['source_sha']"),original)
        with self.assertRaises(ValueError):c.patched_source(new)
    def test_wrong_proof_cannot_install_and_repeated_install_rejected(self):
        with patch.object(c,'verify',return_value=PROOF),self.assertRaisesRegex(ValueError,'before any'):c.install(PIN)
        bad=dict(PIN,sha256='0'*64)
        with self.assertRaises(ValueError):c.verify(bad)
    def test_wrapper_preimport_io_baseline_and_original_counters_are_retained(self):
        source=(HERE/'run_once.py').read_text()
        self.assertLess(source.index("with open('/proc/self/io')"),source.index('import argparse'))
        self.assertIn('baseline = COMPAT_BOOT_IO',source);self.assertIn('INSTALL_COMPAT()',source)
        self.assertNotIn('read_cap +',source);self.assertNotIn('read_cap *',source)
        self.assertIn('original observer boundary',source.lower())

if __name__=='__main__':
    print(json.dumps({'actual_proof_verified':True,'source_control_bytes':c.READ_BYTES,'compute_digest':PROOF['common_compute_blob_sha256'],'module':w.__file__,'deep_signature_reads_stubbed_to_isolate_package_guard':True}),flush=True)
    unittest.main(verbosity=2)

"""CPU safety controls; synthetic receipts do not qualify native tensors."""
import json
from pathlib import Path
import tempfile
import unittest

import prepare_native_e2_comparisons as c
import execution_e2_provenance as p


class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name);self.reader=p.Reader()

    def tearDown(self):self.tmp.cleanup()

    def write(self,path,obj):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))

    def test_original_comparator_and_full_cpu_controls(self):
        command=c.command('python',dict(A='/a',AA='/aa',B='/b'),'/origin',25000,32,'/out')
        self.assertEqual(p.argv_value(command,'--reference-replay'),'/aa/checkpoints/milestone_0025032')
        self.assertEqual(p.argv_value(command,'--origin'),'/origin')
        self.assertEqual(p.argv_value(command,'--threads'),'2')
        self.assertEqual(p.argv_value(command,'--chunk-elements'),'262144')
        self.assertNotIn('--skip-payload-hashes',command);self.assertNotIn('--metadata-only',command)

    def test_no_extra_observation(self):
        with self.assertRaises(ValueError):c.command('python',dict(A='/a',AA='/aa',B='/b'),'/origin',25000,200,'/out')

    def test_missing_role_not_a_triplet(self):
        with self.assertRaises(ValueError):c.command('python',dict(A='/a',B='/b'),'/origin',25000,1,'/out')

    def test_shared_absolute_string_fails_registered_alias_guard(self):
        stage={'image_root':'/var/tmp/current/images','scenes':101592,'source_identity':'s','image_hash_manifest':'h'}
        with self.assertRaisesRegex(ValueError,'registered alias'):c.residency_check('/mnt/project/current/images',stage,{'shared_resolved':'/mnt/project/current'})

    def test_same_alias_keeps_shared_physical_storage_without_metadata_rewrite(self):
        stage={'image_root':'/var/tmp/current/images','scenes':101592,'source_identity':'s','image_hash_manifest':'h'}
        out=c.residency_check('/var/tmp/current/images',stage,{'shared_resolved':'/mnt/project/current'})
        self.assertTrue(out['remote_alias_NOT_locally_resolved'])

    def test_alias_needs_actual_installation_receipt(self):
        stage={'image_root':'/var/tmp/current/images','scenes':101592,'source_identity':'s','image_hash_manifest':'h'}
        with self.assertRaises(ValueError):c.residency_check('/var/tmp/current/images',stage,None)

    def jobs(self):
        return {r:{'run':str(self.root/r),'command':['actual',r],'origin':{'tag':'milestone_0025000'}} for r in p.ROLES}

    def test_missing_endpoints_block_heavy_comparison(self):
        out=c.status(self.reader,self.root/'controller','G1_mid',self.jobs())
        self.assertFalse(out['ready_for_full_CPU_comparison']);self.assertIn('B/PROCESS_EXIT.json',out['missing'])

    def test_controller_failure_is_terminal_even_when_other_metadata_exists(self):
        self.write(self.root/'controller/FAILURE.json',{'error':'residency','completed_jobs':0})
        out=c.status(self.reader,self.root/'controller','G1_mid',self.jobs())
        self.assertFalse(out['ready_for_full_CPU_comparison']);self.assertEqual(out['failures'][0]['completed_jobs'],0)

    def test_worker_failure_preserved_not_zero_cost_success(self):
        self.write(self.root/'A/PROCESS_EXIT.json',{'returncode':1,'monitor_error':None})
        out=c.status(self.reader,self.root/'controller','G1_mid',self.jobs())
        self.assertFalse(out['ready_for_full_CPU_comparison']);self.assertEqual(out['failures'][0]['returncode'],1)

    def test_actual_argv_cannot_be_rewritten(self):
        self.write(self.root/'A/PROCESS.json',{'command':['different']})
        with self.assertRaisesRegex(ValueError,'argv'):c.status(self.reader,self.root/'controller','G1_mid',self.jobs())


if __name__ == '__main__':unittest.main()

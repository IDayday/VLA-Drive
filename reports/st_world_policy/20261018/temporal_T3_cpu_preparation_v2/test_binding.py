import hashlib, importlib.util, json, subprocess, unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'20261017/temporal_T3_cpu_preparation_v1'
spec=importlib.util.spec_from_file_location('v2_binding',HERE/'prepare_once.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
class Binding(unittest.TestCase):
    def test_exact_four_pin_path_edits_only(self):
        old=(OLD/'prepare_once.py').read_text();new=(HERE/'prepare_once.py').read_text()
        for a,b in [('/mnt/project/VLA-Drive-st-temporal-combination-20261017',str(w.RUNTIME)),('temporal_T3_indexed32_preparation_v1','temporal_T3_indexed32_preparation_v2'),('676e3e88d48d9444ab2cb928242d196381631b26','cf516259c8efc24503727f14b9e330ed32231293'),('f699cc48d009791349a65788c9a07c057364fd2edd1e3c085233b7151b91d72d',w.PLAN_SHA)]:
            self.assertEqual(old.count(a),1);old=old.replace(a,b)
        self.assertEqual(old,new)
        self.assertEqual((OLD/'launch_once.sh').read_bytes(),(HERE/'launch_once.sh').read_bytes())
    def test_actual_controller_plan_and_two_commands(self):
        plan=json.loads((HERE/'PLAN.json').read_text());c=w.controller()
        self.assertEqual(hashlib.sha256((HERE/'PLAN.json').read_bytes()).hexdigest(),c.PLAN_SHA)
        self.assertEqual(plan['runtime'],{'path':str(c.RUNTIME),'source_sha':c.COMMIT})
        self.assertEqual(c.HERE,HERE);self.assertEqual(c.ROOT,w.ROOT)
        commands=c.commands(plan);self.assertEqual(len(commands),2)
        self.assertEqual(commands[0][1][0],'prepare-profile-migration')
        self.assertEqual(commands[1][1][1],'T3')
        self.assertIn(str(w.ROOT/'SOURCE_MIGRATION.json'),commands[0][1]);self.assertIn(str(w.ROOT/'SOURCE_MIGRATION.json'),commands[1][1])
        self.assertEqual(subprocess.check_output(['git','-C',str(c.RUNTIME),'rev-parse','HEAD'],text=True).strip(),c.COMMIT)
        self.assertEqual(subprocess.check_output(['git','-C',str(c.RUNTIME),'status','--porcelain=v1'],text=True),'')
        for pin in plan['source_files'].values():self.assertEqual(hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest(),pin['sha256'])
    def test_budget_previous_failure_retained_and_unsigned(self):
        p=json.loads((HERE/'PLAN.json').read_text());a=json.loads((HERE/'ADMISSION_PROPOSAL.json').read_text());r=p['control_reservation']
        self.assertEqual(r['prior_reserved_bytes'],7880<<20);self.assertEqual(r['total_reserved_bytes'],7888<<20);self.assertEqual(r['remaining_bytes'],304<<20)
        self.assertEqual(r['retained_failed_v1_bytes'],8<<20);self.assertEqual(r['control_reservation_bytes'],8<<20)
        self.assertEqual(a['status'],'PROPOSAL_NOT_APPROVED');self.assertIsNone(a['valid_until_unix']);self.assertEqual(a['control_reservation'],r)
        self.assertEqual(p['limits'],dict(wall_seconds=300,cleanup_seconds=40,read_bytes=4<<30,rss_bytes=2<<30))
        self.assertFalse(w.ROOT.exists());self.assertFalse(p['automatic_retry']);self.assertFalse(p['training_started'])
if __name__=='__main__':unittest.main(verbosity=2)

"""Small owned CPU fixtures; no GPU/model probe or scientific receipt produced."""
import argparse
import copy
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.structured_world import run_p0_forward_probes as coord

BINDING='/mnt/project/structured-world-fgtr-execution-artifacts/20261011/p0_forward_v1_prospective/SOURCE_BINDING.json'


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def manifest(root):
    b=coord.load(BINDING)
    return {'schema':'p0_forward_coordinator_manifest_v1','source_binding':BINDING,'host':'owned-CPU-fixture-host',
        'python':sys.executable,'gpu_uuids':[f'GPU-owned-fixture-{r}'for r in range(8)],'artifact_root':str(root/'locks'),
        'cache':str(root/'owned_cache'),'dino_root':str(root/'owned_dino'),
        'origins':[{'case':case,'run':str(root/'owned_native'/case),'tag':f"milestone_{c['origin']['completed']:07d}"}for case,c in b['cases'].items()]}


def fixture(root):
    job=coord.build_plan(manifest(root),root/'out')['jobs'][0]
    folder=Path(job['folder']);folder.mkdir(parents=True)
    frozen=coord.load(BINDING);now=datetime.now(timezone.utc)
    prototype={'binding':job['source_binding'],'created_UTC':(now-timedelta(seconds=1)).isoformat(),'job':job,'fixture':'OWNED_SYNTHETIC_CPU'}
    coord.private_json(folder/'PROBE.json',prototype)
    names=sorted(coord.CORE|{'task_loss/original_FM','raw_loss/depth'})
    exact={n:True for n in ('q0','RNG','BN','BN_finite','query_and_counts','all_buffers','gradient_inventory','buffer_inventory')}
    row={'elements':4,'nonfinite_elements':0,'bitwise_equal':True,'advisory_passed':True,'max_absolute':0.,'difference_norm':0.,'relative_L2':0.}
    per_rank=[{'rank':r,'tokens':[f'OWNED_{r}_{s}'for s in range(4)],
        'roles':{role:{'started_UTC':now.isoformat()}for role in ('A','AA','B')},
        'comparisons':{pair:{'modules':{n:copy.deepcopy(row)for n in names},'exact':exact.copy()}for pair in coord.PAIRS}}for r in range(8)]
    modules={n:{p:copy.deepcopy(row)for p in coord.PAIRS}for n in names+[f'rank{r}/{n}'for r in range(8)for n in names]}
    strict={'passed':False,'ranks':8,'scene_count':32,'checkpoint':job['origin']['tag'],
        'training_identity':frozen['cases'][job['case']]['origin']['identity'],'optimized_mode':'engineering_fast',
        'attention_diagnostic':'native','precision_diagnostic':'native',
        'per_rank':[{'rank':r['rank'],'tokens':r['tokens'],'checks':[{'label':role,'fixture_only':True}for role in ('reference','reference_replay','optimized')]}for r in per_rank],
        'fixture':'OWNED_SYNTHETIC_CPU'}
    observed=Path(job['observed']);put(observed/'strict/COMPLETE.json',strict)
    report={'schema':'execution_p0_exact_forward_stats_v1','preset':coord.PRESET,'producer_sha256':coord.PRODUCER_SHA,
        'registration_sha256':coord.REGISTRATION_SHA,'producer_source_binding':job['source_binding'],
        'model_source':{'root':frozen['model']['root'],'git_sha':coord.MODEL_GIT},'original_acceptance_bindings':frozen['original_acceptance_bindings'],
        'checkpoint':{'run':job['origin']['run'],'tag':job['origin']['tag']},
        'original_strict':{**coord.ref(observed/'strict/COMPLETE.json'),'passed':False,'unchanged_original_json':True,'tool_sha256':coord.LEGACY_SHA},
        'ranks':8,'scene_count':32,'full_tensor_rank_coverage':True,'advisory_forward_passed':True,'per_rank':per_rank,
        'logical_forward_module_inventory':names,'forward_module_inventory':sorted(modules),'modules':modules,
        'forward_exact':{**{n:True for n in coord.CORE},**exact,'BN_float':True,'BN_counter':True},
        'forward_exact_by_pair':{p:exact.copy()for p in coord.PAIRS},
        'physical_q_final':{'full_tensor_rank_coverage':True,**{p:{'finite':True,'max_position_error_m':0.,'max_heading_error_rad':0.}for p in coord.PAIRS}},
        'full_parameter_gradient_statistics_captured':False,'passed':False,'formal_authorized':False,'trainer_consumable':False,'fixture':'OWNED_SYNTHETIC_CPU'}
    put(observed/'P0_FORWARD_COMPARISON.json',report)
    put(Path(job['run'])/'PROCESS_EXIT.json',{'returncode':1,'monitor_error':None,'wall_seconds':12.,'pid':999,'fixture':'OWNED_SYNTHETIC_CPU'})
    return job,report


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='owned-P0-coordinator-');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def test_actual_binding_plan_twelve_unique_repetitions_not_roles(self):
        p=coord.build_plan(manifest(self.root),self.root/'out')
        self.assertEqual(len(p['jobs']),12);self.assertEqual(len({j['observed']for j in p['jobs']}),12)
        self.assertEqual({j['repeat']for j in p['jobs']},{0,1,2})
        self.assertEqual(p['GPU_jobs_started'],0);self.assertFalse(p['passed'])
        for j in p['jobs']:
            self.assertIn('--max_restarts=0',j['command']);self.assertIn('--nproc_per_node=8',j['command'])
            self.assertEqual(j['source_binding']['sha256'],coord.BINDING_SHA)

    def test_completed_strict_failure_keeps_original_false_and_returncode(self):
        j,_=fixture(self.root);strict=Path(j['observed'])/'strict/COMPLETE.json';before=strict.read_bytes()
        obs=coord.classify(j);self.assertEqual(obs['status'],'completed_no_retry')
        self.assertEqual(obs['actual_returncode'],1);self.assertFalse(obs['original_strict_passed']);self.assertFalse(obs['passed'])
        self.assertFalse(obs['full_parameter_AA_B_gradients_captured']);self.assertEqual(strict.read_bytes(),before)
        self.assertEqual(obs['GPU_hours'],12*8/3600)

    def test_missing_forward_report_is_incomplete(self):
        j,_=fixture(self.root);(Path(j['observed'])/'P0_FORWARD_COMPARISON.json').unlink()
        with self.assertRaises(FileNotFoundError):coord.classify(j)

    def test_missing_rank_and_false_exact_rejected(self):
        j,r=fixture(self.root);r['per_rank'].pop();put(Path(j['observed'])/'P0_FORWARD_COMPARISON.json',r)
        with self.assertRaisesRegex(ValueError,'ranks'):coord.classify(j)
        j2,r2=fixture(self.root/'other');r2['forward_exact']['loss']=False;put(Path(j2['observed'])/'P0_FORWARD_COMPARISON.json',r2)
        with self.assertRaisesRegex(ValueError,'exact flags'):coord.classify(j2)

    def test_wrong_source_binding_or_model_rejected(self):
        j,r=fixture(self.root);r['producer_source_binding']=copy.deepcopy(r['producer_source_binding']);r['producer_source_binding']['sha256']='b'*64;put(Path(j['observed'])/'P0_FORWARD_COMPARISON.json',r)
        with self.assertRaisesRegex(ValueError,'source binding'):coord.classify(j)

    def test_nonfinite_forward_and_small_nonzero_not_reinterpreted(self):
        j,r=fixture(self.root);r['modules']['Bt']['A_B']['max_absolute']=1e-12;put(Path(j['observed'])/'P0_FORWARD_COMPARISON.json',r)
        with self.assertRaisesRegex(ValueError,'Nonfinite/nonexact'):coord.classify(j)
        r['modules']['Bt']['A_B']['max_absolute']=float('nan');put(Path(j['observed'])/'P0_FORWARD_COMPARISON.json',r)
        with self.assertRaisesRegex(ValueError,'Nonfinite JSON'):coord.classify(j)

    def test_new_outputs_only_and_private_prototype(self):
        p=self.root/'PROBE.json';coord.private_json(p,{'fixture_only':True})
        self.assertEqual(stat.S_IMODE(p.stat().st_mode),0o600)
        with self.assertRaises(FileExistsError):coord.private_json(p,{'changed':True})
        j,_=fixture(self.root/'other')
        with patch.object(coord.bench,'run_job')as run:
            with self.assertRaisesRegex(ValueError,'no overwritten'):coord.run_attempt({**j,'_locks':[]})
        run.assert_not_called()

    def test_preexisting_gpu_app_refuses_without_any_launch(self):
        m={'host':'owned-host','gpu_uuids':[f'GPU-{r}'for r in range(8)],'minimum_free_MiB':100}
        inventory='\n'.join(f'GPU-{r}, 0, 100000, 0'for r in range(8))
        with patch.object(coord.bench,'authorize'),patch.object(coord.bench.socket,'gethostname',return_value='owned-host'),\
                patch.object(coord.bench,'checked',side_effect=[inventory,'GPU-0, 321, genuine-job']),\
                patch.object(coord.bench,'run_job')as run:
            with self.assertRaisesRegex(ValueError,'compute app'):coord.bench.readiness(m)
        run.assert_not_called()

    def test_unexpected_exit_or_monitor_failure_not_completed(self):
        j,_=fixture(self.root);p=Path(j['run'])/'PROCESS_EXIT.json'
        v=coord.load(p);v['returncode']=2;put(p,v)
        with self.assertRaisesRegex(ValueError,'Unexpected worker'):coord.classify(j)
        v['returncode']=1;v['monitor_error']='owned synthetic error';put(p,v)
        with self.assertRaisesRegex(ValueError,'Unexpected worker'):coord.classify(j)

    def test_prospective_prototype_before_B_and_strict_hash_required(self):
        j,r=fixture(self.root);r['per_rank'][0]['roles']['B']['started_UTC']='2026-10-11T00:00:00+00:00';put(Path(j['observed'])/'P0_FORWARD_COMPARISON.json',r)
        with self.assertRaisesRegex(ValueError,'precede actual B'):coord.classify(j)
        j2,_=fixture(self.root/'other');p=Path(j2['observed'])/'strict/COMPLETE.json';v=coord.load(p);v['passed']=True;put(p,v)
        with self.assertRaisesRegex(ValueError,'exit/strict'):coord.classify(j2)

    def test_actual_owned_cpu_child_exit1_complete_no_retry(self):
        j,r=fixture(self.root);observed=Path(j['observed']);strict=coord.load(observed/'strict/COMPLETE.json')
        # Remove only our synthetic fixtures, so the real child owns publication.
        import shutil
        shutil.rmtree(observed);shutil.rmtree(j['run'])
        payload=self.root/'owned_fixture_payload.json';put(payload,{'strict':strict,'report':r})
        trainer=self.root/'owned_source/tools/structured_world/owned_cpu_child.py';trainer.parent.mkdir(parents=True)
        trainer.write_text('import json,os,sys,hashlib\nfrom pathlib import Path\nassert os.environ["CUDA_VISIBLE_DEVICES"]==""\np=json.loads(Path(sys.argv[1]).read_text());o=Path(sys.argv[2]);(o/"strict").mkdir(parents=True)\ns=o/"strict/COMPLETE.json";s.write_text(json.dumps(p["strict"]))\np["report"]["original_strict"]["sha256"]=hashlib.sha256(s.read_bytes()).hexdigest()\n(o/"P0_FORWARD_COMPARISON.json").write_text(json.dumps(p["report"]))\nprint("owned CPU fixture strict failure",flush=True)\nsys.exit(1)\n')
        j['trainer']=str(trainer);j['gpu_uuids']=[];j['command']=[sys.executable,'-B',str(trainer),str(payload),j['observed']]
        prototype=Path(j['folder'])/'PROBE.json';prototype.unlink()
        coord.private_json(prototype,{'binding':j['source_binding'],'created_UTC':(datetime.now(timezone.utc)-timedelta(seconds=5)).isoformat(),'job':j,'fixture':'OWNED_SYNTHETIC_CPU'})
        obs=coord.run_attempt({**j,'_locks':[]})
        self.assertEqual(obs['actual_returncode'],1);self.assertFalse(obs['original_strict_passed']);self.assertEqual(obs['GPU_hours'],0)
        self.assertEqual(coord.load(Path(j['run'])/'PROCESS_EXIT.json')['returncode'],1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU tests require empty CUDA_VISIBLE_DEVICES')
    started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    no_torch='torch'not in sys.modules
    if a.receipt:
        coord.private_json(a.receipt,{'schema':'p0_forward_coordinator_CPU_tests_v1','tests_run':result.testsRun,'passed':result.wasSuccessful()and no_torch,
            'wall_seconds':time.monotonic()-started,'torch_imported':not no_torch,'GPU_jobs_started':0,'owned_CPU_child_only':True,
            'synthetic_fixture_receipts_are_scientific_evidence':False,'files':[coord.ref(coord.__file__),coord.ref(__file__)]})
    sys.exit(0 if result.wasSuccessful()and no_torch else 1)


if __name__=='__main__':main()

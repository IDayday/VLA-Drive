"""CPU-only contract tests; all process/lock/payload fixtures are owned temporary files."""
import argparse
from contextlib import ExitStack
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world import run_execution_dev as dev

CANONICAL = '/mnt/project/VLA-Drive-structured-world-fgtr-io-formal-frozen-20261009'
UUIDS = ('GPU-11111111-1111-1111-1111-111111111111', 'GPU-22222222-2222-2222-2222-222222222222')


def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def semantic(value):
    return {'identity':hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(), **value}


def make_population(root, count=128):
    name = 'dev128' if count == 128 else 'dev1696'
    inputs = root/name
    rows = [{'token':f'{n:016x}', 'log':f'log{n%16}'}for n in range(count)]
    put(inputs/'index.json', rows)
    put(inputs/'identity.json', {'identity':name+'-fixture', 'dataset':'navsim', 'cameras':3})
    put(inputs/'COMPLETE.json', {'identity':name+'-fixture', 'contains_GT':False, 'scenes':count})
    put(inputs/'score_index.json', [{**r,'cache_path':'owned-fixture-not-opened'}for r in rows])
    return {'name':name, 'scenes':count, 'inputs':str(inputs), 'input_identity':name+'-fixture',
        'input_metadata_sha256':{n:dev.bench.digest(inputs/n)for n in ('index.json','identity.json','COMPLETE.json')},
        'score_index':dev.ref(inputs/'score_index.json')}


def manifest(root):
    return {'schema':'execution_dev_manifest_v1', 'candidate_mode':'engineering_fast', 'source_root':CANONICAL,
        'inference_python':'/qualified/qwen/python', 'scoring_python':'/qualified/navsim/python', 'devkit':'/pinned/navsim',
        'artifact_root':str(root/'leases'), 'score_host':'training-vlawm-zt3', 'CPU_workers':12, 'minimum_free_MiB':40000,
        'qualifications':{}, 'populations':[make_population(root,128), make_population(root,1696)],
        'checkpoints':[{'group':g,'role':r,'completed':n,'tag':f'milestone_{n:07d}', 'native_host':'training-vlawm-zt3',
            'mode':'engineering_fast'if r == 'B'else'io_preserving_v1','run':str(root/'native'/g/r)}
            for g,n in (('G1',25500),('G3',50500))for r in ('A','AA','B')]}


def make_native(job):
    row = job['checkpoint']; run = Path(row['run']); folder = run/'checkpoints'/row['tag']
    identity = {'identity':'native-'+row['role'], 'training_source_sha':dev.NATIVE_SHA, 'execution_mode':row['mode'],
        'dataset':'navsim', 'config':{'structured_world':{'group':row['group']+'_fixture'}}, 'world_size':8,
        'training_seed':42,'effective_global_batch':32,'FM_repeat':8}
    put(run/'identity.json', identity)
    names = ['mp_rank_00_model_states.pt']+[f'rng_rank{r}.pt'for r in range(8)]+[f'zero_pp_rank_{r}_mp_rank_00_optim_states.pt'for r in range(8)]
    folder.mkdir(parents=True)
    for name in names: (folder/name).write_bytes(b'owned CPU fixture, never torch-loaded')
    put(folder/'COMPLETE.json', {'identity':identity['identity'], 'ranks':8, 'tag':row['tag'], 'completed':row['completed'],
        'file_sha256':{n:dev.bench.digest(folder/n)for n in names}})
    put(run/'TRAINING_COMPLETE.json', {'identity':identity['identity'], 'updates':row['completed']})
    return dev.native_binding(job)


def make_predictions(job, path, native):
    path = Path(path)
    identity = semantic({'schema':'structured_world_FP32_predictions_v1', 'training_identity':native['identity'],
        'training_source_sha':dev.NATIVE_SHA,'checkpoint':job['checkpoint']['tag'],'updates':job['checkpoint']['completed'],
        'checkpoint_files_sha256':native['payload_sha256_from_COMPLETE'],'evaluation_source_sha':dev.SOURCE_SHA,
        'evaluation_source_files':{'tools/structured_world/infer_checkpoint.py':dev.INFER_SHA,
            'starVLA/dataloader/structured_world/current_inputs.py':dev.CURRENT_SHA},
        'input_identity':job['population_spec']['input_identity'],'sampling_seed':42,
        'sampling':'independent CPU scene-bound Gaussian, original 10-step FM, one residual and one final candidate',
        'precision':'FP32 native ZeRO master restore, strict=True; FP32 inference; TF32 disabled',
        'supervision_heads_removed':False,'scene_diagnostics':False})
    put(path/'identity.json',identity)
    put(path/'COMPLETE.json',{'identity':identity['identity'],'scenes':job['population_spec']['scenes']})
    (path/'predictions').mkdir()
    for row in dev.population(job['population_spec']):
        payload = path/'predictions'/(row['token']+'.npz')
        np.savez(payload,q0=np.zeros((8,3),np.float32),q_final=np.ones((8,3),np.float32))
        put(path/'records'/(row['token']+'.json'),{'token':row['token'],'identity':identity['identity'],'sha256':dev.bench.digest(payload)})
    return identity


def make_scores(job, path, prediction_identity):
    path = Path(path)
    identity = semantic({'prediction_identity':prediction_identity,'protocol':dev.PROTOCOL, 'adapter_sha256':dev.SCORE_SHA,
        'canonical_adapter_sha256':dev.CANONICAL_ADAPTER_SHA,'navsim_python_tree_sha256':dev.NAVSIM_TREE,
        'nuplan_python_tree_sha256':dev.NUPLAN_TREE,'runtime_versions':dev.SCORE_RUNTIME,
        'index_sha256':job['population_spec']['score_index']['sha256'], 'population':'debug'if job['population'] == 'dev128'else'dev',
        'failure_policy':'keep failed scenes at zero in all summaries and invalidate the benchmark'})
    put(path/'identity.json',identity)
    summaries = {}
    for variant in ('q0','q_final'):
        bank = path/variant; bank.mkdir(parents=True)
        rows = [{'token':r['token'],'log':r['log'],'status':'ok',**{m:.5 for m in dev.METRICS}}for r in dev.population(job['population_spec'])]
        with (bank/'scenes.csv').open('w') as stream:
            writer=csv.DictWriter(stream, ('token','log','status')+dev.METRICS); writer.writeheader(); writer.writerows(rows)
        summary = {'identity':identity['identity'],'valid':True,'failed':0,'scenes':len(rows),'metrics':{m:.5 for m in dev.METRICS}}
        put(bank/'summary.json',summary); summaries[variant]=summary
    put(path/'COMPLETE.json',{'identity':identity['identity'],**summaries})


class DevTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='owned-dev-contract-'); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.m = manifest(self.root)

    def job(self, **kwargs):
        with patch.object(dev,'source'):
            return dev.build_plan(self.m,self.root/'out',groups=('G1',),populations=('dev128',),**kwargs)['jobs'][0]

    def test_actual_frozen_entrypoint_and_process_helper_hashes(self):
        dev.source(CANONICAL)
        self.assertEqual(dev.bench.digest(Path(CANONICAL)/'starVLA/dataloader/structured_world/current_inputs.py'),dev.CURRENT_SHA)
        self.assertEqual(dev.bench.digest(Path(CANONICAL)/'tools/local_interaction_mask_v2/score_async.py'),dev.CANONICAL_ADAPTER_SHA)

    def test_dynamic_process_mask_and_both_populations(self):
        with patch.object(dev,'source'):
            p=dev.build_plan(self.m,self.root/'out',gpu_uuids=UUIDS)
        self.assertEqual(len(p['jobs']),24)
        self.assertFalse(p['execution_ready'])
        for j in p['jobs']:
            if j['phase']=='infer':
                self.assertIn('--nproc_per_node=2',j['command']); self.assertEqual(j['gpu_uuids'],list(UUIDS))
                self.assertEqual(j['command'][-2:],['--sampling-seed','42'])
            else:
                self.assertFalse(j['gpu_uuids']); self.assertEqual(j['command'][-3],'debug'if j['population']=='dev128'else'dev')
        self.assertIn('--nproc_per_node=1',self.job(gpu_uuids=UUIDS[:1])['command'])

    def test_default_plans_no_gpu_and_split_cpu_predictions_path(self):
        self.assertIsNone(self.job()['command'])
        j=self.job(phases=('score',),inference_results_root=self.root/'prior'/'results')
        self.assertEqual(j['command'][j['command'].index('--predictions')+1],str(self.root/'prior/results/G1_A_dev128/predictions'))
        self.assertFalse(j['gpu_uuids'])
        with self.assertRaises(ValueError): self.job(gpu_uuids=('GPU-0',))
        with self.assertRaises(ValueError): self.job(gpu_uuids=(UUIDS[0],UUIDS[0]))

    def test_population_hash_gt_token_and_log_guard(self):
        p=self.m['populations'][0]; self.assertEqual(len(dev.population(p)),128)
        path=Path(p['inputs'])/'COMPLETE.json'; value=dev.bench.read(path); value['contains_GT']=True; put(path,value)
        p['input_metadata_sha256']['COMPLETE.json']=dev.bench.digest(path)
        with self.assertRaisesRegex(ValueError,'no GT'): dev.population(p)
        value['contains_GT']=False; put(path,value)
        with self.assertRaisesRegex(ValueError,'metadata changed'):dev.population(p)

    def test_selected_uuid_genuine_exclusion_and_unselected_busy(self):
        inv=[{'uuid':u,'used_MiB':0,'free_MiB':50000,'utilization':0}for u in UUIDS]
        inv[1]['used_MiB']=20000;inv[1]['utilization']=100
        self.assertTrue(dev.selected_resources(UUIDS,UUIDS[:1],40000,inventory=inv,apps=[UUIDS[1]+', 123, genuine-job'])['passed'])
        with self.assertRaisesRegex(ValueError,'compute job'):dev.selected_resources(UUIDS,UUIDS[:1],40000,inventory=inv,apps=[UUIDS[0]+', 321, genuine-job'])
        with self.assertRaisesRegex(ValueError,'not idle'):dev.selected_resources(UUIDS,UUIDS,40000,inventory=inv,apps=[])

    def test_owned_locks_compatible_with_old_full_pool_and_alias_masks(self):
        with ExitStack() as first:
            locks=dev.gpu_locks(first,self.root/'locks',UUIDS,UUIDS[:1]);self.assertEqual(len(locks),2)
            key=hashlib.sha256(json.dumps(sorted(UUIDS)).encode()).hexdigest()
            with ExitStack() as other:
                with self.assertRaises(BlockingIOError):dev.bench.acquire(other,self.root/'locks/physical_gpu_locks'/(key+'.lock'))
            with ExitStack() as other:
                with self.assertRaises(BlockingIOError):dev.gpu_locks(other,self.root/'locks',tuple(reversed(UUIDS)),UUIDS[1:])
        with ExitStack() as after:dev.gpu_locks(after,self.root/'locks',UUIDS,UUIDS)

    def test_actual_owned_cpu_child_and_failed_process_cost(self):
        trainer=self.root/'source/tools/structured_world/owned_cpu_child.py'
        trainer.parent.mkdir(parents=True); trainer.write_text('import os,sys\nassert os.environ["CUDA_VISIBLE_DEVICES"]==""\nprint("owned CPU child",flush=True)\nsys.exit(2)\n')
        job={'key':'owned_cpu_fail','run':str(self.root/'child'),'trainer':str(trainer),'gpu_uuids':[],
            'command':[sys.executable,'-B',str(trainer)],'host':'owned-temporary-fixture'}
        with self.assertRaisesRegex(ValueError,'worker stage failed'):dev.bench.run_job(job,[])
        c=dev.process_cost(job);self.assertEqual(c['returncode'],2);self.assertGreater(c['actual_pid'],0)
        self.assertEqual(c['GPU_hours'],0);self.assertTrue(c['charged_despite_failure'])
        # Arithmetic on actual CPU wall, with hypothetical allocations explicitly synthetic.
        synthetic=dev.process_cost({**job,'gpu_uuids':list(UUIDS)})
        self.assertEqual(synthetic['GPU_hours'],c['wall_seconds']*2/3600)

    def test_native_completion_and_no_bulk_payload_hash_during_binding(self):
        j=self.job();native=make_native(j)
        self.assertFalse(native['payload_hashes_verified_here']);self.assertEqual(len(native['payload_sha256_from_COMPLETE']),17)
        path=Path(j['checkpoint']['run'])/'TRAINING_COMPLETE.json';value=dev.bench.read(path);value['updates']-=1;put(path,value)
        with self.assertRaisesRegex(ValueError,'not completed'):dev.native_binding(j)

    def test_actual_prediction_hash_precision_nan_and_checkpoint_binding(self):
        j=self.job();native=make_native(j);p=self.root/'prediction';make_predictions(j,p,native)
        self.assertEqual(dev.verify_predictions(j,p,native)['scenes'],128)
        wrong=copy.deepcopy(native);wrong['identity']='other'
        with self.assertRaisesRegex(ValueError,'native payload'):dev.verify_predictions(j,p,wrong)
        payload=next((p/'predictions').glob('*.npz'));np.savez(payload,q0=np.zeros((8,3),np.float32),q_final=np.full((8,3),np.nan,np.float32))
        with self.assertRaisesRegex(ValueError,'payload changed'):dev.verify_predictions(j,p,native)
        recordpath=p/'records'/(payload.stem+'.json');record=dev.bench.read(recordpath);record['sha256']=dev.bench.digest(payload);put(recordpath,record)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):dev.verify_predictions(j,p,native)

    def test_full1696_scores_valid_membership_and_failed_scene_rejected(self):
        with patch.object(dev,'source'):
            j=dev.build_plan(self.m,self.root/'out',groups=('G3',),populations=('dev1696',),phases=('score',))['jobs'][0]
        p=self.root/'scores';make_scores(j,p,'actual-pred-fixture')
        proof=dev.verify_scores(j,p,'actual-pred-fixture');self.assertEqual(set(proof['banks']),{'q0','q_final'})
        csvpath=p/'q_final/scenes.csv';text=csvpath.read_text();csvpath.write_text(text.replace(',ok,',',failed,',1))
        with self.assertRaisesRegex(ValueError,'failed/partial'):dev.verify_scores(j,p,'actual-pred-fixture')

    def test_scorer_summary_backend_and_prediction_pair_guards(self):
        j=self.job(phases=('score',));p=self.root/'scores';make_scores(j,p,'pred')
        with self.assertRaisesRegex(ValueError,'prediction binding'):dev.verify_scores(j,p,'other')
        s=p/'q0/summary.json';summary=dev.bench.read(s);summary['metrics']['score']=.51;put(s,summary)
        c=p/'COMPLETE.json';complete=dev.bench.read(c);complete['q0']=summary;put(c,complete)
        with self.assertRaisesRegex(ValueError,'CSV mean'):dev.verify_scores(j,p,'pred')
        ident=p/'identity.json';identity=dev.bench.read(ident);identity['runtime_versions']['numpy']='wrong';put(ident,identity)
        with self.assertRaisesRegex(ValueError,'backend/runtime'):dev.verify_scores(j,p,'pred')

    def test_explicit_io_reuse_reverifies_payload_and_rejects_B(self):
        j=self.job();native=make_native(j);p=self.root/'pred';ident=make_predictions(j,p,native)
        scores=self.root/'scores';make_scores(j,scores,ident['identity'])
        pred=dev.verify_predictions(j,p,native);proof=dev.verify_scores(j,scores,ident['identity'])
        npath=self.root/'NATIVE_BINDING.json';put(npath,native)
        rpath=self.root/'DEV_RECEIPT.json';put(rpath,{'schema':'execution_dev_receipt_v1','canonical_source_sha':dev.SOURCE_SHA,
            'protocol_complete':True,'role':'A','group':'G1','population':'dev128','native_mode':'io_preserving_v1',
            'artifact_refs':[dev.ref(npath),pred['identity'],pred['COMPLETE'],proof['identity'],proof['COMPLETE']],
            'native_binding':dev.ref(npath),'predictions':pred,'canonical_path':str(scores),'scores':proof})
        j['reuse']=dev.ref(rpath);self.assertEqual(dev.reused_reference(j)['role'],'A')
        with self.assertRaisesRegex(ValueError,'provenance'):dev.reused_reference({**j,'role':'B'})
        payload=next((p/'predictions').glob('*.npz'));payload.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'payload changed'):dev.reused_reference(j)

    def test_runtime_qualified_extras_including_flash_without_torch_import(self):
        packages={'flash_attn':'2.7.4.post1','torch':'2.5.1+cu124','numpy':'1.26.4'}
        with patch.object(dev.bench,'checked',return_value=json.dumps(packages))as checked:
            r=dev.runtime('/qualified/python',packages)
        self.assertFalse(r['CUDA_imported']);self.assertIn('flash_attn',checked.call_args.args[0][-1])
        self.assertEqual(checked.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
        with patch.object(dev.bench,'checked',return_value=json.dumps({'torch':'wrong'})):
            with self.assertRaisesRegex(ValueError,'runtime'):dev.runtime('/qualified/python',packages)

    def test_split_score_dispatch_uses_owned_cpu_children_and_never_accepts(self):
        out=self.root/'score_dispatch';out.mkdir()
        with patch.object(dev,'source'):
            plan=dev.build_plan(self.m,out,groups=('G1',),populations=('dev128',),phases=('score',),inference_results_root=self.root/'prior/results')
        trainer=self.root/'owned-source/tools/structured_world/owned_score_fixture.py'
        trainer.parent.mkdir(parents=True)
        trainer.write_text('import os\nassert os.environ["CUDA_VISIBLE_DEVICES"]==""\n')
        for j in plan['jobs']:
            native=make_native(j);prior=Path(j['prediction_result']);ident=make_predictions(j,prior/'predictions',native)
            put(prior/'NATIVE_BINDING.json',native)
            put(prior/'INFERENCE_RECEIPT.json',{'schema':'execution_dev_inference_v1',
                'native_binding':dev.ref(prior/'NATIVE_BINDING.json'),'predictions':dev.verify_predictions(j,prior/'predictions',native),
                'process_cost':{'fixture_only':True,'GPU_hours':0}})
            jobpath=self.root/(j['key']+'.json');put(jobpath,j)
            code=('import os,sys;assert os.environ["CUDA_VISIBLE_DEVICES"]=="";sys.path.insert(0,'+repr(str(ROOT))+');'
                'from tools.structured_world.test_run_execution_dev import make_scores;'
                'from tools.structured_world.run_execution_dev import bench;'
                'j=bench.read('+repr(str(jobpath))+');'
                'i=bench.read(j["prediction_result"]+"/predictions/identity.json");'
                'make_scores(j,j["result"]+"/canonical",i["identity"]);print("owned synthetic CPU scorer")')
            j['trainer']=str(trainer);j['command']=[sys.executable,'-B','-c',code]
        with patch.object(dev,'source'),patch.object(dev.bench,'authorize'),patch.object(dev.bench,'immutable_source'),\
                patch.object(dev.bench,'checked',return_value='owned-fixture-tracked'),\
                patch.object(dev,'runtime',return_value={'fixture_only':True}),\
                patch.object(dev.socket,'gethostname',return_value='training-vlawm-zt3'):
            dev.execute(self.m,plan,out)
        done=dev.bench.read(out/'COMPLETE.json');self.assertFalse(done['passed']);self.assertFalse(done['formal_authorized'])
        self.assertEqual(done['GPU_hours'],0);self.assertEqual(len(done['outputs']),3)
        for j in plan['jobs']:
            receipt=dev.bench.read(Path(j['result'])/'DEV_RECEIPT.json')
            self.assertTrue(receipt['protocol_complete']);self.assertFalse(receipt['full_dev_evidence'])
            self.assertFalse(receipt['passed']);self.assertGreater(receipt['scoring_process_cost']['actual_pid'],0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path);args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':raise ValueError('Run CPU tests with CUDA_VISIBLE_DEVICES empty')
    start=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(DevTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    require_no_torch='torch'not in sys.modules
    if args.receipt:
        put(args.receipt,{'schema':'execution_dev_CPU_contract_v1','passed':result.wasSuccessful()and require_no_torch,
            'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'wall_seconds':time.monotonic()-start,
            'torch_imported':not require_no_torch,'GPU_jobs_started':0,'scoring_jobs_started':0,
            'owned_CPU_child_processes_only':True,'fixture_outputs_are_scientific_evidence':False,
            'source_files':[dev.ref(dev.__file__),dev.ref(__file__)], 'canonical_source_git_SHA':dev.SOURCE_SHA})
    sys.exit(0 if result.wasSuccessful()and require_no_torch else 1)


if __name__=='__main__':main()

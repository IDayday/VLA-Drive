"""CPU controls with synthetic small metadata, not native numerical qualification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import execution_e2_provenance as p
import freeze_native_e2_phase as f


class Gate:
    @staticmethod
    def scalar(v, label):
        if type(v) not in (int, float) or not __import__('math').isfinite(v):
            raise ValueError(label)
        return float(v)

    @staticmethod
    def time_value(value):
        return __import__('datetime').datetime.fromisoformat(value)


class ProvenanceControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.reader = p.Reader()

    def tearDown(self):
        self.tmp.cleanup()

    def doc(self, name, obj):
        path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj)); return p.pins.ref(path)

    def process(self, **updates):
        argv = ['python', '-m', 'torch.distributed.run', '--nproc_per_node=8', '--max_restarts=0', '--', 'real.py']
        process = {'pid': 7, 'started_unix': 200., 'command': argv}
        exited = {'pid': 7, 'returncode': 0, 'monitor_error': None, 'wall_seconds': 12.}
        process.update(updates.pop('process', {})); exited.update(updates.pop('exit', {}))
        return self.doc('PROCESS.json', process), self.doc('EXIT.json', exited), argv

    def run_process(self, **changes):
        a,b,cmd = self.process(**changes)
        return p.process_facts(self.reader, a,b,cmd,'1970-01-01T00:02:00+00:00',Gate)

    def test_actual_process_cost(self):
        out = self.run_process(); self.assertEqual(out['GPU_hours'], 12*8/3600)
        self.assertEqual(out['actual_command'][3], '--nproc_per_node=8')

    def test_refuse_pid_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'PID'): self.run_process(exit={'pid':8})

    def test_failed_process_not_acceptance(self):
        with self.assertRaisesRegex(ValueError, 'failed'): self.run_process(exit={'returncode':255})

    def test_monitor_failure_not_acceptance(self):
        with self.assertRaisesRegex(ValueError, 'Infrastructure'): self.run_process(exit={'monitor_error':'brokenpipe'})

    def test_retrospective_process(self):
        with self.assertRaisesRegex(ValueError, 'predates'): self.run_process(process={'started_unix':100.})

    def test_command_is_not_expected_header(self):
        a,b,cmd = self.process(); wrong=cmd+['--secretly-new-mode']
        with self.assertRaisesRegex(ValueError, 'argv'): p.process_facts(self.reader,a,b,wrong,'1970-01-01T00:02:00+00:00',Gate)

    def test_changed_bytes_not_header_trusted(self):
        a=self.doc('observed.json',{'source':'actual'}); Path(a['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'hash'): self.reader.read(a)

    def test_mutation_during_review(self):
        a=self.doc('observed.json',{'source':'actual'}); self.reader.read(a); Path(a['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'changed'): self.reader.recheck()

    def test_duplicate_argv_refused(self):
        with self.assertRaisesRegex(ValueError,'duplicate'): p.argv_value(['--mode','a','--mode=b'],'--mode')

    def boundary(self, completed=25016, **updates):
        x={'completed':completed,'identity':'Bactual','passed':True,'FP32_model_and_buffers':'bitwise',
           'FP32_masters':'bitwise all ranks, actual parameters','Adam_moments_steps_and_LR':'bitwise all ranks',
           'all_task_RNG':'exact all ranks','timing':'before next forward/backward/update'}
        x.update(updates); return self.doc('boundary.json',x)

    def test_actual_split16_distinct_from_endpoint32(self):
        out=p.restore_boundary(self.reader,self.boundary(),{'identity':'Bactual'},25000)
        self.assertEqual((out['actual_restore_completed'],out['endpoint_completed']),(25016,25032))

    def test_endpoint32_cannot_masquerade_load16(self):
        with self.assertRaisesRegex(ValueError,'split16'): p.restore_boundary(self.reader,self.boundary(25032),{'identity':'Bactual'},25000)

    def test_boundary_wrong_identity(self):
        with self.assertRaisesRegex(ValueError,'candidate'): p.restore_boundary(self.reader,self.boundary(),{'identity':'other'},25000)

    def test_boundary_rng_not_claimed_exact(self):
        with self.assertRaisesRegex(ValueError,'RNG'): p.restore_boundary(self.reader,self.boundary(all_task_RNG='sampled'),{'identity':'Bactual'},25000)

    def test_runtime_observed_does_not_include_recipe_flags(self):
        obs={'host':'h','observed_utc':'2026-10-11T16:00:00+00:00','CUDA_initialized':False,
             'python':'3.10','cuda_runtime':'12.4','cudnn':90100,'nccl':[2,21,5],
             'packages':{'torch':'2.5','deepspeed':'.16','transformers':'4.57','flash_attn':'2.7','numpy':'1.26'},
             'gpus':[f'GPU-{r}, A800, 595' for r in range(8)]}
        env={k:obs[k] for k in ('python','cuda_runtime','cudnn','nccl')};env.update(obs['packages'],gpu_model='A800',cuda_driver='595',TF32=False)
        out=p.runtime_facts(self.reader,self.doc('runtime.json',obs),env)
        self.assertNotIn('TF32',out['observed_runtime']);self.assertIn('TF32',out['not_live_GPU_observations'])


class QualificationControls(ProvenanceControls):
    def fixture(self):
        run=self.root/'run'; source={'git_sha':'be5','files_sha256':{'a.py':'a'*64}}
        identity={'identity':'actual','training_source_sha':'be5','training_source_tree_sha256':p.metadata_hash(source['files_sha256']),
                  'execution_mode':'io_preserving_v1','engineering_continue_origin':{'completed':25000}}
        self.doc('run/identity.json',identity)
        names={'mp_rank_00_model_states.pt'}|{f'zero_pp_rank_{r}_mp_rank_00_optim_states.pt' for r in range(8)}|{f'rng_rank{r}.pt' for r in range(8)}
        folder=run/'checkpoints/milestone_0025001'; folder.mkdir(parents=True)
        manifest={}
        for name in names:
            (folder/name).write_bytes(b'synthetic small CPU payload');manifest[name]=p.pins.digest(folder/name)
        complete={'identity':'actual','completed':25001,'ranks':8,'status':'COMPLETE','file_sha256':manifest}
        self.doc('run/checkpoints/milestone_0025001/COMPLETE.json',complete)
        self.doc('run/RESUME_BOUNDARY_VERIFIED.json',{'passed':True,'completed':25000,'identity':'actual','all_task_RNG':'exact all ranks'})
        self.doc('run/PROCESS.json',{'pid':9,'command':['python','--nproc_per_node=8','--',str(p.pins.MODEL/'tools/structured_world/train_vla.py'),'--execution-mode','io_preserving_v1']})
        self.doc('run/PROCESS_EXIT.json',{'pid':9,'returncode':0,'wall_seconds':11.,'monitor_error':None})
        q=self.doc('q.json',{'passed':True,'host':'training-vlawm-zt3','physical_gpu_UUIDs':[f'GPU-{r}' for r in range(8)]})
        return q,run,source,[f'GPU-{r}' for r in range(8)]

    def test_qualification_metadata_not_initialization(self):
        out=f.qualification(self.reader,*self.fixture());self.assertTrue(out['qualified_checkpoint_is_NOT_500_initialization'])

    def test_incomplete_bank(self):
        args=self.fixture(); (args[1]/'checkpoints/milestone_0025001/rng_rank7.pt').unlink()
        with self.assertRaisesRegex(ValueError,'payload'): f.qualification(self.reader,*args)

    def test_17_arbitrary_files_are_not_native_inventory(self):
        args=self.fixture(); path=args[1]/'checkpoints/milestone_0025001/COMPLETE.json';x=json.loads(path.read_text());x['file_sha256']['rogue.pt']=x['file_sha256'].pop('rng_rank7.pt');path.write_text(json.dumps(x))
        with self.assertRaisesRegex(ValueError,'bank'): f.qualification(self.reader,*args)

    def test_wrong_source_in_actual_identity(self):
        args=self.fixture(); path=args[1]/'identity.json'; x=json.loads(path.read_text());x['training_source_sha']='other';path.write_text(json.dumps(x))
        with self.assertRaisesRegex(ValueError,'source'): f.qualification(self.reader,*args)

    def test_failed_qualification_cannot_be_enrolled(self):
        args=list(self.fixture()); args[0]=self.doc('q.json',{'passed':False,'host':'training-vlawm-zt3','physical_gpu_UUIDs':args[3]})
        with self.assertRaisesRegex(ValueError,'passed'): f.qualification(self.reader,*args)

    def test_qualified_update_cannot_replace_original_origin(self):
        args=self.fixture(); path=args[1]/'identity.json';x=json.loads(path.read_text());x['engineering_continue_origin']['completed']=25001;path.write_text(json.dumps(x))
        with self.assertRaisesRegex(ValueError,'original'): f.qualification(self.reader,*args)


if __name__ == '__main__':
    unittest.main()

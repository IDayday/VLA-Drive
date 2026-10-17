"""Small CPU fixtures: no real prep, GPU, database or checkpoint access."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
import itertools
from unittest import mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('full_profile_control',HERE/'controller.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def recipe():
    return {'execution':{'scope':'PROFILE','family':'temporal','world_size':2,'micro_batch':1,'global_batch':2,
        'max_updates':20000,'stop_after':64,'sample_limit':101592,'seed':42,
        'precision_policy':'S0_BF16_execution_FP32_master_Adam_ZeRO2_TF32_off'},
        'optimizer':{'warmup':1000},'runtime':{'max_seconds':1500,'save_every':64,'loader_workers':0,
        'deterministic':True,'gradient_checkpointing':False,'profile_mode':'none',
        'output_root':str(c.FACTORY/'configured/runs')}}


def population():
    names=['factory_result','config','teacher_manifest','provider_pca_scope','config_lineage','source_migration',
        'data_manifest','request_index_complete','base_binding']
    pins={name:{'path':'/fixture/'+name,'sha256':'a'*64} for name in names}
    inputs={'schema':'st_full_index_profile64_controller_inputs_v1','source_sha':c.SOURCE,
        'status':'ACTUAL_FACTORY_PINNED_NOT_DISPATCHED','pins':pins}
    cfg=recipe();cfg['data']={'teacher_provider':{'descriptor':pins['teacher_manifest'],
        'pca_channel_scope':pins['provider_pca_scope']}}
    outputs={n:pins[k] for k,n in [('config','configured/training_config.json'),('teacher_manifest','PROVIDER.json'),
        ('provider_pca_scope','configured/PROVIDER_CHANNEL_SCOPE.json'),
        ('config_lineage','configured/PROVIDER_CONFIG_LINEAGE.json'),('source_migration','SOURCE_MIGRATION.json')]}
    controls={'/fixture/factory_result':{'status':'PREPARATION_COMPLETE','training_started':False,
        'cli_result':{'status':'PROVIDER_PREPARATION_COMPLETE','outputs':outputs}},
        '/fixture/request_index_complete':{'status':'COMPLETE','scene_count':101592,'manifest_header':{'split':'train'}},
        '/fixture/teacher_manifest':{'request_index':{'complete':pins['request_index_complete']},'profile_admission':{
            'schema':'st_full_index_provider_profile_admission_v1','max_scenes':101592,
            'data_manifest':pins['data_manifest'],'consumption':{
            'schema':'st_full_index_profile_consumption_v1','initial_completed_updates':0,'initial_epoch':0,'initial_offset':0,
            'cumulative_optimizer_updates':64,'cumulative_scene_exposures':128,'world_size':2,'micro_batch':1,'global_batch':2,
            'gradient_accumulation_steps':1,'per_rank_scene_exposures':[64,64],
            'resume_budget':'same_run_cumulative_not_per_invocation','formal_global32_throughput_qualified':False}}}}
    return inputs,cfg,controls


class ControlTests(unittest.TestCase):
    def test_exact_commands_and_two_locks(self):
        self.assertEqual(c.command()[:4],['timeout','--signal=TERM','--kill-after=180s','1620s'])
        self.assertIn('--nproc_per_node=2',c.command());self.assertNotIn('--resume',c.command())
        self.assertEqual(c.command(True),c.command()+['--resume','--acknowledge-stop'])
        self.assertEqual(c.GPU,'0,1');self.assertEqual([str(x) for x in c.GPU_LOCKS],
            ['/tmp/foresight_gpu_0.lock','/tmp/foresight_gpu_1.lock'])

    def test_recipe_and_only_output_relocation(self):
        cfg=recipe();c.validate_recipe(cfg)
        moved=c.registered_config(cfg);self.assertEqual(moved['runtime']['output_root'],str(c.OUT/'runs'))
        moved['runtime']['output_root']=cfg['runtime']['output_root'];self.assertEqual(moved,cfg)
        changes=[('execution','scope','FORMAL'),('execution','world_size',1),('execution','global_batch',32),
            ('execution','sample_limit',256),('execution','stop_after',256),('execution','max_updates',64),
            ('optimizer','warmup',32),('runtime','save_every',1),('runtime','max_seconds',7200)]
        for section,key,val in changes:
            bad=deepcopy(cfg);bad[section][key]=val
            with self.subTest(key=key),self.assertRaises(RuntimeError):c.validate_recipe(bad)

    def test_current_template_cannot_be_accepted(self):
        inputs={'status':'WAITING_ACTUAL_FACTORY_AND_RESOURCE_PINS'}
        with self.assertRaises(RuntimeError):c.validate_population(inputs,recipe())

    def test_pinned_complete_population_without_full_manifest_read(self):
        inputs,cfg,controls=population()
        with mock.patch.object(c,'read',side_effect=lambda p:controls[str(p)]) as read:
            c.validate_population(inputs,cfg)
        self.assertEqual(read.call_count,3)
        self.assertNotIn(mock.call('/fixture/data_manifest'),read.call_args_list)

    def test_wrong_factory_index_source_budget(self):
        inputs,cfg,controls=population()
        changes=[('factory','FAILED'),('population',32),('split','dev'),('budget',128),('source','wrong'),('output','wrong')]
        for which,val in changes:
            ii,cc,dd=deepcopy(inputs),deepcopy(cfg),deepcopy(controls)
            if which=='factory':dd['/fixture/factory_result']['status']=val
            if which=='population':dd['/fixture/request_index_complete']['scene_count']=val
            if which=='split':dd['/fixture/request_index_complete']['manifest_header']['split']=val
            if which=='budget':dd['/fixture/teacher_manifest']['profile_admission']['consumption']['cumulative_optimizer_updates']=val
            if which=='source':ii['source_sha']=val
            if which=='output':dd['/fixture/factory_result']['cli_result']['outputs']['PROVIDER.json']={'path':'/wrong','sha256':'b'*64}
            with self.subTest(which=which),mock.patch.object(c,'read',side_effect=lambda p:dd[str(p)]),self.assertRaises(RuntimeError):
                c.validate_population(ii,cc)

    def test_storage_separate_new80_and_old2303(self):
        self.assertEqual(c.storage_requirement(0),(1024+2303+80)*(1<<30))
        for x in (-1,True,c.RESERVE+1):
            with self.assertRaises(RuntimeError):c.storage_requirement(x)

    def test_gpu_pair_busy_missing_or_duplicate(self):
        with mock.patch.object(c.subprocess,'check_output',return_value='GPU-a,0,0\nGPU-b,0,0\n'):
            self.assertEqual(len(c.gpu_record(True)),2)
        for text in ['GPU-a,0,0\n','GPU-a,0,0\nGPU-b,65,90\n','GPU-a,0,0\nGPU-a,0,0\n']:
            with mock.patch.object(c.subprocess,'check_output',return_value=text),self.assertRaises(RuntimeError):c.gpu_record(True)

    def admission(self):
        return {'schema':'st_full_index_profile64_resource_admission_v1','status':'ADMITTED','host':c.HOST,
            'gpu':c.GPU,'source_sha':c.SOURCE,'run_id':c.RUN_ID,'output_root':str(c.OUT),'controller_sha256':'pin',
            'inputs_sha256':'pin','checkpoint_reservation_bytes':c.RESERVE,'other_commitments_bytes':c.OTHER_COMMITMENTS,
            'free_floor_bytes':c.FREE_FLOOR,'max_invocation_seconds':1500,'outer_timeout_seconds':1620,
            'cleanup_seconds':180,'rss_limit_bytes':c.RSS_LIMIT,'rss_interval_seconds':c.RSS_INTERVAL,
            'storage_interval_seconds':c.STORAGE_INTERVAL,'valid_from_unix':0,'valid_until_unix':4000,'allowed_actions':['prepare','run'],
            'cpu_affinity':[4,5,6,7],'checkpoint_commitments':{'path':'/fixture/CP.json','sha256':'pin'}}

    def test_admission_and_fresh_checkpoint_commitment(self):
        a=self.admission();cp={'remaining_bytes':c.OTHER_COMMITMENTS,'free_floor_bytes':c.FREE_FLOOR,'valid_until_unix':4000}
        def reads(p):return cp if str(p)=='/fixture/CP.json' else a
        with mock.patch.object(c,'read',side_effect=reads),mock.patch.object(c,'digest',return_value='pin'),mock.patch.object(c.time,'time',return_value=1000):
            c.admission('/fixture/a','pin','prepare')
            for key,value in [('gpu','0'),('checkpoint_reservation_bytes',0),('other_commitments_bytes',2103<<30),
                ('valid_until_unix',2799),('valid_until_unix',float('inf')),('cpu_affinity',[4,4])]:
                old=a[key];a[key]=value
                with self.subTest(key=key),self.assertRaises(RuntimeError):c.admission('/fixture/a','pin','prepare')
                a[key]=old
            with self.assertRaises(RuntimeError):c.admission('/fixture/a','pin','resume')
            cp['valid_until_unix']=2799
            with self.assertRaises(RuntimeError):c.admission('/fixture/a','pin','prepare')

    def test_native_resume_cannot_expand_total_budget(self):
        good={'status':'PAUSED','source_sha':c.SOURCE,'completed':32,'exposure':64};c.validate_resume_state(good)
        for key,val in [('completed',64),('completed',0),('completed',True),('exposure',65),('source_sha','wrong'),('status','RUNNING')]:
            bad=dict(good);bad[key]=val
            with self.subTest(key=key),self.assertRaises(RuntimeError):c.validate_resume_state(bad)

    def test_cleanup_retains_limit_failure_and_180s_bound(self):
        class Owner:
            verified=3
            def __init__(self):self.calls=0;self.closed=False;self.finishes=0
            def discover(self):self.calls+=1;return [1] if self.calls==1 else []
            def leader_done(self):return False
            def finish(self,*,terminate,deadline):
                self.finishes+=1;self.terminate=terminate;self.deadline=deadline
                self.discover();return 0
            def close(self):self.closed=True
        with tempfile.TemporaryDirectory() as tmp:
            o=Owner();ticks=itertools.count(100)
            with mock.patch.object(c,'OUT',Path(tmp)),mock.patch.object(c,'proc_io',return_value={'rchar':10}),\
                mock.patch.object(c,'storage',return_value={}),mock.patch.object(c,'rank_telemetry',return_value={'0':{},'1':{}}),\
                mock.patch.object(c.time,'monotonic',side_effect=lambda:next(ticks)):
                code,result=c.monitor_training(None,o,{'signal':None},mock.Mock(side_effect=[c.RSS_LIMIT+1,c.RSS_LIMIT+99]),
                    Path(tmp)/'run','fixture',started=100,absolute_end=250)
            self.assertEqual(code,0);self.assertEqual(result['reason'],'RSS_LIMIT')
            self.assertEqual(result['rss_peak_bytes'],c.RSS_LIMIT+99)
            self.assertEqual(o.deadline,250);self.assertTrue(o.terminate);self.assertTrue(o.closed)
            self.assertEqual(o.finishes,1);self.assertEqual(result['live_owned_remaining'],0)

    def test_monitor_error_cleanup_once_and_unknown_scope_not_success(self):
        class Owner:
            verified=1
            def discover(self):return [1]
            def leader_done(self):return False
            def finish(self,**kwargs):self.finishes+=1;raise RuntimeError('Unwitnessed adopted child')
            def close(self):self.closed=True
        with tempfile.TemporaryDirectory() as tmp:
            o=Owner();o.finishes=0;o.closed=False
            with mock.patch.object(c,'OUT',Path(tmp)),mock.patch.object(c,'proc_io',return_value={'rchar':10}),\
                mock.patch.object(c,'storage',side_effect=RuntimeError('free floor')),\
                mock.patch.object(c,'rank_telemetry',return_value={}):
                code,result=c.monitor_training(None,o,{'signal':None},lambda pid:1,
                    Path(tmp)/'run','fixture',started=c.time.monotonic(),absolute_end=c.time.monotonic()+1800)
            self.assertIsNone(code);self.assertEqual(result['reason'],'MONITOR_ERROR')
            self.assertEqual(result['cleanup_error']['message'],'Unwitnessed adopted child')
            self.assertIsNone(result['live_owned_remaining']);self.assertEqual(o.finishes,1);self.assertTrue(o.closed)

    def test_normal_exit_race_rediscovery_and_true_orphan(self):
        class Owner:
            verified=2
            def __init__(self,second):self.snapshots=iter([[1],second,[]]);self.closed=False
            def discover(self):return next(self.snapshots,[])
            def leader_done(self):return True
            def finish(self,*,terminate,deadline):self.terminate=terminate;self.discover();return 0
            def close(self):self.closed=True
        for second,reason in [([],None),([2],'OWNED_DESCENDANTS_AFTER_LEADER_EXIT')]:
            with self.subTest(second=second),tempfile.TemporaryDirectory() as tmp:
                o=Owner(second)
                with mock.patch.object(c,'OUT',Path(tmp)),mock.patch.object(c,'proc_io',return_value={'rchar':10}),\
                    mock.patch.object(c,'storage',return_value={}),mock.patch.object(c,'rank_telemetry',return_value={}):
                    code,result=c.monitor_training(None,o,{'signal':None},lambda pid:1,
                        Path(tmp)/'run','race',started=c.time.monotonic(),absolute_end=c.time.monotonic()+1800)
                self.assertEqual(code,0);self.assertEqual(result['reason'],reason)
                self.assertEqual(o.terminate,reason is not None);self.assertTrue(o.closed)

    def test_partial_checkpoint_counts_against_two_slot_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);folder=run/'checkpoints';folder.mkdir()
            (folder/'step_000032').mkdir();self.assertEqual(c.require_resume_slot(run),['step_000032'])
            (folder/'step_000033_partial').mkdir()
            with self.assertRaises(RuntimeError):c.require_resume_slot(run)

    def test_readonly_default(self):
        with mock.patch.object(c,'common',return_value=({},{})),mock.patch.object(c,'prepare') as p,\
            mock.patch.object(c,'execute') as e,mock.patch.object(c,'gpu_record') as gpu,\
            mock.patch.object(c.sys,'argv',['controller.py']):c.main()
        p.assert_not_called();e.assert_not_called();gpu.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)

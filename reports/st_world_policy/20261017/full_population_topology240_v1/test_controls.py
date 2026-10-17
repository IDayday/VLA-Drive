"""Only small sources and synthetic control fixtures; no actual preparation/dispatch."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from adapter import load, bindings, raw, HERE

c=load('controller'); p=load('prepare'); b=bindings()
recipe=json.loads(raw(HERE/'PROFILE_RECIPE.json'))

class Controls(unittest.TestCase):
    def test_fixed_command_and_all_eight_locks(self):
        self.assertIn('--nproc_per_node=8',c.command())
        self.assertEqual(c.command()[3],'3720s')
        self.assertEqual(len(c.GPU_LOCKS),8)
        self.assertEqual(c.SOURCE,b['runtime']['sha'])
        with self.assertRaises(RuntimeError):c.command(True)
        with mock.patch.object(sys,'argv',['controller.py','--action','resume']),self.assertRaises(SystemExit):c.main()

    def test_fixed_recipe_and_only_output_relocation(self):
        c.validate_recipe(recipe)
        moved=c.registered_config(recipe);moved['runtime']['output_root']=recipe['runtime']['output_root']
        self.assertEqual(moved,recipe)
        for section,key,value in [('execution','world_size',2),('execution','micro_batch',1),
            ('execution','global_batch',2),('execution','scope','FORMAL'),('execution','stop_after',64),
            ('execution','max_updates',240),('runtime','profile_mode','none'),('runtime','save_every',240),
            ('optimizer','warmup',20)]:
            bad=deepcopy(recipe);bad[section][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):c.validate_recipe(bad)

    def test_original_monitor_and_factory_bodies_unchanged(self):
        transform=json.loads(raw(b['transform']['path'],b['transform']['sha256']))
        old=raw(transform['donor']['path'],transform['donor']['sha256']).decode();new=old
        for e in transform['edits']:new=new.replace(e['old'],e['new'])
        def body(source,name):return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name))
        self.assertEqual(body(old,'monitor_training'),body(new,'monitor_training'))
        self.assertEqual(body(old,'monitor_helpers'),body(new,'monitor_helpers'))
        self.assertEqual(p.LIMITS,dict(wall_seconds=1500,cleanup_seconds=40,read_bytes=16<<30,rss_bytes=4<<30))
        self.assertEqual(p.ROOT,Path(b['factory_root']))
        self.assertEqual(p.COMMIT,c.SOURCE)
        plan=json.loads(raw(HERE/'PLAN.json'))
        commands=p.commands(plan)
        self.assertEqual([x[0] for x in commands],['tools.st_world_policy.qualify_indexed_provider','tools.st_world_policy.teacher_provider'])
        self.assertIn(str(p.ROOT/'CONSUMPTION.json'),commands[1][1])
        self.assertEqual(plan['consumption']['per_rank_scene_exposures'],[960]*8)

    def test_valid_and_old_consumption_population(self):
        # Reuse the original synthetic fixture, never its old actual input files.
        path=HERE.parent/'full_population_profile_launch_v1/test_controller.py'
        spec=importlib.util.spec_from_file_location('old_fixtures',path);f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
        inputs,cfg,controls=f.population();inputs['source_sha']=c.SOURCE
        controls['/fixture/teacher_manifest']['profile_admission']['consumption']=json.loads(raw(HERE/'CONSUMPTION.json'))
        with mock.patch.object(c,'read',side_effect=lambda key:controls[str(key)]):c.validate_population(inputs,cfg)
        controls['/fixture/teacher_manifest']['profile_admission']['consumption']['cumulative_optimizer_updates']=64
        with mock.patch.object(c,'read',side_effect=lambda key:controls[str(key)]),self.assertRaises(RuntimeError):c.validate_population(inputs,cfg)

    def test_fixed_official_backend_overrides(self):
        # Execute the actual pure resolver source, excluding unrelated Torch imports.
        path=Path(b['runtime']['path'])/'tools/st_world_policy/official_runtime.py'
        tree=ast.parse(path.read_text());selected=[]
        for node in tree.body:
            if isinstance(node,ast.Assign) or isinstance(node,ast.FunctionDef) and node.name in ('_hash','_exact','_number','resolve_official_execution'):selected.append(node)
        ns=dict(deepcopy=deepcopy,hashlib=hashlib,json=json,math=math,Path=Path,__file__=str(path))
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)
        result=ns['resolve_official_execution'](recipe['runtime']['official_execution'],execution=recipe['execution'],optimizer=recipe['optimizer'],gradient_checkpointing=False)
        self.assertEqual(set(result['recipe_overrides']),{'max_updates','warmup'})
        self.assertEqual(result['accumulation_steps'],1)
        self.assertFalse(result['gpu_qualified'])

    def test_busy_or_missing_gpu_rejected(self):
        text='\n'.join('GPU-'+str(i)+',0,0' for i in range(8))
        with mock.patch.object(c.subprocess,'check_output',return_value=text):self.assertEqual(len(c.gpu_record(True)),8)
        for bad in (text.rsplit('\n',1)[0],text.replace('GPU-7,0,0','GPU-7,80,50'),text.replace('GPU-7','GPU-6')):
            with mock.patch.object(c.subprocess,'check_output',return_value=bad),self.assertRaises(RuntimeError):c.gpu_record(True)

    def test_cpu_ram_storage_are_explicit_not_approvals(self):
        self.assertEqual(c.RSS_LIMIT,256<<30);self.assertEqual(c.RESERVE,80<<30)
        self.assertEqual(c.storage_requirement(0),(1024+2383+80)<<30)
        a=json.loads(raw(HERE/'RESOURCE_ADMISSION_TEMPLATE.json'))
        self.assertEqual(a['status'],'TEMPLATE_NOT_ADMITTED')
        self.assertIsNone(a['checkpoint_commitments']['sha256'])
        self.assertEqual(a['allowed_actions'],['prepare','run'])

    def test_no_preflight_gpu_or_dispatch(self):
        with mock.patch.object(c,'common',return_value=({},{})),mock.patch.object(c,'prepare') as prep,mock.patch.object(c,'execute') as run,mock.patch.object(c,'gpu_record') as gpu,mock.patch.object(sys,'argv',['controller.py']):c.main()
        prep.assert_not_called();run.assert_not_called();gpu.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)

"""CPU module bridge from actual frozen 0c to the current reference math path.

No weights, model/guard edits, GPU use or full-model acceptance. Example:
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
    /tmp/structured-world-round1/envs/qwen/bin/python -B \
    tools/structured_world/test_canonical_execution_bridge.py --output NEW.json

Both sides execute their real source bytes. The old future has its own old
TokenLearner/Fuser/MLN/decoder imports; no current implementation is substituted.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
import types
import unittest

import numpy as np
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
OLD = Path('/mnt/project/VLA-Drive-structured-world-fgtr-io-formal-frozen-20261009')
OLD_SHA = '0c54f1c49e7ed2a76b22826b654d6c72625f5b6d'
ATOL, RTOL = 1e-6, 1e-4  # Existing CPU module tolerance; never configurable.
SOURCES, CASES = {}, []
GRID_ARGS = dict(xmin=-3., xmax=3., ymin=-2., ymax=2., dx=.5, dy=.5)
BODY = dict(length=1.2, width=.8, rear_axle_to_center=.17)


def source(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(data).hexdigest()
    return data


def package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module
    if '.' in name:
        parent, leaf = name.rsplit('.', 1)
        if parent in sys.modules:
            setattr(sys.modules[parent], leaf, module)
    return module


def direct(name, path):
    """Compile the exact file bytes, avoiding a possibly stale .pyc."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Required by dataclass and relative imports.
    parent, _, leaf = name.rpartition('.')
    if parent in sys.modules:
        setattr(sys.modules[parent], leaf, module)
    exec(compile(source(path), str(path), 'exec'), module.__dict__)
    return module


def modules(root, label):
    """Isolate relative SW imports and each future's absolute ported imports."""
    prefix = '_canonical_execution_bridge_' + label
    package(prefix, root)
    sw = root/'starVLA/model/modules/structured_world'
    package(prefix+'.sw', sw)
    result = {stem: direct(prefix+'.sw.'+stem, sw/(stem+'.py'))
              for stem in ('grid', 'queries', 'losses', 'semantics')}
    saved = {k: v for k, v in sys.modules.items()
             if k == 'third_party' or k.startswith('third_party.')}
    try:
        for name in list(saved):
            del sys.modules[name]
        package('third_party', root/'third_party')
        package('third_party.resworld', root/'third_party/resworld')
        port = root/'third_party/resworld/ported'
        package('third_party.resworld.ported', port)
        for stem in ('precision', 'res_layer', 'resnet', 'custom_module', 'mln', 'tokenlearner'):
            result[stem] = direct('third_party.resworld.ported.'+stem, port/(stem+'.py'))
        result['future'] = direct(prefix+'.sw.future', sw/'future.py')
    finally:
        for name in list(sys.modules):
            if name == 'third_party' or name.startswith('third_party.'):
                del sys.modules[name]
        sys.modules.update(saved)
    framework = root/'starVLA/model/framework/vla_structured_fgtr.py'
    tree = ast.parse(source(framework), filename=str(framework))
    registration = next(n for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'REGISTERED_GROUPS' for t in n.targets))
    result['groups'] = ast.literal_eval(registration.value)
    anchors = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                       and t.value.id == 'losses' and isinstance(t.slice, ast.Constant)
                       and t.slice.value == 'future_semantics' for t in n.targets)
               and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Mult)
               and isinstance(n.value.right, ast.Constant) and n.value.right.value == 0]
    if len(anchors) != 1:
        raise ValueError('Expected one actual G0 future readout zero-anchor')
    result['zero_anchor'] = compile(ast.Expression(anchors[0]), str(framework), 'eval')
    return result


def rng():
    return {'python': hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
            'numpy': hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
            'torch_CPU': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}


def tensor_bytes(value):
    return value.detach().contiguous().reshape(-1).view(torch.uint8)


def compare(a, b, path=''):
    """Evidence only: exact bytes AND unchanged preexisting CPU tolerance."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype):
            return [{'path': path, 'structure_equal': False, 'bitwise': False, 'within_CPU_tolerance': False}]
        exact = torch.equal(tensor_bytes(a), tensor_bytes(b))
        tolerance = exact
        maximum = 0.
        nonfinite = 0
        if a.is_floating_point():
            finite = torch.isfinite(a) & torch.isfinite(b)
            nonfinite = int((~finite).sum())
            if finite.any():
                maximum = float((a[finite]-b[finite]).abs().max())
            try:
                torch.testing.assert_close(a, b, atol=ATOL, rtol=RTOL, equal_nan=True)
                tolerance = True
            except AssertionError:
                tolerance = False
        return [{'path': path, 'elements': a.numel(), 'structure_equal': True,
                 'bitwise': exact, 'within_CPU_tolerance': tolerance,
                 'max_absolute_finite_difference': maximum, 'nonfinite_elements_either_side': nonfinite}]
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return [{'path': path, 'structure_equal': False, 'bitwise': False, 'within_CPU_tolerance': False}]
        return [row for k in a for row in compare(a[k], b[k], path+'/'+str(k))]
    if isinstance(a, (tuple, list)) and type(a) is type(b) and len(a) == len(b):
        return [row for i, (x, y) in enumerate(zip(a, b)) for row in compare(x, y, path+'/'+str(i))]
    same = type(a) is type(b) and a == b
    return [{'path': path, 'structure_equal': same, 'bitwise': same, 'within_CPU_tolerance': same}]


def clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone_tree(v) for v in value)
    return copy.deepcopy(value)


def floating_leaves(value):
    if torch.is_tensor(value):
        return [value] if value.is_floating_point() and value.requires_grad else []
    if isinstance(value, dict):
        return [x for v in value.values() for x in floating_leaves(v)]
    if isinstance(value, (tuple, list)):
        return [x for v in value for x in floating_leaves(v)]
    return []


def snapshot(fn, values, *, model=None, total_only=False, seed=427):
    inputs = [x.detach().clone().requires_grad_(x.requires_grad) for x in values]
    if model is not None:
        model.zero_grad(set_to_none=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    before = rng()
    out = fn(*inputs)
    selected = [out['total']] if total_only else floating_leaves(out)
    if selected:
        # Deterministic unequal element weights expose every differentiable path.
        objective = sum((v*torch.linspace(.3, 1.1, v.numel(), dtype=v.dtype).reshape(v.shape)).sum()/max(1, v.numel())
                        for v in selected)
        objective.backward()
    return {'output': clone_tree(out),
            'input_gradients': [None if x.grad is None else x.grad.detach().clone() for x in inputs],
            'parameter_gradients': {} if model is None else {n: None if p.grad is None else p.grad.detach().clone() for n, p in model.named_parameters()},
            'buffers': {} if model is None else {n: b.detach().clone() for n, b in model.named_buffers()},
            'RNG_before': before, 'RNG_after': rng()}


def control(test, name, old_fn, new_fn, values, *, old_model=None, new_model=None, total_only=False, metadata=None):
    a = snapshot(old_fn, values, model=old_model, total_only=total_only)
    aa = snapshot(old_fn, values, model=old_model, total_only=total_only)
    b = snapshot(new_fn, values, model=new_model, total_only=total_only)
    pairs = {'A_A': compare(a, aa), 'A_B': compare(a, b)}
    record = {'case': name, 'metadata': metadata or {}, 'pairs': {}}
    for label, rows in pairs.items():
        record['pairs'][label] = {'all_bitwise': all(r['bitwise'] for r in rows),
            'all_within_existing_CPU_tolerance': all(r['within_CPU_tolerance'] for r in rows),
            'checked_leaf_count': len(rows),
            'non_bitwise_leaves': [r for r in rows if not r['bitwise']],
            'max_absolute_finite_difference': max((r.get('max_absolute_finite_difference', 0.) for r in rows), default=0.),
            'leaf_details': rows}
    CASES.append(record)
    test.assertTrue(record['pairs']['A_A']['all_bitwise'], name+' A/A replay discrepancy')
    test.assertTrue(record['pairs']['A_B']['all_within_existing_CPU_tolerance'], name+' original discrepancy (retained in report)')
    return a, b


def labels(grid, steps, *, kind='regular', batch=3):
    g = torch.Generator().manual_seed(371)
    shape = (batch, steps+1, grid.height, grid.width)
    valid = torch.rand(shape, generator=g) > .17
    occupancy = torch.randint(2, shape, generator=g).byte()
    road = torch.randn(batch, grid.height, grid.width, generator=g)
    if kind == 'empty': valid.zero_()
    elif kind == 'tie': valid.fill_(True); occupancy.fill_(1); road.fill_(1.)
    elif kind == 'no_boundary': valid.fill_(True); occupancy.zero_(); road.fill_(10.)
    if kind == 'regular': valid[-1].zero_()
    occupancy[~valid] = 255
    return dict(road_distance=road, road_valid=valid[:, 0].clone(), occupancy=occupancy,
                occupancy_valid=valid, query_candidates=[object()]*batch)


class CanonicalModuleBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=OLD, text=True).strip() != OLD_SHA:
            raise ValueError('Actual frozen formal0c root changed')
        cls.old, cls.new = modules(OLD, 'old0c'), modules(ROOT, 'current')
        cfg = OLD/'configs/structured_world/navsim_G1_FULL_UNIFORM_seed42.yaml'
        options = yaml.safe_load(source(cfg))['structured_world']['class_statistics']
        cls.stats = {k: options[k] for k in ('current_positive_weight', 'future_positive_weight', 'arrival_positive_weight', 'release_positive_weight')}

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def grids(self):
        return self.old['grid'].GridSpec(**GRID_ARGS), self.new['grid'].GridSpec(**GRID_ARGS)

    def test_actual_source_package_isolation(self):
        self.assertIs(self.old['queries'].GridSpec, self.old['grid'].GridSpec)
        self.assertIs(self.old['losses'].read_time_field, self.old['queries'].read_time_field)
        self.assertIs(self.old['future'].GridSpec, self.old['grid'].GridSpec)
        for name in ('TokenLearner', 'TokenFuser'):
            old_class, new_class = getattr(self.old['future'], name), getattr(self.new['future'], name)
            self.assertIsNot(old_class, new_class)
            self.assertTrue(Path(old_class.forward.__code__.co_filename).is_relative_to(OLD))
            self.assertTrue(Path(new_class.forward.__code__.co_filename).is_relative_to(ROOT))
        self.assertIs(self.old['future'].CustomTransformerDecoder, self.old['custom_module'].CustomTransformerDecoder)
        self.assertIs(self.new['future'].CustomTransformerDecoder, self.new['custom_module'].CustomTransformerDecoder)

    def test_host_fast_math_mapping_from_actual_expression(self):
        path = ROOT/'starVLA/model/modules/structured_world/execution.py'
        tree = ast.parse(source(path))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'configure_execution')
        expr = next(n.value for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'fast' for t in n.targets))
        self.assertIs(eval(compile(ast.Expression(expr), str(path), 'eval'), {'mode':'engineering_host_fast'}), False)
        self.assertIs(eval(compile(ast.Expression(expr), str(path), 'eval'), {'mode':'engineering_fast'}), True)

    def test_grid_output_input_gradients_and_edges(self):
        old, new = self.grids()
        for dtype in (torch.float32, torch.float64):
            for mode in ('bilinear', 'nearest'):
                with self.subTest(dtype=str(dtype), mode=mode):
                    g = torch.Generator().manual_seed(96)
                    field = torch.randn(2, 2, old.height, old.width, generator=g, dtype=dtype, requires_grad=True)
                    xy = torch.tensor([[[-3.,-2.],[-2.75,-1.75],[0.,0.],[3.,2.],[4.,0.]]]*2, dtype=dtype, requires_grad=True)
                    control(self, f'grid_{dtype}_{mode}_boundaries', lambda f,x:old.sample(f,x,mode=mode), lambda f,x:new.sample(f,x,mode=mode), (field,xy))
            for key in ('centers', 'normalized_reference'):
                a = old.centers(dtype=dtype) if key=='centers' else old.normalized_reference(torch.tensor([[-3.,-2.],[3.,2.],[float('nan'),0.],[float('inf'),0.]],dtype=dtype))
                b = new.centers(dtype=dtype) if key=='centers' else new.normalized_reference(torch.tensor([[-3.,-2.],[3.,2.],[float('nan'),0.],[float('inf'),0.]],dtype=dtype))
                rows=compare(a,b);CASES.append({'case':f'grid_{key}_{dtype}','pairs':{'A_B':{'all_bitwise':all(r['bitwise'] for r in rows),'leaf_details':rows}}})
                self.assertTrue(all(r['bitwise'] for r in rows))

    def test_body_template_default_dtype_and_pose_gradients(self):
        original = torch.get_default_dtype()
        try:
            for default in (torch.float32, torch.float64):
                torch.set_default_dtype(default)
                for dtype in (torch.float32, torch.float64):
                    pose=torch.tensor([[.2,-.1,.27],[0.,0.,0.]],dtype=dtype,requires_grad=True)
                    control(self,f'body_default{default}_pose{dtype}',lambda p:self.old['queries'].body_points(p,BODY),lambda p:self.new['queries'].body_points(p,BODY,execution_fast=False),(pose,))
        finally: torch.set_default_dtype(original)

    def test_query_sampling_exact_detachment_strata_and_RNG(self):
        old,new=self.grids()
        for steps in (6,8):
            for mode in ('uniform','local'):
                for kind in ('regular','empty','no_boundary','outside'):
                    with self.subTest(steps=steps,mode=mode,kind=kind):
                        target=labels(old,steps,kind='regular' if kind=='outside' else kind)
                        proposal=torch.full((3,steps,3),1000. if kind=='outside' else 0.,requires_grad=True)
                        runs=[]
                        for side,grid in ((self.old,old),(self.new,new)):
                            gen=torch.Generator().manual_seed(42);before=rng()
                            out=side['queries'].sample_queries(proposal,target,mode=mode,generator=gen,count=16,grid=grid,
                                **({'execution_fast':False} if side is self.new else {}))
                            self.assertFalse(out['poses'].requires_grad)
                            self.assertEqual(before,rng())
                            runs.append((out,gen.get_state()))
                        rows=compare(runs[0],runs[1]);CASES.append({'case':f'query_{steps}_{mode}_{kind}','pairs':{'A_B':{'all_bitwise':all(r['bitwise'] for r in rows),'leaf_details':rows}}})
                        self.assertTrue(all(r['bitwise'] for r in rows))

    def test_time_reads_real_times_invalid_times_and_gradients(self):
        old,new=self.grids()
        for steps in (6,8):
            for mode in ('bilinear','nearest'):
                g=torch.Generator().manual_seed(18)
                field=torch.randn(2,steps,2,old.height,old.width,generator=g,requires_grad=True)
                xy=torch.tensor([[[0.,0.],[.2,-.3],[-3.,-2.],[3.,2.],[4.,0.]]]*2,requires_grad=True)
                times=torch.tensor([[0,steps-1,-1,steps,2]]*2)
                control(self,f'time_{steps}_{mode}_invalidtime',lambda f,x:self.old['queries'].read_time_field(f,x,times,old,mode=mode),lambda f,x:self.new['queries'].read_time_field(f,x,times,new,mode=mode,execution_fast=False),(field,xy))

    def test_body_relation_ties_boundaries_and_all_input_gradients(self):
        old,new=self.grids()
        for steps in (6,8):
            for kind in ('tie','regular','boundary'):
                g=torch.Generator().manual_seed(81)
                road=torch.ones(2,1,old.height,old.width) if kind=='tie' else torch.randn(2,1,old.height,old.width,generator=g)
                occ=torch.full((2,steps,1,old.height,old.width),.25) if kind=='tie' else torch.rand(2,steps,1,old.height,old.width,generator=g)
                pose=torch.tensor([[[0.,0.,.0],[.2,.1,.21],[0.,-.2,-.32]]]*2)
                if kind=='boundary':pose[0,:2,:2]=torch.tensor([[-3.,-2.],[3.,2.]])
                for value in (road,occ,pose):value.requires_grad_()
                times=torch.tensor([[0,steps-1,2]]*2)
                control(self,f'body_relation_{steps}_{kind}',lambda r,o,p:self.old['queries'].body_relations(r,o,p,times,[BODY]*2,grid=old),lambda r,o,p:self.new['queries'].body_relations(r,o,p,times,[BODY]*2,grid=new,execution_fast=False),(road,occ,pose))

    def test_actual_semantic_modules_G0_G1_G3_losses_and_parameter_gradients(self):
        old_grid,new_grid=self.grids()
        groups=('G0_NO_FUTURE_LABEL','G1_FULL_UNIFORM','G3_EVENT_LOCAL')
        for steps in (6,8):
            for group in groups:
                semantic,query_mode=self.old['groups'][group]
                self.assertEqual(self.new['groups'][group],(semantic,query_mode))
                for kind in ('regular','tie','empty','boundary'):
                    with self.subTest(steps=steps,group=group,kind=kind):
                        torch.manual_seed(89)
                        old_model=self.old['semantics'].StructuredSemantics(semantic,channels=4).train()
                        new_model=self.new['semantics'].StructuredSemantics(semantic,channels=4).train()
                        self.assertEqual(list(old_model.state_dict()),list(new_model.state_dict()))
                        new_model.load_state_dict(old_model.state_dict(),strict=True)
                        target=labels(old_grid,steps,kind=kind if kind!='boundary' else 'regular')
                        g=torch.Generator().manual_seed(74)
                        B0=torch.randn(3,4,old_grid.height,old_grid.width,generator=g,requires_grad=True)
                        Bt=torch.randn(3,steps,4,old_grid.height,old_grid.width,generator=g,requires_grad=True)
                        pose=torch.rand(3,8,3,generator=g)*.4-.2
                        if kind=='boundary':pose[0,:4,:2]=torch.tensor([[-3.,-2.],[3.,2.],[10.,0.],[0.,10.]])
                        pose.requires_grad_();times=torch.arange(8)[None].expand(3,-1)%steps
                        def call(side,grid,model):
                            def run(b0,bt,p):
                                prediction=model(b0,bt)
                                raw,counts=side['losses'].compute_structured_losses(prediction,target,{'poses':p,'time_index':times},[BODY]*3,
                                    semantic_mode=semantic,statistics=self.stats,global_scenes=5,grid=grid,
                                    **({'execution_fast':False} if side is self.new else {}))
                                zero=eval(side['zero_anchor'],{'semantic':prediction}) if semantic=='none' else 0.
                                return {'raw':raw,'counts':counts,'total':sum(raw.values())+zero}
                            return run
                        a,b=control(self,f'{group}_T{steps}_{kind}',call(self.old,old_grid,old_model),call(self.new,new_grid,new_model),(B0,Bt,pose),old_model=old_model,new_model=new_model,total_only=True,
                            metadata={'semantic_mode':semantic,'query_mode_from_actual_registry':query_mode,'query_sampling_tested_separately':True,'G0_zero_anchor_actual_AST':semantic=='none'})
                        if semantic=='none':
                            for name,grad in b['parameter_gradients'].items():
                                if name.startswith('future.'):
                                    self.assertIsNotNone(grad);self.assertEqual(int(torch.count_nonzero(grad)),0)

    def test_future_reference_outputs_all_input_parameter_gradients_and_RNG(self):
        old_grid,new_grid=self.grids()
        for steps in (6,8):
            for train in (False,True):
                for dropout_stress in (False,True):
                    with self.subTest(steps=steps,train=train,dropout_stress=dropout_stress):
                        torch.manual_seed(40)
                        a=self.old['future'].SingleFrameFuture(24,steps,channels=32,scene_tokens=4,grid=old_grid).train(train)
                        b=self.new['future'].SingleFrameFuture(24,steps,channels=32,scene_tokens=4,grid=new_grid).train(train)
                        with torch.no_grad():
                            for layer in (a.action_mln.gamma,a.action_mln.beta):layer.weight.normal_(std=.08);layer.bias.normal_(std=.08)
                        if dropout_stress:
                            for model in (a,b):
                                for m in model.modules():
                                    if isinstance(m,nn.Dropout):m.p=.2
                        b.execution_fast=False;b.future_time_vectorized=False
                        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
                        self.assertEqual(list(dict(a.named_parameters())),list(dict(b.named_parameters())))
                        b.load_state_dict(a.state_dict(),strict=True)
                        g=torch.Generator().manual_seed(51)
                        inputs=(torch.randn(2,32,old_grid.height,old_grid.width,generator=g,requires_grad=True),torch.randn(2,7,24,generator=g,requires_grad=True),torch.randn(2,steps,3,generator=g))
                        control(self,f'future_T{steps}_train{train}_dropout{dropout_stress}',a,b,inputs,old_model=a,new_model=b,
                            metadata={'execution_fast':False,'future_time_vectorized':False,'actual_old_ported_dependencies':True,'dropout_stress_not_registered_recipe':dropout_stress})

    def test_future_detached_proposal_and_horizon_guards(self):
        for side in (self.old,self.new):
            grid=side['grid'].GridSpec(**GRID_ARGS)
            with self.assertRaises(ValueError):side['future'].SingleFrameFuture(24,7,channels=32,scene_tokens=4,grid=grid)
            model=side['future'].SingleFrameFuture(24,6,channels=32,scene_tokens=4,grid=grid)
            with self.assertRaisesRegex(ValueError,'detached'):
                model(torch.zeros(1,32,grid.height,grid.width),torch.zeros(1,2,24),torch.zeros(1,6,3,requires_grad=True))

    def test_final_trajectory_loss_heading_zero_and_masks(self):
        for steps in (6,8):
            g=torch.Generator().manual_seed(76)
            encoded=torch.randn(3,steps,4,generator=g);encoded[0,:,2:]=0.;encoded.requires_grad_()
            target=torch.randn(3,steps,4,generator=g);valid=torch.rand(3,steps,generator=g)>.4;valid[-1]=False
            control(self,f'final_loss_T{steps}_zero_heading_masks',lambda e:self.old['losses'].final_trajectory_loss(e,target,valid,global_scenes=5),lambda e:self.new['losses'].final_trajectory_loss(e,target,valid,global_scenes=5),(encoded,))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('NEW report required; no original evidence overwrite')
    torch.set_num_threads(2)
    started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CanonicalModuleBridge))
    changed=[p for p,sha in SOURCES.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha]
    report={'schema':'canonical0c_current_reference_CPU_module_bridge_v1','observed_UTC':datetime.now(timezone.utc).isoformat(),
        'old_root':str(OLD),'old_git_SHA':OLD_SHA,'current_root':str(ROOT),
        'current_git_SHA':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'actual_source_SHA256':SOURCES,'source_changes_during_run':changed,
        'test_file_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'CPU_runtime':{'python':sys.version,'executable':sys.executable,'packages':{n:importlib.metadata.version(n) for n in ['torch','mmcv','mmengine','numpy']},'threads':torch.get_num_threads(),'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'CUDA_initialized':torch.cuda.is_initialized()},
        'fixed_existing_CPU_tolerance':{'atol':ATOL,'rtol':RTOL,'changed':False},
        'methods_run':result.testsRun,'errors':[str(e) for _,e in result.errors],'failures':[str(e) for _,e in result.failures],
        'comparison_cases':CASES,'case_count':len(CASES),'wall_seconds':time.monotonic()-started,
        'non_bitwise_cases':[r['case'] for r in CASES if not r['pairs']['A_B']['all_bitwise']],
        'module_bridge_tests_passed':result.wasSuccessful() and not changed and not torch.cuda.is_initialized(),
        'scope':'Direct actual0c grid/query/loss/semantics/future and old ported dependencies vs current reference operators with math reuse/timevec off. Actual G0 registry/zero-anchor AST exercised; tiny synthetic CPU tensors, shared random module state only.',
        'excluded':['Full VLA/Qwen/GeoBEV/DiT/FGTR forward and optimizer continuation','Actual host IO, prepared current batches and Qwen metadata adapter','Native full-model AA variability, CUDA kernels, multi-rank or world16 execution','Full dataset/calibration recipe equivalence (including nuScenes config differences)','Predicted driving trajectories and canonical evaluation'],
        'full_model_or_GPU_acceptance':False,'formal_authorized':False,'thresholds_or_models_modified':False,'weights_loaded':False,'GPU_jobs_started':0,'committed':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'report':str(args.output),'methods':result.testsRun,'cases':len(CASES),'module_bridge_tests_passed':report['module_bridge_tests_passed'],'non_bitwise_cases':report['non_bitwise_cases'],'full_model_or_GPU_acceptance':False}))
    raise SystemExit(0 if report['module_bridge_tests_passed'] else 1)


if __name__=='__main__':main()

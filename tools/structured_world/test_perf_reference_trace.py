"""Owned CPU fixtures only; no DeepSpeed/model weights, CUDA or GPU trace capture."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import io
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world import perf_reference_trace as trace


class Processor:
    def __init__(self):
        self.calls = []
        self.error = None

    def apply_chat_template(self, value, **kwargs):
        self.calls.append((value, kwargs))
        if self.error is not None:
            raise self.error
        return value


class Language(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1., 2., 3.]))
        self.register_buffer('seen', torch.tensor(0))

    def forward(self, value, *, generator=None):
        self.seen.add_(1)
        if generator is None:
            return value
        return (value*self.weight+torch.rand(value.shape, generator=generator)+
                torch.rand_like(value)+random.random()+float(np.random.random()))


class Qwen:
    def __init__(self, language):
        self.language_model = language

    def get_rope_index(self, value, **kwargs):
        return value


class Depth:
    def get_depth_loss(self, value, **kwargs):
        return value


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language = Language()
        self.qwen_vl_interface = SimpleNamespace(processor=Processor(),
            model=SimpleNamespace(model=Qwen(self.language)))
        self.geometry = SimpleNamespace(view=Depth())

    def _build_qwen_batch(self, value, instructions):
        value = self.qwen_vl_interface.processor.apply_chat_template(value, instructions=instructions)
        return self.qwen_vl_interface.model.model.get_rope_index(value, instructions=instructions)


class Log:
    def flush(self, value=None, **kwargs):
        return value


def fixtures():
    framework = ModuleType('starVLA.model.framework.vla_structured_fgtr')
    prepared = ModuleType('starVLA.dataloader.structured_world.prepared_current')
    calls = []
    def passthrough(value, **kwargs):
        calls.append((value, kwargs))
        return value
    for name in ('sample_queries', 'compute_structured_losses', 'final_trajectory_loss'):
        setattr(framework, name, passthrough)
    prepared.transfer_optimizer_batch = passthrough
    return Model(), framework, prepared, calls


def owned_attributes(model, framework, prepared):
    qwen = model.qwen_vl_interface.model.model
    return [(model.language, 'forward'), (model.qwen_vl_interface.processor, 'apply_chat_template'),
        (qwen, 'get_rope_index'), (model, '_build_qwen_batch'), (model.geometry.view, 'get_depth_loss'),
        (framework, 'sample_queries'), (framework, 'compute_structured_losses'),
        (framework, 'final_trajectory_loss'), (prepared, 'transfer_optimizer_batch'), (Log, 'flush')]


class ExtendedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ext = trace._ExtendedStages(self.temp.name, 3)
        self.addCleanup(self.ext.restore)
        self.model, self.framework, self.prepared, self.calls = fixtures()

    def attach(self):
        self.ext.attach(self.model, self.framework, self.prepared, Log)

    def test_all_ten_targets_return_same_object_and_original_called_once(self):
        value, token = object(), object()
        self.attach()
        qwen = self.model.qwen_vl_interface.model.model
        targets = [self.model.language.forward, self.model.qwen_vl_interface.processor.apply_chat_template,
            qwen.get_rope_index, self.framework.sample_queries, self.framework.compute_structured_losses,
            self.framework.final_trajectory_loss, self.model.geometry.view.get_depth_loss,
            self.prepared.transfer_optimizer_batch, Log().flush]
        for function in targets:
            self.assertIs(function(value), value)
        self.assertIs(self.model._build_qwen_batch(value, [token]), value)
        self.assertEqual(set(row['label'] for row in self.ext.rows), set(trace.EXTENDED_STAGE_SCOPES))
        self.assertEqual(len(self.ext.rows), 12)  # build nests processor + RoPE
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(len(self.model.qwen_vl_interface.processor.calls), 2)
        self.assertIs(self.model.qwen_vl_interface.processor.calls[-1][1]['instructions'][0], token)

    def test_positional_keywords_no_grad_and_exception_identity_are_unchanged(self):
        value, token, seen = object(), object(), []
        error = RuntimeError('private message must not be serialized')
        def native(arg, *, marker):
            seen.append((arg, marker, torch.is_grad_enabled()))
            raise error
        wrapped = self.ext._wrap('sample_queries_inclusive', native)
        with torch.no_grad():
            with self.assertRaises(RuntimeError) as failure:
                wrapped(value, marker=token)
        self.assertIs(failure.exception, error)
        self.assertEqual(seen, [(value, token, False)])
        self.assertTrue(self.ext.rows[0]['raised'])
        self.assertIsNone(self.ext.rows[0]['parent_call_index'])

    def test_cpu_worker_records_processor_rope_without_any_cuda_api(self):
        self.attach()
        self.ext.profile = SimpleNamespace(step_num=6)
        value = torch.tensor([1, 2])
        def worker():
            return self.model._build_qwen_batch(value, ['owned CPU fixture'])
        with ExitStack() as stack:
            forbidden = [stack.enter_context(patch.object(torch.cuda, name,
                side_effect=AssertionError('CPU worker must not call CUDA')))
                for name in ('Event', 'synchronize', 'current_stream', 'default_stream', 'init')]
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertIs(pool.submit(worker).result(), value)
            for mock in forbidden:
                mock.assert_not_called()
        self.assertEqual(len(self.ext.rows), 3)
        parents = [row for row in self.ext.rows if row['label']=='Qwen_build_batch_including_vision']
        children = [row for row in self.ext.rows if row not in parents]
        self.assertTrue(all(row['parent_call_index']==parents[0]['call_index'] for row in children))
        self.assertTrue(all(row['thread_id']!=__import__('threading').get_ident() for row in self.ext.rows))
        self.assertTrue(all(row['profiler_step_at_start']==6 and row['profiler_step_at_end']==6 for row in self.ext.rows))

    def test_restore_exact_instance_dictionary_and_idempotent_attach_restore(self):
        attributes = owned_attributes(self.model, self.framework, self.prepared)
        before = [(owner, name, name in vars(owner), vars(owner).get(name)) for owner,name in attributes]
        self.attach()
        self.attach()
        self.model._build_qwen_batch(object(), [])
        self.assertEqual(len(self.ext.rows), 3)
        self.ext.restore()
        self.ext.restore()
        for owner,name,owned,original in before:
            self.assertEqual(name in vars(owner), owned)
            self.assertIs(vars(owner).get(name), original)

    def test_install_restore_do_not_change_parameters_gradients_buffers_or_rng(self):
        self.model.language.weight.grad = torch.tensor([.25, .5, .75])
        parameters = [(name,id(value),value.data_ptr(),value.grad,id(value.grad),value.detach().clone(),value.grad.clone())
                      for name,value in self.model.named_parameters()]
        state = {name:value.clone() for name,value in self.model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        self.attach()
        self.ext.restore()
        for (name,value), expected in zip(self.model.named_parameters(), parameters):
            self.assertEqual((name,id(value),value.data_ptr(),id(value.grad)), (expected[0],expected[1],expected[2],expected[4]))
            self.assertIs(value.grad, expected[3])
            self.assertTrue(torch.equal(value,expected[5]))
            self.assertTrue(torch.equal(value.grad,expected[6]))
        self.assertTrue(all(torch.equal(value,state[name]) for name,value in self.model.state_dict().items()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(python_rng,random.getstate())
        self.assertTrue(np.array_equal(numpy_rng[1],np.random.get_state()[1]))
        self.assertEqual(numpy_rng[2:],np.random.get_state()[2:])

    def test_original_sync_and_collective_are_called_once_with_original_arguments(self):
        value = object()
        def native():
            torch.cuda.synchronize()
            torch.distributed.all_reduce(value)
            return value
        with patch.object(torch.cuda,'synchronize') as sync,patch.object(torch.distributed,'all_reduce') as reduce:
            self.assertIs(self.ext._wrap('execution_log_flush_inclusive',native)(),value)
        sync.assert_called_once_with()
        reduce.assert_called_once_with(value)
        self.assertEqual([row['label']for row in self.ext.rows],['execution_log_flush_inclusive'])

    def test_real_cpu_forward_backward_and_rng_match_native_exactly(self):
        torch.manual_seed(137); random.seed(138); np.random.seed(139)
        rng, prng, nrng = torch.get_rng_state(), random.getstate(), np.random.get_state()
        generator = torch.Generator().manual_seed(140)
        grng = generator.get_state()
        value = torch.tensor([.125, .25, .5], requires_grad=True)
        expected = self.model.language(value, generator=generator)
        expected.sum().backward()
        gradient, input_gradient = self.model.language.weight.grad.clone(), value.grad.clone()
        after = (torch.get_rng_state(),random.getstate(),np.random.get_state(),generator.get_state())
        self.model.language.weight.grad = None;value.grad = None;self.model.language.seen.zero_()
        torch.set_rng_state(rng);random.setstate(prng);np.random.set_state(nrng);generator.set_state(grng)
        self.attach()
        actual = self.model.language(value, generator=generator)
        actual.sum().backward()
        self.assertTrue(torch.equal(actual,expected))
        self.assertTrue(torch.equal(gradient,self.model.language.weight.grad))
        self.assertTrue(torch.equal(input_gradient,value.grad))
        self.assertEqual(self.model.language.seen.item(),1)
        self.assertTrue(torch.equal(after[0],torch.get_rng_state()))
        self.assertEqual(after[1],random.getstate())
        self.assertTrue(np.array_equal(after[2][1],np.random.get_state()[1]))
        self.assertEqual(after[2][2:],np.random.get_state()[2:])
        self.assertTrue(torch.equal(after[3],generator.get_state()))
        self.assertEqual(len(self.ext.rows),1)

    def test_cpu_sidecar_has_no_payload_or_exception_message_and_no_overwrite(self):
        secret = 'OWNED_PRIVATE_IMAGE_TOKEN_NEVER_SERIALIZE'
        self.attach()
        self.model.qwen_vl_interface.processor.error = RuntimeError(secret)
        with self.assertRaises(RuntimeError):
            self.model._build_qwen_batch({'private_image':secret}, [secret])
        self.ext.restore()
        self.ext.flush()
        path = Path(self.temp.name)/'rank3_extended_cpu.jsonl'
        saved = path.read_text()
        self.assertNotIn(secret,saved)
        self.assertNotIn('private_image',saved)
        rows = [json.loads(line) for line in saved.splitlines()]
        self.assertEqual(rows[0]['CUDA_events_added'],0)
        self.assertEqual(set(rows[0]['stage_scopes']),set(trace.EXTENDED_STAGE_SCOPES))
        self.assertEqual(set(rows[0]['installed_stage_labels']),set(trace.EXTENDED_STAGE_SCOPES))
        self.assertEqual(len(rows),3)
        self.assertTrue(all(row['raised'] and row['host_wall_seconds']>=0 for row in rows[1:]))
        self.assertEqual(path.stat().st_mode & 0o777,0o600)
        with self.assertRaises(FileExistsError): self.ext.flush()
        self.assertEqual(path.read_text(),saved)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())


class MainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model, self.framework, self.prepared, _ = fixtures()
        stage = ModuleType('tools.structured_world.stage_timing')
        class Timing:
            def __init__(timing, model, engine=None):
                self.native_init_calls += 1
                self.assertNotIn('forward',vars(model.language))
                timing.wrap = timing._wrap('Qwen_current_inclusive',model._build_qwen_batch)
            def _wrap(timing, name, original):
                return original
        stage.TrainingStageTiming = Timing
        self.Timing, self.native_init_calls = Timing, 0
        self.original_wrap, self.original_init = Timing._wrap, Timing.__init__
        ds = ModuleType('deepspeed')
        class Engine:
            def step(engine, *args, **kwargs):
                self.step_native.append((args,kwargs))
                return self.token
        self.Engine, self.original_step = Engine, Engine.step
        ds.DeepSpeedEngine = Engine
        log = ModuleType('tools.structured_world.execution_log');log.ExecutionLogWindow = Log
        self.modules = {stage.__name__:stage,ds.__name__:ds,self.framework.__name__:self.framework,
            self.prepared.__name__:self.prepared,log.__name__:log}
        self.token, self.step_native, self.profiles, self.schedule_args = object(), [], [], []
        self.trainer = Path(self.temp.name)/'tools'/'structured_world'/'train_vla.py'
        self.trace_dir = Path(self.temp.name)/'trace'

    def invoke(self, run, extended=False, entry_error=None, flush_error=None):
        owner = self
        class Profile:
            step_num = 0
            def __init__(self, **kwargs):
                self.kwargs = kwargs;owner.profiles.append(self)
            def __enter__(self):
                if entry_error is not None:raise entry_error
                return self
            def __exit__(self,*args):pass
            def step(self):self.step_num += 1
        def schedule(**kwargs):
            self.schedule_args.append(kwargs)
            return 'owned CPU schedule fixture'
        argv = ['perf_reference_trace.py','--trainer',str(self.trainer),'--trace-dir',str(self.trace_dir),
            '--scope','profile','--profile-stages','--debug-updates','32']
        if extended:argv.append('--extended-stages')
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules,self.modules))
            stack.enter_context(patch.object(sys,'path',list(sys.path)))
            stack.enter_context(patch.object(sys,'argv',argv))
            stack.enter_context(patch.dict(os.environ,{'RANK':'0'}))
            stack.enter_context(patch.object(torch.profiler,'profile',Profile))
            stack.enter_context(patch.object(torch.profiler,'schedule',schedule))
            stack.enter_context(patch.object(trace.runpy,'run_path',run))
            if flush_error is not None:
                stack.enter_context(patch.object(trace._ExtendedStages,'flush',side_effect=flush_error))
            trace.main()
            self.assertIs(sys.argv,argv)

    def assert_restored(self):
        self.assertIs(self.Timing._wrap,self.original_wrap)
        self.assertIs(self.Timing.__init__,self.original_init)
        self.assertIs(self.Engine.step,self.original_step)
        self.assertNotIn('forward',vars(self.model.language))
        self.assertNotIn('_build_qwen_batch',vars(self.model))

    def test_default_keeps_old_schedule_scopes_export_and_trainer_arguments(self):
        def run(path, run_name):
            self.assertEqual(path,str(self.trainer));self.assertEqual(run_name,'__main__')
            self.assertEqual(sys.argv,[str(self.trainer),'--scope','profile','--profile-stages','--debug-updates','32'])
            timing = self.Timing(self.model,self.Engine())
            self.assertIs(timing.wrap(self.token,[]),self.token)
            self.assertIs(self.Engine().step(self.token,marker=self.token),self.token)
        with patch.object(trace._ExtendedStages,'install',side_effect=AssertionError('Default must not install extended stages')):
            self.invoke(run)
        self.assert_restored()
        self.assertEqual(self.native_init_calls,1)
        self.assertEqual(len(self.step_native),1)
        self.assertEqual(self.profiles[0].step_num,1)
        self.assertEqual(self.schedule_args,[{'wait':5,'warmup':1,'active':3,'repeat':1}])
        settings = self.profiles[0].kwargs
        self.assertEqual(settings['activities'],[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
        self.assertTrue(settings['record_shapes']);self.assertFalse(settings['with_stack'])
        exported = []
        class Export:
            def export_chrome_trace(export,path):exported.append(path)
            def key_averages(export):return export
            def table(export,**kwargs):
                self.assertEqual(kwargs,{'sort_by':'self_cuda_time_total','row_limit':80})
                return 'owned CPU text fixture'
        settings['on_trace_ready'](Export())
        self.assertEqual(exported,[str(self.trace_dir/'rank0.json')])
        self.assertEqual((self.trace_dir/'rank0_operators.txt').read_text(),'owned CPU text fixture')
        self.assertFalse((self.trace_dir/'rank0_extended_cpu.jsonl').exists())

    def test_extended_installs_after_native_init_and_exception_restores_flushes(self):
        error = ValueError('owned private error never serialize')
        originals = [(owner,name,name in vars(owner),vars(owner).get(name))
                     for owner,name in owned_attributes(self.model,self.framework,self.prepared)]
        def run(*args,**kwargs):
            self.Timing(self.model,self.Engine())
            self.assertNotIn('--extended-stages',sys.argv)
            self.model.qwen_vl_interface.processor.error = error
            self.model._build_qwen_batch(self.token,[])
        with self.assertRaises(ValueError) as failure:self.invoke(run,extended=True)
        self.assertIs(failure.exception,error)
        self.assert_restored();self.assertEqual(self.native_init_calls,1)
        for owner,name,owned,original in originals:
            self.assertEqual(name in vars(owner),owned);self.assertIs(vars(owner).get(name),original)
        saved = (self.trace_dir/'rank0_extended_cpu.jsonl').read_text()
        self.assertNotIn(str(error),saved)
        self.assertEqual(len(saved.splitlines()),3)

    def test_extended_normal_exit_restores_and_writes_step_hints(self):
        def run(*args,**kwargs):
            self.Timing(self.model,self.Engine())
            self.assertIs(self.model.language(self.token),self.token)
            self.assertIs(self.Engine().step(),self.token)
            self.assertIs(Log().flush(self.token),self.token)
        self.invoke(run,extended=True)
        self.assert_restored()
        rows = [json.loads(line)for line in (self.trace_dir/'rank0_extended_cpu.jsonl').read_text().splitlines()]
        self.assertEqual(set(rows[0]['installed_stage_labels']),set(trace.EXTENDED_STAGE_SCOPES))
        self.assertEqual([row['profiler_step_at_start']for row in rows[1:]],[0,1])
        self.assertEqual([row['label']for row in rows[1:]],['Qwen_language_inclusive','execution_log_flush_inclusive'])
        self.assertEqual(len(self.step_native),1)

    def test_partial_attachment_failure_restores_every_already_installed_wrapper(self):
        del self.framework.final_trajectory_loss
        originals = [(owner,name,name in vars(owner),vars(owner).get(name))
                     for owner,name in owned_attributes(self.model,self.framework,self.prepared)]
        def run(*args,**kwargs):self.Timing(self.model,self.Engine())
        with self.assertRaises(AttributeError):self.invoke(run,extended=True)
        self.assert_restored()
        for owner,name,owned,original in originals:
            self.assertEqual(name in vars(owner),owned);self.assertIs(vars(owner).get(name),original)
        header = json.loads((self.trace_dir/'rank0_extended_cpu.jsonl').read_text().splitlines()[0])
        self.assertEqual(len(header['installed_stage_labels']),6)

    def test_profiler_entry_failure_also_restores_and_flushes_empty_rows(self):
        error = ValueError('owned profiler entry failure')
        with self.assertRaises(ValueError) as failure:self.invoke(lambda *a,**kw:None,extended=True,entry_error=error)
        self.assertIs(failure.exception,error);self.assert_restored()
        self.assertEqual(len((self.trace_dir/'rank0_extended_cpu.jsonl').read_text().splitlines()),1)

    def test_sidecar_io_failure_does_not_replace_native_exception(self):
        error = ValueError('owned native failure')
        def run(*args,**kwargs):raise error
        with patch.object(sys,'stderr',io.StringIO()) as stderr:
            with self.assertRaises(ValueError) as failure:
                self.invoke(run,extended=True,flush_error=OSError('owned disk failure'))
        self.assertIs(failure.exception,error);self.assert_restored()
        self.assertIn('original exception preserved',stderr.getvalue())

    def test_extended_requires_existing_profile_stages_flag(self):
        with patch.object(sys,'argv',['perf_reference_trace.py','--trainer',str(self.trainer),
              '--trace-dir',str(self.trace_dir),'--extended-stages']),patch.object(sys,'stderr',io.StringIO()):
            with self.assertRaises(SystemExit) as failure:trace.main()
        self.assertEqual(failure.exception.code,2)
        self.assertFalse(self.trace_dir.exists())

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main(verbosity=2)

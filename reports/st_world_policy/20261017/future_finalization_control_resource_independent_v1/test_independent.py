"""Independent control fixtures: no payload, actual ledger, GPU or dispatch."""
import os, time
ENTRY = time.monotonic()
def proc_io():
    return {k:int(v) for k,v in (line.split(':') for line in open('/proc/self/io'))}
BOOT = proc_io()
import ast, contextlib, copy, hashlib, importlib.util, io, json, math, re, socket, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
TREE=HERE.parents[3]
PROPOSAL=TREE/'reports/st_world_policy/20261017/future_finalization_actual_proposal_v1'
LEAD=Path('/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261017/future_finalization_actual_admission_v1')
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def doc(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=load('independent_finalizer_adapter',PROPOSAL/'run_finalize_once.py')
D=doc(PROPOSAL/'DRAFT_ADMISSION.json')
REG=doc(PROPOSAL/'DRAFT_STORAGE_REGISTRATION.json')
FREG=doc(PROPOSAL/'DRAFT_FINALIZATION_REGISTRATION.json')
corepath=a.LIBRARY/'full_future_metadata_preparation_v1/run_build_once.py'
tree=ast.parse(corepath.read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('terminal_read_accounting','charge_after_reap')]
arithmetic={};exec(compile(ast.Module(body=functions,type_ignores=[]),str(corepath),'exec'),arithmetic)

class IndependentControls(unittest.TestCase):
    def live(self):
        x=copy.deepcopy(D); now=time.time()
        x.update(status='APPROVED_FUTURE_FINALIZATION_ONLY', reviewer='/root/st_owned_descendants_review_v2',hostname=socket.gethostname(),issued_at_unix=now-1,latest_start_unix=now+60,valid_until_unix=now+3700)
        return x
    def test_01_fixed_pins_and_writer_dependencies(self):
        for pin in D['controller_sources']+REG['source_pins']+[D['registration'],D['finalization_registration'],D['policy'],D['old_policy'],D['checkpoint_receipt']]:
            self.assertTrue(Path(pin['path']).is_file()); self.assertFalse(Path(pin['path']).is_symlink())
            self.assertEqual(sha(pin['path']),pin['sha256'])
        self.assertEqual(REG['source_pins'],FREG['implementation'])
        self.assertEqual(len(D['controller_sources']),5);self.assertEqual(len(REG['source_pins']),9)
    def test_02_exact_runtime_command(self):
        cmd=doc(PROPOSAL/'DRAFT_COMMAND.json')
        self.assertEqual(cmd['finalizer_worker_argv'],REG['command'])
        self.assertEqual(cmd['cwd'],str(a.RUNTIME)); self.assertEqual(REG['cwd'],str(a.RUNTIME))
        self.assertEqual(REG['command'][-2:],['--output',str(a.ROOT)])
        self.assertFalse(cmd['direct_worker_invocation_authorized'])
    def test_03_old_claims_preserved_and_only_2GiB_added(self):
        old=doc(D['old_policy']['path']);new=doc(D['policy']['path'])
        for k in ('state_dir','cache_cap_bytes','free_reserve_bytes','ledger_reserve_bytes','allowed_roots','legacy_reservations','metadata_package','frozen_w_package'):
            self.assertEqual(new[k],old[k],k)
        self.assertEqual(len(old['claims']),11)
        before={r['root']:r for r in old['claims']};after={r['root']:r for r in new['claims']}
        for k,v in before.items():self.assertEqual(after[k],v)
        self.assertEqual(set(after)-set(before),{str(a.ROOT)})
        self.assertEqual(sum(after[str(a.ROOT)][k] for k in ('output_bytes','staging_bytes','metadata_bytes')),2<<30)
        self.assertEqual(new['metadata_package']['maximum_bytes'],8<<30)
    def test_04_separate_control_reserve_and_quiescence(self):
        reserve=doc(LEAD/'CONTROL_RESERVATION_ADOPTED.json');q=doc(LEAD/'QUIESCENCE.json')
        self.assertEqual(sha(LEAD/'CONTROL_RESERVATION_ADOPTED.json'),'3bd2880105bbddd5738e9bf2433f203fe561c13a2ac078157fd250eada5c9005')
        self.assertEqual(reserve['reserved_bytes'],a.CONTROL_CAP)
        self.assertEqual(reserve['combined_artifact_control_bytes'],a.LIMITS['max_disk_bytes'])
        self.assertEqual(reserve['new_planned_metadata_bytes'],reserve['prior_planned_metadata_bytes']+a.LIMITS['max_disk_bytes'])
        self.assertLess(reserve['new_planned_metadata_bytes'],reserve['metadata_envelope_bytes'])
        self.assertTrue(reserve['not_borrowed_from_existing_monitor_4MiB'])
        self.assertEqual(q['bindings']['new_policy_sha256'],D['policy']['sha256'])
        self.assertEqual(q['active_or_unknown_writers'],[]);self.assertTrue(q['queued_writers_disabled'])
    def test_05_exact_limits_and_source_failure_preserved(self):
        self.assertEqual(a.LIMITS,dict(wall_seconds=3600,cleanup_seconds=40,read_bytes=16<<30,rss_bytes=4<<30,cpu_affinity=[0,1],max_disk_bytes=(2<<30)+(1<<20),max_invocations=1))
        self.assertEqual(FREG['limits']['max_metadata_bytes'],2<<30)
        self.assertEqual(FREG['limits']['max_source_read_bytes'],16<<30)
        self.assertEqual(FREG['source_terminal'],D['source_terminal'])
        self.assertFalse(FREG['training_authorized']);self.assertFalse(FREG['teacher_extraction_authorized']);self.assertFalse(FREG['gpu_authorized'])
    def test_06_draft_and_author_rejected(self):
        with self.assertRaises(ValueError):a.validate(D,time.time())
        x=self.live();a.validate(x,time.time());x['reviewer']='/root/st_future_finalization_cost'
        with self.assertRaises(ValueError):a.validate(x,time.time())
    def test_07_nonfinite_expired_short_lease_rejected(self):
        for k in ('issued_at_unix','latest_start_unix','valid_until_unix'):
            for v in (float('nan'),float('inf'),-float('inf')):
                x=self.live();x[k]=v
                with self.assertRaises(ValueError):a.validate(x,time.time())
        x=self.live();x['valid_until_unix']=time.time()+3600
        with self.assertRaises(ValueError):a.validate(x,time.time())
    def test_08_resource_or_identity_change_rejected(self):
        for k,v in [('limits',{**a.LIMITS,'read_bytes':32<<30}),('runtime','/tmp/wrong'),('output_root','/tmp/wrong'),('control_reserved_bytes',0),('hostname','wrong')]:
            x=self.live();x[k]=v
            with self.assertRaises(ValueError):a.validate(x,time.time())
    def test_09_control_hash_and_symlink_refused(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            p=Path(td)/'control.json';p.write_text('{}'); pin={'path':str(p),'sha256':sha(p)}
            self.assertEqual(a.read(pin),{})
            p.write_text('{"changed":1}')
            with self.assertRaises(ValueError):a.read(pin)
            s=p.with_name('symlink');s.symlink_to(p)
            with self.assertRaises(ValueError):a.read({'path':str(s),'sha256':sha(p)})
    def test_10_real_lock_expression_matches_draft_allowlist(self):
        writer=ast.parse((a.RUNTIME/'tools/st_world_policy/finalize_future_metadata.py').read_text())
        run=next(n for n in writer.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        calls=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='metadata_lock']
        self.assertEqual(len(calls),1);self.assertIn('FINALIZE.lock',ast.unparse(calls[0]))
        self.assertTrue(any(re.fullmatch(p,'FINALIZE.lock') for p in REG['file_patterns']))
        for p in ('TEACHER_REQUESTS.json','RAW_LOG_PROVENANCE.json','SHARD_RECEIPTS.json','FINALIZATION_COMPLETE.json','CONTROL_FINALIZATION_RESULT.json'):
            self.assertTrue(any(re.fullmatch(x,p) for x in REG['file_patterns']),p)
        self.assertFalse(any(re.fullmatch(x,'unexpected.pt') for x in REG['file_patterns']))
    def test_11_control_no_overwrite_or_retry(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            a.control_once(td,'RESULT.json',{'status':'FAILED_NO_AUTO_RETRY'});p=Path(td)/'RESULT.json';first=p.read_bytes()
            with self.assertRaises(FileExistsError):a.control_once(td,'RESULT.json',{'status':'COMPLETE'})
            self.assertEqual(p.read_bytes(),first);self.assertTrue(p.with_name('RESULT.json.tmp').exists())
    def test_12_control_size_entries_and_alias_reject(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            with self.assertRaises(ValueError):a.control_once(td,'RESULT.json',{'data':'x'*(256<<10)})
            p=Path(td)/'BAD';p.write_text('')
            with self.assertRaises(ValueError):a.control_census(td)
            p.unlink();p=Path(td)/'RESULT.json';p.write_text('{}');os.link(p,Path(td)/'INVOCATION.json')
            with self.assertRaises(ValueError):a.control_census(td)
    def test_13_actual_control_lock_nonblocking(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            with a.control_lock(td):
                with self.assertRaises(BlockingIOError):
                    with a.control_lock(td):self.fail('overlapping lock')
    def test_14_original_ledger_acquisition_nonblocking(self):
        import fcntl
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            p=Path(td)/'fixture.lock'
            with p.open('wb') as f,p.open('rb') as g:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with a.nonblocking_original_ledger(SimpleNamespace(fcntl=fcntl)):
                    with self.assertRaises(BlockingIOError):fcntl.flock(g,fcntl.LOCK_EX)
    def test_15_absolute_deadline_and_existing_alarm(self):
        with self.assertRaises(TimeoutError):a.deadline_check(time.monotonic()-1)
        with self.assertRaises(TimeoutError):
            with a.whole_invocation_alarm(time.monotonic()+.02):time.sleep(.2)
        import signal
        signal.setitimer(signal.ITIMER_REAL,1)
        try:
            with self.assertRaises(ValueError):
                with a.whole_invocation_alarm(time.monotonic()+.1):self.fail('replaced foreign alarm')
        finally:signal.setitimer(signal.ITIMER_REAL,0)
    def test_16_reaped_child_charged_once(self):
        result=arithmetic['terminal_read_accounting']({'rchar':100},{'rchar':120},{'rchar':1125},1000,5000,final_child_available=True)
        self.assertEqual(result['charged_parent_child_rchar_bytes'],1025)
        self.assertEqual(arithmetic['charge_after_reap'](result,{'rchar':1130}),1030)
        self.assertEqual(result['parent_activity_during_wait_rchar_bytes'],5)
    def test_17_bad_terminal_counters_conservative(self):
        for final in (False,True):
            r=arithmetic['terminal_read_accounting']({'rchar':100},{'rchar':120},{'rchar':125},1000,5000,final_child_available=final)
            self.assertEqual(r['status'],'INCOMPLETE_CONSERVATIVE');self.assertEqual(r['charged_parent_child_rchar_bytes'],5000)
    def fixture(self,*,failure=None,artifact=True,extra=0,monitor_error=False):
        calls=[]
        def publish(m,b,g,p,v):
            calls.append(str(p))
            if failure=='ledger':raise BlockingIOError('synthetic ledger busy')
            p.write_text(json.dumps(v))
        def monitor(*args,**kw):
            self.assertEqual(kw['cpu_affinity'],(0,1));self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],'')
            self.assertLess(kw['budget']['wall_seconds'],30)
            if monitor_error:raise RuntimeError('synthetic monitor failure')
            return dict(returncode=0,failure=None,cleanup={'live_owned_remaining':0},read_accounting={'parent_post_reap':{'rchar':1125},'charged_parent_child_rchar_bytes':1025},child_accounting={'cli_result':{'status':'GUARDED_METADATA_RETURN','registration_sha256':D['registration']['sha256'],'artifact':{'schema':'st_future_metadata_finalization_complete_v1' if artifact else 'WRONG','status':'COMPLETE'}}})
        core=SimpleNamespace(publish_guarded=publish,monitor=monitor,charge_after_reap=arithmetic['charge_after_reap'])
        obs=SimpleNamespace(proc_io=lambda:{'rchar':1130+extra})
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            root=Path(td)/'metadata';root.mkdir();control=Path(td)/'controls';control.mkdir()
            with patch.object(a,'ROOT',root),patch.object(a,'CONTROL_ROOT',control),contextlib.redirect_stdout(io.StringIO()):
                x=self.live();args=SimpleNamespace(admission='synthetic.json',admission_sha256='0'*64)
                code=a.run_parent(args,x,{}, {},None,obs,None,SimpleNamespace(fcntl=a.fcntl),core,lambda:1<<50,work_deadline=time.monotonic()+30)
                r=doc(control/'RESULT.json')
            return code,r,calls
    def test_18_full_control_success_requires_complete_artifact(self):
        code,r,_=self.fixture();self.assertEqual(code,0);self.assertEqual(r['status'],'FINALIZATION_COMPLETE');self.assertFalse(r['scientific_result'])
        self.assertEqual(r['charged_rchar_bytes'],1030);self.assertEqual(r['source_terminal'],D['source_terminal'])
        code,r,_=self.fixture(artifact=False);self.assertEqual(code,2);self.assertEqual(r['status'],'FAILED_NO_AUTO_RETRY')
    def test_19_terminal_budget_and_monitor_exception_fail_closed(self):
        code,r,_=self.fixture(extra=16<<30);self.assertEqual(code,2)
        code,r,_=self.fixture(monitor_error=True);self.assertEqual(code,2);self.assertEqual(r['cleanup']['status'],'UNKNOWN_MONITOR_EXCEPTION')
        self.assertEqual(r['charged_rchar_bytes'],16<<30)
    def test_20_ledger_conflict_before_spawn_single_attempt(self):
        code,r,calls=self.fixture(failure='ledger');self.assertEqual(code,2);self.assertEqual(len(calls),1)
        self.assertEqual(r['cleanup']['status'],'NOT_STARTED');self.assertFalse(r['automatic_retry'])

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(IndependentControls)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    summary=dict(passed=result.testsRun-len(result.failures)-len(result.errors),failed=len(result.failures),errors=len(result.errors),tests=result.testsRun,body_seconds=time.monotonic()-ENTRY,own_rchar_delta=proc_io()['rchar']-BOOT['rchar'],cpu_affinity=sorted(os.sched_getaffinity(0)),real_payload_reads=0,actual_ledger_mutations=0,dispatches=0,author_tests_inherited=False)
    (HERE/'TEST_RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)

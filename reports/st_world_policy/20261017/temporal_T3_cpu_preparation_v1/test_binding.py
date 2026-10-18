"""CPU control fixtures only: no composition CLI, DB, teacher or ledger access."""
import copy,hashlib,importlib.util,io,json,os,resource,socket,sys,tempfile,time,types,unittest
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
from unittest.mock import Mock,patch
START=time.monotonic()
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('t3_binding',HERE/'prepare_once.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
PLAN=json.loads((HERE/'PLAN.json').read_text())
class Reads:
    def raw(self,path,sha):
        if str(path) in (str(HERE/'prepare_once.py'),str(HERE/'launch_once.sh')):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise ValueError('Bad controller SHA')
        return b''
    def document(self,pin):return {'status':'COMPLETE','scenes':32}

class Binding(unittest.TestCase):
    def setUp(self):
        self.c=w.controller();self.plan=copy.deepcopy(PLAN);now=time.time()
        self.a=dict(schema='st_T3_cpu_preparation_admission_v1',status='APPROVED_T3_METADATA_PREPARATION_ONLY',reviewer='independent_fixture_NOT_REAL_APPROVAL',hostname=socket.gethostname(),limits=w.LIMITS,cpu_affinity=[2,3],output_root=str(w.ROOT),max_output_bytes=w.OUTPUT_LIMIT,control_reservation=copy.deepcopy(PLAN['control_reservation']),gpu=False,automatic_retry=False,ledger_written=False,checkpoint_reservation_created=False,plan={'path':str(HERE/'PLAN.json'),'sha256':w.PLAN_SHA},issued_at_unix=now-1,latest_start_unix=now+30,valid_until_unix=now+400,controller_sources=[{'path':str(HERE/p),'sha256':hashlib.sha256((HERE/p).read_bytes()).hexdigest()} for p in ('prepare_once.py','launch_once.sh')])
    def validate(self):
        with patch.object(self.c.os,'sched_getaffinity',return_value={2,3}):self.c.validate(self.a,self.plan,None,Reads(),time.time())
    def test_bound_budget_and_original_monitor(self):
        self.validate()
        self.assertEqual(hashlib.sha256((HERE/'PLAN.json').read_bytes()).hexdigest(),w.PLAN_SHA)
        src=w.bind_source(w.original_bytes())
        self.assertIn('core.monitor(',src);self.assertIn('core.charge_after_reap(',src)
        self.assertIn('lock=s.acquire(s.LOCK,stack)',src);self.assertIn("if ROOT.exists():raise ValueError",src)
        self.assertEqual(self.c.LIMITS,dict(wall_seconds=300,cleanup_seconds=40,read_bytes=4<<30,rss_bytes=2<<30))
        self.assertEqual(self.c.OUTPUT_LIMIT,8<<20)
        self.assertEqual(self.c.COMMIT, '676e3e88d48d9444ab2cb928242d196381631b26')
        self.assertEqual(PLAN['runtime']['source_sha'], self.c.COMMIT)
        self.assertEqual(self.c.bootstrap().CORE_SHA,'4b2f0fccfd4bf3d655198f61179c88f6b2a51840731465c720b8addb6169b1a2')
    def test_only_T3_two_commands_and_same_order(self):
        calls=self.c.commands(self.plan)
        self.assertEqual([x[0] for x in calls],['tools.st_world_policy.qualify_indexed_provider','tools.st_world_policy.temporal_composition'])
        self.assertEqual(calls[0][1][0],'prepare-profile-migration')
        args=calls[1][1];self.assertEqual(args[args.index('--cell')+1],'T3')
        self.assertEqual(args[args.index('--history-complete')+1],PLAN['inputs']['history_complete']['path'])
        self.assertEqual(args[args.index('--source-migration')+1],calls[0][1][-1])
        self.assertEqual(args[args.index('--max-seconds')+1],'1800')
        self.assertNotIn('teacher_provider',[x[0] for x in calls]);self.assertNotIn('PROFILE_RECIPE.json',w.OUTPUTS)
    def test_pins_limits_role_and_lease_negative(self):
        for key,val in [('status','PROPOSAL'),('gpu',True),('automatic_retry',True),('output_root','/tmp/other'),('max_output_bytes',9<<20),('cpu_affinity',[0,1]),('valid_until_unix',time.time()+100)]:
            saved=self.a[key];self.a[key]=val
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate()
            self.a[key]=saved
        self.a['controller_sources'][0]['sha256']='0'*64
        with self.assertRaises(ValueError):self.validate()
    def test_incomplete_history_and_changed_original_rejected(self):
        with patch.object(Reads,'document',return_value={'status':'COMPLETE','scenes':31}),self.assertRaises(ValueError):self.validate()
        with self.assertRaises(ValueError):w.bind_source(w.original_bytes().replace(b'else 1540',b'else 1550'))
    def test_cli_JSON_SystemExit_and_failure_order(self):
        out=io.StringIO();err=io.StringIO();calls=[]
        def fake(module,run_name):
            calls.append(module);print(json.dumps({'original':module}));raise SystemExit(0)
        with redirect_stdout(out),redirect_stderr(err),patch.object(self.c.runpy,'run_module',side_effect=fake):
            for module,args in self.c.commands(self.plan):self.c.original_cli(module,args)
            print(json.dumps({'envelope':True}))
        self.assertEqual(json.loads(out.getvalue()),{'envelope':True});self.assertEqual(len(calls),2)
        calls=[]
        with patch.object(self.c.runpy,'run_module',side_effect=lambda *a,**kw:(_ for _ in ()).throw(SystemExit(2))),self.assertRaises(RuntimeError):
            for module,args in self.c.commands(self.plan):calls.append(module);self.c.original_cli(module,args)
        self.assertEqual(len(calls),1)
    def metadata(self):
        sys.path.insert(0,str(w.RUNTIME))
        from tools.st_world_policy import metadata_storage
        return metadata_storage
    def test_local_publication_counted_no_double_count(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);stage=root/'.configured.staging-test';stage.mkdir();(stage/'a.json').write_text('{}')
            before=w.composition_inventory(self.metadata(),root)['charged_bytes']
            link=root/'configured';link.symlink_to(stage.name)
            after=w.composition_inventory(self.metadata(),root)['charged_bytes'];s=link.lstat()
            self.assertGreaterEqual(after,before+max(s.st_size,s.st_blocks*512))
            self.assertEqual(w.output_path(link/'a.json'),stage/'a.json')
    def test_foreign_broken_hardlink_and_nested_link_rejected(self):
        for target in ('../outside','/tmp/outside','.configured.staging-missing'):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);(root/'configured').symlink_to(target)
                with self.subTest(target=target),self.assertRaises(ValueError):w.composition_inventory(self.metadata(),root)
        for kind in ('foreign_link','shared_file'):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);p=root/'real';p.write_text('{}')
                if kind=='foreign_link':(root/'extra').symlink_to('real')
                else:os.link(p,root/'extra')
                with self.subTest(kind=kind),self.assertRaises(Exception):w.composition_inventory(self.metadata(),root)
    def test_child_mount_escape_rejected_before_delegated_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);child=root/'mounted';child.mkdir()
            original=Path.lstat
            def lstat(path):
                result=original(path)
                if path==child:
                    fields=list(result);fields[2]+=1;return os.stat_result(fields)
                return result
            metadata=types.SimpleNamespace(tree_inventory=Mock())
            with patch.object(Path,'lstat',lstat),self.assertRaisesRegex(ValueError,'mount escape'):
                w.composition_inventory(metadata,root)
            metadata.tree_inventory.assert_not_called()
    def test_no_admission_or_real_dispatch_created(self):
        self.assertFalse(PLAN['training_started']);self.assertFalse(PLAN['T4_automatic'])
        launch=(HERE/'launch_once.sh').read_text()
        self.assertIn('timeout --signal=TERM --kill-after=40s 300s',launch)
        self.assertIn('taskset -c 2,3',launch)
        self.assertNotIn('torch',sys.modules)

if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Binding))
    print('RESOURCE_JSON='+json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),wall_seconds=time.monotonic()-START,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,actual_preparation=False,database_or_teacher_read=False,ledger_written=False,independent_approval=False)))
    raise SystemExit(0 if result.wasSuccessful() else 1)

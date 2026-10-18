"""Real CPU migration CLI preflight plus precise protected-source negative tests."""
import ast,copy,contextlib,importlib.util,io,json,os,resource,runpy,sys,time,unittest
from pathlib import Path
START=time.monotonic()
# Shared-library virtual mappings are not resident memory. Preserve the real
# 2-GiB resident-memory reporting instead of blocking torch's library mapping.
HERE=Path(__file__).resolve().parent
BASE=Path('/mnt/project/st-world-policy-20261016')
PARENT=Path('/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017')
CURRENT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(CURRENT))
BEFORE=int(next(x.split(':')[1] for x in Path('/proc/self/io').read_text().splitlines() if x.startswith('rchar:')))
BLOCKED=[]
CLI_COMPLETED=False
def no_payload(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=os.fsdecode(args[0])
        if Path(path).suffix.lower() in {'.sqlite','.db','.safetensors','.pt','.pth','.pkl','.jpg','.jpeg','.png'}:
            BLOCKED.append(path);raise AssertionError('Migration preflight must not open payload/DB: '+path)
sys.addaudithook(no_payload)
class Migration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global CLI_COMPLETED
        original_path=Path('/mnt/project/VLA-Drive-st-temporal-combination-20261017/tools/st_world_policy/qualify_indexed_provider.py')
        spec=importlib.util.spec_from_file_location('original_migration_failure',original_path)
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        old_output=HERE/'ORIGINAL_REFUSAL_MUST_NOT_EXIST.json'
        try:
            original.prepare_control_migration(str(BASE/'smoke/world_indexed_provider_native2_v1/PROVIDER.json'),str(PARENT),str(old_output))
        except ValueError as exc:
            if str(exc)!='PROFILE migration changed protected mathematical/reader/PCA/sampler source':raise
            (HERE/'ACTUAL_ORIGINAL_REFUSAL.json').write_text(json.dumps(dict(
                exception=type(exc).__name__,message=str(exc),actual_function_called=True,
                source=str(original_path),output_created=old_output.exists()))+'\n')
        else:raise AssertionError('Original actual migration must refuse the protected-source mismatch')
        path=HERE/'SOURCE_MIGRATION_PREFLIGHT.json'
        args=['qualify_indexed_provider','prepare-profile-migration','--parent-descriptor',str(BASE/'smoke/world_indexed_provider_native2_v1/PROVIDER.json'),'--parent-runtime',str(PARENT),'--output',str(path)]
        before=sys.argv;out=io.StringIO();err=io.StringIO()
        try:
            sys.argv=args
            with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
                try:runpy.run_module('tools.st_world_policy.qualify_indexed_provider',run_name='__main__')
                except SystemExit as exc:
                    if exc.code not in (None,0):raise
        finally:sys.argv=before
        (HERE/'ACTUAL_CLI.log').write_text(out.getvalue()+err.getvalue())
        cls.migration=json.loads(path.read_text())
        from tools.st_world_policy import qualify_indexed_provider as q
        cls.q=q;cls.plan=q.tp.pinned(cls.migration['parent_plan'])
        CLI_COMPLETED=True
    def tearDown(self):
        now=int(next(x.split(':')[1] for x in Path('/proc/self/io').read_text().splitlines() if x.startswith('rchar:')))
        self.assertLess(now-BEFORE,512<<20)
    def test_actual_cli_and_real_validator(self):
        parent=self.q.validate_control_migration(self.plan,self.migration)
        self.assertEqual(parent['identity'],'90a908a48eba2e3d94e1d07700178dd08c61cb4aa0943459bbf572bd1b17ff8b')
        self.assertEqual(self.migration['mathematics_compatibility']['relative'],self.q.PACKED_HISTORY_PATH)
        self.assertFalse(self.migration['formal_approved']);self.assertFalse(BLOCKED)
    def test_all_other_fourteen_protected_sources_equal(self):
        rows=self.migration['mathematics'];changed=[r for r in rows if r['old']['sha256']!=r['new']['sha256']]
        self.assertEqual(len(rows),15);self.assertEqual([r['relative'] for r in changed],[self.q.PACKED_HISTORY_PATH])
    def test_explicit_missing_or_tampered_compatibility_rejected(self):
        for fault in ('missing','tampered'):
            bad=copy.deepcopy(self.migration)
            if fault=='missing':del bad['mathematics_compatibility']
            else:bad['mathematics_compatibility']['new_sha256']='0'*64
            bad['identity']=self.q.identity_hash({k:v for k,v in bad.items() if k!='identity'})
            with self.subTest(fault=fault),self.assertRaisesRegex(ValueError,'exact reviewed'):
                self.q.validate_control_migration(self.plan,bad)
    def test_other_protected_change_and_third_reader_hash_rejected(self):
        for name in ('other','third','reversed'):
            rows=copy.deepcopy(self.migration['mathematics'])
            target=next(r for r in rows if r['relative']==self.q.PACKED_HISTORY_PATH)
            if name=='other':rows[0]['new']['sha256']='0'*64
            elif name=='third':target['new']['sha256']='1'*64
            else:target['old'],target['new']=target['new'],target['old']
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'protected'):
                self.q.packed_history_compatibility(rows)
    def test_legacy_equal_sources_need_no_new_record(self):
        rows=copy.deepcopy(self.migration['mathematics'])
        for row in rows:row['new']['sha256']=row['old']['sha256']
        self.assertIsNone(self.q.packed_history_compatibility(rows))
    def test_original_semantic_functions_and_reader_init_unchanged(self):
        def nodes(root):return {n.name:n for n in ast.parse((root/self.q.PACKED_HISTORY_PATH).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        old,new=nodes(PARENT),nodes(CURRENT)
        for name,node in old.items():
            if name!='HistoryReader':self.assertEqual(ast.dump(node),ast.dump(new[name]),name)
        methods=lambda node:{n.name:n for n in node.body if isinstance(n,ast.FunctionDef)}
        for name,node in methods(old['HistoryReader']).items():
            if name!='read':self.assertEqual(ast.dump(node),ast.dump(methods(new['HistoryReader'])[name]),name)
    def test_actual_current_source_inventory_bound(self):
        for row in self.migration['controls']+self.migration['mathematics']:
            self.assertEqual(row['new'],self.q.tp.pin(CURRENT/row['relative']))
        self.assertEqual(self.migration['parent_commit'],'31c3dca2fbc697b419bf343ce05d10be0a4fe13f')

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Migration))
    read=int(next(x.split(':')[1] for x in Path('/proc/self/io').read_text().splitlines() if x.startswith('rchar:')))-BEFORE
    evidence=dict(author_tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),wall_seconds=time.monotonic()-START,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,self_rchar_bytes=read,payload_open_attempts=BLOCKED,actual_source_migration_cli=CLI_COMPLETED,actual_T3_preparation=False,ledger_written=False,GPU_used=False,independent_approval=False)
    (HERE/'TEST_RESOURCE.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence));raise SystemExit(0 if result.wasSuccessful() else 1)

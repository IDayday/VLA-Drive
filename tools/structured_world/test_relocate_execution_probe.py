"""Filesystem/transaction safety with full15 x17 tiny byte fixtures, no Torch/GPU."""
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.structured_world import relocate_execution_probe as tool


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, sort_keys=True))


def fixture(root, group='G1_mid'):
    start = tool.ORIGINS[group]
    for role in tool.ROLES:
        run = root/group/role
        ancestry = {'completed':start, 'identity':'1'*64, 'complete_sha256':'2'*64,
                    'tag':f'milestone_{start:07d}', 'run':'/immutable/formal/origin'}
        ident = {'scope':'profile','dataset':'navsim','training_source_sha':tool.TRAINING_SOURCE,
            'world_size':8,'effective_global_batch':32,'microbatch':4,'FM_repeat':8,
            'schedule_horizon':100000,'schedule_warmup':5000,'scenes_per_epoch':101592,
            'execution_mode':'engineering_fast' if role=='B' else 'io_preserving_v1',
            'benchmark_no_end_checkpoint':False,'engineering_continue_origin':ancestry,
            'engineering_native_capture':{'origin_completed':start,'end_update':start+500,
                'relative_updates':[1,32,100,250],'absolute_updates':[start+x for x in (1,32,100,250)],
                'stop_at_capture':False},'config':{'structured_world':{'group':'G1_FULL_UNIFORM' if group=='G1_mid' else 'G3_EVENT_LOCAL'},'fixture':'CPU filesystem safety only'}}
        ident['identity'] = tool.json_hash(ident)
        write(run/'identity.json', ident)
        write(run/'PROCESS.json', {'pid':4100})
        write(run/'PROCESS_EXIT.json', {'pid':4100,'returncode':0,'wall_seconds':500.,'monitor_error':None})
        write(run/'RESUME_BOUNDARY_VERIFIED.json', {'identity':ident['identity'],'completed':start,
            'exposure':start*32,'passed':True,'all_task_RNG':'exact all ranks'})
        write(run/'TRAINING_COMPLETE.json', {'identity':ident['identity'],'updates':start+500,'exposure':(start+500)*32})
        write(run/'optimizer_groups.json', [{'names':['a.weight'],'base_lr':1e-5}])
        (run/'metrics.jsonl').write_text(''.join(json.dumps({'updates':start+x,'scene_exposure':(start+x)*32})+'\n' for x in range(1,501)))
        for point in tool.POINTS:
            tag=f'milestone_{start+point:07d}'; bank=run/'checkpoints'/tag;bank.mkdir(parents=True)
            hashes={}
            for name in tool.PAYLOADS:
                data=f'CPU_SYNTHETIC_NOT_TENSOR:{group}:{role}:{point}:{name}'.encode()
                (bank/name).write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
            write(bank/'COMPLETE.json', {'identity':ident['identity'],'tag':tag,'completed':start+point,
                'ranks':8,'status':'COMPLETE' if point==500 else 'RUNNING','file_sha256':hashes})
        (run/'checkpoints/latest').write_text(f'milestone_{start+500:07d}\n')
    return root/group


class LocalSource:
    """Same production validator/release; only SSH/rsync transport substituted."""
    def __init__(self, source):
        self.source=source;self.calls=[];self.copy_calls=0
    def call(self, action, group, ticket=None):
        self.calls.append(action)
        if action=='inspect': return tool.inspect_group(self.source/group,group)
        if action=='receipt':
            return tool.record_source_receipt(self.source/group,group,ticket)
        return tool.release_payloads(self.source/group,group,ticket)
    def copy(self, group, stage, inventory):
        self.copy_calls+=1
        for relative in inventory['files']:
            p=stage/relative;p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(self.source/group/relative,p)


class RelocationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.source=self.base/'source/native_zt4_recollected_p0_v1';self.group=fixture(self.source)
        self.archive=self.base/'parent/native_recollected_archive';self.backend=LocalSource(self.source)
        self.stack=ExitStack();self.stack.enter_context(patch.object(tool,'SOURCE_ROOT',self.source))
        self.stack.enter_context(patch.object(tool,'ARCHIVE_PARENT',self.base/'parent'))
        self.stack.enter_context(patch.object(tool.socket,'gethostname',return_value=tool.PARENT_HOST))
        self.stack.enter_context(patch.object(tool.shutil,'disk_usage',return_value=SimpleNamespace(free=2*(1<<40))))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
    def tearDown(self):
        self.stack.close();self.tmp.cleanup()
    def operate(self, **kw):
        return tool.relocate('G1_mid',self.archive,backend=self.backend,**kw)
    def payloads(self, root=None):
        return [p for p in (root or self.group).rglob('*.pt')]
    def unchanged(self):
        self.assertEqual(len(self.payloads()),255)
        self.assertEqual(len(list(self.group.glob('*/checkpoints/*/COMPLETE.json'))),15)
        self.assertFalse(any((self.group/r/'RELOCATED.json').exists() for r in tool.ROLES))
    def bank(self):return self.group/'A/checkpoints/milestone_0025001'
    def mutate_json(self,p,change):
        d=json.loads(p.read_text());change(d);write(p,d)
    def test_copy_default_full_inventory_and_no_source_deletion(self):
        result=self.operate();self.assertEqual(result['status'],'COPY_ONLY');self.unchanged()
        self.assertEqual(len(self.payloads(self.archive/'G1_mid')),255)
        self.assertEqual(tool.inspect_group(self.group,'G1_mid'),tool.inspect_group(self.archive/'G1_mid','G1_mid'))
        self.assertFalse(list(self.archive.glob('.writing_*')))
    def test_explicit_release_archives_COMPLETE_and_preserves_verified_destination(self):
        originals={str(p.relative_to(self.group)):p.read_bytes() for p in self.group.glob('*/checkpoints/*/COMPLETE.json')}
        result=self.operate(release=True);self.assertEqual(result['status'],'RELEASED')
        self.assertEqual(len(self.payloads()),0);self.assertEqual(len(self.payloads(self.archive/'G1_mid')),255)
        for name,data in originals.items():
            self.assertEqual((self.group/Path(name).with_name('COMPLETE.original.json')).read_bytes(),data)
            self.assertFalse((self.group/name).exists())
        self.assertFalse(tool.read(self.group/'A/RELOCATED.json')['restorable_here'])
        self.assertEqual(tool.read(self.group/'A/RELEASE_RECEIPT.json')['deleted_payload_count'],255)
        self.assertEqual({p.name for p in self.group.iterdir()},set(tool.ROLES))
        with self.assertRaises(ValueError):tool.inspect_group(self.group,'G1_mid')
    def test_delayed_explicit_release_reverifies_archive_without_recopy(self):
        self.operate();self.operate(release=True,release_only=True)
        self.assertEqual(self.backend.copy_calls,1);self.assertEqual(len(self.payloads()),0)
    def test_G3_same_full17_rule_and_offset(self):
        group=fixture(self.source,'G3_mid');inv=tool.inspect_group(group,'G3_mid')
        self.assertEqual(inv['payload_count'],255)
        self.assertEqual([b['completed'] for b in inv['banks']['B']],[50001,50032,50100,50250,50500])
    def test_wrong_source_hash_no_copy_or_deletion(self):
        next(self.bank().glob('*.pt')).write_bytes(b'corrupt')
        with self.assertRaises(ValueError):self.operate(release=True)
        self.unchanged();self.assertEqual(self.backend.copy_calls,0)
    def test_missing_payload_no_deletion(self):
        next(self.bank().glob('*.pt')).unlink()
        with self.assertRaises(ValueError):self.operate(release=True)
        self.assertEqual(len(self.payloads()),254);self.assertFalse((self.group/'A/RELOCATED.json').exists())
    def test_16_file_manifest_not_accepted_as_synthetic_shortcut(self):
        self.mutate_json(self.bank()/'COMPLETE.json',lambda d:d['file_sha256'].pop('rng_rank7.pt'))
        with self.assertRaises(ValueError):self.operate(release=True)
        self.unchanged()
    def test_missing_fifth_bank_or_missing_AA_never_releases(self):
        shutil.rmtree(self.group/'AA/checkpoints/milestone_0025500')
        with self.assertRaises((ValueError,FileNotFoundError)):self.operate(release=True)
        self.assertEqual(len(self.payloads()),238);self.assertFalse((self.group/'A/RELOCATED.json').exists())
    def test_bad_status_tag_step_identity_each_blocked(self):
        p=self.bank()/'COMPLETE.json';original=p.read_bytes()
        for key,value in [('status','PAUSED'),('tag','latest_0025001'),('completed',25002),('identity','a'*64)]:
            with self.subTest(key=key):
                p.write_bytes(original);self.mutate_json(p,lambda d:d.update({key:value}))
                with self.assertRaises(ValueError):tool.inspect_group(self.group,'G1_mid')
                self.unchanged()
    def test_failed_process_or_incomplete500_never_copies(self):
        p=self.group/'B/PROCESS_EXIT.json';self.mutate_json(p,lambda d:d.update(returncode=255))
        with self.assertRaises(ValueError):self.operate(release=True)
        self.unchanged();self.assertEqual(self.backend.copy_calls,0)
    def test_499_metrics_not_success_even_when_summary_claims500(self):
        p=self.group/'B/metrics.jsonl';p.write_text(''.join(p.read_text().splitlines(True)[:-1]))
        with self.assertRaises(ValueError):self.operate(release=True)
        self.unchanged()
    def test_payload_symlink_and_hardlink_rejected_without_outside_mutation(self):
        p=next(self.bank().glob('*.pt'));original=p.read_bytes();outside=self.base/'outside';outside.write_bytes(original)
        p.unlink();p.symlink_to(outside)
        with self.assertRaises(ValueError):tool.inspect_group(self.group,'G1_mid')
        self.assertEqual(outside.read_bytes(),original);p.unlink();os.link(outside,p)
        with self.assertRaises(ValueError):tool.inspect_group(self.group,'G1_mid')
        self.assertEqual(outside.read_bytes(),original)
    def test_symlink_metadata_or_directory_rejected(self):
        p=self.group/'A/metrics.jsonl';outside=self.base/'metrics';p.rename(outside);p.symlink_to(outside)
        with self.assertRaises(ValueError):tool.inspect_group(self.group,'G1_mid')
        self.assertTrue(outside.exists())
    def test_source_symlink_ancestor_cannot_reach_unrelated_banks(self):
        outside=self.base/'unrelated_formal';self.group.rename(outside);self.group.symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):self.operate(release=True)
        self.assertEqual(len(list(outside.rglob('*.pt'))),255)
        self.assertEqual(len(list(outside.glob('*/checkpoints/*/COMPLETE.json'))),15)
    def test_destination_collision_no_copy_delete_or_overwrite(self):
        final=self.archive/'G1_mid';final.mkdir(parents=True,mode=0o700);self.archive.chmod(0o700)
        keep=final/'preserve';keep.write_text('existing owned archive')
        with self.assertRaises(ValueError):self.operate(release=True)
        self.unchanged();self.assertEqual(keep.read_text(),'existing owned archive');self.assertEqual(self.backend.calls,[])
    def test_atomic_rename_does_not_replace_empty_directory(self):
        a=self.base/'writing';b=self.base/'collision';a.mkdir();b.mkdir();(a/'keep').write_text('partial')
        with self.assertRaises(FileExistsError):tool.rename_new(a,b)
        self.assertEqual((a/'keep').read_text(),'partial');self.assertEqual(list(b.iterdir()),[])
    def test_copy_hash_error_retains_partial_no_retry_no_source_delete(self):
        original=self.backend.copy
        def corrupt(*args):
            original(*args);next(args[1].rglob('rng_rank0.pt')).write_bytes(b'bad transport')
        self.backend.copy=corrupt
        with self.assertRaises(ValueError):self.operate(release=True)
        self.unchanged();self.assertEqual(len(list(self.archive.glob('.writing_*'))),1)
        with self.assertRaises(ValueError):self.operate(release=True)
        self.assertEqual(self.backend.copy_calls,1)
    def test_transport_failure_partial_retained(self):
        def failed(group,stage,inv):
            (stage/'partial').write_bytes(b'partial');raise OSError('SSH closed')
        self.backend.copy=failed
        with self.assertRaises(OSError):self.operate(release=True)
        self.unchanged();self.assertEqual(len(list(self.archive.glob('.writing_*/partial'))),1)
    def test_source_mutation_after_copy_blocks_receipt_and_release(self):
        original=self.backend.copy
        def changed(*args):original(*args);next(self.bank().glob('*.pt')).write_bytes(b'changed after copy')
        self.backend.copy=changed
        with self.assertRaises(ValueError):self.operate(release=True)
        self.unchanged();self.assertTrue((self.archive/'G1_mid/ARCHIVE_VERIFIED.json').exists())
    def test_release_only_corrupted_archive_leaves_source(self):
        self.operate();next((self.archive/'G1_mid').rglob('*.pt')).write_bytes(b'archive corruption')
        with self.assertRaises(ValueError):self.operate(release=True,release_only=True)
        self.unchanged()
    def test_release_failure_retains_archive_and_truthful_partial_markers(self):
        unlink=tool.os.unlink;calls=[]
        def fail(name,*,dir_fd=None):
            if dir_fd is not None and name in tool.PAYLOADS:
                if calls:raise OSError('simulated storage failure')
                calls.append(name)
            return unlink(name,dir_fd=dir_fd)
        with patch.object(tool.os,'unlink',side_effect=fail),self.assertRaises(OSError):self.operate(release=True)
        self.assertEqual(len(self.payloads()),254);self.assertEqual(len(self.payloads(self.archive/'G1_mid')),255)
        self.assertEqual(len(tool.read(self.group/'A/RELOCATION_FAILED.json')['deleted_payloads']),1)
        self.assertFalse(tool.read(self.group/'A/RELOCATED.json')['restorable_here'])
        with self.assertRaises(ValueError):tool.inspect_group(self.group,'G1_mid')
    def test_changed_file_after_final_hash_is_not_unlinked(self):
        original=tool.write_new;target=self.bank()/'mp_rank_00_model_states.pt'
        def changed(p,value):
            original(p,value)
            if Path(p)==self.group/'B/checkpoints/milestone_0025500/RELOCATED.json':
                target.write_bytes(b'changed after final hash')
        with patch.object(tool,'write_new',side_effect=changed),self.assertRaises(ValueError):self.operate(release=True)
        self.assertEqual(len(self.payloads()),255)
        self.assertEqual(len(self.payloads(self.archive/'G1_mid')),255)
        self.assertEqual(tool.read(self.group/'A/RELOCATION_FAILED.json')['deleted_payloads'],[])
    def test_insufficient_archive_free_no_copy(self):
        with patch.object(tool.shutil,'disk_usage',return_value=SimpleNamespace(free=tool.RESERVE_BYTES)):
            with self.assertRaises(ValueError):self.operate(release=True)
        self.unchanged();self.assertEqual(self.backend.copy_calls,0)
    def test_owned_pool_cap_no_copy_or_delete_and_registered_alias(self):
        with patch.object(tool,'GROUP_UPPER_BYTES',1):
            with self.assertRaises(ValueError):self.operate(release=True)
        self.unchanged();self.assertEqual(self.backend.copy_calls,0)
        alias=self.archive.with_name('native_zt4_recollected_archive_p0_v1')
        self.assertEqual(tool.archive_path(alias,'G3_mid'),alias/'G3_mid')
    def test_path_group_and_host_admission_reject_outside(self):
        for group in ['G1_early','G0','../G1_mid','G3_mid/../../formal']:
            with self.subTest(group=group),self.assertRaises(ValueError):tool.source_path(group)
        for p in [self.base/'parent/other',self.base/'outside/native_recollected_archive']:
            with self.subTest(path=p),self.assertRaises(ValueError):tool.archive_path(p,'G1_mid')
        with patch.object(tool.socket,'gethostname',return_value='unrelated-host'),self.assertRaises(ValueError):self.operate()
    def test_archive_symlink_ancestor_rejected_before_source_contact(self):
        self.archive.parent.mkdir();self.archive.symlink_to(self.source,target_is_directory=True)
        with self.assertRaises(ValueError):self.operate(release=True)
        self.unchanged();self.assertEqual(self.backend.calls,[])
    def test_real_remote_entry_host_bound(self):
        with self.assertRaises(ValueError):tool.remote_action('inspect','G1_mid')
        with patch.object(tool.socket,'gethostname',return_value=tool.SOURCE_HOST+'-worker-0'):
            self.assertEqual(tool.remote_action('inspect','G1_mid')['payload_count'],255)
    def test_plan_only_does_not_touch_transport(self):
        with patch.object(sys,'argv',['relocate','--group','G1_mid','--archive-root',str(self.archive)]),patch.object(tool,'SSHSource',side_effect=AssertionError('No SSH')):
            self.assertEqual(tool.main(),0)
    def test_cold_code_gate_blocks_mutated_helper_before_transport(self):
        import subprocess
        with patch.object(sys,'argv',['relocate','--group','G1_mid','--archive-root',str(self.archive),'--execute']),patch.object(tool.subprocess,'check_output',return_value=b'other committed code'),patch.object(tool,'SSHSource',side_effect=AssertionError('No SSH')):
            with self.assertRaises(ValueError):tool.main()
        with patch.object(sys,'argv',['relocate','--group','G1_mid','--archive-root',str(self.archive),'--execute']),patch.object(tool.subprocess,'check_output',side_effect=subprocess.CalledProcessError(128,['git'])),patch.object(tool,'SSHSource',side_effect=AssertionError('No SSH')):
            with self.assertRaises(subprocess.CalledProcessError):tool.main()


if __name__=='__main__':unittest.main()

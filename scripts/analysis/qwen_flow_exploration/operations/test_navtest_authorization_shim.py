"""Finite CPU synthetic coverage; no global policy/capability/GPU mutation."""
import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest

source=Path(__file__).with_name('navtest_authorization_shim.py')
spec=importlib.util.spec_from_file_location('shim_under_test',source);shim=importlib.util.module_from_spec(spec);spec.loader.exec_module(shim)
H1='training-vlawm-zt';H2='training-vlawm-tsinghua-zt6';ORD='training-vla-zt'


def sealed(value):
    value=copy.deepcopy(value);value.pop('sha256',None);value['sha256']=shim.fingerprint(value);return value


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.helper=self.root/'native_navtest_spare_lease.py'
        self.helper.write_text('synthetic immutable helper marker')
        self.record=self.root/'real_user_record.json';self.record.write_text(json.dumps({'user_instruction':'spare Navtest only'}))
        grants={shim.GRANT:dict(allowed_hosts=[H1,H2],authorized_gpu_indices={H1:list(range(8)),H2:list(range(1,8))},
                               user_instructions=['spare Navtest only','zt6 GPU1-7'],scope='Navtest'),
                shim.TASK:dict(allowed_hosts=[ORD],preserve_all_existing_training=True),shim.LEGACY_TASK:dict(allowed_hosts=[ORD])}
        self.baseline=dict(task_authorizations=grants,allowed_hosts={ORD:dict(aliases=[ORD,'local'])},
            global_dispatch_prohibitions={h:dict(host=h,aliases=[h],scope='Standing native Navtest grant retained; legacy activation closed',
                new_task_dispatch_allowed=False,authorized_gpu_indices=grants[shim.GRANT]['authorized_gpu_indices'][h],
                user_instruction='historical withdrawal',utc='old_UTC',record=str(self.record)) for h in (H1,H2)},
            latest_host_restriction=dict(hosts=[H1,H2],new_task_dispatch_allowed=False,utc='old_UTC',user_instruction='historical withdrawal'))
        self.anchor=sealed(dict(status='ACTIVE',expires_unix=time.time()+10000,created_unix=time.time(),configuration_lock_sha256='locked',
            evaluation_entry_sha256='immutable345',evaluation_budget_sha256='same_finite_budget',allow_TRAIN_or_DEV=False,allow_optimizer=False,
            hosts={h:dict(gpu_indices=grants[shim.GRANT]['authorized_gpu_indices'][h],gpu_uuids=['synthetic'+str(i) for i in grants[shim.GRANT]['authorized_gpu_indices'][h]],
                          authorization_sha256=shim.full_fingerprint(self.baseline,h)) for h in (H1,H2)}))
        self.receipt=shim.seal_semantics(self.baseline,self.anchor,self.helper)

    def tearDown(self):self.temp.cleanup()

    def enter(self,before,name,*,publish=True):
        control=self.root/name;control.mkdir();shim.atomic(control/'policy_before.json',before)
        value=copy.deepcopy(before);token='token_'+name
        for h in (H1,H2):
            indices=value['task_authorizations'][shim.GRANT]['authorized_gpu_indices'][h]
            value['allowed_hosts'][h]=dict(aliases=before['global_dispatch_prohibitions'][h]['aliases'],lease_id=token,
                task_scope=shim.LEGACY_TASK,purpose='this registered Navtest only',authorized_gpu_indices=indices)
            value['global_dispatch_prohibitions'][h].update(lease_id=token,lease_record=str(control/'lease.json'),
                scope='this registered native Navtest only',new_task_dispatch_allowed=True,authorized_gpu_indices=indices)
        if publish:shim.atomic(control/'lease.json',dict(lease_id=token,hosts={h:value['task_authorizations'][shim.GRANT]['authorized_gpu_indices'][h] for h in (H1,H2)},standing_grant=shim.GRANT))
        return value

    def check(self,policy):return shim.validate_current(policy,self.receipt,self.anchor,self.anchor)

    def test_original_closed_and_legacy_on_off(self):
        self.assertEqual(set(self.check(self.baseline).values()),{'CLOSED_BASELINE'})
        active=self.enter(self.baseline,'one');self.assertEqual(set(self.check(active).values()),{'KNOWN_LEGACY_ACTIVE'})
        newer=self.enter(active,'two');self.check(newer)
        self.check(active);self.check(self.baseline)

    def test_policy_published_before_legacy_lease_record(self):
        active=self.enter(self.baseline,'publishing',publish=False)
        self.assertEqual(set(self.check(active).values()),{'KNOWN_LEGACY_PUBLISHING'})

    def test_refresh_only_our_hashes_never_policy_or_scope(self):
        policy=self.enter(self.baseline,'on');p=self.root/'policy.json';l=self.root/'our_lease.json';r=self.root/'semantics.json';t=self.root/'template.json'
        for path,value in [(p,policy),(l,self.anchor),(r,self.receipt),(t,self.anchor)]:shim.atomic(path,value)
        before=p.read_bytes();result=shim.refresh(p,l,r,t)
        self.assertTrue(result['updated_hosts']);self.assertEqual(before,p.read_bytes())
        refreshed=shim.read(l);self.assertEqual(shim.capability_invariant(refreshed),shim.capability_invariant(self.anchor))
        self.assertEqual(refreshed['hosts'][H1]['authorization_sha256'],shim.full_fingerprint(policy,H1))
        shim.atomic(p,self.baseline);shim.refresh(p,l,r,t)
        self.assertEqual(shim.read(l)['hosts'][H1]['authorization_sha256'],self.anchor['hosts'][H1]['authorization_sha256'])

    def test_standing_host_GPU_or_instruction_reduction_refused(self):
        for kind in ('host','GPU','instruction'):
            p=copy.deepcopy(self.baseline)
            if kind=='host':p['task_authorizations'][shim.GRANT]['allowed_hosts'].remove(H2)
            if kind=='GPU':p['task_authorizations'][shim.GRANT]['authorized_gpu_indices'][H2].remove(7)
            if kind=='instruction':p['task_authorizations'][shim.GRANT]['user_instructions']=['no more Navtest']
            with self.subTest(kind=kind),self.assertRaises(PermissionError):self.check(p)

    def test_our_TASK_or_ordinary_permanent_permission_change_refused(self):
        p=copy.deepcopy(self.baseline);p['task_authorizations'][shim.TASK]['allowed_hosts']=[]
        with self.assertRaises(PermissionError):self.check(p)
        ordinary=copy.deepcopy(self.anchor);ordinary['hosts']={ORD:dict(gpu_indices=[0],gpu_uuids=['ordinary0'],authorization_sha256=shim.full_fingerprint(self.baseline,ORD))};ordinary=sealed(ordinary)
        receipt=shim.seal_semantics(self.baseline,ordinary,self.helper);p=copy.deepcopy(self.baseline);p['allowed_hosts'][ORD]['unknown_permission_field']='changed'
        with self.assertRaises(PermissionError):shim.validate_current(p,receipt,ordinary,ordinary)

    def test_real_withdrawal_flag_UTC_record_unknown_fields_refused(self):
        original=self.enter(self.baseline,'active')
        for key,value in [('new_task_dispatch_allowed',False),('scope','user revoked this host'),('utc','new_UTC'),('unknown_new_permission',False)]:
            p=copy.deepcopy(original);p['global_dispatch_prohibitions'][H1][key]=value
            with self.subTest(key=key),self.assertRaises(PermissionError):self.check(p)
        p=copy.deepcopy(original);p['latest_exact_host_withdrawal']=dict(hosts=[H2],record='new_user_withdrawal',utc='now')
        with self.assertRaises(PermissionError):self.check(p)
        self.record.write_text(json.dumps({'user_instruction':'withdraw both target hosts now'}))
        with self.assertRaises(PermissionError):self.check(original)

    def test_false_flag_cannot_be_hidden_by_removing_only_lease_id(self):
        p=self.enter(self.baseline,'active');p['global_dispatch_prohibitions'][H1].pop('lease_id');p['global_dispatch_prohibitions'][H1]['new_task_dispatch_allowed']=False
        with self.assertRaises(PermissionError):self.check(p)

    def test_unknown_temporary_allowed_field_and_smaller_GPU_scope_refused(self):
        p=self.enter(self.baseline,'active');p['allowed_hosts'][H1]['unknown_field']='do not ignore'
        with self.assertRaises(PermissionError):self.check(p)
        p=self.enter(self.baseline,'active2');p['allowed_hosts'][H2]['authorized_gpu_indices']=[1]
        with self.assertRaises(PermissionError):self.check(p)

    def test_zt6_GPU0_expiry_and_capability_mutation_refused(self):
        bad=copy.deepcopy(self.anchor);bad['hosts'][H2]['gpu_indices']=[0,1];bad=sealed(bad)
        with self.assertRaises(PermissionError):shim.seal_semantics(self.baseline,bad,self.helper)
        expired=copy.deepcopy(self.anchor);expired['expires_unix']=0;expired=sealed(expired)
        with self.assertRaises(PermissionError):shim.validate_current(self.baseline,self.receipt,expired,expired)
        changed=copy.deepcopy(self.anchor);changed['expires_unix']+=100;changed=sealed(changed)
        with self.assertRaises(PermissionError):shim.validate_current(self.baseline,self.receipt,changed,self.anchor)

    def test_refusal_never_refreshes_auth_and_can_revoke_our_lease_only(self):
        policy=self.enter(self.baseline,'active');policy['global_dispatch_prohibitions'][H1]['utc']='real_user_change'
        p=self.root/'policy.json';l=self.root/'our_lease.json';r=self.root/'receipt.json';t=self.root/'template.json'
        for path,value in [(p,policy),(l,self.anchor),(r,self.receipt),(t,self.anchor)]:shim.atomic(path,value)
        before=p.read_bytes()
        with self.assertRaises(PermissionError):shim.refresh(p,l,r,t,invalidate_on_refusal=True)
        blocked=shim.read(l);self.assertEqual(blocked['status'],'BLOCKED_PERMISSION_CHANGE');self.assertEqual(before,p.read_bytes())
        self.assertEqual(blocked['hosts'],self.anchor['hosts'])

    def test_immutable345_actual_guard_oracle_on_off_and_GPU0(self):
        original=Path('/mnt/project/qwen-flow-three-routes-20261024/snapshots/evaluation_audit_v2/starVLA/rl/qwen_flow_exploration/evaluation_admission.py')
        syntax=ast.parse(original.read_text())
        functions=[node for node in syntax.body if isinstance(node,ast.FunctionDef) and node.name in ('authorization_fingerprint','validate_capability')]
        self.assertEqual(len(functions),2)
        namespace=dict(fingerprint=shim.fingerprint,GRANT=shim.GRANT,TASK=shim.TASK)
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(original),'exec'),namespace)
        locked=sealed(dict(plan='synthetic fixed model/seed budget'))
        for policy in (self.baseline,self.enter(self.baseline,'actual_oracle')):
            lease=copy.deepcopy(self.anchor);lease['configuration_lock_sha256']=locked['sha256']
            for host in lease['hosts']:
                self.assertEqual(namespace['authorization_fingerprint'](policy,host),shim.full_fingerprint(policy,host))
                lease['hosts'][host]['authorization_sha256']=shim.full_fingerprint(policy,host)
            lease=sealed(lease)
            for host,indices in ((H1,[0]),(H2,[1])):
                self.assertEqual(namespace['validate_capability'](policy,lease,locked,host,indices,'immutable345',time.time()),host)
            with self.assertRaises(PermissionError):namespace['validate_capability'](policy,lease,locked,H2,[0],'immutable345',time.time())
            lease['status']='BLOCKED_PERMISSION_CHANGE';lease=sealed(lease)
            with self.assertRaises(PermissionError):namespace['validate_capability'](policy,lease,locked,H1,[0],'immutable345',time.time())


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests))
    value=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        scope='CPU-only synthetic permission semantics and legacy metadata; global policy/real lease/actors untouched',
        no_actor_math_or_qualification_change=True,no_GPU_dispatch=True,
        race_boundary='Separate external owner cannot remove the transient window in immutable345 unlocked policy+lease readers. Refresh successfully before dispatch; do not promise race-free live metadata changes.')
    out=Path(a.output)
    if out.exists():raise FileExistsError('Keep prior CPU evidence')
    shim.atomic(out,value)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()

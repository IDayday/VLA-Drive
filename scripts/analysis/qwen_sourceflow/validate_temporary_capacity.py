"""CPU-only lease boundary checks; no change to live policy or training."""
import json,tempfile,unittest
from pathlib import Path
from starVLA.rl.qwen_sourceflow.resource_authorization import authorized_hosts


class TemporaryCapacity(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.host='training-vla-zt3'
        self.policy=dict(allowed_hosts={'training-vla-zt':{},self.host:{}},global_dispatch_prohibitions={},task_authorizations={
            'qwen_sourceflow_native_20261018':dict(allowed_hosts=['training-vla-zt',self.host],temporary_host_exceptions={self.host:dict(
                enabled=True,scope='auxiliary_native_RL_completion',end_condition='both_registered_RL_runs_reach_50000')})})
        self.write(20000,20000)
    def write(self,a,b):
        for n,u in [('RL_CANONICAL_seed42',a),('RL_SOURCE_seed42',b)]:
            p=self.root/'runs'/n;p.mkdir(parents=True,exist_ok=True);(p/'progress.json').write_text(json.dumps(dict(updates=u)))
    def hosts(self):return authorized_hosts(self.policy,self.root)
    def test_active_lease(self):self.assertIn(self.host,self.hosts())
    def test_both_endpoints_close_even_before_global_guard(self):self.write(50000,50000);self.assertNotIn(self.host,self.hosts())
    def test_one_remaining_keeps_lease(self):self.write(50000,49996);self.assertIn(self.host,self.hosts())
    def test_global_withdrawal_wins(self):self.policy['global_dispatch_prohibitions'][self.host]={'new_task_dispatch_allowed':False};self.assertNotIn(self.host,self.hosts())
    def test_task_revoke_wins(self):self.policy['task_authorizations']['qwen_sourceflow_native_20261018']['temporary_host_exceptions'][self.host]['enabled']=False;self.assertNotIn(self.host,self.hosts())
    def test_missing_progress_fails_closed(self):(self.root/'runs/RL_SOURCE_seed42/progress.json').unlink();self.assertNotIn(self.host,self.hosts())
    def test_bad_progress_fails_closed(self):(self.root/'runs/RL_SOURCE_seed42/progress.json').write_text('{');self.assertNotIn(self.host,self.hosts())
    def test_lease_cannot_expand_to_withdrawn_host(self):
        revoked='training-vlawm-tsinghua-zt2';auth=self.policy['task_authorizations']['qwen_sourceflow_native_20261018']
        self.policy['allowed_hosts'][revoked]={};auth['allowed_hosts'].append(revoked)
        auth['temporary_host_exceptions'][revoked]=dict(auth['temporary_host_exceptions'][self.host])
        self.assertNotIn(revoked,self.hosts())
    def test_normal_host_unchanged(self):self.write(50000,50000);self.assertEqual(self.hosts(),['training-vla-zt'])


if __name__=='__main__':unittest.main()

"""Regression for the actual shallow-copy signing mistake; no live signature IO."""
from pathlib import Path
import importlib.util,unittest
p=Path(__file__).with_name('sign_v2.py');spec=importlib.util.spec_from_file_location('sign_v2',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class SigningTests(unittest.TestCase):
 def proposal(self):return {'status':'PROPOSAL_NOT_APPROVED','reviewer':None,'controller_sources':[{'path':'/fixture','sha256':m.WRAPPER_SHA}],'policy':{'path':'/fixture_policy','sha256':'1'*64}}
 def test_legal_transform_does_not_alias_input(self):
  p=self.proposal();s=m.authorized_transform(p);m.verify_transform(p,s);s['controller_sources'][-1]['sha256']='old';self.assertEqual(p['controller_sources'][-1]['sha256'],m.WRAPPER_SHA)
 def test_exact_shallow_copy_regression_refuses_output(self):
  p=self.proposal();s=m.authorized_transform(p);shallow=dict(s);shallow['controller_sources'][-1]['sha256']='14f70bab1bc5f1d6d66946d71a411b05bef4b05e7d614166d948cf81f1dc0f40'
  with self.assertRaisesRegex(ValueError,'beyond status/reviewer'):m.verify_transform(p,s)
 def test_policy_mutation_refused(self):
  p=self.proposal();s=m.authorized_transform(p);s['policy']['sha256']='2'*64
  with self.assertRaises(ValueError):m.verify_transform(p,s)
 def test_unexpected_added_field_refused(self):
  p=self.proposal();s=m.authorized_transform(p);s['implicit_override']=True
  with self.assertRaises(ValueError):m.verify_transform(p,s)
if __name__=='__main__':unittest.main(verbosity=2)

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location('retire', Path(__file__).with_name('retire_states_parallel.py'))
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class RetirementTest(unittest.TestCase):
    def test_end_to_end_preserves_weights_protection_and_unknown_containers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); old_root = module.ROOT; module.ROOT = root
            try:
                cases = {
                    'adapter': dict(adapters={'A':torch.arange(8,dtype=torch.float32).reshape(4,2)}, optimizer={'state':'Adam'}, scheduler={'step':50}),
                    'pure': dict(optimizer={'state':{},'param_groups':[]},scheduler={'step':10}),
                    'protected': dict(optimizer={'state':{},'param_groups':[]}),
                    'unknown': dict(model={'weight':torch.ones(2)}, optimizer={'state':'Adam'}),
                }
                rows = []; protected = None
                for name, payload in cases.items():
                    checkpoint = root/name; checkpoint.mkdir(); p=checkpoint/'optimizer.pt';torch.save(payload,p)
                    (checkpoint/'model.pt').write_bytes(b'inference weights remain unchanged')
                    s=p.stat()
                    rows.append(dict(path=str(checkpoint),files=[dict(path=str(p),name=p.name,device=s.st_dev,inode=s.st_ino,size=s.st_size,blocks=s.st_blocks*512,nlink=s.st_nlink)]))
                    if name=='protected':protected=[s.st_dev,s.st_ino]
                (root/'ACTIVE_PROTECTED.json').write_text(json.dumps(dict(inodes=[protected])))
                inventory=root/'inventory.json';inventory.write_text(json.dumps(dict(checkpoints=rows)))
                with patch.object(sys,'argv',['retire','--inventory',str(inventory),'--apply']):module.main()
                self.assertFalse((root/'adapter/optimizer.pt').exists())
                self.assertFalse((root/'pure/optimizer.pt').exists())
                self.assertTrue((root/'protected/optimizer.pt').exists())
                self.assertTrue((root/'unknown/optimizer.pt').exists())
                compact=torch.load(root/'adapter/inference_adapters.pt',weights_only=False)
                self.assertEqual(set(compact),{'adapters'})
                self.assertTrue(torch.equal(compact['adapters']['A'],cases['adapter']['adapters']['A']))
                for name in cases:self.assertEqual((root/name/'model.pt').read_bytes(),b'inference weights remain unchanged')
                summary=json.loads((root/'PHASE2_COMPLETE.json').read_text())
                self.assertEqual(summary['removed_paths'],2)
                self.assertEqual(summary['skipped_protected_inodes'],1)
                self.assertEqual(summary['skipped_unknown_inodes'],1)
            finally:module.ROOT=old_root

    def test_mixed_precision_tensor_values_survive(self):
        original=dict(adapters={'A':torch.randn(2,3,dtype=torch.float32),'B':torch.randn(3,2,dtype=torch.bfloat16)},optimizer={'state':{}})
        keep=module.split_optimizer(original)
        self.assertTrue(module.same_tree(keep['adapters'],original['adapters']))
        self.assertNotIn('optimizer',keep)

    def test_unknown_master_weights_are_not_discarded(self):
        with self.assertRaises(ValueError):module.split_optimizer(dict(fp32_master=torch.ones(5),optimizer={}))
        with self.assertRaises(ValueError):module.split_optimizer(dict(optimizer=dict(fp32_flat_groups=[torch.ones(5)])))

    def test_handoff_reuses_exact_capsule_with_one_source_alias_already_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old_root=module.ROOT;module.ROOT=root
            try:
                first=root/'a';second=root/'b';first.mkdir();second.mkdir()
                original=first/'optimizer.pt';payload=dict(adapters={'A':torch.arange(6,dtype=torch.float32)},optimizer={'state':{}})
                torch.save(payload,original);os.link(original,second/'optimizer.pt');st=original.stat()
                rows=[]
                for checkpoint in [first,second]:
                    rows.append(dict(path=str(checkpoint),files=[dict(path=str(checkpoint/'optimizer.pt'),name='optimizer.pt',device=st.st_dev,inode=st.st_ino,size=st.st_size,blocks=st.st_blocks*512,nlink=2)]))
                weights=root/'inference_weights';weights.mkdir();capsule=weights/f'{st.st_dev}_{st.st_ino}.pt'
                torch.save(module.split_optimizer(payload),capsule)
                for checkpoint in [first,second]:os.link(capsule,checkpoint/'inference_adapters.pt')
                digest=module.sha(capsule);original.unlink()
                (root/'ACTIVE_PROTECTED.json').write_text(json.dumps(dict(inodes=[])))
                inventory=root/'inventory.json';inventory.write_text(json.dumps(dict(checkpoints=rows)))
                with patch.object(sys,'argv',['retire','--inventory',str(inventory),'--apply']):module.main()
                self.assertFalse((second/'optimizer.pt').exists())
                self.assertEqual(module.sha(capsule),digest)
                self.assertTrue(os.path.samefile(first/'inference_adapters.pt',second/'inference_adapters.pt'))
            finally:module.ROOT=old_root


if __name__=='__main__':unittest.main()

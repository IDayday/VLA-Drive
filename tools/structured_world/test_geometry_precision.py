"""CPU precision-policy checks; small shapes are tests, never speed evidence."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import copy
from pathlib import Path
import sys
import unittest
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.model.modules.structured_world.geo_bev import SingleFrameGeoBEV
from starVLA.model.modules.structured_world.grid import GridSpec


def inputs(cameras):
    images = torch.randn(2, cameras, 3, 64, 96, generator=torch.Generator().manual_seed(8))
    transforms = torch.eye(4).expand(2, cameras, 4, 4).clone()
    transforms[..., :3, :3] = torch.tensor([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    transforms[..., 2, 3] = 1.5
    intrinsics = torch.tensor([[60.,0.,48.],[0.,60.,32.],[0.,0.,1.]]).expand(2,cameras,3,3).clone()
    calib = {'sensor2ego': transforms, 'intrinsics': intrinsics,
             'post_rots': torch.eye(3).expand(2,cameras,3,3).clone(),
             'post_trans': torch.zeros(2,cameras,3)}
    return images, calib, torch.ones(2,cameras,64,96,dtype=torch.bool)


class PrecisionTests(unittest.TestCase):
    def test_canonical_bypass_is_exact_and_inventory_unchanged(self):
        torch.set_num_threads(2)
        torch.manual_seed(14)
        model = SingleFrameGeoBEV(GridSpec(-4, 4, -4, 4, .5, .5), (64,96)).eval()
        names = list(model.state_dict())
        for cameras in (3,6):
            x, calib, valid = inputs(cameras)
            with torch.no_grad():
                reference = model(x, calib, valid)
                model.conv_autocast_bf16 = True
                canonical = model(x, calib, valid, canonical_fp32=True)
                mixed = model(x, calib, valid)
                model.conv_autocast_bf16 = False
            torch.testing.assert_close(reference['B0'], canonical['B0'], rtol=0, atol=0)
            torch.testing.assert_close(reference['depth'][0], canonical['depth'][0], rtol=0, atol=0)
            self.assertTrue(torch.equal(reference['projection_support'], mixed['projection_support']))
            self.assertEqual(mixed['B0'].dtype, torch.float32)
            self.assertEqual(mixed['depth'][0].dtype, torch.float32)
            self.assertTrue(torch.isfinite(mixed['B0']).all())
            self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
            self.assertEqual(names, list(model.state_dict()))

    def test_syncbn_population_and_buffers_stay_registered(self):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            SingleFrameGeoBEV(GridSpec(-4,4,-4,4,.5,.5), (64,96)))
        before = {name: (type(module), id(module.running_mean), id(module.running_var), module.eps, module.momentum)
                  for name, module in model.named_modules() if isinstance(module, torch.nn.SyncBatchNorm)}
        model.conv_autocast_bf16 = True
        after = {name: (type(module), id(module.running_mean), id(module.running_var), module.eps, module.momentum)
                 for name, module in model.named_modules() if isinstance(module, torch.nn.SyncBatchNorm)}
        self.assertTrue(before)
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main(verbosity=2)

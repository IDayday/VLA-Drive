"""Targeted CPU arithmetic/mask/gradient and real nine-image processor checks."""
import argparse,copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from starVLA.model.modules.qwen_sourceflow.temporal_dino import TemporalQwenProcessor,CurrentDINOHead,current_dino_loss
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenEncoder
from starVLA.rl.qwen_sourceflow.temporal_data import TemporalSceneData
from starVLA.rl.qwen_sourceflow.contracts import QWEN_PATH
from starVLA.rl.qwen_sourceflow.temporal_contracts import ROOT


class Arithmetic(unittest.TestCase):
    def test_deterministic_resampling_forward_and_gradient(self):
        x=torch.randn(1,7,11,21,requires_grad=True);y=x.detach().clone().requires_grad_()
        head=CurrentDINOHead(7,5,(6,8))
        a=head.resize(x);b=torch.nn.functional.interpolate(y,size=(6,8),mode='bilinear',align_corners=False)
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        a.square().sum().backward();b.square().sum().backward()
        torch.testing.assert_close(x.grad,y.grad,atol=3e-6,rtol=2e-5)

    def test_current_only_readout_and_gradient(self):
        h=torch.randn(2,12,7,requires_grad=True)
        layouts=[[dict(camera=c,frame_time=t,offset=3*(t+2)+c,height=1,width=1) for t in [-2,-1,0] for c in range(3)] for _ in range(2)]
        e=dict(final_hidden=h,visual_indices=torch.arange(9).repeat(2,1),image_layout=layouts)
        head=CurrentDINOHead(7,5,(2,2));p,valid=head(e)
        y=torch.randn_like(p,requires_grad=True);loss=current_dino_loss(p,y,torch.ones(2,3,2,2,dtype=torch.bool),valid)
        self.assertTrue(torch.allclose(loss,(p-y.detach()).square().mean()))
        loss.backward();self.assertIsNone(y.grad)
        self.assertEqual(float(h.grad[:,:6].abs().sum()),0.)
        self.assertGreater(float(h.grad[:,6:9].abs().sum()),0.)
        self.assertGreater(float(head.projection.weight.grad.abs().sum()),0.)

    def test_invalid_padding_nan_is_excluded(self):
        p=torch.randn(1,3,5,2,2,requires_grad=True);y=torch.randn_like(p)
        valid=torch.ones(1,3,2,2,dtype=torch.bool);valid[:,:,0,0]=False
        y[:,:, :,0,0]=float('nan')
        loss=current_dino_loss(p,y,valid,torch.ones(1,3,dtype=torch.bool));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(p.grad).all())

    def test_zero_valid_teacher_refused(self):
        p=torch.zeros(1,3,5,2,2)
        with self.assertRaises(ValueError):current_dino_loss(p,p,torch.zeros(1,3,2,2,dtype=torch.bool),torch.ones(1,3,dtype=torch.bool))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Arithmetic))
    if not result.wasSuccessful():raise RuntimeError('Temporal auxiliary arithmetic failed')
    data=TemporalSceneData(ROOT/'data/TRAIN_v3',cohort='TRAIN');o=data.observation(data.tokens[0]);processor=TemporalQwenProcessor(QWEN_PATH)
    batch=processor([o]);assert batch['camera_indices']==[[0,1,2]*3]
    assert batch['image_time_indices']==[[-2]*3+[-1]*3+[0]*3]
    assert len(batch['image_grid_thw'])==9
    dirty=dict(o,requested_GT=torch.randn(8,3),reward=999.,future_images='not actor input')
    other=processor([dirty])
    for k in batch:
        if torch.is_tensor(batch[k]):assert torch.equal(batch[k],other[k]),k
        else:assert batch[k]==other[k],k
    missing=copy.deepcopy(o);missing['frames'][0]['images']['cam_r0']=None
    padded=processor([o,missing]);assert len(padded['image_grid_thw'])==17
    receipt=dict(status='PASS',CPU_arithmetic_tests=result.testsRun,real_processor_images=9,
        camera_order='chronological then L/F/R',frame_coordinates=[-2,-1,0],
        actual_grids=batch['image_grid_thw'].tolist(),future_labels_ignored=True,missing_historical_view_explicit=True,
        current_head_only_reads_current_tokens=True,teacher_detached=True,
        qualification_scope='CPU/processor only; real full-model/GPU qualification still required')
    Path(a.output).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()

"""The calibrated recipe is a distinct algorithm, tested at its actual boundary."""
import copy
import importlib.util
from pathlib import Path
import pytest
import torch
from starVLA.rl.d46_augustus.config import defaults,validate
from starVLA.rl.d46_augustus.runtime import advantages,recipe_credit,AugustusExtension
from starVLA.rl.d46_augustus.output_precision import FP32ActionOutput


def test_physical_noise_and_precision_are_bound_to_independent_recipe():
    cfg=defaults('AUG6_NATIVE_CALIBRATED');spec=cfg['augustus']
    scale=torch.tensor(spec['noise_scale'])
    assert scale.shape==(8,3)and (scale>0).all()and not torch.equal(scale[0],scale[-1])
    assert cfg['group']==16 and cfg['save_every']==100 and spec['action_output_precision']=='float32_linear'
    for field,value in [('noise_scale',[.05,.05,.02]),('action_output_precision','bfloat16')]:
        bad=copy.deepcopy(cfg);bad['augustus'][field]=value
        with pytest.raises(ValueError):validate(bad)


def test_global_normalization_keeps_safety_contrast_above_small_EP_and_ignores_padding():
    raw=torch.stack((torch.linspace(-.01,.01,16),torch.linspace(-.5,.5,16),torch.linspace(-10,10,16))).double()
    valid=torch.tensor([True,True,False]);actual,std=advantages(raw,valid,16,1,global_normalization=True)
    expected=(raw[:2]/(raw[:2].std(unbiased=False)+1e-8)).clamp(-5,5).float()
    torch.testing.assert_close(actual[:2],expected)
    assert actual[2].count_nonzero()==0 and actual[1].abs().sum()>40*actual[0].abs().sum()
    zeros,_=advantages(raw*0,valid,16,1,global_normalization=True)
    assert torch.isfinite(zeros).all()and zeros.count_nonzero()==0


def test_calibrated_collector_fresh_ratio_and_two_inner_updates():
    spec=importlib.util.spec_from_file_location('calibrated_tiny',Path(__file__).with_name('test_d46_augustus_actor.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        actor,_,cfg,_,condition=m.tiny('AUG6_NATIVE_CALIBRATED');condition['context']=torch.zeros(16,1,1)
        assert isinstance(actor.action_expert.decision.action_output,FP32ActionOutput)
        assert isinstance(actor.ref_action_expert.decision.action_output,FP32ActionOutput)
        scene=m.collect_scene(actor,condition,'TRAIN_CPU_0','TRAIN_cpu',m.SyntheticService(),torch.Generator().manual_seed(42),cfg,0.,{'update':0})
        raw,_=recipe_credit([scene],None,0.,cfg)
        a,_=advantages(raw,torch.tensor([True]),16,1,global_normalization=True)
        scene.update(advantages=a[0],padding=False,beta=cfg['beta'],expect_fresh=True)
        optimizer=torch.optim.AdamW(AugustusExtension.optimizer_groups(actor,cfg),lr=cfg['learning_rate'])
        for inner in range(2):
            loss,metrics=actor(scene);loss.backward()
            assert torch.isfinite(loss)and all(torch.isfinite(p.grad).all()for p in actor.parameters()if p.grad is not None)
            if inner==0:assert metrics['fresh_ratio_error']==0
            torch.nn.utils.clip_grad_norm_([p for p in actor.parameters()if p.requires_grad],1.)
            optimizer.step();optimizer.zero_grad();scene['expect_fresh']=False
        assert scene['chain'].shape==(16,4,8,3)
    finally:torch.set_num_threads(old)

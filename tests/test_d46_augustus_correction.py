"""Regression tests for the observed failures; no scoring-code changes."""
import importlib.util
import json
from pathlib import Path
import pytest
import torch
from starVLA.rl.d46_augustus.config import defaults,validate
from starVLA.rl.d46_augustus.runtime import recipe_credit,advantages
from starVLA.model.modules.qwen_world_postcot.flow import shift_sigma
from tools.d44_native_rl.train import publish_evaluation_requests


def test_first_exposure_gate_publishes_into_missing_directory(tmp_path):
    root=tmp_path/'new_run'
    assert not root.exists()
    publish_evaluation_requests(root,[.5,1.],'/sealed/native','run_sha',404)
    first=json.loads((root/'evaluation_requests/epoch0.5.json').read_text())
    assert first['checkpoint']=='/sealed/native'and first['updates']==404
    assert first['protocol']=='single_B1_cleanx_ODE_K10_shift5_canonical_v1.1'
    assert (root/'evaluation_requests/epoch1.json').exists()


def test_TTC_counterexample_cannot_receive_positive_credit_with_safe_option():
    # Real formal-prefix pattern: TTC0 score7/12 exceeds group mean, although a
    # truly safe score1 candidate exists. Old canonical-only GRPO rewarded it.
    factors=torch.tensor([[1,1,1,1,1,1],[1,1,0,1,1,1]]+[[0,1,0,.5,1,1]]*6,dtype=torch.float64)
    scores=[1.,7/12]+[0.]*6
    from starVLA.rl.d44_native_rl.safety_credit import FACTORS
    scene=dict(factors=factors,chain=torch.zeros(8,4,8,3),records=[dict(metrics=dict(zip(FACTORS,f.tolist()),score=s))for f,s in zip(factors,scores)])
    old,_=recipe_credit([scene],None,0.,defaults('AUG2_LORA'))
    new,_=recipe_credit([scene],None,0.,defaults('AUG4_LORA_ALIGNED'))
    a,_=advantages(new,torch.tensor([True]),8,1)
    assert old[0,1]>0 and new[0,1]<0 and a[0,1]<0 and a[0,0]>0


@pytest.mark.parametrize('experiment',['AUG3_FULL_ALIGNED','AUG4_LORA_ALIGNED','AUG5_LORA_TOKEN_ROOT'])
def test_exact_native_grid_and_initial_parity_then_clipped_reuse(experiment):
    spec=importlib.util.spec_from_file_location('tiny_aug_correction',Path(__file__).with_name('test_d46_augustus_actor.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        actor,parent,cfg,inputs,condition=m.tiny(experiment)
        grid=shift_sigma(torch.linspace(1,0,11),5.).tolist()
        assert cfg['augustus']['training_sigma_grid']==grid[7:]
        assert cfg['augustus']['deterministic_prefix_grid']==grid[:8]
        assert cfg['beta']==.1 and cfg['inner_updates']==2
        root=torch.randn((1,8,3),generator=torch.Generator().manual_seed(2101))
        with torch.no_grad():
            native=parent.predict(inputs,torch.zeros(1,8),root)
            chain,_,_,_=actor.sample_action_chain(condition,deterministic=True,init_actions=root.expand(8,-1,-1))
        # CPU BF16 batch kernels may accumulate differently for B1 vs repeatedG;
        # comparison with native repeatedG also covers the exact ActionCodec.
        torch.testing.assert_close(chain[:1,-1],native['normalized_action'],atol=.02,rtol=.02)
        scene=m.collect_scene(actor,condition,'TRAIN_CPU_0','TRAIN_cpu',m.SyntheticService(),torch.Generator().manual_seed(420000),cfg,0.,{'update':0})
        raw,_=recipe_credit([scene],None,0.,cfg);a,_=advantages(raw,torch.tensor([True]),8,1)
        scene.update(advantages=a[0],padding=False,beta=cfg['beta'],expect_fresh=True)
        optimizer=torch.optim.AdamW([p for p in actor.parameters()if p.requires_grad],lr=cfg['learning_rate'])
        loss,metrics=actor(scene);loss.backward();assert metrics['fresh_ratio_error']==0
        torch.nn.utils.clip_grad_norm_([p for p in actor.parameters()if p.requires_grad],1.);optimizer.step();optimizer.zero_grad()
        scene['expect_fresh']=False
        second,metrics=actor(scene);second.backward()
        assert torch.isfinite(second)and metrics['reference_KL']>=0
        assert all(torch.isfinite(p.grad).all()for p in actor.parameters()if p.grad is not None)
        bad=dict(cfg,inner_updates=1)
        with pytest.raises(ValueError):validate(bad)
    finally:torch.set_num_threads(old_threads)


def test_production_collector_binds_root_to_scene_like_official_evaluator():
    """Old parity tests supplied manual roots and missed the real entry point."""
    from starVLA.rl.simwam_recipe.contracts import seed_for
    from starVLA.rl.d46_augustus.collector import scene_initial_actions
    spec=importlib.util.spec_from_file_location('tiny_aug_root',Path(__file__).with_name('test_d46_augustus_actor.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        actor,_,cfg,_,condition=m.tiny('AUG5_LORA_TOKEN_ROOT')
        original=actor.sample_action_chain;seen=[]
        def capture(*args,**kwargs):
            seen.append(kwargs['init_actions'].clone())
            return original(*args,**kwargs)
        actor.sample_action_chain=capture
        generator=torch.Generator().manual_seed(420000)
        scenes=[]
        for token in ('TRAIN_CPU_0','TRAIN_CPU_1','TRAIN_CPU_0'):
            before=generator.get_state().clone()
            scenes.append(m.collect_scene(actor,condition,token,'TRAIN_cpu',m.SyntheticService(),generator,cfg,0.,{'update':0}))
            assert not torch.equal(generator.get_state(),before)  # Fresh JaT innovations.
            expected=torch.randn((1,8,3),generator=torch.Generator().manual_seed(
                seed_for('native_qwen_sourceflow_ODE',2101,token)))
            assert torch.equal(seen[-1],expected.expand(8,-1,-1))
        assert not torch.equal(seen[0],seen[1])
        assert torch.equal(seen[0],seen[2])
        assert torch.equal(scenes[0]['chain'][:,0],scenes[2]['chain'][:,0])
        assert not torch.equal(scenes[0]['chain'][:,-1],scenes[2]['chain'][:,-1])
        with pytest.raises(ValueError,match='explicit scene initial actions'):
            original(condition,generator=generator)
        # Old recipe identities remain unchanged; there is no silent resume migration.
        old=defaults('AUG4_LORA_ALIGNED')
        assert scene_initial_actions(old,'TRAIN_CPU_0','cpu') is None
        assert 'initial_root_scheme' not in old['augustus']
        from starVLA.rl.d46_augustus.config import kernel_identity
        assert kernel_identity(old)!=kernel_identity(cfg)
        bad=dict(cfg,experiment_id='AUG4_LORA_ALIGNED')
        with pytest.raises(ValueError,match='protocol'):validate(bad)
    finally:torch.set_num_threads(old_threads)

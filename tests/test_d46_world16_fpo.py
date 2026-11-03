"""Real Tiny-native FPO MC4 / world16 recovery; CPU only, not full IL46."""
import copy,json,os
from pathlib import Path
import pytest,torch
from tools.d46_world16.qualify_native import make_actor,condition,SyntheticService,qualify
from starVLA.rl.d44_native_rl.config import defaults46
from starVLA.rl.d44_native_rl.batch_runtime import collect_scene
from starVLA.rl.d44_native_rl.recovery import material_digest
from starVLA.rl.d46_world16.planning import fixed_scene_rng_plan,assert_scene_rng_end
from starVLA.rl.d46_world16.native import World16Boundary,validate_supported_recipe,FPO_PROTOCOL


def test_original_FPO_MC4_actual_draw_sequence_and_energy_not_Gaussian(monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *a,**kw:None)
    torch.manual_seed(42);torch.set_num_threads(1)
    cfg=defaults46('D3_FPO');cfg['checkpoint_policy']='all_off'
    actor=make_actor(cfg);g=torch.Generator().manual_seed(420003)
    preview=fixed_scene_rng_plan(g.get_state(),'D3_FPO',mc=cfg['mc_samples'])
    for j in range(8):
        assert torch.equal(g.get_state(),preview['starts'][j])
        before=g.get_state().clone()
        scene=collect_scene(actor,condition(actor,j),f'TRAIN_CPU_{j}','TRAIN_log',SyntheticService(),g,cfg,0.,{'run':'CPU','update':0,'collection':1})
        assert_scene_rng_end(g,preview['ends'][j])
        expected=torch.Generator().set_state(before)
        for _ in range(4):torch.randn((16,8,3),generator=expected)
        sigma=.2+.6*torch.rand((4,16),generator=expected)
        eps=torch.randn((4,16,8,3),generator=expected)
        assert torch.equal(sigma,scene['fm_sigma'])
        clean=scene['chain'][:,-1]
        assert torch.equal(scene['fm_state'],(1-sigma[...,None,None])*clean[None]+sigma[...,None,None]*eps)
        assert torch.equal(scene['fm_target'],eps-clean[None])
        assert torch.equal(expected.get_state(),g.get_state())
        if j==0:
            scene.update(advantages=torch.linspace(-1,1,16),beta=.001,expect_fresh=True)
            loss,metrics=actor(scene)
            assert metrics['fpo_joint_logratio_min']==0 and metrics['fpo_joint_logratio_max']==0
            assert metrics['fresh_ratio_error']==0
            loss.backward()
            assert any(p.grad is not None and p.grad.abs().max()>0 for p in actor.parameters()if p.requires_grad)
            assert all(torch.isfinite(p.grad).all()for p in actor.parameters()if p.grad is not None)
    assert torch.equal(g.get_state(),preview['next_virtual_lane_state'])
    with pytest.raises(ValueError,match='MC4'):fixed_scene_rng_plan(g.get_state(),'D3_FPO',mc=2)
    with pytest.raises(ValueError,match='MC4'):validate_supported_recipe(dict(cfg,mc_samples=2))
    with pytest.raises(ValueError,match='not admitted'):validate_supported_recipe(defaults46('D2'))


@pytest.fixture(scope='module')
def fpo_qualified(tmp_path_factory):
    root=Path(os.environ['WORLD16_FPO_CPU_QUALIFICATION_ROOT'])if 'WORLD16_FPO_CPU_QUALIFICATION_ROOT'in os.environ else tmp_path_factory.mktemp('FPO16')/'native'
    return root,qualify(root,arm='D3_FPO')


def test_actual_FPO_native8_to16_PENDING_MC4_Adam_RNG_exact(fpo_qualified):
    root,result=fpo_qualified
    assert result['status']=='PASS'and result['arm']=='D3_FPO'and result['mc_samples']==4
    assert result['actual_fresh_FPO_energy_ratio_checked']
    assert result['READY16_parameters_Adam_scheduler_exact']and result['all16_adapters_Adam_scheduler_RNG_scene_stream_virtual8_and_material_bitexact']
    assert result['post_native_B1_parity_all16']and result['terminal_JSON_all16']
    identity=json.loads((root/'split16/identity.json').read_text())
    assert identity['native_world_transition']['flow_energy_protocol']==FPO_PROTOCOL
    pending=root/'split16/checkpoints/update000003_inner1'
    for rank in range(16):
        own=torch.load(pending/f'rank{rank}.pt',weights_only=False)
        assert len(own['rollout']['scenes'])==4
        assert all(r['fm_state'].shape==(4,16,8,3)and r['old_energy'].shape==(16,)for r in own['rollout']['scenes'])
        assert own['state']['updates']==3 and own['state']['next_inner']==1


def test_FPO_MC_material_corruption_rejected_after_consistent_digest(fpo_qualified):
    root,_=fpo_qualified;cp=root/'split16/checkpoints/update000003_inner1'
    own=torch.load(cp/'rank0.pt',weights_only=False);identity=json.loads((cp/'identity.json').read_text())
    for key in('fm_state','fm_sigma','fm_target','old_energy'):
        bad=copy.deepcopy(own['rollout']);bad['scenes'][0][key]=bad['scenes'][0][key][:1]
        bad['digest']=material_digest(bad)
        with pytest.raises(ValueError,match='FPO MC4 material'):World16Boundary.validate_boundary(own['state'],bad,identity)
    bad=copy.deepcopy(own['rollout']);bad['scenes'][0]['fm_sigma'].fill_(.99);bad['digest']=material_digest(bad)
    with pytest.raises(ValueError,match='sigma/energy'):World16Boundary.validate_boundary(own['state'],bad,identity)

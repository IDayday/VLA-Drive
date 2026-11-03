"""Bounded CPU contracts: real parent metadata, M1/pending, canonical group A."""
import copy
from types import SimpleNamespace
import pytest
import torch
from starVLA.rl.qwen_world_compare.contracts import fingerprint
from starVLA.rl.d44_native_rl.recovery import material_digest
from starVLA.rl.d46_augustus.config import defaults,validate,kernel_identity,SCHEMA
from starVLA.rl.d46_augustus.runtime import AugustusExtension,AugustusBoundary,FixedKL,advantages,credit


def test_two_explicit_configs_deployment_grid_ownworld_and_M1_are_strict():
    for name,scope,batch in [('AUG1_FULL','full_text_action',128),('AUG2_LORA','lora_text_action',64)]:
        cfg=defaults(name)
        assert cfg['augustus']['scope']==scope and cfg['global_scenes']==batch and cfg['inner_updates']==1
        assert cfg['steps']==10 and cfg['shift']==5 and cfg['augustus']['training_steps']==3
        assert cfg['augustus']['training_sigma_grid']==[1.,2/3,1/3,0.]
        ext=AugustusExtension();ext.cfg=cfg;ext.saved_identity=None
        for world in(8,16,32):ext.validate_world(world)
        with pytest.raises(ValueError,match='world8/16/32'):ext.validate_world(64)
        ext.saved_identity={'world':8}
        with pytest.raises(ValueError,match='silently change world'):ext.validate_world(16)
        for path,value in [('group',16),('learning_rate',2e-6),('steps',3),('inner_updates',4)]:
            bad=copy.deepcopy(cfg);bad[path]=value
            with pytest.raises(ValueError):validate(bad)
        bad=copy.deepcopy(cfg);bad['augustus']['second_global_normalization']=True
        with pytest.raises(ValueError,match='protocol'):validate(bad)


def test_canonical_group_sample_std_padding_and_fixed_KL_restore():
    scores=torch.tensor([[0.,.1,.2,.3,.4,.5,.6,1.],[1.]*8,[.2]*8],dtype=torch.float64)
    scenes=[dict(chain=torch.zeros(8,4,8,3),records=[{'metrics':{'score':float(x),'no_at_fault_collisions':1.,'drivable_area_compliance':1.,'time_to_collision_within_bound':1.}}for x in row])for row in scores]
    raw,feasible=credit(scenes,None,0.)
    a,std=advantages(raw,torch.tensor([True,True,False]),8,1)
    expected=((scores-scores.mean(1,keepdim=True))/(scores.std(1,unbiased=True,keepdim=True)+1e-6)).float();expected[2].zero_()
    torch.testing.assert_close(a,expected,atol=0,rtol=0)
    assert torch.isfinite(a).all()and a[1:].count_nonzero()==0 and feasible.all()
    assert not torch.allclose(a[0],(raw/std).float()[0])  # No second global scaling.
    cfg=defaults('AUG2_LORA');dual=FixedKL(cfg)
    for value in(0.,.1,100.):assert dual.update(value)==.001
    restored=FixedKL(cfg);restored.load_state_dict(dual.state_dict())
    assert restored.collections==3 and restored.beta==.001
    with pytest.raises(ValueError,match='fixed KL'):restored.load_state_dict(dict(dual.state_dict(),beta=.01))


def pending_fixture(cfg):
    m=cfg['inner_updates'];c=2;i=0 if m==1 else 1;u=(c-1)*m+i
    identity=dict(recipe=cfg,world=8,algorithm_schema=SCHEMA,augustus_protocol_sha256=fingerprint(cfg['augustus']))
    identity['sha256']=fingerprint(identity)
    dual=FixedKL(cfg);dual.update(0.)
    state=dict(updates=u,collections=c,next_inner=i,phase='ROLLOUT_COMMITTED'if i==0 else'INNER_COMMITTED',scene_cursor=128,
        raw_scene_exposures=128,valid_scene_exposures=128,padding_scene_exposures=0,new_candidates=1024,valid_PG_candidates=1024,
        extra_B1_queries=0,adaptive_G32_queries=0,adaptive_kl=dual.state_dict())
    scenes=[];grid=torch.tensor(cfg['augustus']['training_sigma_grid'],dtype=torch.float32)
    score=torch.linspace(.1,.8,8,dtype=torch.float64);a=((score-score.mean())/(score.std(unbiased=True)+1e-6)).float()
    for j in range(8):
        scene=dict(chain=torch.zeros(8,4,8,3),old_mean=torch.zeros(8,3,8,3),ref_mean=torch.zeros(8,3,8,3),std=torch.ones(8,3,8,3)*.03,
            old_clean_x=torch.zeros(8,3,8,3),ref_clean_x=torch.zeros(8,3,8,3),advantages=a.clone(),timesteps=grid,deltas=grid[1:]-grid[:-1],
            augustus_kernel=kernel_identity(cfg),records=[{'metrics':{'score':float(x)}}for x in score],padding=False,beta=.001,
            behavior_version={'run':identity['sha256'],'update':u-i})
        scenes.append(scene)
    batch=dict(scenes=scenes,behavior_update=u-i,global_std=.2,valid_scenes=64,epoch=0.);batch['digest']=material_digest(batch)
    opt={'state':{0:{'step':torch.tensor(float(u))}}};sch={'last_epoch':u}
    return state,batch,identity,opt,sch


def test_native_M1_M2_pending_and_material_Adam_controller_tamper_refused(tmp_path):
    for m in(1,2):
        cfg=defaults('AUG2_LORA',inner_updates=m);s,b,i,opt,sch=pending_fixture(cfg)
        row=AugustusBoundary.validate_boundary(s,b,i,opt,sch)
        assert row['pending']and row['updates']==s['updates']and row['augustus_pending_material_sha256']
        # Native torch roundtrip retains the pending material/counters, no model.
        path=tmp_path/f'M{m}.pt';torch.save(dict(state=s,rollout=b,optimizer=opt,scheduler=sch),path)
        restored=torch.load(path,weights_only=False)
        assert AugustusBoundary.validate_boundary(restored['state'],restored['rollout'],i,restored['optimizer'],restored['scheduler'])==row
        bad=copy.deepcopy(b);bad['scenes'][0]['old_clean_x'][0,0,0,0]+=1
        with pytest.raises(ValueError,match='material|probabilities|advantages'):AugustusBoundary.validate_boundary(s,bad,i,opt,sch)
        bad=copy.deepcopy(b);bad['scenes'][0]['advantages']*=.5;bad['digest']=material_digest(bad)
        with pytest.raises(ValueError,match='group normalization'):AugustusBoundary.validate_boundary(s,bad,i,opt,sch)
        with pytest.raises(ValueError,match='optimizer steps'):AugustusBoundary.validate_boundary(s,b,i,{'state':{}},sch)
        bad=copy.deepcopy(s);bad['world16_virtual_sampler']={}
        with pytest.raises(ValueError,match='virtual sampler'):AugustusBoundary.validate_boundary(bad,b,i,opt,sch)

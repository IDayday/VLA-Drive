"""Meaningful CPU math/real TinyScope recovery; no driving-score claims."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml

from test_world_performance_nominal import actual,primary,Service,Capture
from starVLA.rl.qwen_world_performance import nominal_improvement as N,ode_target as O
from starVLA.rl.qwen_world_performance.adaptive_group import collect_extra_if_needed
from starVLA.rl.qwen_world_verified_quality import calibration as C
from starVLA.rl.qwen_world_verified_quality.runtime import VerifiedQualityLoss
from starVLA.rl.qwen_world_safety_exploration.retention_material import digest,sealed,verify_seal
from starVLA.rl.qwen_world_structured_progress.geometry import propose_family
from starVLA.rl.qwen_world_structured_progress import material as M
from starVLA.rl.qwen_world_structured_progress.recipe import build_recipe,validate_recipe,PROTOCOL,PROTOCOL_SHA256
from starVLA.rl.qwen_world_structured_progress.runtime import StructuredProgressLoss,new_state


class Reader:
    identity=dict(schema='CPU_fixture_source_not_actual_TRAIN')
    def __init__(self,valid=True):self.valid=valid
    def current_ego(self,t,log):return torch.tensor([1.,0.,0.,0.])
    def read(self,t,log):
        x=torch.zeros(8,3);x[:,0]=torch.arange(1,9)*.5
        return sealed(dict(valid=self.valid,requested=x if self.valid else None,reason=None if self.valid else 'invalid_time',
            token=t,log=log),'source_sha256')
    def validate_saved(self,source,t,log):
        verify_seal(source,'source_sha256');assert source['token']==t and source['log']==log
        return source


class StructuredService(Service):
    def __init__(self):super().__init__(primary_high=False,extra_high=False,nominal_ep=.75)
    def score(self,job):
        synthetic=len(job[2])>1 and self.group_calls>=2
        result=super().score(job)
        if synthetic:
            for r in result:
                r['metrics'].update(no_at_fault_collisions=1.,drivable_area_compliance=1.,time_to_collision_within_bound=1.,comfort=1.,ego_progress=1.,score=1.)
                r['readouts'].update(canonical_PDMS=1.,train_reward_source=1.)
        return result


def attached():
    torch.set_num_threads(1);a,c,cfg=actual();service=StructuredService();leaf=primary(a,c,service)
    reader=Reader();M.bind_reader(reader);audit=a._probability_audit;rng=torch.get_rng_state().clone()
    nominal=O.probe_nominal(a,c,'t0','TRAIN_log',service,leaf['behavior_version'])
    extra=collect_extra_if_needed(a,leaf,nominal,service,rank=0)
    pool=M.collect_pool(leaf,nominal,extra,service,reader,0)
    N.attach_material(a,leaf,c,service,exposure_epoch=0.,rank=0,config=N.QUALITY_CONFIG,nominal=nominal,extra=extra,
        teacher_selector=lambda x,y,z,k:M.select_teacher(x,y,z,k,pool))
    p={k:v for k,v in leaf['nominal_improvement'].items()if k!='material_sha256'};p['structured_progress']=pool
    leaf['nominal_improvement']=sealed(p,'material_sha256');a._probability_audit=audit
    assert torch.equal(rng,torch.get_rng_state())
    return a,c,cfg,service,leaf


def test_real_scored112_proposal_bits_and_initial_velocity_and_invalid_GT_only_skips_one_family():
    fixture=json.loads((Path(__file__).parent/'fixtures/structured_progress_diagnostic7.json').read_text())
    for case in fixture['cases']:
        hashes=[]
        for family in ('GT_path','B1_path'):
            requests,clean,desc=propose_family(case['bases'][family],case['ego'][0],family)
            hashes.extend(hashlib.sha256(str((str(x.dtype),x.shape)).encode()+x.tobytes()).hexdigest()for x in requests)
            assert requests.shape==(8,8,3) and clean.shape==(8,8,3)
            assert all(np.allclose(d['initial_velocity_xy'],[case['ego'][0],0.],rtol=0,atol=1e-12) for d in desc)
        assert hashes==case['expected_requested_typed_sha256']
    straight=torch.zeros(8,3);straight[:,0]=torch.arange(1,9)*.5
    generation=M.generate(dict(token='t0',log='TRAIN_log'),dict(requested=straight),Reader(valid=False))
    assert generation['requests'].shape==(8,8,3) and all(d['family']=='B1_path' for d in generation['descriptors'])
    assert generation['family_status']['GT_path'].startswith('GT_source_rejected')


def test_actual_scope_teacher_only_fourinner_pending_Adam_rng_and_calibration_restore():
    a,c,cfg,service,r=attached();p=M.validate_nominal(r,rank=0,expected_epoch=0.)
    assert p['enabled'] and p['active_alpha']==.05 and p['teacher']['generation']=='explicit_synthetic_not_a_model_sample'
    assert service.counts==[16,1,16,16] and p['structured_progress']['queries']==16
    primary_before=digest({k:r[k]for k in ('chain','old_logp','rewards','advantages','adv_mask')})
    a._loss=StructuredProgressLoss(cfg);a._loss.auxiliary_state=C.new_calibration_state()
    opt=torch.optim.AdamW([p for p in a.parameters()if p.requires_grad],lr=1e-5)
    for _ in range(2):
        opt.zero_grad(set_to_none=True);loss,metrics=a(r);loss.backward();opt.step()
    buffer=io.BytesIO();torch.save(dict(actor=a.state_dict(),optimizer=opt.state_dict(),rollout=r,
        calibration=a._loss.auxiliary_state,RNG=torch.get_rng_state(),structured_ledger=new_state(0)),buffer)
    buffer.seek(0);saved=torch.load(buffer,weights_only=False);right,_,_=actual();right.load_state_dict(saved['actor'])
    right._loss=StructuredProgressLoss(cfg);right._loss.auxiliary_state=saved['calibration']
    ropt=torch.optim.AdamW([p for p in right.parameters()if p.requires_grad],lr=1e-5);ropt.load_state_dict(saved['optimizer'])
    M.validate_nominal(saved['rollout'],rank=0,expected_epoch=0.)
    for _ in range(2):
        opt.zero_grad(set_to_none=True);ropt.zero_grad(set_to_none=True)
        left,_=a(r);other,_=right(saved['rollout']);assert torch.equal(left,other)
        left.backward();other.backward()
        for p,q in zip(a.parameters(),right.parameters()):
            if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all() and torch.equal(p.grad,q.grad)
        for prefix in ('action_expert.','scene_provider.','predictive_provider.'):
            assert sum(float(p.grad.norm())for n,p in a.named_parameters()if p.requires_grad and n.startswith(prefix))>0
        opt.step();ropt.step();assert digest(opt.state_dict())==digest(ropt.state_dict())
    assert primary_before==digest({k:r[k]for k in ('chain','old_logp','rewards','advantages','adv_mask')})
    assert a._loss.auxiliary_state['effective_updates']==4
    for field in ('queries','requests','origin','requires_grad'):
        bad=copy.deepcopy(r);bp=bad['nominal_improvement'];pool=bp['structured_progress']
        if field=='queries':pool['queries']-=1
        elif field=='requests':pool['generation']['requests'][0,0,0]+=.01
        elif field=='origin':bp['teacher']['origin']='primary_G16'
        else:pool['generation']['clean'].requires_grad_(True)
        bp['structured_progress']=sealed({k:v for k,v in pool.items()if k!='material_sha256'},'material_sha256')
        bad['nominal_improvement']=sealed({k:v for k,v in bp.items()if k!='material_sha256'},'material_sha256')
        with pytest.raises(ValueError):M.validate_nominal(bad,rank=0,expected_epoch=0.)


def test_default_S6_material_loss_gradients_bitexact_to_frozen8067_and_recipe_is_one_pool_only():
    from test_world_verified_quality import material
    a,c,cfg,service,r=material();p=r['nominal_improvement']
    path=Path('/mnt/project/qwen-world-performance-rl-20261103/runtime/verified_quality_v2/starVLA/rl/qwen_world_performance/nominal_improvement.py')
    spec=importlib.util.spec_from_file_location('starVLA.rl.qwen_world_performance.frozen8067_fixture',path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    left,right=copy.deepcopy(r),copy.deepcopy(r)
    for leaf in (left,right):leaf.pop('nominal_improvement')
    old.attach_material(a,left,c,service,exposure_epoch=0.,rank=0,config=old.QUALITY_CONFIG,nominal=p['nominal'],extra=p['extra'])
    N.attach_material(a,right,c,service,exposure_epoch=0.,rank=0,config=N.QUALITY_CONFIG,nominal=p['nominal'],extra=p['extra'])
    assert digest(left['nominal_improvement'])==digest(right['nominal_improvement'])
    OldLoss=type('OldS6Loss',(old.NominalLossMixin,Capture),dict(nominal_config=old.QUALITY_CONFIG,calibrate_nominal_components=staticmethod(C.calibrate_components)))
    gradients=[];values=[]
    for klass,leaf in ((OldLoss,left),(VerifiedQualityLoss,right)):
        a._loss=klass(cfg);a._loss.auxiliary_state=C.new_calibration_state();value,metrics=a(leaf)
        gradients.append(torch.autograd.grad(value,[p for p in a.parameters()if p.requires_grad]));values.append((value,metrics))
    assert torch.equal(values[0][0],values[1][0]) and digest(values[0][1])==digest(values[1][1])
    assert all(torch.equal(x,y)for x,y in zip(*gradients))
    base=yaml.safe_load(Path('configs/qwen_world_verified_quality/S6_VERIFIED_QUALITY.yaml').read_text());new=build_recipe(base)
    assert new['grpo']==base['grpo'] and new['learning_rate']==base['learning_rate'] and new['research']['performance']==base['research']['performance']
    validate_recipe(new)
    bad=copy.deepcopy(new);bad['grpo']['sample']['group_size']=32
    with pytest.raises(ValueError):validate_recipe(bad)


def test_qualification_plan_and_comparator_keep_physical_time_and_noise(tmp_path):
    from tools.qwen_world_structured_progress.qualify_native import prepare,scientific_pending
    cfg=Path('configs/qwen_world_structured_progress/S7_STRUCTURED_PROGRESS.yaml').resolve()
    plan=prepare(tmp_path/'qual',cfg.parents[2],cfg)
    assert len(plan['stages'])==3 and plan['world']==8 and plan['B']==64
    assert all('tools.qwen_world_structured_progress.train'in s['command'] and '--qualification-force-auxiliary'not in s['command']for s in plan['stages'])
    original=dict(invocation='a'*32,noise=torch.arange(4),actual_future_seconds=[.5,1.],official_first_ttc_infraction_seconds=1.4,
        record=dict(scoring_seconds=.01,metrics=dict(score=1.)),seconds=dict(forward=1.,scoring=2.),material_sha256='b'*64)
    other=copy.deepcopy(original);other['invocation']='c'*32;other['record']['scoring_seconds']=4.;other['seconds']['scoring']=9.
    assert digest(scientific_pending(original))==digest(scientific_pending(other))
    other['actual_future_seconds'][0]=.6
    assert digest(scientific_pending(original))!=digest(scientific_pending(other))


def test_real_TRAIN_reader_exact_GT_provenance_time_rejection_and_saved_gate_not_refreshed():
    from starVLA.rl.qwen_world_research_data import ResearchSceneData
    from starVLA.rl.qwen_world_structured_progress.gt_reader import GTReader
    root=Path('/mnt/project/qwen-world-rl-optimization-20261030/data/NAVTRAIN')
    if not root.exists():pytest.skip('Real TRAIN source not available; portable geometry fixture is tested separately')
    data=ResearchSceneData(root,role='NAVTRAIN');reader=GTReader(data)
    fixture=json.loads((Path(__file__).parent/'fixtures/structured_progress_diagnostic7.json').read_text())
    for case in fixture['cases']:
        t=case['token'];source=reader.read(t,data.logs[t]);assert source['valid']
        assert torch.equal(source['requested'],data.target(t));reader.validate_saved(source,t,data.logs[t])
    t=fixture['cases'][0]['token'];source=reader.read(t,data.logs[t])
    reader.logs.clear();reader.raw_root=Path('/nonexistent_RAW_on_resume')
    assert reader.validate_saved(source,t,data.logs[t])['valid'] #No new raw-source I/O for saved valid material.
    changed=copy.copy(data);changed.records=dict(data.records);record=copy.deepcopy(data.records[t]);record['raw_future_dt_seconds'][0]+=1.
    changed.records[t]=record;invalid=GTReader(changed).read(t,data.logs[t])
    assert not invalid['valid'] and invalid['requested'] is None and 'invalid_fixed_time' in invalid['reason']
    #A later source recovery must not upgrade a pending unavailable-GT pool.
    dummy=Reader(valid=False);straight=torch.zeros(8,3);straight[:,0]=torch.arange(1,9)*.5
    before=M.generate(dict(token='t0',log='TRAIN_log'),dict(requested=straight),dummy)
    dummy.valid=True
    after=M.generate(dict(token='t0',log='TRAIN_log'),dict(requested=straight),dummy,saved_GT_source=before['GT_source'])
    assert digest(before)==digest(after) and len(after['requests'])==8


def test_native_cost_ledger_all8rank_and_foreign_rank_rejection(monkeypatch):
    from starVLA.rl.qwen_world_structured_progress import recovery as R
    from starVLA.rl.qwen_world_verified_quality import recipe as S6
    reader=Reader();M.bind_reader(reader)
    cfg=build_recipe(yaml.safe_load(Path('configs/qwen_world_verified_quality/S6_VERIFIED_QUALITY.yaml').read_text()))
    identity=dict(recipe=cfg,structured_progress_protocol=PROTOCOL,structured_progress_protocol_sha256=PROTOCOL_SHA256,
        verified_quality_protocol=S6.PROTOCOL,quality_protocol_sha256=S6.QUALITY_PROTOCOL_SHA256,
        performance_implementation_protocols=dict(nominal=N.QUALITY_CONFIG,calibration=C.PROTOCOL),structured_GT_source_identity=reader.identity)
    called=[]
    def base(state,window,identity,*args,**kwargs):
        called.append(kwargs);return dict(rank=kwargs['expected_rank'],pending=False)
    monkeypatch.setattr(R.base,'validate_boundary',base);monkeypatch.setattr(R.base,'validate_ranks',lambda s:s)
    zero=dict(queries=0,triggered=0,enabled=0,GT_rejected=0,GT_qualified=0,B1_qualified=0)
    total=dict(queries=16,triggered=1,enabled=1,GT_rejected=0,GT_qualified=8,B1_qualified=8)
    summaries=[]
    for rank in range(8):
        own=new_state(rank);own['ledger']=[dict(collection=1,local=total if rank==0 else zero,global_counts=total)]
        state=dict(collections=1,structured_progress=own)
        summaries.append(R.validate_boundary(state,None,identity,expected_rank=rank))
    R.validate_ranks(summaries)
    assert all(v['recipe_validator'] is validate_recipe and v['nominal_validator'] is M.validate_nominal and v['calibration_validator'] is C.validate_calibration_state for v in called)
    bad=copy.deepcopy(summaries);bad[2]['structured_total_local_queries']=1
    with pytest.raises(ValueError,match='sum'):R.validate_ranks(bad)
    own=new_state(3);own['ledger']=[dict(collection=1,local=zero,global_counts=total)]
    with pytest.raises(ValueError,match='rank-private'):R.validate_boundary(dict(collections=1,structured_progress=own),None,identity,expected_rank=4)

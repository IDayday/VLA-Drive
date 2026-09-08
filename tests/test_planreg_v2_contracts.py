from types import SimpleNamespace
import copy
import json
import torch
import pytest
from navsim.agents.EpisodeDrive.planreg_v2.runtime import ExactExposureSampler,same_batch_gradient_audit
from navsim.agents.EpisodeDrive.planreg_v2.data import reject_cached_representations,preprocess_fixed_layout
from navsim.agents.EpisodeDrive.planreg_v2.optimizer import build_optimizer,audit_adam_state,multiplier
from navsim.agents.EpisodeDrive.planreg_v2.normalizers import measured_statistics
from navsim.agents.EpisodeDrive.planreg_v2.checkpoint import convert_physical_output_head
from navsim.agents.EpisodeDrive.planreg_v2.language import SoftTaskQueries,inject_language_lora
from navsim.agents.EpisodeDrive.planreg_v2.predictor import ActionCausalPredictor
from navsim.agents.EpisodeDrive.planreg_v2.motion import CandidateKinematicsCodec
from navsim.agents.EpisodeDrive.planreg_v2.losses import trajectory_loss


def test_exact_padded_exposure_and_resume():
    all_ids=[]
    for rank in range(16):
        sampler=ExactExposureSampler(103288,128,16,rank,3)
        assert sampler.steps == 807 and len(sampler) == 6456
        all_ids.extend(list(sampler))
        resume=ExactExposureSampler(103288,128,16,rank,3,start_step=12)
        assert list(resume) == list(sampler)[12*8:]
    assert len(set(all_ids)) == 103288 and len(all_ids) == 103296


def test_dynamic_cache_guard_recursive():
    for field in ('last_hidden_state','future_registers','ema_registers','semantic_tokens'):
        with pytest.raises(ValueError): reject_cached_representations({'features':{field:torch.zeros(1)}})


def test_fixed_layout_applies_to_different_future_image_size(tmp_path):
    from PIL import Image
    Image.new('RGB',(1920,1080),(10,30,50)).save(tmp_path/'current.png')
    Image.new('RGB',(1280,720),(40,20,5)).save(tmp_path/'future.png')
    x,metadata=preprocess_fixed_layout(tmp_path/'current.png')
    y,future_metadata=preprocess_fixed_layout(tmp_path/'future.png',metadata)
    assert x.shape==y.shape and torch.equal(metadata,future_metadata)


def test_real_llm_class_small_config_padding_and_gradient_boundary():
    # A small genuine HF Qwen2 class checks API semantics, NOT a full InternVL validation claim.
    from transformers import Qwen2Config,Qwen2ForCausalLM
    cfg=Qwen2Config(vocab_size=31,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
    lm=Qwen2ForCausalLM(cfg).eval().requires_grad_(False)
    assert len(inject_language_lora(lm,4,8))==8
    queries=SoftTaskQueries(32,16)
    prefix=torch.randn(2,5,32,requires_grad=True)
    mask=torch.ones(2,5,dtype=torch.bool)
    result=queries(lm,prefix,mask)
    padded=torch.cat((prefix,torch.randn(2,4,32)*100),1)
    masked=torch.cat((mask,torch.zeros(2,4,dtype=torch.bool)),1)
    torch.testing.assert_close(result,queries(lm,padded,masked))
    result.square().mean().backward()
    assert prefix.grad.norm()>0 and queries.queries.grad.norm()>0
    assert all(p.grad is None for n,p in lm.named_parameters() if '.lora_' not in n)
    assert any(p.grad is not None and p.grad.norm()>0 for n,p in lm.named_parameters() if '.lora_b.' in n)


def test_physical_head_migration_and_four_stage_gradients():
    normalizer=measured_statistics([('x',torch.randn(8,3),torch.ones(8,dtype=torch.bool)),('y',torch.randn(8,3),torch.ones(8,dtype=torch.bool))],'train','unit')
    w,b=torch.randn(24,32),torch.randn(24)
    nw,nb=convert_physical_output_head(w,b,normalizer)
    hidden=torch.randn(3,32)
    restored=(hidden@nw.T+nb)*normalizer.std.flatten()+normalizer.mean.flatten()
    torch.testing.assert_close(restored,hidden@w.T+b,atol=3e-6,rtol=1e-5)
    stages=torch.randn(4,2,64,8,3,requires_grad=True)
    target=dict(trajectory=torch.zeros(2,8,3),trajectory_valid=torch.ones(2,8,dtype=torch.bool),
                trajectory_long=torch.ones(2,8,3),trajectory_long_valid=torch.tensor([1,0]).bool())
    loss,_=trajectory_loss(stages,target,normalizer.std)
    loss.backward()
    assert all(stages.grad[i].norm()>0 for i in range(4))


@pytest.mark.parametrize('k',[1,8,64])
def test_candidate_rollout_interface_chunk_invariance(k):
    torch.manual_seed(8)
    model=ActionCausalPredictor(dim=32,layers=2,heads=4,ffn=64).eval()
    p=torch.randn(1,k,8,3)
    codec=CandidateKinematicsCodec()(p,torch.zeros(1,4),torch.arange(1,9)*.5,torch.ones(8,dtype=torch.bool))
    z=torch.randn(1,16,32); geometry=torch.randn(1,16,5); valid=torch.ones(1,16,dtype=torch.bool); semantic=torch.randn(1,16,32)
    with torch.no_grad():
        a,coverage=model.rollout_candidates(z,codec,geometry,valid,semantic,chunk_size=7)
        b,_=model.rollout_candidates(z,codec,geometry,valid,semantic,chunk_size=64)
    assert a.shape==(1,k,3,16,32) and coverage.all()
    torch.testing.assert_close(a,b,atol=3e-6,rtol=1e-5)


def test_same_batch_grad_does_not_pollute_grad_or_rng():
    p=torch.nn.Parameter(torch.randn(4)); p.grad=torch.ones_like(p)
    rng=torch.get_rng_state().clone()
    report=same_batch_gradient_audit(p.square().sum(),(p-1).square().sum(),[('backbone.model.vision_model.q_lora_a.weight',p)],.1)
    assert report['vision_lora']['plan_norm']>0
    assert torch.equal(p.grad,torch.ones_like(p)) and torch.equal(rng,torch.get_rng_state())


def test_adam_groups_scheduler_resume_fp32():
    class SmallAgent(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.scene_memory=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.LayerNorm(4))
    a=SmallAgent(); opt,scheduler,summary=build_optimizer(a,100)
    for _ in range(8):
        opt.zero_grad(); a.scene_memory(torch.ones(2,3)).square().mean().backward(); opt.step(); scheduler.step()
    audit_adam_state(opt)
    b=SmallAgent(); b.load_state_dict(a.state_dict()); opt2,scheduler2,_=build_optimizer(b,100)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict())); scheduler2.load_state_dict(copy.deepcopy(scheduler.state_dict()))
    for _ in range(12):
        opt.step(); scheduler.step(); opt2.step(); scheduler2.step()
        assert scheduler.get_last_lr()==scheduler2.get_last_lr()
    assert multiplier(0,100)==.01 and multiplier(5,100)==1 and multiplier(99,100)==.1


def test_vqa_config_does_not_resolve_overridden_base_environment(monkeypatch):
    from navsim.agents.EpisodeDrive.planreg_v2.runtime import load_config
    monkeypatch.delenv('PLANREG_BASE_VLM_PATH',raising=False)
    monkeypatch.setenv('PLANREG_VQA_VLM_PATH','/unit/vqa')
    monkeypatch.setenv('PLANREG_V2_NORMALIZER','/unit/stats.json')
    config=load_config('navsim/planning/script/config/common/agent/planreg_wm_v2_vqa.yaml')
    assert config['vlm_path']=='/unit/vqa' and config['variant']=='driving_vqa'

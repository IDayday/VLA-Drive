"""Native precision override preserves identities and reaches every text path."""
import copy
import importlib.util
from pathlib import Path
import torch
from starVLA.rl.d46_augustus.config import defaults,validate
from starVLA.rl.d46_augustus.text_precision import install_text_action_precision,PrecisionQwenFlowAdapter


def test_native_prefix_suffix_and_reference_override_outer_bfloat16():
    spec=importlib.util.spec_from_file_location('native_precision_fixture',Path(__file__).with_name('test_d46_augustus_actor.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        actor,_,cfg,_,condition=m.tiny('AUG7_NATIVE_FP32')
        for expert in (actor.action_expert,actor.ref_action_expert):
            assert isinstance(expert.decision,PrecisionQwenFlowAdapter)
        decision=actor.action_expert.decision
        before={n:id(p) for n,p in decision.named_parameters()};rng=torch.get_rng_state().clone()
        install_text_action_precision(decision,'float32')
        assert before=={n:id(p) for n,p in decision.named_parameters()}
        assert torch.equal(rng,torch.get_rng_state())
        seen=[]
        hook=decision.language_model.layers[0].self_attn.q_proj.register_forward_hook(lambda _,args,result:seen.append(result.dtype))
        with actor._context_call(condition) as material:
            prefix=actor._prepare(material,False)
            value=actor.predict_clean(torch.randn(8,8,3),torch.full((8,),.5),prefix)
            assert all(k.dtype==v.dtype==torch.float32 for k,v in prefix.tensors.layer_kv)
        value.square().mean().backward();hook.remove()
        assert len(seen)==3 and set(seen)=={torch.float32}  # C, P and A
        assert any(p.grad is not None and p.grad.count_nonzero() for n,p in actor.named_parameters() if '.lora_B' in n)
        assert all(p.grad is None for p in actor.ref_action_expert.parameters())
        assert validate(cfg)['augustus']['text_action_precision']=='float32'
    finally:torch.set_num_threads(old)

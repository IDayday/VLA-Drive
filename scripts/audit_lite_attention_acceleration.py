#!/usr/bin/env python3
"""Real InternVL eager/split-SDPA audit, not a final-PDMS equivalence claim."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import torch
from hydra.utils import instantiate
from smoke_task_future_lite_real import agent_config
from planreg_audit_runtime import to_device_non_paths
from navsim.planning.training.dataset import CacheOnlyDataset, drivevla_cached_collate


def difference(reference, result):
    a, b = reference.detach().float().flatten(), result.detach().float().flatten()
    return dict(max_abs=float((a-b).abs().max()),
                relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-20)),
                cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=0)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'shared-init', 'cache-root', 'scene-manifest', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='RUNNING', kind='real_model_numerical_backend_audit',
                  source_checkpoint=str(a.checkpoint.resolve()),
                  gates=dict(fp32_block_relative_l2=1e-4,
                             fp32_gradient_relative_l2=1e-3),
                  final_pdms_equivalence='NOT_ESTABLISHED', blocks=[], scenes=[])
    def save():
        a.output.write_text(json.dumps(report, indent=2)+'\n')
    save()
    torch.manual_seed(20260905)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = agent_config('base', a.shared_init)
    agent = instantiate(cfg)
    agent.initialize()
    checkpoint = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    state = {k.removeprefix('agent.'): v for k,v in checkpoint['state_dict'].items()}
    agent.load_state_dict(state, strict=True)
    report['checkpoint_optimizer_step'] = checkpoint['global_step']
    del checkpoint, state
    agent.to('cuda').eval()
    adapter = agent.backbone.planning_register_adapter
    vision = agent.backbone.model.vision_model
    report['vision_class'] = str(type(vision))
    report['trainable_dtypes'] = sorted({str(p.dtype) for p in agent.parameters() if p.requires_grad})
    rows = [r for r in json.loads(a.scene_manifest.read_text())['rows'] if r['partition']=='train']
    selected = [next(r for r in rows if r['category']==kind)
                for kind in ('turn','stop','crowded','boundary')]
    dataset = CacheOnlyDataset(str(a.cache_root), agent.get_feature_builders(), agent.get_target_builders(),
        log_names=[r['log'] for r in selected], preprocess_images=True, preprocess_future_images=True,
        input_only_cache_name='planreg_input_only', reject_dynamic_feature_keys=True)
    captured = {}
    handles = []
    for i, block in enumerate(vision.encoder.layers):
        def capture(module, inputs, index=i):
            # Real thumbnail activations, not synthetic substitutes for model tests.
            captured[index] = inputs[0][-1:].detach().cpu()
        handles.append(block.attn.register_forward_pre_hook(capture))
    for row in selected:
        features, targets = drivevla_cached_collate([dataset[dataset.tokens.index(row['token'])]])
        features = to_device_non_paths(features, torch.device('cuda'))
        targets = to_device_non_paths(targets, torch.device('cuda'))
        scene = dict(token=row['token'], log=row['log'], category=row['category'])
        outputs = {}
        for backend in ('eager', 'split_sdpa'):
            adapter.read_only_attention_backend = backend
            teacher = agent.ema_register_target
            teacher.planning_register_adapter.read_only_attention_backend = backend
            start = time.perf_counter()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                pred = agent(features)
                tc, tf, valid = agent._encode_ema_register_targets(features, targets, batch_size=1)
            torch.cuda.synchronize()
            outputs[backend] = {key: pred[key].detach().cpu() for key in ('proposals','trajectory','planning_registers')}
            outputs[backend]['ema_current'] = tc.cpu()
            outputs[backend]['ema_future'] = tf.cpu()
            scene[backend+'_seconds_unwarmed'] = time.perf_counter()-start
            del pred,tc,tf
        scene['bf16_differences'] = {k:difference(outputs['eager'][k], outputs['split_sdpa'][k])
                                    for k in outputs['eager']}
        report['scenes'].append(scene)
        save()
        if handles:
            for handle in handles: handle.remove()
            handles = []
        del features,targets,outputs
    # Audit all 24 actual attention modules, including learned Q/V adapters and
    # Q/K normalization, in FP32 so BF16 kernel rounding is not confused with
    # a changed mathematical function. Full policy BF16 deltas remain above.
    for i, block in enumerate(vision.encoder.layers):
        reference = copy.deepcopy(block.attn).float().eval()
        candidate = copy.deepcopy(reference).eval()
        reference._planreg_read_only_backend = 'eager'
        candidate._planreg_read_only_backend = 'split_sdpa'
        x = captured[i].cuda().float().requires_grad_(True)
        y = x.detach().clone().requires_grad_(True)
        r, s = reference(x), candidate(y)
        upstream = torch.randn_like(r)
        (r*upstream).mean().backward()
        (s*upstream).mean().backward()
        result = dict(block=i, forward=difference(r,s), input_gradient=difference(x.grad,y.grad))
        grad_r = torch.cat([p.grad.flatten() for p in reference.parameters() if p.grad is not None])
        grad_s = torch.cat([p.grad.flatten() for p in candidate.parameters() if p.grad is not None])
        result['parameter_gradient'] = difference(grad_r,grad_s)
        assert result['forward']['relative_l2'] <= report['gates']['fp32_block_relative_l2'], result
        assert result['input_gradient']['relative_l2'] <= report['gates']['fp32_gradient_relative_l2'], result
        assert result['parameter_gradient']['relative_l2'] <= report['gates']['fp32_gradient_relative_l2'], result
        assert torch.isfinite(grad_s).all()
        report['blocks'].append(result)
        save()
        del reference,candidate,x,y,r,s,upstream,grad_r,grad_s
        gc.collect()
    assert len(report['blocks'])==24 and report['trainable_dtypes']==['torch.float32']
    report['status']='PASS'
    report['interpretation']='All actual blocks preserve FP32 function/gradients within preregistered tolerances; BF16 policy rounding is measured, not bitwise parity or a final-score guarantee.'
    save()
    print(json.dumps(dict(status=report['status'], blocks=len(report['blocks']), scenes=report['scenes']),indent=2))


if __name__=='__main__':
    main()

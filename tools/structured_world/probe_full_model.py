"""Actual generic Qwen/DDP/full-world training and label-free deployment probe.

This measures an engineering probe only. Formal throughput must additionally
include the distributed optimizer and checkpoint paths.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.model.framework.vla_structured_fgtr import VLAStructuredFGTR
from starVLA.dataloader.structured_world.dataset import StructuredNAVSIMDataset, collate_structured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenes', type=int, default=1)
    args = parser.parse_args()
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    config = OmegaConf.load(args.config)
    data = StructuredNAVSIMDataset(args.cache, '/var/tmp/ddp-full-foresight-20260929/student_train_v1',
        dino_root='/var/tmp/ddp-full-foresight-20260929/targets/C1',
        dino_index='/mnt/project/ddp-full-foresight-study-artifacts/20260929/dino_index_v1',
        expected_dino='7663c45b77dd711e8304e8d202644dcaa8467f7e265ff8b4c1816d59d2c131cb',
        image_root='/var/tmp/ddp-full-foresight-20260929/images', allow_debug=True)
    observations, targets = collate_structured([data[i] for i in range(args.scenes)])
    start = time.monotonic()
    model = VLAStructuredFGTR(config).float().cuda().train()
    loading_seconds = time.monotonic()-start
    generators = {name: torch.Generator(device='cuda').manual_seed(42+offset)
                  for name, offset in [('noise_generator', 1), ('time_generator', 2),
                                       ('proposal_generator', 3), ('query_generator', 4)]}
    current_calls = []
    original_current = model.encode_current
    def counted_current(observation):
        current_calls.append(1)
        return original_current(observation)
    model.encode_current = counted_current
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    output = model(observations, targets, **generators)
    if len(current_calls) != 1:
        raise AssertionError('Training repeated the current Qwen call')
    output['loss'].backward()
    torch.cuda.synchronize()
    training_seconds = time.monotonic()-start
    gradients = {}
    for name in ('qwen_vl_interface', 'action_model', 'geometry', 'future_space', 'refiner', 'scene_semantics'):
        squared = sum(p.grad.float().square().sum() for p in getattr(model, name).parameters() if p.grad is not None)
        gradients[name] = float(squared.sqrt()) if torch.is_tensor(squared) else 0.
    if any(value <= 0 for value in gradients.values()):
        raise AssertionError(f'An enabled branch has no learning gradient: {gradients}')
    report = {'kind': 'full_model_real_training_probe_not_formal_run', 'group': config.structured_world.group,
              'loading_seconds': loading_seconds, 'scenes': args.scenes,
              'training_seconds': training_seconds, 'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
              'raw_losses': {k: float(v) for k, v in output['metrics']['raw'].items()},
              'weighted_losses': {k: float(v) for k, v in output['losses'].items()},
              'gradient_norms': gradients, 'Qwen_current_calls': len(current_calls),
              'shared_geometry_preparation': 'not yet trained; this probe checks random branch wiring'}
    model.zero_grad(set_to_none=True)
    del output
    torch.cuda.empty_cache()
    model.eval(); model.inference_fp32 = True
    # Add fake GT keys, then remove them. Whitelisted forward cannot depend on
    # either the fields or the separately generated training-label files.
    private_keys = ('gt_map', 'gt_boxes', 'lidar', 'gt_depth', 'future_image', 'ego_future')
    poisoned = [dict(o, **{key: torch.randn(3, 4) for key in private_keys}) for o in observations]
    torch.cuda.synchronize(); start = time.monotonic()
    prediction = model.predict_action(observations, sampling_seed=42, return_diagnostics=True)
    torch.cuda.synchronize(); report['FP32_deployment_seconds'] = time.monotonic()-start
    changed = model.predict_action(poisoned, sampling_seed=42, return_diagnostics=True)
    for key in ('q0', 'q_final'):
        torch.testing.assert_close(prediction[key], changed[key], rtol=0, atol=0)
    model.strip_auxiliary_heads()
    stripped = model.predict_action(observations, sampling_seed=42, return_diagnostics=True)
    for key in ('q0', 'q_final'):
        torch.testing.assert_close(prediction[key], stripped[key], rtol=0, atol=0)
    report.update(label_invariance=True, deployment_head_stripping_invariance=True,
                  q0=prediction['q0'].cpu().tolist(), q_final=prediction['q_final'].cpu().tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

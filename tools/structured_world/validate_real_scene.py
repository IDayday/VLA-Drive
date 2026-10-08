"""Real NAVSIM geometry/label probe; not a formal training experiment."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.dataloader.structured_world.adapters import navsim_scene
from starVLA.dataloader.structured_world.labels import RoadMap, build_labels
from starVLA.dataloader.foresight_dataset import encode_ego, decode_ego
from starVLA.model.modules.structured_world.geo_bev import SingleFrameGeoBEV
from starVLA.model.modules.structured_world.future import SingleFrameFuture
from starVLA.model.modules.structured_world.fgtr import DDPProposalFGTR
from starVLA.model.modules.structured_world.semantics import StructuredSemantics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--imagenet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels-only', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    record = json.loads(args.current.read_text())
    scene = navsim_scene(record)
    originals = [None if f is None else f.boxes.tensor.clone() for f in scene.frames]
    labels = build_labels(scene, RoadMap('navsim', '/mnt/navsim/maps'))
    if any(not torch.equal(before, frame.boxes.tensor) for before, frame in zip(originals, scene.frames) if frame is not None):
        raise AssertionError('UniAD label generation mutated adapter boxes')
    report = {'token': scene.token, 'kind': 'interface_and_gradient_probe_not_formal_run',
              'road_valid_fraction': float(labels['road_valid'].mean()),
              'occupancy_valid_fractions': labels['occupancy_valid'].mean(axis=(1, 2)).tolist(),
              'occupied_valid_cells': ((labels['occupancy'] == 1) & labels['occupancy_valid']).sum(axis=(1, 2)).tolist(),
              'depth_labeled_pixels': (labels['depth'] > 0).sum(axis=(1, 2)).tolist(),
              'height_quality': labels['height_quality'], 'protocol': scene.protocol,
              'label_seconds': time.monotonic()-start}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/(scene.token+'.npz'), **{k: v for k, v in labels.items() if isinstance(v, np.ndarray)})
    if not args.labels_only:
        geometry = SingleFrameGeoBEV(imagenet_checkpoint=args.imagenet).cuda().train()
        H_A = torch.randn(1, 8, 2048, device='cuda', requires_grad=True)
        future = SingleFrameFuture(2048, 8).cuda().train()
        refine = DDPProposalFGTR(2048, 8).cuda().train()
        semantic = StructuredSemantics('event').cuda().train()
        images = scene.observations['geometry_images'][None].cuda()
        calibration = {k: v[None].cuda() for k, v in scene.observations['calibration'].items()}
        current = geometry(images, calibration)
        # Deliberately fixed random proposal only tests module wiring. It is
        # never a formal substitute for the native ten-step DDP proposal.
        q0 = torch.randn(1, 8, 4, device='cuda')
        proposal = decode_ego(q0).detach()
        fields = future(current['B0'], H_A, proposal)
        output = refine(H_A, fields, q0, proposal)
        if not torch.equal(output['q_final_encoded'], q0):
            raise AssertionError('Zero residual layer did not preserve the proposal')
        gt = torch.from_numpy(encode_ego(scene.ego_physical))[None].cuda()
        optimizer = torch.optim.AdamW(list(future.parameters())+list(refine.parameters())+
                                      list(semantic.parameters())+list(geometry.parameters()), lr=1e-4)
        (output['q_final_encoded']-gt).square().mean().backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        # Zero residual output initially blocks its upstream gradients for one
        # update only. Validate live H_A/Bt/geometry gradients after that update.
        current = geometry(images, calibration)
        fields = future(current['B0'], H_A, proposal)
        fields.retain_grad()
        output = refine(H_A, fields, q0, proposal)
        loss = (output['q_final_encoded']-gt).square().mean()
        loss.backward()
        gradients = {'H_A': float(H_A.grad.norm()), 'Bt': float(fields.grad.norm()),
                     'geometry_fpn': float(geometry.fpn.fpn_convs[0].conv.weight.grad.norm()),
                     'FGTR_value': float(refine.col_attn.value_proj.weight.grad.norm())}
        if any(value <= 0 or not np.isfinite(value) for value in gradients.values()):
            raise AssertionError(f'Missing trainable deployment dependency: {gradients}')
        report.update(B0_shape=list(current['B0'].shape), Bt_shape=list(fields.shape),
                      refine_gradient_norms=gradients, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                      total_seconds=time.monotonic()-start)
    (args.output/'REAL_SCENE_PROBE.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

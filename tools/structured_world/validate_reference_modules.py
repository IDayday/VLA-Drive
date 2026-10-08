"""Generate upstream fixtures in the legacy environment and verify the port.

Reference mode executes original class bodies with native MMCV/mmdet imports.
Only registry decorators are removed, avoiding unrelated mmdet3d import side
effects; no operator is replaced. Fixtures contain random test weights only.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def upstream_classes(path, namespace, selected=None):
    module = ast.parse(path.read_text())
    classes = []
    for node in module.body:
        if isinstance(node, ast.ClassDef) and (selected is None or node.name in selected):
            node.decorator_list = []
            classes.append(node)
    body = ast.Module(body=classes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(body), str(path), 'exec'), namespace)
    return namespace


def modules(reference):
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
    if not reference:
        from third_party.resworld.ported.tokenlearner import TokenLearner, TokenFuser
        from third_party.resworld.ported.mln import MLN
        from third_party.resworld.ported.resnet import ResNet
        from third_party.resworld.ported.custom_module import CustomFPN, CustomResNet, FPN_LSS, CustomTransformerDecoder
        from third_party.resworld.ported.rcsample import DepthNet, RCSample
        return locals()
    from mmcv.cnn import ConvModule, build_norm_layer, build_conv_layer
    from mmcv.runner import BaseModule, auto_fp16, force_fp32
    from mmcv.cnn.bricks.transformer import TransformerLayerSequence
    from mmdet.models.backbones.resnet import BasicBlock, Bottleneck, ResNet
    from torch.cuda.amp.autocast_mode import autocast
    namespace = dict(locals(), torch=torch, nn=nn, F=F, checkpoint=checkpoint)
    directory = ROOT / 'third_party/resworld/upstream/projects/mmdet3d_plugin/resworld'
    for path, selected in [
            (directory/'tokenlearner.py', None),
            (directory/'resworld_head.py', {'MLN'}),
            (directory/'custom_module.py', {'CustomFPN', 'CustomResNet', 'FPN_LSS', 'CustomTransformerDecoder'}),
            (directory/'rcsample.py', None)]:
        upstream_classes(path, namespace, selected)
    return namespace


class SequentialGeometry(nn.Module):
    def __init__(self, types):
        super().__init__()
        self.backbone = types['ResNet'](50, out_indices=(2, 3), frozen_stages=-1,
                                       norm_cfg=dict(type='BN', requires_grad=True), norm_eval=False)
        self.fpn = types['CustomFPN']([1024, 2048], 32, 1, out_ids=[0])

    def forward(self, image):
        return self.fpn(self.backbone(image))


class BEVNeck(nn.Module):
    def __init__(self, types):
        super().__init__()
        self.backbone = types['CustomResNet'](8, num_channels=[16, 32, 64])
        self.neck = types['FPN_LSS'](80, 16)

    def forward(self, value):
        return self.neck(self.backbone(value))


def fixed_calibration(batch=1, cameras=3):
    sensor = torch.eye(4).repeat(batch, cameras, 1, 1)
    # Camera forward=z; ego forward=x, left=y, up=z.
    sensor[:, :, :3, :3] = torch.tensor([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    sensor[:, :, 2, 3] = 1.5
    intrinsic = torch.tensor([[32., 0., 32.], [0., 32., 16.], [0., 0., 1.]]).repeat(batch, cameras, 1, 1)
    identity = torch.eye(4).repeat(batch, cameras, 1, 1)
    post = torch.eye(3).repeat(batch, cameras, 1, 1)
    trans = torch.zeros(batch, cameras, 3)
    bda = torch.eye(3).repeat(batch, 1, 1)
    return [sensor, identity, intrinsic, post, trans, bda]


def cases(types):
    decoder = types['CustomTransformerDecoder'](num_layers=2, return_intermediate=False,
        transformerlayers=dict(type='BaseTransformerLayer',
            attn_cfgs=[dict(type='MultiheadAttention', embed_dims=32, num_heads=4)],
            ffn_cfgs=dict(type='FFN', embed_dims=32, feedforward_channels=64, num_fcs=2),
            operation_order=('self_attn', 'norm', 'ffn', 'norm')))
    rc = types['RCSample'](dict(x=[-4., 4., 1.], y=[-4., 4., 1.], z=[-2., 2., 4.], depth=[1., 17., 1.]),
        (32, 64), scale_num=1, ins_channels=[32], out_channels=8, downsamples=[8],
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=8), loss_depth_weight=[1.])
    cal = fixed_calibration()
    mlp = rc.get_mlp_input(*cal)
    return {
        'TokenLearner': (types['TokenLearner'](4, 32), [torch.randn(2, 30, 32)], {}),
        'TokenFuser': (types['TokenFuser'](4, 32), [torch.randn(2, 4, 32), torch.randn(2, 32, 5, 6)], {}),
        'MLN': (types['MLN'](12, 32), [torch.randn(2, 4, 32), torch.randn(2, 1, 12)], {}),
        'latent_decoder': (decoder, [torch.randn(4, 2, 32)], {}),
        'DepthNet': (types['DepthNet'](32, 32, 8, 16, use_dcn=False, aspp_mid_channels=8),
                     [torch.randn(3, 32, 4, 8), mlp], {}),
        'RCSample': (rc, [[torch.randn(1, 3, 32, 4, 8)] + cal + [mlp]], {}),
        'ResNet50_FPN': (SequentialGeometry(types), [torch.randn(1, 3, 64, 96)], {}),
        'BEV_encoder_neck': (BEVNeck(types), [torch.randn(1, 8, 16, 24)], {}),
        'MSDeformAttn': (types['MultiScaleDeformableAttention'](32, num_heads=4, num_levels=1, num_points=4),
            [torch.randn(2, 1, 32)], dict(value=torch.randn(20, 1, 32),
            reference_points=torch.tensor([[[[.2, .3]], [[.7, .8]]]]),
            spatial_shapes=torch.tensor([[4, 5]]), level_start_index=torch.tensor([0])))
    }


def map_tensors(value, function):
    if torch.is_tensor(value):
        return function(value)
    if isinstance(value, dict):
        return {k: map_tensors(v, function) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(map_tensors(v, function) for v in value)
    return value


def flatten(value):
    if torch.is_tensor(value):
        return [value]
    result = []
    for child in value:
        result += flatten(child)
    return result


def execution(model, args, kwargs, name):
    model.cuda().eval()
    leaves = []
    position = 0
    def convert(value):
        nonlocal position
        result = value.cuda().clone()
        # Camera calibration is fixed metadata. Upstream projection modifies
        # temporary coordinates in place and does not support calibration grads.
        differentiable = name != 'RCSample' or position == 0
        position += 1
        if result.is_floating_point() and differentiable:
            result.requires_grad_(True)
            leaves.append(result)
        return result
    inputs = map_tensors(args, convert)
    keywords = map_tensors(kwargs, convert)
    outputs = flatten(model(*inputs, **keywords))
    loss = sum((value.square().mean()*(i+1) for i, value in enumerate(outputs)))
    loss.backward()
    return {'outputs': [v.detach().cpu() for v in outputs],
            'input_gradients': [v.grad.detach().cpu() if v.grad is not None else None for v in leaves],
            'parameter_gradients': {n: p.grad.detach().cpu() if p.grad is not None else None
                                    for n, p in model.named_parameters()}}


def comparison(actual, expected, path='', failures=None, metrics=None):
    failures = [] if failures is None else failures
    metrics = [] if metrics is None else metrics
    if torch.is_tensor(expected):
        difference = (actual-expected).abs()
        scale = float(expected.abs().max())
        maximum = float(difference.max())
        # FP32 CUDA kernels changed across PyTorch/CUDA generations. Require a
        # small error relative to the entire tensor scale, plus absolute floor.
        tolerance = 3e-5 + 3e-4*scale
        metrics.append({'tensor': path, 'max_abs_error': maximum, 'reference_scale': scale,
                        'tolerance': tolerance})
        if maximum > tolerance or not torch.isfinite(actual).all():
            failures.append(path)
    elif expected is None:
        if actual is not None:
            failures.append(path + ': unexpected gradient')
    else:
        keys = expected if isinstance(expected, dict) else range(len(expected))
        for key in keys:
            comparison(actual[key], expected[key], f'{path}/{key}', failures, metrics)
    return failures, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['reference', 'port'], required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261008)
    torch.cuda.manual_seed_all(20261008)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    types = modules(args.mode == 'reference')
    args.fixtures.mkdir(parents=True, exist_ok=True)
    report = {'mode': args.mode, 'torch': torch.__version__, 'cases': {}}
    for name, (model, inputs, keywords) in cases(types).items():
        target = args.fixtures / (name + '.pt')
        if args.mode == 'reference':
            payload = {'state': copy.deepcopy(model.state_dict()), 'inputs': inputs, 'keywords': keywords}
            payload['expected'] = execution(model, inputs, keywords, name)
            torch.save(payload, target)
            entry = {'status': 'fixture_generated', 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        else:
            payload = torch.load(target, map_location='cpu', weights_only=False)
            model.load_state_dict(payload['state'], strict=True)
            actual = execution(model, payload['inputs'], payload['keywords'], name)
            failed, metrics = comparison(actual, payload['expected'])
            entry = {'status': 'PASS' if not failed else 'FAIL', 'failed_tensors': failed,
                     'comparisons': metrics}
        report['cases'][name] = entry
        print(json.dumps({'module': name, 'status': entry['status']}), flush=True)
        model.cpu()
        torch.cuda.empty_cache()
    (args.fixtures / (args.mode + '_report.json')).write_text(json.dumps(report, indent=2))
    if any(value['status'] == 'FAIL' for value in report['cases'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

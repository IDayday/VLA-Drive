"""Real eight-rank loss/gradient comparison, including tails and empty labels.

The original native FM head and actual structured loss functions are used.
Small learned feature tensors isolate reduction correctness from Qwen/GeoBEV
throughput; this is a reduction test, not a substitute full-model experiment.
"""
import argparse
import json
import os
import socket
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn, distributed as dist
from torch.nn.parallel import DistributedDataParallel
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.model.modules.structured_world.semantics import StructuredSemantics
from starVLA.model.modules.structured_world.queries import sample_queries
from starVLA.model.modules.structured_world.losses import compute_structured_losses, scene_reduce
from tools.structured_world.build_cache import atomic_json


class ReductionProbe(nn.Module):
    def __init__(self, steps, mode):
        super().__init__()
        config = OmegaConf.create({'framework': {'qwenvl': {'vl_hidden_dim': 64}, 'action_model': {
            'hidden_size': 64, 'action_dim': 4, 'action_horizon': steps, 'num_inference_timesteps': 10,
            'noise_beta_alpha': 1.5, 'noise_beta_beta': 1., 'noise_s': .999, 'num_timestep_buckets': 1000,
            'add_pos_embed': True, 'max_seq_len': 32,
            'DiTConfig': {'num_layers': 2, 'input_embedding_dim': 64, 'attention_head_dim': 16, 'num_attention_heads': 4},
            'diffusion_model_cfg': {'cross_attention_dim': 64, 'dropout': 0., 'final_dropout': False,
                'interleave_self_attention': True, 'norm_type': 'ada_norm', 'output_dim': 64, 'positional_embeddings': None}}}})
        self.action = FlowmatchingActionHead(config)
        self.fields = nn.Conv2d(4, 4, 1)
        self.semantics = StructuredSemantics(mode, channels=4)
        self.mode = mode

    def forward(self, inputs, targets, queries, *, denominator):
        repeat = 8
        y = targets['ego'].repeat(repeat, 1, 1)
        loss = self.action(inputs['H_A'].repeat(repeat, 1, 1), y,
            noise=inputs['noise'].transpose(0, 1).flatten(0, 1),
            times=inputs['time'].transpose(0, 1).flatten(0, 1))
        fm = scene_reduce(loss.expand(len(y)//repeat), global_scenes=denominator)
        fields = self.fields(inputs['B0'])
        future = self.fields(inputs['Bt'].flatten(0, 1)).reshape_as(inputs['Bt'])
        predicted = self.semantics(fields, future)
        bodies = [dict(length=4.084, width=1.85, rear_axle_to_center=.5)]*len(fields)
        aux, _ = compute_structured_losses(predicted, targets, queries, bodies, semantic_mode=self.mode,
            statistics=dict(current_positive_weight=2., future_positive_weight=2., arrival_positive_weight=3., release_positive_weight=2.),
            global_scenes=denominator, grid=GridSpec(-4., 4., -4., 4., .5, .5))
        anchor = predicted['future_logits'].sum()*0.  # Same empty/G0 graph contract.
        return fm+sum(aux.values())+anchor


def global_data(number, steps, distribution):
    generator = torch.Generator().manual_seed(4821+steps+number)
    normal = lambda *shape: torch.randn(shape, generator=generator)
    grid = GridSpec(-4., 4., -4., 4., .5, .5)
    inputs = {'H_A': normal(number, 3, 64), 'B0': normal(number, 4, 16, 16),
        'Bt': normal(number, steps, 4, 16, 16), 'noise': normal(number, 8, steps, 4),
        'time': torch.rand(number, 8, generator=generator)*.998}
    valid = torch.ones(number, steps+1, 16, 16, dtype=torch.bool)
    # Every scene assigned to rank zero has no auxiliary labels. All of them
    # retain original ego supervision and stay in the global denominator.
    valid[::8] = False
    targets = {'ego': normal(number, steps, 4), 'road_distance': normal(number, 16, 16),
        'road_valid': valid[:, 0].clone(), 'occupancy_valid': valid,
        'occupancy': torch.randint(2, (number, steps+1, 16, 16), generator=generator).byte()}
    targets['occupancy'][~valid] = 255
    proposal = torch.zeros(number, steps, 3)
    queries = sample_queries(proposal, targets, mode=distribution, generator=generator, count=16, grid=grid)
    queries.pop('strata_records')
    return inputs, targets, queries


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    if socket.gethostname().removesuffix('-worker-0') not in policy['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']:
        raise ValueError('Unauthorized validation host')
    rank, world, local = [int(os.environ[name]) for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')]
    if world != 8: raise ValueError('Eight real ranks required')
    torch.cuda.set_device(local); torch.set_num_threads(2); dist.init_process_group('nccl')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    records = []
    for steps in (6, 8):
        for mode, distribution in (('none', 'uniform'), ('full', 'uniform'), ('event', 'uniform'), ('event', 'local')):
            for number in (32, 24, 30):
                torch.manual_seed(42); model = ReductionProbe(steps, mode).float().cuda().eval()
                original = {k: v.detach().clone() for k, v in model.state_dict().items()}
                inputs, targets, queries = global_data(number, steps, distribution)
                local_ids = torch.arange(rank, number, world)
                sliced = lambda v: {k: x[local_ids].cuda() for k, x in v.items()}
                distributed = DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True)
                value = distributed(sliced(inputs), sliced(targets), sliced(queries), denominator=number)
                value.backward()
                averaged_loss = value.detach().clone(); dist.all_reduce(averaged_loss); averaged_loss /= world
                gradients = {k: p.grad.detach().clone() if p.grad is not None else None for k, p in model.named_parameters()}
                if rank == 0:
                    reference = ReductionProbe(steps, mode).float().cuda().eval(); reference.load_state_dict(original, strict=True)
                    all_cuda = lambda v: {k: x.cuda() for k, x in v.items()}
                    # scene_reduce normally compensates the distributed mean.
                    # Its reference denominator is multiplied by world to make
                    # one full-population, undistributed objective in this live
                    # process group; the loss/operator definitions are unchanged.
                    reference_loss = reference(all_cuda(inputs), all_cuda(targets), all_cuda(queries), denominator=number*world)
                    reference_loss.backward()
                    maximum = 0.
                    for name, parameter in reference.named_parameters():
                        if parameter.grad is None:
                            if gradients[name] is not None: raise AssertionError('Unused parameter reduction changed: '+name)
                            continue
                        maximum = max(maximum, float((gradients[name]-parameter.grad).abs().max()))
                        torch.testing.assert_close(gradients[name], parameter.grad, rtol=2e-4, atol=2e-5,
                            msg='Distributed native loss reduction changed: '+name)
                    torch.testing.assert_close(averaged_loss, reference_loss.detach(), rtol=2e-5, atol=2e-6)
                    records.append({'steps': steps, 'mode': mode, 'distribution': distribution, 'global_scenes': number,
                        'empty_auxiliary_label_rank': 0, 'FM_repeat': 8, 'max_absolute_gradient_error': maximum,
                        'averaged_native_loss': float(averaged_loss), 'passed': True})
                    del reference
                dist.barrier(); del distributed, model
    if rank == 0:
        atomic_json(args.output, {'scope': 'real_eight_rank_native_loss_reduction_test_not_training',
            'world_size': world, 'cases': records, 'all_passed': True, 'tolerance': {'rtol': 2e-4, 'atol': 2e-5}})
        print(json.dumps({'cases': len(records), 'all_passed': True}), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__': main()

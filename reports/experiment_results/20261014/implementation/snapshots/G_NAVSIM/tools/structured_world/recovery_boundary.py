"""Check exact saved FP32 model/master/moment state before the next update."""
import numpy as np
import torch
from torch import distributed as dist
from tools.foresight.student_state import capture_rng


def exact(left, right, path='state'):
    if torch.is_tensor(left):
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left.cpu(), right.cpu()):
            raise AssertionError(f'Non-lossless recovery at {path}')
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        if set(left) != set(right): raise AssertionError(f'Recovery keys differ: {path}')
        for name in left: exact(left[name], right[name], path+'/'+str(name))
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right): raise AssertionError(f'Recovery length differs: {path}')
        for index, (a, b) in enumerate(zip(left, right)): exact(a, b, path+'/'+str(index))
    elif left != right:
        raise AssertionError(f'Recovery scalar differs: {path}')


def verify_boundary(engine, folder, generators):
    rank = dist.get_rank()
    rng = torch.load(folder/f'rng_rank{rank}.pt', map_location='cpu', weights_only=False)
    exact(capture_rng(generators), rng, 'RNG')
    optimizer = torch.load(folder/f'zero_pp_rank_{rank}_mp_rank_00_optim_states.pt',
        map_location='cpu', mmap=True, weights_only=False)['optimizer_state_dict']
    # DeepSpeed saves FP32 master partitions WITHOUT alignment padding. Use its
    # own extraction routine; padding is not a model parameter or trained value.
    exact(engine.optimizer._get_groups_without_padding(engine.optimizer.single_partition_of_fp32_groups),
          optimizer['single_partition_of_fp32_groups'], 'FP32_masters')
    # These are actual optimizer moment tensors, step counters and LR groups.
    exact(engine.optimizer.optimizer.state_dict(), optimizer['base_optimizer_state'], 'Adam_state')
    del optimizer
    if rank == 0:
        saved = torch.load(folder/'mp_rank_00_model_states.pt', map_location='cpu',
                           mmap=True, weights_only=False)['module']
        exact(engine.module.state_dict(), saved, 'model_parameters_and_buffers')
    dist.barrier()
    return {'passed': True, 'FP32_model_and_buffers': 'bitwise', 'FP32_masters': 'bitwise all ranks, actual parameters',
        'alignment_padding': 'excluded using DeepSpeed original _get_groups_without_padding',
        'Adam_moments_steps_and_LR': 'bitwise all ranks', 'all_task_RNG': 'exact all ranks',
        'timing': 'before next forward/backward/update'}

"""Independent module construction RNG; never disturbs training/resume streams."""
from contextlib import contextmanager
import random
import numpy as np
import torch
from ..io import digest
from .checkpoint import capture_rng, restore_rng


def module_seed(base_seed, module_kind, expansion_id):
    return int(digest({'seed': base_seed, 'kind': module_kind, 'expansion': expansion_id})[:8], 16)


@contextmanager
def initialization_rng(seed):
    state = capture_rng()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        # Module constructors allocate on CPU. Avoid manual_seed_all changing
        # other devices or lazily queued CUDA streams.
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        restore_rng(state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

"""Deterministic global scene batches and complete stochastic training state."""
import random
import numpy as np
import torch


def epoch_batches(size, batch, seed, epoch):
    if size < 1 or batch < 1 or epoch < 0:
        raise ValueError("Invalid dataset/batch/epoch")
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed+epoch)).tolist()
    return [order[i:i+batch] for i in range(0, size, batch)]


def capture_rng(model, noise, roles):
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
             "noise": noise.get_state(), "roles": roles.state_dict()}
    if hasattr(model, "rgb_model"):
        wan = model.rgb_model
        state["wan"] = {"rng": wan.rng.bit_generator.state, "rng_2": wan.rng_2.bit_generator.state,
                        "torch_rng": wan.torch_rng.get_state()}
    return state


def restore_rng(state, model, noise, roles):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    noise.set_state(state["noise"])
    roles.load_state_dict(state["roles"])
    if "wan" in state:
        wan = model.rgb_model
        wan.rng.bit_generator.state = state["wan"]["rng"]
        wan.rng_2.bit_generator.state = state["wan"]["rng_2"]
        wan.torch_rng.set_state(state["wan"]["torch_rng"])

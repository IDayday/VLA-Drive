"""Deterministic sample keys; checkpoint cursors track consumption, not prefetch."""
from contextlib import contextmanager
import random
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .contracts import digest


@contextmanager
def data_rng(seed):
    py = random.getstate()
    npstate = np.random.get_state()
    ts = torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.random.default_generator.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(npstate)
        torch.set_rng_state(ts)


class KeyedDataset(Dataset):
    def __init__(self, sft_cfg, split="train"):
        from starVLA.dataloader.navsim_dataset import NavSimDataset

        self.dataset = NavSimDataset(
            sft_cfg.datasets.vla_data.datalist_path,
            split=split,
            video_data_cfg=sft_cfg.datasets.video_data,
            gs_data_cfg=sft_cfg.datasets.gs_data,
            reward_data_cfg=sft_cfg.datasets.reward_data,
            ver_1225=sft_cfg.ver_1225,
            dataset_cfg=sft_cfg.datasets.vla_data,
            all_cfg=sft_cfg,
            data_root=sft_cfg.datasets.vla_data.data_root,
        )
        self.indices = {token: i for i, token in enumerate(self.dataset.raw_list)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        token, seed = key
        # Pre-read makes the SFT dataset's historical fallback to previous sample
        # unreachable for missing/corrupt metadata; never replace a failed scene.
        path = Path(self.dataset.base_dir) / f"{token}.pkl"
        with path.open("rb") as stream:
            pickle.load(stream)
        self.dataset.raw_data = None
        self.dataset.raw = None
        with data_rng(seed):
            sample = self.dataset[self.indices[token]]
        if sample["token"] != token:
            raise RuntimeError("SFT dataset replaced a scene")
        sample["_flow_sample_seed"] = int(seed)
        return sample


def keys_for_positions(tokens, seed, positions):
    n = len(tokens)
    perms = {}
    for pos in positions:
        epoch, index = divmod(pos, n)
        if epoch not in perms:
            generator = torch.Generator().manual_seed(seed + epoch)
            perms[epoch] = torch.randperm(n, generator=generator).tolist()
        token = tokens[perms[epoch][index]]
        yield token, int(digest([seed, pos, token])[:8], 16)


class SceneStream:
    def __init__(
        self,
        dataset,
        tokens,
        seed,
        rank=0,
        world_size=1,
        cursor=0,
        workers=0,
        prefetch=2,
        timeout=180,
    ):
        self.dataset = dataset
        self.tokens = tokens
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.cursor = cursor
        self.workers = workers
        self.prefetch = prefetch
        self.timeout = timeout

    def take(self, count):
        # Only bounded requested consumption is prefetched. No hidden advancing cursor.
        positions = [
            (self.cursor + i) * self.world_size + self.rank for i in range(count)
        ]
        keys = list(keys_for_positions(self.tokens, self.seed, positions))
        if self.workers:
            loader = DataLoader(
                self.dataset,
                batch_size=None,
                sampler=keys,
                num_workers=self.workers,
                prefetch_factor=self.prefetch,
                multiprocessing_context="spawn",
                collate_fn=identity,
                timeout=self.timeout,
            )
            iterator = iter(loader)
            try:
                samples = list(iterator)
            finally:
                # PyTorch 2.5 DataLoader iterator owns only this call's workers.
                iterator._shutdown_workers()
        else:
            samples = [self.dataset[key] for key in keys]
        self.cursor += len(samples)
        return samples

    def state_dict(self):
        return dict(
            cursor=self.cursor,
            seed=self.seed,
            token_hash=digest(self.tokens),
            rank=self.rank,
            world_size=self.world_size,
        )

    def load_state_dict(self, state):
        expected = self.state_dict()
        if any(
            state[k] != expected[k]
            for k in ("seed", "token_hash", "rank", "world_size")
        ):
            raise ValueError("sampler resume contract changed")
        self.cursor = state["cursor"]


def identity(value):
    return value


def to_device(value, device):
    import dataclasses

    if dataclasses.is_dataclass(value):
        return dataclasses.replace(
            value,
            **{
                f.name: to_device(getattr(value, f.name), device)
                for f in dataclasses.fields(value)
            },
        )
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(to_device(v, device) for v in value)
    return value

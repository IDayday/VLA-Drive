#!/usr/bin/env python3
"""Current-only inference adapter for the existing PlanReg candidate exporter.

No target builder, Scene construction, metric cache, or offline scorer is used
by this dataset. The existing agent and Lightning prediction path are unchanged.
Whole-log host shards and atomic rank files avoid NCCL Python-object gathering.
"""
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from navsim.common.dataloader import SceneLoader
from navsim.planning.script.run_pdm_score_multi_gpu import save_formal_candidate_bank
from navsim.planning.training.agent_lightning_module import AgentLightningModule


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class CurrentOnlyDataset(Dataset):
    def __init__(self, loader, builders):
        self.loader, self.builders = loader, builders

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, index):
        token = self.loader.tokens[index]
        agent_input = self.loader.get_agent_input_from_token(token)
        features = {}
        for builder in self.builders:
            features.update(builder.compute_features(agent_input))
        if any('future' in key or 'target' in key for key in features):
            raise RuntimeError('Future/target data in inference features')
        features['scenario_token'] = token
        return features, {}, token


@hydra.main(config_path='../navsim/planning/script/config/pdm_scoring',
            config_name='default_run_pdm_score_gpu', version_base=None)
def main(cfg):
    if str(cfg.trainer.params.precision) not in ('32', '32-true'):
        raise ValueError('Benchmark requires trainer precision 32')
    if str(cfg.agent.vlm_config.compute_dtype) != 'float32':
        raise ValueError('Benchmark requires FP32 VLM compute')
    if cfg.agent.world_model.enabled or cfg.agent.ema.enabled:
        raise ValueError('Deployment must not construct EMA/auxiliary modules')
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pl.seed_everything(int(cfg.get('seed', 0)), workers=True)
    output = Path(cfg.output_dir)
    agent = instantiate(cfg.agent)
    agent.initialize()
    for name in ('ema_register_target', 'physical_query_decoder', 'future_register_predictor'):
        if getattr(agent, name, None) is not None:
            raise RuntimeError(f'Training-only module constructed: {name}')
    dtypes = Counter()
    for parameter in agent.parameters():
        dtypes[str(parameter.dtype)] += parameter.numel()
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            raise RuntimeError(f'Non-FP32 model parameter: {parameter.dtype}')
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    parts = int(os.getenv('PLANREG_EVAL_HOST_PARTS', '1'))
    part = int(os.getenv('PLANREG_EVAL_HOST_PART', '0'))
    if not 0 <= part < parts:
        raise ValueError('Invalid host shard')
    if parts > 1:
        if not scene_filter.log_names:
            raise ValueError('Expected explicit Navtest logs for host sharding')
        scene_filter.log_names = sorted(scene_filter.log_names)[part::parts]
    loader = SceneLoader(sensor_blobs_path=Path(cfg.sensor_blobs_path),
                         data_path=Path(cfg.navsim_log_path), scene_filter=scene_filter,
                         sensor_config=agent.get_sensor_config(), load_image_path=True)
    dataset = CurrentOnlyDataset(loader, agent.get_feature_builders())
    trainer = pl.Trainer(**cfg.trainer.params)
    predictions = trainer.predict(AgentLightningModule(agent, for_analysis=True),
        DataLoader(dataset, **cfg.dataloader.params, shuffle=False), return_predictions=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    local = {}
    for batch in predictions:
        if set(local) & set(batch):
            raise RuntimeError('Duplicate local inference token')
        local.update(batch)
    path = output / f'predictions.rank{rank:03d}.pkl'
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(local, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    if dist.is_initialized():
        dist.barrier()
    if rank:
        return
    merged = {}
    for index in range(world):
        with (output / f'predictions.rank{index:03d}.pkl').open('rb') as stream:
            batch = pickle.load(stream)
        if set(merged) & set(batch):
            raise RuntimeError('Duplicate distributed inference token')
        merged.update(batch)
    if set(merged) != set(loader.tokens):
        raise RuntimeError('Inference scene coverage differs from requested shard')
    for token, row in merged.items():
        for key in ('proposals', 'predicted_log_pdm', 'planning_registers'):
            if not np.isfinite(row[key]).all():
                raise RuntimeError(f'Nonfinite {key}: {token}')
        if row['selected_index'] != int(np.argmax(row['predicted_log_pdm'])):
            raise RuntimeError('Selection differs from unchanged scorer argmax')
    bank = output / 'candidate_bank.npz'
    manifest = save_formal_candidate_bank(merged, list(loader.tokens), bank)
    # Schema consumed by the existing validated per-log CPU evaluation scripts.
    cache = output / 'proposal_predictions.pkl'
    with cache.open('wb') as stream:
        pickle.dump({token: {'proposals': row['proposals'],
                            'predicted_scores': row['predicted_log_pdm']}
                     for token, row in merged.items()}, stream, protocol=pickle.HIGHEST_PROTOCOL)
    cls = type(agent)
    manifest.update(agent_class=f'{cls.__module__}.{cls.__qualname__}',
        checkpoint_path=str(cfg.agent.checkpoint_path),
        checkpoint_sha256=sha256(cfg.agent.checkpoint_path),
        proposal_predictions_sha256=sha256(cache),
        parameter_storage=dict(dtypes), precision='FP32 VLM + FP32 action/scorer; TF32 off',
        host_shard=part, host_shard_count=parts,
        log_count=len(loader.get_tokens_list_per_log()),
        target_builders_used=False, evaluator_called_during_inference=False,
        aux_modules_constructed=False,
        source_files={str(p): sha256(p) for p in (Path(__file__), Path(inspect.getfile(cls)))})
    (output / 'inference_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output / 'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()

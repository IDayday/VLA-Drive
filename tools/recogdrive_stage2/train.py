"""Run unchanged official Stage2 agent/optimizer/Lightning step on 2 x 8 GPUs.

Local changes: identified label overlay, development-log exclusion, full cache
contract, multi-node launch, local logging and recoverable latest checkpoint.
"""
import argparse
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .assets import atomic_json, check_official, digest, read


class IdentifiedCacheDataset(Dataset):
    def __init__(self, manifest, root, split):
        self.rows = [r for r in manifest['rows'] if r['split'] == split]
        self.root = Path(root)
        self.manifest_hash = manifest['manifest_sha256']
        if not self.rows:
            raise ValueError('Empty official split')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        x = torch.load(self.root / row['token'][:2] / (row['token'] + '.pt'),
                       map_location='cpu', weights_only=True)
        if x['token'] != row['token'] or x['manifest_sha256'] != self.manifest_hash:
            raise ValueError('Wrong Stage1 cache identity')
        features = {k: x[k] for k in ('history_trajectory', 'high_command_one_hot',
                                     'status_feature', 'last_hidden_state')}
        target = torch.tensor(row['trajectory'], dtype=torch.float32)
        if not torch.isfinite(target).all():
            raise ValueError('Illegal physical target')
        return features, {'trajectory': target}, row['token']


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('official-source', 'official-revision', 'manifest', 'cache', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--resume')
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--gpu-hours-limit', type=float)
    a = p.parse_args()
    source = check_official(a.official_source, a.official_revision)
    sys.path.insert(0, str(source))
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import pytorch_lightning as pl
    from navsim.planning.script.run_training_recogdrive import custom_collate_fn
    from navsim.planning.training.agent_lightning_module import AgentLightningModule
    local, rank, world = [int(os.environ.get(k, d)) for k, d in [('LOCAL_RANK', 0), ('RANK', 0), ('WORLD_SIZE', 1)]]
    if world != 16 and not a.smoke_steps:
        raise ValueError('Formal Stage2 requires exactly two nodes of eight GPUs')
    torch.cuda.set_device(local)
    torch.distributed.init_process_group('nccl')
    torch.backends.cuda.matmul.allow_tf32 = False
    data = read(a.manifest); data['manifest_sha256'] = digest(a.manifest)
    if not a.smoke_steps:
        if (data['identity'].get('scope', 'formal') != 'formal' or
                len(data['rows']) != data['identity']['optimized_labels']['source_scenes']):
            raise ValueError('A smoke or prefix manifest cannot initialize formal Stage2')
        finished = [read(Path(a.cache) / f'COMPLETE_rank{i:02d}.json') for i in range(world)]
        if sum(x['scenes'] for x in finished) != len(data['rows']):
            raise ValueError('Full official training/validation cache is incomplete')
        for x in finished:
            if x['signature']['manifest_sha256'] != data['manifest_sha256'] or x['signature']['actual_scenes'] != len(data['rows']):
                raise ValueError('Partial/foreign extraction cannot initialize formal Stage2')
    output = Path(a.output); output.mkdir(parents=True, exist_ok=True)
    begin = time.time()
    stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopping.__setitem__(0, True))
    with initialize_config_dir(config_dir=str(source / 'navsim/planning/script/config/training'), version_base=None):
        cfg = compose(config_name='default_training', overrides=[
            'agent=recogdrive_agent', 'agent.lr=1e-4', 'agent.grpo=False',
            'agent.vlm_path=' + data['identity']['stage1']['path'],
            'agent.cam_type=single', 'agent.cache_hidden_state=True', 'agent.vlm_type=internvl',
            'agent.dit_type=small', 'agent.vlm_size=small', 'agent.sampling_method=ddim',
            'agent.checkpoint_path=', 'trainer.params.max_epochs=200',
            'trainer.params.num_nodes=' + str(world // 8 if world == 16 else 1),
            'trainer.params.devices=' + str(8 if world == 16 else world),
            'train_test_split=navtrain', 'experiment_name=recogdrive_stage2_optimized_16gpu',
            'cache_path=' + a.cache, 'use_cache_without_dataset=True', 'force_cache_computation=False',
            'output_dir=' + str(output)])
    pl.seed_everything(cfg.seed, workers=True)
    agent = instantiate(cfg.agent)
    if agent.backbone is not None or agent.grpo or agent.checkpoint_path:
        raise ValueError('Stage2 must have cached fixed Stage1 and random imitation-learning planner')
    module = AgentLightningModule(agent)
    identity = dict(schema='official_recogdrive_stage2_optimized_16gpu_v1',
        scope='pipeline_smoke' if a.smoke_steps else 'formal', smoke_steps=a.smoke_steps,
        official_revision=a.official_revision, training_source=__import__('subprocess').check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(), manifest_sha256=data['manifest_sha256'],
        stage1=data['identity']['stage1'], optimized_labels=data['identity']['optimized_labels']['identity'],
        planner_initialization='random official small DiT; no Stage2 IL or Stage3 RL checkpoint loaded',
        world_size=world, seed=int(cfg.seed), epochs=200, per_gpu_batch=int(cfg.dataloader.params.batch_size),
        global_batch=world * int(cfg.dataloader.params.batch_size),
        precision=cfg.trainer.params.precision, tf32=False,
        gpu_hours_limit=a.gpu_hours_limit,
        architecture=vars(agent.action_head.config),
        data_counts=data['identity']['counts'], split_adjustment=data['identity']['split_adjustment'])
    # Exact official configuration is stored without the 100k-entry scene-filter token list.
    resolved = OmegaConf.to_container(cfg, resolve=True)
    identity['optimizer_recipe'] = dict(lr=1e-4, weight_decay=1e-4, betas=[.9,.95],
        scheduler='official WarmupCosLR', epochs=200, warmup_epochs=3, min_lr=1e-6)
    identity['architecture'] = str(agent.action_head.config)
    if (output / 'identity.json').exists() and read(output / 'identity.json') != identity:
        raise ValueError('Training identity changed')
    if rank == 0:
        atomic_json(output / 'identity.json', identity)
        atomic_json(output / 'resolved_official_config.json', resolved)
    train = IdentifiedCacheDataset(data, a.cache, 'train')
    val = IdentifiedCacheDataset(data, a.cache, 'val')
    if a.smoke_steps:
        for dataset in (train, val):
            dataset.rows = [r for r in dataset.rows if
                (Path(a.cache) / r['token'][:2] / (r['token'] + '.pt')).exists()]
            if len(dataset) < world * int(cfg.dataloader.params.batch_size):
                raise ValueError('Real full-batch smoke requires enough cached train and validation scenes')
    train_loader = DataLoader(train, collate_fn=custom_collate_fn, **cfg.dataloader.params, shuffle=True)
    val_loader = DataLoader(val, collate_fn=custom_collate_fn, **cfg.dataloader.params, shuffle=False)

    class Progress(pl.Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if trainer.is_global_zero and (trainer.global_step == 1 or trainer.global_step % 20 == 0):
                atomic_json(output / 'status.json', dict(status='TRAINING', update=trainer.global_step,
                    epoch=trainer.current_epoch, train_scenes=len(train), val_scenes=len(val), world_size=world,
                    epoch_batch=batch_idx, loss=float(outputs['loss'].detach()),
                    peak_allocated=torch.cuda.max_memory_allocated(), timestamp=time.time()))
            exhausted = a.gpu_hours_limit is not None and world * (time.time() - begin) / 3600 >= a.gpu_hours_limit
            stop = torch.tensor(int(stopping[0] or exhausted or (output / 'STOP_REQUESTED').exists()), device='cuda')
            torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)
            if stop.item():
                trainer.save_checkpoint(output / 'checkpoints' / 'paused.ckpt')
                trainer.should_stop = True
                stopping[0] = True
        def on_save_checkpoint(self, trainer, module, checkpoint):
            checkpoint['recogdrive_identity'] = identity
            rng = dict(python=random.getstate(), numpy=np.random.get_state(),
                       torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())
            states = [None] * world
            torch.distributed.all_gather_object(states, rng)
            checkpoint['recogdrive_rng_by_rank'] = states
        def on_load_checkpoint(self, trainer, module, checkpoint):
            if checkpoint.get('recogdrive_identity') != identity:
                raise ValueError('Checkpoint protocol/target/source mismatch')
            rng = checkpoint['recogdrive_rng_by_rank'][rank]
            random.setstate(rng['python']); np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'].cpu()); torch.cuda.set_rng_state(rng['cuda'].cpu())
        def on_train_end(self, trainer, module):
            if trainer.is_global_zero:
                atomic_json(output / 'status.json', dict(status='PAUSED' if stopping[0] else 'COMPLETE',
                    updates=trainer.global_step, epochs=trainer.current_epoch,
                    gpu_hours=world * (time.time()-begin) / 3600, timestamp=time.time()))
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=output / 'checkpoints', monitor='val/loss_epoch',
        mode='min', save_top_k=5, every_n_epochs=1, save_last=True)
    kwargs = dict(cfg.trainer.params)
    if a.smoke_steps:
        kwargs.update(max_steps=a.smoke_steps, limit_val_batches=1)
    trainer = pl.Trainer(**kwargs, callbacks=[checkpoint, Progress()])
    trainer.fit(module, train_loader, val_loader, ckpt_path=a.resume)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()

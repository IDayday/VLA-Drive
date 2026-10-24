"""Fixed512 TRAIN scenes, unfiltered initial SFT source SDE; no actor update."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from starVLA.rl.qwen_sourceflow.safe_admission import require_dispatch
from starVLA.rl.qwen_sourceflow.rl_runtime import NativeRuntime
from starVLA.rl.qwen_sourceflow.contracts import fingerprint, file_sha, configure_native_numerics
from starVLA.rl.qwen_sourceflow.safe_recipe import COSTS
from starVLA.rl.qwen_sourceflow.safe_advantages import event_costs
from starVLA.rl.qwen_sourceflow.collection import collect
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch
from starVLA.rl.flow_grpo.transactions import atomic_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', default='configs/qwen_sourceflow/rl_canonical.yaml')
    p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    NativeRuntime.add_arguments(p)
    a = p.parse_args(); a.qualification = False
    require_dispatch(); configure_native_numerics()
    rank = int(os.environ.get('RANK', 0)); world = int(os.environ.get('WORLD_SIZE', 1))
    if world not in (1, 2, 4, 8) or 512 % world: raise ValueError('Explicit finite calibration layout required')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0))); device = torch.device('cuda', torch.cuda.current_device())
    if world > 1: dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    torch.set_num_threads(4); random.seed(42); np.random.seed(42); torch.manual_seed(42)
    runtime = NativeRuntime(); cfg = runtime.load_recipe(a)
    stream = runtime.make_scene_stream(cfg, a, world)
    actor, _, _ = runtime.make_actor(cfg, a, device); conditions = runtime.make_conditions(actor)
    service = RewardService(Path.cwd(), workers=1); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    if (out/f'rank{rank}_COMPLETE.json').exists(): raise FileExistsError('Never overwrite calibration results')
    generator = torch.Generator().manual_seed(seed_for('native_safe_budget_calibration', 42, rank))
    sums = torch.zeros(3, dtype=torch.float64); rows = []; started = time.monotonic()
    try:
        for cursor in range(0, 512, world):
            token, position = stream.at(cursor, rank)
            batch = collect(actor, conditions.get(token, device), token, stream.logs[token], service, generator,
                variant='R_CANONICAL', version=dict(update=0, SFT_sha256=runtime.parent['files']['model.pt']))
            costs = event_costs(batch['records']); sums += costs.sum((0, 1))
            atomic_torch(out/'scenes'/f'{token}.pt', dict(token=token, log=stream.logs[token],
                costs=costs, records=batch['records'], chain=batch['chain'], stoch_idx=batch['stoch_idx'], position=position))
            rows.append(dict(token=token, costs=costs[0].tolist(), canonical_PDMS=[r['readouts']['canonical_PDMS'] for r in batch['records']]))
            if len(rows) % 8 == 0: print(json.dumps(dict(rank=rank,scenes=len(rows),seconds=time.monotonic()-started)), flush=True)
        record = dict(rank=rank, scenes=len(rows), candidates=len(rows)*8, cost_sums=sums.tolist(),
            tokens=[r['token'] for r in rows], seconds=time.monotonic()-started,
            peak_GPU_bytes=torch.cuda.max_memory_allocated(), prefill_calls=conditions.prefill_calls)
        atomic_json(out/f'rank{rank}_COMPLETE.json', record)
        all_rows = [None] * world
        if world > 1: dist.all_gather_object(all_rows, record)
        else: all_rows = [record]
        if rank == 0:
            tokens = [t for r in all_rows for t in r['tokens']]
            if len(tokens) != 512 or len(set(tokens)) != 512: raise ValueError('Calibration scene population duplicated')
            total = np.array([r['cost_sums'] for r in all_rows]).sum(0)
            budget = dict(schema='native_SFT_event_cost_budget_v1', status='COMPLETE',
                SFT_sha256=runtime.parent['files']['model.pt'], parent=runtime.parent['path'],
                hard_manifest_sha256=file_sha(a.manifest), population='TRAIN_only_public_hard',
                scenes=512, candidates=4096, world=world, training_seed=42,
                calibration_RNG='native_safe_budget_calibration namespace; independent of formal rollout and evaluation2101',
                cost_metrics=list(COSTS), cost_definition='I[official_metric < 1]', cost_sums=total.tolist(),
                budgets=(total/4096).tolist(), tokens=sorted(tokens), rank_results=all_rows,
                sampling='initial SFT, G8 K10 source shift5 rigorous SDE, stochastic[7,8,9]',
                score_protocol=service.protocol, no_updates=True,
                caveat='Finite TRAIN stochastic reference expected costs; not a hard or per-scene safety guarantee',
                source_files=runtime.source_files())
            budget['sha256'] = fingerprint(budget); atomic_json(out/'budget.json', budget)
            print(json.dumps(dict(status='COMPLETE', budgets=budget['budgets'], scenes=512, candidates=4096)))
    finally:
        service.close()
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__': main()

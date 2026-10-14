"""Execution-only overlap: unchanged B1 rollout and canonical scene workers.

All local scene jobs are fixed before dispatch. Resolve in input order and
drain before returning to the native collector/checkpoint boundary. This
module changes scheduling only; it never performs actor/dual updates.
"""
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from starVLA.rl.constrained_flow_noise.component_advantages import component_advantages
from starVLA.rl.constrained_flow_noise.contracts import seed_for
from starVLA.rl.constrained_flow_noise.runtime import cpu_tree
from starVLA.rl.constrained_flow_noise.safety_costs import costs_from_record, score_scene
from starVLA.rl.safe_flow_improvement.policy_adapter import decode_request


@torch.no_grad()
def collect_batch(behavior, reference, store, service, stream, pop, collector,
                  raw_start, count, calibration, rank, world, *, failure_dir=None,
                  include_reference=True):
    offset=len(collector.rows)
    pending=[]
    local=[]
    try:
        for i in range(offset,offset+count):
            if i%world != rank:continue
            position=raw_start+i-offset;token=stream.at(position)
            raw=store.get(token)
            if raw is None:raise FileNotFoundError('Missing sealed TRAIN observation: '+token)
            device=next(behavior.parameters()).device
            t0=time.monotonic()
            try:
                chain=behavior.sample_chain(raw.to(device),[token],[position],version=collector.behavior_identity)
            except Exception as error:
                if failure_dir is not None:
                    root=Path(failure_dir);root.mkdir(parents=True,exist_ok=True)
                    torch.save(dict(token=token,position=position,version=collector.behavior_identity,raw=raw.cpu(),
                                    RNG_root_seed=seed_for('cfn_root',42,token,position),
                                    RNG_innovation_seed=seed_for('cfn_innovation',42,token,position),
                                    partial_chain=getattr(error,'chain_evidence',None),traceback=traceback.format_exc()),
                               root/f'rollout_rank{rank}_position{position}.pt')
                raise
            torch.cuda.synchronize();rollout_seconds=time.monotonic()-t0
            request=decode_request(chain['chain'][0,:,-1].float().cpu().numpy(),act_norm=1)
            submitted=time.monotonic()
            future=service.pool.submit(score_scene,(token,pop['token_logs'][token],request))
            # Keep this pending item even if reference computation fails. The
            # finally block drains every owned future before native failure save.
            item=dict(index=i,token=token,position=position,chain=chain,requests=request,
                      future=future,submitted=submitted,rollout_seconds=rollout_seconds)
            pending.append(item)
            t0=time.monotonic();ref=None
            if i<16 and include_reference:
                values=reference.saved_statistics(chain['raw'],chain['chain'])
                ref={k:values[k] for k in ('mean','std')}
            torch.cuda.synchronize()
            item.update(reference_seconds=time.monotonic()-t0,reference=cpu_tree(ref),chain=cpu_tree(chain))
        for item in pending:
            wait_start=time.monotonic()
            try:records=item['future'].result()
            except Exception:
                if failure_dir is not None:
                    root=Path(failure_dir);root.mkdir(parents=True,exist_ok=True)
                    torch.save(dict(token=item['token'],position=item['position'],chain=item['chain'],
                                    requested_trajectory=item['requests'],traceback=traceback.format_exc(),
                                    status='SCORING_FAILURE_NOT_ZERO_REWARD'),
                               root/f'score_rank{rank}_position{item["position"]}.pt')
                raise
            wait_seconds=time.monotonic()-wait_start
            rewards=torch.tensor([[r['metrics']['score'] for r in records]],dtype=torch.float64)
            costs=torch.from_numpy(np.stack([costs_from_record(r,calibration['thresholds']) for r in records]))[None]
            advantage,statistics=component_advantages(rewards,costs,collector.lambda_snapshot,calibration['component_floors'])
            row={key:item[key] for key in ('token','position','chain','requests','reference',
                                         'rollout_seconds','reference_seconds')}
            row.update(scene_key=f'{collector.behavior_identity["collection"]}:{item["position"]}:{item["token"]}',
                       records=records,costs=costs[0],rewards=rewards[0],advantage=advantage[0],
                       advantage_statistics=statistics,scoring_seconds=time.monotonic()-item['submitted'],
                       scoring_cpu_seconds=records[0]['scene_scoring_cpu_seconds'],
                       score_wait_seconds=wait_seconds,
                       collection_execution='overlap_v1; scoring_seconds is submission-to-resolution latency and overlaps GPU work')
            local.append((item['index'],cpu_tree(row)))
    finally:
        # No pending work crosses a behavior refresh, fault checkpoint, or the
        # return to the collective. Infrastructure errors never become zeros.
        for item in pending:
            if not item['future'].done():
                try:item['future'].result()
                except Exception:pass  # The affected collection already aborts.
    return local

"""Predeclared TRAIN monitoring with fresh real chains; never consumed by updates."""
from pathlib import Path
import json
import os
import torch
from starVLA.rl.flow_grpo import distributed as control
from starVLA.rl.flow_grpo.transactions import atomic_json
from .effective_group_collector import EffectiveGroupCollector
from .runtime import collect_batch, gather_rows
from .diagnostics import summarize_rows


def run_monitor(behavior, reference, store, service, pop, registration, calibration, multipliers,
                version, output, rank, world):
    tokens=registration['monitor_tokens']
    if len(tokens)!=128 or len(set(tokens))!=128 or not set(tokens)<=set(pop['train_tokens']):
        raise ValueError('Registered fixed TRAIN monitoring population changed')
    root=Path(output)
    def prepare():
        if (root/'COMPLETE').exists():
            old=json.loads((root/'summary.json').read_text())
            if old['behavior']!=version or old['lambda_snapshot']!=multipliers.tolist():raise ValueError('Monitor identity changed')
            return False
        root.mkdir(parents=True,exist_ok=True)
        return True
    if not control.rank0_result(prepare,next(behavior.parameters()).device):return
    # Namespace separation by disjoint positive positions; same monitor roots at
    # successive checkpoints, independent of original deployment seed2101.
    start_position=1_000_000_000
    class Stream:
        def at(self,position):return tokens[position-start_position]
    collector=EffectiveGroupCollector(version,multipliers)
    device=next(behavior.parameters()).device
    local=control.synchronized_call(lambda:collect_batch(behavior,reference,store,service,Stream(),pop,collector,start_position,len(tokens),
                        calibration,rank,world,failure_dir=root/'failures',include_reference=False),device)
    temporary=root/f'.rank{rank}.incomplete'
    def save_local():
        torch.save(local,temporary);os.replace(temporary,root/f'rank{rank}.pt')
    control.synchronized_call(save_local,device)
    rows=gather_rows(local)
    def publish():
        summary=summarize_rows([row for _,row in rows],behavior.spec)
        summary.update(behavior=version,lambda_snapshot=multipliers.tolist(),population='fixed_128_RL_TRAIN_monitor',
                       scope='stochastic_G16; excluded from PG, dual, calibration and data cursor',root_positions_start=start_position)
        atomic_json(root/'summary.json',summary)
        (root/'COMPLETE').write_text('All128 real G16 TRAIN monitoring scenes; never update data\n')
    control.rank0_call(publish,next(behavior.parameters()).device)

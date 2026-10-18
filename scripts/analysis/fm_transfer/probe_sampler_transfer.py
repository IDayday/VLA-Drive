"""Matched SFT/RL, native/shift5 ODE/source SDE on fixed TRAIN observations.

This is a bounded diagnostic, not a new formal inference-seed matrix. G8 mean
and finite candidate opportunity are separate; no optimizer or policy selection.
Uses the original actor, scheduler, decoder, conditions and canonical scorer.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.fm_transfer.rl_adapter import make_transfer_actor, TransferConditions
from starVLA.rl.fm_transfer.model import TransferPolicy
from starVLA.rl.simwam_recipe.contracts import POPULATION,SFT_EXPORT,file_sha,fingerprint,seed_for
from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
from starVLA.rl.simwam_recipe.reward_readouts import RewardService
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
from starVLA.rl.safe_flow_improvement.policy_adapter import decode_request
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe.checkpoint import atomic_torch


def probe_tokens(population, diagnostic):
    tokens=diagnostic['explicit_tokens']
    if len(tokens)!=512 or len(set(tokens))!=512 or not set(tokens)<=set(population['train_tokens']):
        raise ValueError('Expected the existing fixed TRAIN diagnostic population')
    if set(tokens)&(set(population['dev_tokens'])|set(population['navtest_tokens'])):
        raise ValueError('Diagnostic may not select DEV/Navtest scenes')
    return sorted(tokens,key=lambda t:fingerprint(dict(selection_seed=42,token=t,task='failure_sampler_probe')))[:64]


def means(rows):
    keys=('score','no_at_fault_collisions','drivable_area_compliance',
          'time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance')
    v=np.array([[r['metrics'][k] for k in keys] for r in rows],dtype=np.float64)
    if v.shape!=(8,7) or not np.isfinite(v).all() or any(r['status']!='OK' for r in rows):
        raise ValueError('Invalid or incomplete real candidate scores')
    safe=(v[:,1:4]==1).all(1)
    return dict(mean=dict(zip(keys,v.mean(0).tolist())),safe_fraction=float(safe.mean()),
                finite_best_safe=float(v[safe,0].max()) if safe.any() else None,
                feasible=bool(safe.any()),any_zero=bool((v[:,0]==0).any()),all_zero=bool((v[:,0]==0).all()))


@torch.no_grad()
def run_probe(run,output,registration):
    host=require_dispatch();configure_numerics();torch.set_num_threads(1)
    if output.exists():raise FileExistsError('Preserve diagnostic attempts: '+str(output))
    output.mkdir(parents=True);started=time.monotonic()
    identity=json.loads((run/'identity.json').read_text())
    transfer=identity['recipe']['supervised_transfer']; cfg=dict(identity['recipe']);cfg.pop('supervised_transfer')
    if identity['sha256']!=registration['runs'][transfer['variant']]['identity_sha256']:
        raise ValueError('Run differs from preregistered diagnostic')
    pop=json.loads(Path(POPULATION).read_text())
    if file_sha(POPULATION)!=registration['population_sha256']:raise ValueError('Population changed')
    tokens=registration['tokens']
    if len(tokens)!=64 or not set(tokens)<=set(pop['train_tokens']):raise ValueError('Non-TRAIN probe')
    actor,parent=make_transfer_actor(transfer['SFT_identity']['checkpoint'],cfg)
    if parent!=transfer['SFT_identity']:raise ValueError('Wrong permanent SFT')
    conditions=TransferConditions(actor,parent)
    policy=TransferPolicy(actor.action_expert,transfer['architecture']).eval()
    checkpoint=run/'checkpoints/update020000_inner0'
    files=json.loads((checkpoint/'files.json').read_text())
    seal=json.loads((checkpoint/'COMPLETE').read_text())
    if (seal['updates']!=20000 or seal['phase']!='COMPLETE' or
        file_sha(checkpoint/'files.json')!=seal['files_sha256'] or
        file_sha(checkpoint/'optimizer.pt')!=files['optimizer.pt']):raise ValueError('Unsealed RL endpoint')
    service=RewardService(Path.cwd(),workers=1)
    act_norm=int(OmegaConf.load(str(Path(SFT_EXPORT)/'config.yaml')).datasets.vla_data.act_norm)
    record_identity=dict(run_identity=identity['sha256'],checkpoint_files_sha256=seal['files_sha256'],
        parent=parent,registration_sha256=fingerprint(registration),condition=conditions.sha,
        scoring=service.protocol,source_sha256=file_sha(__file__),host=host,
        root_seed=2101,root_and_innovation_RNG='separate scene-bound streams shared across samplers/checkpoints',
        inference='cached original observation only; G8 mean is diagnostic, not formal single-candidate score')
    atomic_json(output/'identity.json',record_identity)
    rows=[];reference_hash=tensor_digest({n:p for n,p in actor.ref_action_expert.named_parameters()})
    try:
        for stage in ('SFT','RL'):
            if stage=='RL':
                saved=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
                if {int(v['step']) for v in saved['optimizer']['state'].values()}!={20000}:
                    raise ValueError('Wrong optimizer boundary')
                restore_adapters(actor,saved['adapters']);del saved
            for index,token in enumerate(tokens):
                raw=conditions.raw.get(token).to(actor.device,dtype=torch.float32)
                context=conditions.get(token,actor.device).expand(8,-1,-1).contiguous()
                root_rng=torch.Generator(device=actor.device).manual_seed(seed_for('failure_probe_root',2101,token))
                root=torch.randn((8,8,4),device=actor.device,generator=root_rng)
                modes={}
                for mode in ('E_NATIVE','E_SHIFT5','SOURCE_SDE'):
                    actor._noise_audit=[]
                    generator=torch.Generator(device=actor.device).manual_seed(seed_for('failure_probe_innovation',2101,token))
                    if mode=='E_NATIVE':
                        actions=policy.sample_ode(raw.expand(8,-1,-1).contiguous(),root,grid='native')
                        chain=None;indices=[]
                    else:
                        chain,times,deltas,idx=actor.sample_action_chain({'context':context},8,
                            deterministic=mode=='E_SHIFT5',init_actions=root.clone(),generator=generator)
                        actions=chain[:,-1];indices=idx.tolist()
                        if indices!=([7,8,9] if mode=='SOURCE_SDE' else []):raise ValueError('Wrong stochastic window')
                        if index==0 and mode=='E_SHIFT5':
                            direct=policy.sample_ode(raw.expand(8,-1,-1).contiguous(),root,grid='shift5')
                            torch.testing.assert_close(actions,direct,atol=1e-6,rtol=1e-6)
                    request=decode_request(actions.cpu().numpy(),act_norm=act_norm)
                    records=service.score((token,pop['token_logs'][token],request))
                    stat=means(records)
                    modes[mode]=dict(actions=actions.cpu(),chain=chain.cpu() if chain is not None else None,
                        records=records,summary=stat,random_steps=indices)
                    rows.append(dict(stage=stage,token=token,log=pop['token_logs'][token],mode=mode,**stat))
                path=output/stage/(token+'.pt');path.parent.mkdir(exist_ok=True)
                atomic_torch(path,dict(token=token,stage=stage,root=root.cpu(),modes=modes))
                atomic_json(path.with_suffix('.json'),dict(payload_sha256=file_sha(path),stage=stage,token=token,
                    registration_sha256=fingerprint(registration),root_sha256=tensor_digest(root)))
                atomic_json(output/'progress.json',dict(stage=stage,scenes_done=index+1,scenes_per_stage=64))
        if reference_hash!=tensor_digest({n:p for n,p in actor.ref_action_expert.named_parameters()}):
            raise ValueError('Reference changed during read-only probe')
        result=dict(status='COMPLETE_MATCHED_TRAIN_SAMPLER_PROBE',identity=record_identity,rows=rows,
            scenes=64,group_size=8,new_candidates=64*8*3*2,actual_scoring_calls=64*8*3*2,
            optimizer_updates=0,wall_seconds=time.monotonic()-started,
            prefix_grid_parity='first scene both checkpoints compared to original policy ODE',
            limits=['Fixed TRAIN diagnostic subset, not unbiased full TRAIN or Navtest.',
                'Eight roots are within-group sampling, not independent training replications.',
                'SDE/ODE means use matched scene/root populations; max is finite opportunity only.'])
        atomic_json(output/'summary_private.json',result)
        atomic_json(output/'COMPLETE.json',dict(status=result['status'],summary_sha256=file_sha(output/'summary_private.json')))
        print(json.dumps({k:result[k] for k in ('status','scenes','new_candidates','wall_seconds')}))
    finally:service.close()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--registration',required=True,type=Path)
    a=p.parse_args();run_probe(a.run,a.output,json.loads(a.registration.read_text()))


if __name__=='__main__':main()

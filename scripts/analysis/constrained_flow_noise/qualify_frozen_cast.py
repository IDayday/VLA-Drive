"""Test same-layout full-chain equality and version refresh of frozen FP32 casts."""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch

from starVLA.rl.constrained_flow_noise.contracts import require_dispatch,file_sha
from starVLA.rl.constrained_flow_noise.loading import make_noise_policy
from starVLA.rl.constrained_flow_noise.rollout_graph import configure
from starVLA.rl.flow_grpo.checkpoint import capture_rng
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json
from .frozen_weight_cast import install
from .qualify_collection_overlap import exact


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();host=require_dispatch();configure_numerics();torch.cuda.set_device(0)
    root=Path(a.run_root);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    cal=json.loads((root/'calibration/calibration_lock.json').read_text())
    report=dict(status='RUNNING',host=host,checks=[],source_sha256=file_sha(__file__),
                adapter_sha256=file_sha(Path(__file__).with_name('frozen_weight_cast.py')),
                scope='Frozen behavior only; unchanged batch1 matmul and RNG; no scoring or optimizer')
    atomic_json(out,report)
    for variant,position in [('C_SDE',12760),('N_FLOW_NOISE',12744)]:
        actor,_,_=make_noise_policy(variant,cal['noise_std_bounds'])
        if actor.noise_head is not None:
            actor.noise_head.load_state_dict(torch.load(cal['noise_initial_path'],map_location='cpu',weights_only=True))
        actor.requires_grad_(False).eval();cached=copy.deepcopy(actor)
        casts=install(cached)
        configure(actor,True);configure(cached,True)
        path=root/f'formal_{variant}_seed42_v1/behavior_collections/000512.pt'
        if file_sha(path)!=json.loads(path.with_suffix('.json').read_text())['sha256']:raise ValueError('Changed archive')
        rows=torch.load(path,map_location='cpu',weights_only=False)['rows'][:2]
        cp=root/f'formal_{variant}_seed42_v1/checkpoints'/f'update001024_raw{position:06d}'
        model=cp/'pytorch_model/mp_rank_00_model_states.pt'
        if file_sha(model)!=json.loads((cp/'files.json').read_text())[str(model.relative_to(cp))]:raise ValueError('Changed model')
        for version in ('SFT','update1024'):
            if version=='update1024':
                state=torch.load(model,map_location='cpu',mmap=True,weights_only=False)['module']
                actor.load_state_dict(state,strict=True);cached.load_state_dict(state,strict=True)
                del state
                if casts.refreshes!=2:raise AssertionError('Behavior change did not refresh casts')
            for row in rows:
                raw=row['chain']['raw'].cuda();measured=[];expected=None
                for index,label in enumerate(['original','cached','original','cached','cached','original']):
                    policy=actor if label=='original' else cached
                    rng=capture_rng();t0=time.monotonic()
                    value=policy.sample_chain(raw,[row['token']],[row['position']],version={'probe':version})
                    torch.cuda.synchronize();elapsed=time.monotonic()-t0
                    if expected is None:expected=value
                    else:exact(expected,value,'entire_saved_chain')
                    exact(rng,capture_rng(),'RNG')
                    measured.append(dict(mode=label,seconds=elapsed,warmup=index<2))
                old=statistics.median(v['seconds'] for v in measured if v['mode']=='original' and not v['warmup'])
                new=statistics.median(v['seconds'] for v in measured if v['mode']=='cached' and not v['warmup'])
                entry=dict(variant=variant,behavior=version,position=row['position'],measurements=measured,
                           original_seconds=old,cached_seconds=new,speedup=old/new,all_chain_mu_sigma_logp_RNG_exact=True,
                           frozen_linear_modules=len(casts.entries),refreshes=casts.refreshes)
                report['checks'].append(entry);atomic_json(out,report)
                print(json.dumps({k:v for k,v in entry.items() if k!='measurements'}),flush=True)
        # The same cache must refuse grad-enabled calls, even on frozen params.
        try:
            cached.condition(rows[0]['chain']['raw'].cuda())
        except ValueError:pass
        else:raise AssertionError('Grad-enabled cast path was not rejected')
        casts.close();del actor,cached,casts,value,expected;torch.cuda.empty_cache()
    report.update(status='PASS',no_model_parameter_or_probability_changes=True)
    atomic_json(out,report)


if __name__=='__main__':main()

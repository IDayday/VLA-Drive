"""Record actual sampling means/stds without adding or reordering RNG calls."""
import argparse
import json
from pathlib import Path
import torch
from starVLA.rl.simwam_recipe.contracts import ROOT,load_recipe,require_dispatch,file_sha
from starVLA.rl.simwam_recipe.trainer_adapter import make_actor
from starVLA.rl.simwam_recipe.condition import ProjectedConditions
from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json


@torch.no_grad()
def qualify(actor,cond):
    initial=torch.Generator().manual_seed(42);start=initial.get_state().clone()
    baseline=actor.sample_action_chain(cond,8,generator=initial);end=initial.get_state().clone()
    initial.set_state(start);captured=[];innovations=[]
    sde=actor._sde_mean_std;randn=actor._randn
    def record_sde(x,v,sigma,dt):
        mean,std=sde(x,v,sigma,dt)
        captured.append(dict(mean=mean.to(x.dtype).clone(),std=std,sigma=sigma,dt=dt))
        return mean,std
    def record_noise(*args,**kwargs):
        value=randn(*args,**kwargs);innovations.append(value.clone());return value
    actor._sde_mean_std=record_sde;actor._randn=record_noise
    try:chain,ts,dt,indices=actor.sample_action_chain(cond,8,generator=initial)
    finally:del actor._sde_mean_std;del actor._randn
    if not torch.equal(chain,baseline[0]) or not torch.equal(initial.get_state(),end):
        raise ValueError('Diagnostic recording changed chain or RNG sequence')
    lp,terms=actor.action_chain_logprobs(cond,chain,ts,dt,stoch_idx=indices,return_terms=True)
    rows=[]
    for j,k in enumerate(indices.tolist()):
        entry=captured[j];eps=innovations[j+1].clamp(-5,5)
        reconstructed=entry['mean']+entry['std']*eps
        if not torch.equal(reconstructed,chain[:,k+1]):raise ValueError('Recorded distribution did not generate the actual next state')
        error=float((entry['mean']-terms['mean'][:,j]).abs().max())
        std_error=abs(entry['std']-float(terms['std'][j]))
        if error>1e-6 or std_error>1e-8:raise ValueError('Sampling/recomputation mean or std differs')
        rows.append(dict(step=k,sigma=entry['sigma'],delta=entry['dt'],std=entry['std'],
            mean_max_abs_error=error,std_abs_error=std_error,exact_next_state=True,
            innovation_clip_hits=int((innovations[j+1].abs()>5).sum())))
    audit=actor._probability_audit['current']
    return dict(steps=rows,recording_preserves_RNG_and_chain=True,
        source_clamp_hit_rate=audit['source_clamp_hit_rate'],
        raw_gaussian_and_source_clamped_fields_separate=True,
        ratio_field='source relative-clamped dimension-mean logprob, not joint density',
        native_shape=list(chain.shape),new_scoring_calls=0)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/flow_rl/simwam_source_ddp.yaml');a=p.parse_args()
    require_dispatch();configure_numerics();cfg=load_recipe(a.config)
    actor,_,_=make_actor(cfg)
    token=json.loads(Path(ROOT+'/qualification/real_TRAIN_tokens.json').read_text())['tokens'][0]
    condition=ProjectedConditions(Path(ROOT)/'projected_condition',actor.action_expert).get(token)
    cond={'context':condition.expand(8,-1,-1).contiguous()}
    zero=qualify(actor,cond)
    checkpoint=Path(a.checkpoint);files=json.loads((checkpoint/'files.json').read_text())
    if file_sha(checkpoint/'optimizer.pt')!=files['optimizer.pt']:raise ValueError('Checkpoint changed')
    saved=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
    restore_adapters(actor,saved['adapters'])
    if not any(torch.count_nonzero(v) for n,v in saved['adapters'].items() if n.endswith('lora_B')):raise ValueError('Expected trained nonzero adapter')
    nonzero=qualify(actor,cond)
    result=dict(status='PASS',recipe_variant=cfg['variant'],zero_LoRA=zero,nonzero_LoRA=nonzero,
        checkpoint_files_sha256=file_sha(checkpoint/'files.json'),
        source_3D_oracle='Reuse existing 64 mathematical checks; this is actual DDP 4D sampling instrumentation',
        heading_mapping='[x_norm,y_norm,sin_norm,cos_norm], weights [.2,1,.5,.5]; not angular MSE')
    atomic_json(Path(a.output),result);print(json.dumps(result))


if __name__=='__main__':main()

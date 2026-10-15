"""Write auditable field-level provenance, adaptations and fixed evaluation rules."""
import ast
import json
from pathlib import Path
from omegaconf import OmegaConf
from starVLA.rl.simwam_recipe.contracts import ROOT,UPSTREAM_COMMIT,file_sha,fingerprint


def flatten(d,prefix=''):
    result={}
    for k,v in d.items():
        key=prefix+'.'+k if prefix else k
        if isinstance(v,dict):result.update(flatten(v,key))
        else:result[key]=v
    return result


def main():
    out=Path('reports/ddp_simwam_recipe_20261015');up=Path(ROOT)/'upstream/SimWAM'
    source=OmegaConf.to_container(OmegaConf.load(out/'resolved_source_recipe.yaml'),resolve=True)
    a=OmegaConf.to_container(OmegaConf.load('configs/flow_rl/simwam_source_ddp.yaml'),resolve=True)
    b=OmegaConf.to_container(OmegaConf.load('configs/flow_rl/simwam_canonical_ddp.yaml'),resolve=True)
    sa,aa,bb=map(flatten,(source,a,b))
    diff={k:dict(R_SOURCE=aa.get(k),R_CANONICAL=bb.get(k)) for k in set(aa)|set(bb) if aa.get(k)!=bb.get(k)}
    assert set(diff)=={'variant','grpo.reward.progress_weight'}
    fields=[]
    task='configs/task/navsim_grpo_action_pdm_384x672_flowgrpo_lora.yaml'
    for key,value in aa.items():
        active=True;reason='Source behavior retained';origin=task
        runtime='src/simwam/trainer_grpo.py';port='scripts/analysis/simwam_recipe/train.py'
        if key.startswith('grpo.sample.'):
            runtime='src/simwam/models/wan22/simwam_grpo.py::_configure_action_grpo/sample_action_chain/action_chain_logprobs'
            port='starVLA/rl/simwam_recipe/vendor/action_math.py'
            if key in ('grpo.sample.sample_noise_std','grpo.sample.logprob_noise_std','grpo.sample.anneal_noise'):
                active=False;reason='Parsed but only bridge branch consumes it; rigorous uses noise_level and actual delta'
            elif key in ('grpo.sample.sde_step_indices','grpo.sample.infer_shift'):
                reason='Null override; source window selection and model scheduler shift remain active'
        elif key.startswith('grpo.lora.'):
            runtime='src/simwam/models/wan22/simwam_grpo.py::_configure_action_grpo; lora.py'
            port='starVLA/rl/simwam_recipe/lora_adapter.py + vendor/lora.py'
            reason='Exact DDP attention projection paths; source module/initialization/scaling preserved'
        elif key.startswith('grpo.train.'):
            runtime='src/simwam/trainer_grpo.py::_compute_advantages/_policy_loss'
            port='starVLA/rl/simwam_recipe/vendor/trainer_math.py + source_advantages.py'
            if key=='grpo.train.kl_dim_weights':reason='Required 3D to existing DDP 4D heading-pair-average adaptation; not angular-MSE equivalence'
            elif key=='grpo.train.bc_coeff' or key.startswith('grpo.train.kl_gate.') and key!='grpo.train.kl_gate.enabled':
                active=False;reason='BC/old-policy KL gate is disabled; no associated gradient/data I/O'
            elif key=='grpo.train.adv_global_std':reason='Same raw-global population std; stable FP64 group centering is an explicit numerical adaptation'
        elif key.startswith('grpo.reward.'):
            runtime='src/simwam/datasets/navsim/pdm_reward.py';port='starVLA/rl/simwam_recipe/reward_readouts.py'
            if key=='grpo.reward.metric_cache_path':active=False;reason='Source path retained for provenance; actual sealed v1 cache roots in contracts.py'
        elif key.startswith(('grpo.eval.','grpo.vis.')):
            active=False;reason='Source preview-only settings retained as provenance; replaced by fixed DEV/TRAIN paired-ODE checkpoints, no private visualization upload'
        elif key not in sa:
            origin=None;runtime=None;reason='Explicit DDP execution/evaluation/identity adaptation; see patch categories'
        else:origin='configs/train_grpo.yaml' if key not in ('learning_rate','lr_scheduler_type','lr_warmup_steps') else task
        if key=='action_infer_shift':
            origin='configs/model/simwam_navsim.yaml';runtime='src/simwam/runtime_grpo.py -> infer_action_scheduler';reason='model.action_scheduler.infer_shift=5, preserved under DDP time/sign conversion'
        if key=='gradient_clipping_effective':reason='Actual source DS engine has 0.0, despite YAML max_grad_norm=1; see qualification/source_engine.json'
        fields.append(dict(field=key,commit=UPSTREAM_COMMIT,source_file=origin,
                           resolved_source_value=sa.get(key),DDP_value=value,source_runtime=runtime,DDP_runtime=port,
                           command_line_override=False,active=active,reason=reason))
    patches={
      'verbatim_reuse':[
        dict(source='src/simwam/models/wan22/lora.py',destination='vendor/lora.py',changes='none'),
        dict(source='src/simwam/models/wan22/schedulers/scheduler_continuous.py',destination='vendor/scheduler_continuous.py',changes='none'),
        dict(source='src/simwam/utils/samplers.py',destination='vendor/samplers.py',changes='none'),
        dict(source='simwam_grpo.py and trainer_grpo.py',destination='vendor/action_math.py and vendor/trainer_math.py',changes='function bodies/decorators unchanged; imports/logger/class shell isolated; no video platform')],
      'required_DDP_adaptations':[
        'Original F-SFT Qwen/history/action configuration and frozen exact-prefix cache; no SFT architecture change',
        'DDP 8x4 relative-to-t0 normalized [x,y,sin,cos] instead of source 8x3; decode unchanged',
        'sigma=1-t_DDP, velocity_source=-velocity_DDP; original integer time buckets',
        'Exact 96 DDP self/cross-attention projection paths; 192 A/B tensors, 4718592 trainable parameters',
        'Logical [.2,1,1] -> native [.2,1,.5,.5], denominator H*2.2; not equivalent angular heading error',
        'FP32 DDP parameters/latent/velocity/means/probabilities/Adam versus source BF16 latent; source casting order retained within chosen dtype',
        'Stable FP64 centered statistics prevent equal decimal groups receiving rounding advantages; no floor or second normalization',
        'torch DDP FP32 LoRA optimizer instead of source Accelerate/ZeRO1; explicit global rank mean and real 4 updates tested',
        'Fixed DDP reduction buckets: find_unused_parameters=True, static_graph=False, bucket_cap_mb=25. All 192 gradients remain mandatory. Avoid restarting the post-first-step bucket rebuild; eight-rank exact recovery is qualified separately.',
        'Per-rank independent persistent CPU FP32 sampling RNG, identical LoRA initialization seed42',
        'DDP own complete TRAIN single-ODE canonical score<.9 hard list instead of SimWAM token list',
        'v1 canonical independent-candidate scorer reused; same simulation dual reward readout; infrastructure failures abort instead of source fallback zero',
        'Atomic native LoRA/Adam/scheduler/RNG/data/inner-state checkpoints; permanent SFT reference and cached frozen qwen projection identities',
        'Fixed complete DEV/512 TRAIN diagnostic at 0/5k/10k/15k/20k; original raw-observation paired native/shift5 deployment evaluation'],
      'intentional_algorithm_changes':[
        dict(variant='R_CANONICAL',change='Train reward EP weight 10 -> 5; TTC=5, Comfort=2; gate unchanged',comparison='R_SOURCE vs R_CANONICAL, all other resolved values equal')],
      'not_changed':['source last-three SDE including final step','source innovation and per-dimension relative logprob clamps','source dimension-mean PPO clip .005',
                     'source global raw-reward std and equal rank mean','source x-mean MSE beta50 with weaker longitudinal anchor','source four inner optimizer updates',
                     'source effective DS gradient clipping 0.0','source no BC, no learned noise/costdual/dynamic resampling'],
      'theory_boundary':'Clamped innovation/logprob and saved-state partial-stochastic-step loss are source empirical proxies, not an exact unbiased full mixed-chain policy gradient.'}
    hf=json.loads((Path(ROOT)/'upstream/HF_model_card_identity.json').read_text())
    versions=dict(paper_v1=dict(url='https://arxiv.org/abs/2608.07468v1',reported_PDMS=91.5),
                  github=dict(commit=UPSTREAM_COMMIT,README_PDMS=91.9,default_optimizer_updates=20000,README_long_command_steps=170000),
                  model_card=hf,release_training_config_identity='UNKNOWN: no authenticated original run config/log links current defaults to released 91.9 weights',
                  no_weight_download=True)
    plan=dict(status='REGISTERED_BEFORE_FORMAL_TRAINING',variants=['R_SOURCE','R_CANONICAL'],training_seed=42,inference_seeds=[2101],
              per_variant=dict(optimizer_updates=20000,fresh_global_rollouts=5000,raw_scene_exposures=40000,new_candidates=320000,
                               world=8,scene_per_rank=1,G=8,K=10,inner_updates=4,stochastic_steps=[7,8,9]),
              evaluation_updates=[0,5000,10000,15000,20000],evaluation_populations=dict(DEV=1696,TRAIN_diagnostic=512,NAVTEST=12146),
              deployment_modes=['E_NATIVE','E_SHIFT5'],reference='SFT separately evaluated on each same grid',
              final_selection_rule='Fixed update20000 endpoints for both variants, plus matching SFT; no best-checkpoint selection. Navtest lock created only after endpoint identities exist.',
              uncertainty='Fixed seed log-cluster bootstrap; excludes training/inference seed uncertainty',
              history='Navtest was observed previously; no renewed blind-test claim',
              failure_policy='Preserve native state and abort affected distributed run on implementation, scoring infrastructure or nonfinite failures; no bad-candidate removal/reward replacement',
              resource_policy='Permanent authorized Tsinghua hosts only; capacity-qualified GPU sharing allowed; old C/N remain stopped; no additional seeds or variants',
              budget_extras='Qualification, complete SFT hard screen and stage/final evaluation accounted separately')
    for name,obj in [('recipe_delta.json',dict(schema='simwam_recipe_delta_v1',formal_resolved_diff=diff,fields=fields,patches=patches)),
                     ('version_boundaries.json',versions),('experiment_registration.json',plan)]:
        (out/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(fields=len(fields),resolved_diff=diff,output=str(out))))


if __name__=='__main__':main()

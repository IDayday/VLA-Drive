"""Finite TRAIN-only, fixed-theta, matched-root ODE8 / rigorous-SDE8 probe.

prepare/summarize are CPU metadata operations; run requires the ordinary explicit
operator dispatch grant. This tool never dispatches, trains, or selects NAVTEST.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

SCHEMA = 'TRAIN_fixed_theta_matched_root_ODE8_SDE8_probe_v1'
POPULATION = Path('/mnt/project/flow-safe-improvement-20261013/population/manifest.json')
DEFAULT_ROOT = Path('/mnt/project/qwen-world-rl-optimization-20261030')
PINNED = {'PPO_ADAPTIVE':26000,'PPO_FULL':35000,'PPO_SAFETY_EP':10000}
DIAGNOSTIC_NOISE_LEVELS = (.1, .03)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def fingerprint(body):
    return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def sealed(body):
    return dict(body,sha256=fingerprint(body))


def read_sealed(path):
    d=json.loads(Path(path).read_text());b=dict(d);s=b.pop('sha256',None)
    if s!=fingerprint(b):raise ValueError('Changed seal: '+str(path))
    return d


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def validate_train(tokens,pop):
    train=set(pop['train_tokens'])|set(pop['dev_tokens']);test=set(pop['navtest_tokens'])
    import re
    physical=lambda x:re.sub(r'_\d+_\d+$','',x)
    test_logs={physical(pop['token_logs'][t]) for t in test}
    if (not tokens or len(tokens)>32 or len(set(tokens))!=len(tokens) or not set(tokens)<=train
            or set(tokens)&test or any(physical(pop['token_logs'][t]) in test_logs for t in tokens)):
        raise ValueError('Probe requires at most32 TRAIN-only physical-disjoint scenes')


def prepare_lower_noise(root,output,baseline_manifests):
    """Clone completed .1 evidence, never reselect scenes or rewrite baseline."""
    root=Path(root);output=Path(output);baseline=Path(baseline_manifests)
    if output.resolve()==baseline.resolve():raise ValueError('Lower-noise output must be separate from baseline')
    registered=read_sealed(baseline/'REGISTRATION.json');plans=[];tokens=None
    pop=json.loads(POPULATION.read_text())
    for row in registered['arms']:
        arm=row['arm'];source=Path(row['manifest']);old=read_sealed(source);folder=Path(row['output'])
        identity=read_sealed(folder/'identity.json');summary=read_sealed(folder/'SUMMARY.json')
        if (arm not in PINNED or old['sampling']['noise_level']!=.1 or old['updates']!=PINNED[arm]
                or summary['status']!='COMPLETE' or summary['identity_sha256']!=identity['sha256']
                or identity['manifest_sha256']!=old['sha256'] or identity['checkpoint']!=old['rl_checkpoint']):
            raise ValueError('Completed fixed-theta .1 baseline required')
        if tokens is None:tokens=old['tokens'];validate_train(tokens,pop)
        if old['tokens']!=tokens or old['count']!=15:raise ValueError('Keep the same completed15-scene cohort')
        for name,expected in old['checkpoint_receipts'].items():
            if sha(Path(old['rl_checkpoint'])/name)!=expected:raise ValueError('Pinned native metadata changed')
        body=copy.deepcopy(old);body.pop('sha256')
        body['sampling']['noise_level']=.03
        body['sampling']['ODE8_source']='reuse sealed baseline chain/feedback; no ODE forward or scoring'
        body.update(diagnostic_noise_level=.03,required_output=str(output/arm),
            diagnostic_delta='Only actor.grpo_noise_level .1 -> .03 in the rigorous sampler; native recipe and theta unchanged',
            baseline_probe=dict(manifest=str(source),manifest_sha256=old['sha256'],manifest_file_sha256=sha(source),
                output=str(folder),identity_sha256=identity['sha256'],canonical_protocol_sha256=identity['protocol']['sha256'],summary_file_sha256=sha(folder/'SUMMARY.json'),
                scene_file_sha256={t:sha(folder/'scenes'/f'{t}.json') for t in tokens}))
        manifest=output/(arm+'.json');value=sealed(body)
        if manifest.exists() and read_sealed(manifest)!=value:raise FileExistsError('Immutable lower-noise manifest exists')
        atomic(manifest,value)
        plans.append(dict(arm=arm,manifest=str(manifest),output=str(output/arm),
            command=['python','-m','tools.qwen_world_research.probe_zero_rollouts','run','--manifest',str(manifest),
                     '--output',str(output/arm),'--diagnostic-noise-level','0.03']))
    if {p['arm'] for p in plans}!=set(PINNED):raise ValueError('All three pinned baseline arms required')
    result=sealed(dict(schema=SCHEMA+'_registration',status='PREPARED_NOT_DISPATCHED',at=time.time(),
        diagnostic_noise_level=.03,baseline_registration=str(baseline/'REGISTRATION.json'),
        baseline_registration_file_sha256=sha(baseline/'REGISTRATION.json'),TRAIN_unique_scenes=len(tokens),arms=plans,
        total_diagnostic_candidates=len(tokens)*8*len(plans),new_generated_candidates=len(tokens)*8*len(plans),
        new_scored_candidates=len(tokens)*8*len(plans),reused_ODE_candidates=len(tokens)*8*len(plans),
        paired_population_candidates_including_reuse=len(tokens)*16*len(plans),canonical_single_ODE_reused=True,
        same_theta_same8roots_same_raw_innovations=True,NAVTEST_tokens_included=False,training_seed_unchanged=42))
    if not (output/'REGISTRATION.json').exists():atomic(output/'REGISTRATION.json',result)
    return read_sealed(output/'REGISTRATION.json')


def prepare(root,output,diagnostic_noise_level=.1,baseline_manifests=None):
    if diagnostic_noise_level not in DIAGNOSTIC_NOISE_LEVELS:raise ValueError('Diagnostic noise must be .1 or .03')
    if diagnostic_noise_level==.03:
        baseline=baseline_manifests or Path(root)/'data/probes/ZERO_ODE8_SDE8_FIXED_THETA_20261031'
        return prepare_lower_noise(root,output,baseline)
    root=Path(root);output=Path(output)
    traces=json.loads((root/'analysis/TRAIN_ONLY_ZERO_TRAJECTORIES.json').read_text())
    pop=json.loads(POPULATION.read_text());labels={};union=set()
    for arm,u in PINNED.items():
        t=traces['series'][arm];j=t['updates'].index(u)
        for i,token in enumerate(t['tokens']):
            hist=[m[i][0]<=1e-12 for m in t['metrics'][:j+1]]
            if not any(hist):continue
            flags=[]
            if all(hist):flags.append('persistent_all_observed_ODE_zero')
            if hist[-1]:flags.append('zero_at_pinned_checkpoint')
            if hist[-1] and not hist[0]:flags.append('new_zero_vs_IL')
            if any(hist) and not all(hist):flags.append('intermittent_or_recovered_zero')
            union.add(token);labels.setdefault(token,{})[arm]=flags
    # Mandatory current/persistent cases precede historical-zero controls.
    priority=lambda t:(not any('persistent_all_observed_ODE_zero' in f for f in labels[t].values()),
        not any('zero_at_pinned_checkpoint' in f for f in labels[t].values()),hashlib.sha256(('42:'+t).encode()).hexdigest())
    tokens=sorted(union,key=priority)[:32];validate_train(tokens,pop)
    data=root/'data/NAVTRAIN';data_sha=sha(data/'manifest.json');plan=[]
    for arm,u in PINNED.items():
        state=json.loads((root/f'owners/{arm}/STATE.json').read_text());stage=state['stages'][str(u)]
        cp=Path(stage['checkpoint']);native=json.loads((cp/'identity.json').read_text());complete=json.loads((cp/'COMPLETE').read_text())
        if complete['updates']!=u or complete['next_inner']!=0:raise ValueError('Pinned complete gate checkpoint changed')
        ev=Path(stage['TRAIN_DIAGNOSTIC']['spec']['output']);ei=read_sealed(ev/'identity.json')
        if Path(ei['native_RL_checkpoint']).resolve()!=cp.resolve():raise ValueError('Cached ODE has different checkpoint')
        scene_hashes={t:sha(ev/'scenes'/f'{t}.json') for t in tokens}
        body=dict(schema=SCHEMA,purpose='finite_sampler_diagnosis_not_actor_update',arm=arm,
            cohort='TRAIN_DIAGNOSTIC',NAVTEST_tokens_included=False,used_for_actor_update=False,
            tokens=tokens,count=len(tokens),token_logs={t:pop['token_logs'][t] for t in tokens},selection_labels=labels,
            selection_rule='TRAIN1024 longitudinal persistent/current/new-zero, then historical-zero controls; shared cohort; cap32',
            data=str(data),data_manifest_file_sha256=data_sha,population_file_sha256=sha(POPULATION),
            parent_checkpoint=native['IL_parent']['path'],rl_checkpoint=str(cp),updates=u,
            checkpoint_receipts={n:sha(cp/n) for n in ['COMPLETE','identity.json','files.json']},
            checkpoint_optimizer_sha256=json.loads((cp/'files.json').read_text())['optimizer.pt'],
            original_scientific_source=state['source'],
            canonical_ODE_cache=dict(output=str(ev),identity_sha256=ei['sha256'],scene_file_sha256=scene_hashes),
            sampling=dict(candidates_each=8,steps=10,shift=5.,SDE_mode='rigorous',noise_level=.1,SDE_steps=[7,8,9],
                root_seed=42,root_namespace='TRAIN_fixed_theta_ODE_SDE_probe',CPU_generator_restore_before_each_call=True,
                deployment_candidates=1,deployment_seed=2101,deployment_namespace='native_qwen_sourceflow_ODE'),
            actors_see_current_observations_only=True,scoring='official independent reference-candidate pair canonical PDM',
            no_optimizer_or_training_update=True)
        manifest=output/(arm+'.json');value=sealed(body)
        if manifest.exists() and read_sealed(manifest)!=value:raise FileExistsError('Immutable probe manifest exists: '+str(manifest))
        atomic(manifest,value)
        plan.append(dict(arm=arm,manifest=str(manifest),output=str(output/arm),
            command=['python','-m','tools.qwen_world_research.probe_zero_rollouts','run','--manifest',str(manifest),'--output',str(output/arm)]))
    registration=sealed(dict(schema=SCHEMA+'_registration',status='PREPARED_NOT_DISPATCHED',
        at=time.time(),TRAIN_unique_scenes=len(tokens),arms=plan,total_diagnostic_candidates=len(tokens)*16*len(plan),
        canonical_single_ODE_reused=True,NAVTEST_tokens_included=False,training_seed_unchanged=42))
    # Repeated preparation adopts the pinned manifests; retain first registration.
    if not (output/'REGISTRATION.json').exists():atomic(output/'REGISTRATION.json',registration)
    return json.loads((output/'REGISTRATION.json').read_text())


def tensor_hash(tensor):
    t=tensor.detach().cpu().contiguous()
    return dict(shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(t.numpy().tobytes()).hexdigest())


def configure_diagnostic_noise(actor,level):
    """Change the actual rigorous g/mean/std field, not bridge-only std fields."""
    if level not in DIAGNOSTIC_NOISE_LEVELS:raise ValueError('Diagnostic noise must be .1 or .03')
    names=('grpo_noise_level','grpo_sample_noise_std','grpo_logprob_noise_std','grpo_min_std','grpo_randn_clip')
    before={k:float(getattr(actor,k)) for k in names}
    if (actor.grpo_sde_mode!='rigorous' or before['grpo_noise_level']!=.1
            or before['grpo_min_std']!=.001 or before['grpo_randn_clip']!=5.
            or actor._stoch_step_indices(10)!=[7,8,9] or actor.grpo_merge_sde_window):
        raise ValueError('Only the native rigorous last3-step sampler can receive this diagnostic override')
    actor.grpo_noise_level=float(level)
    after={k:float(getattr(actor,k)) for k in names}
    if any(after[k]!=before[k] for k in names if k!='grpo_noise_level'):
        raise ValueError('Bridge std/minimum/clipping fields must remain unchanged')
    return dict(before=before,after=after,active_field='grpo_noise_level',
        rigorous_formula='g=eta*sqrt(sigma/(1-sigma)); mean correction proportional eta^2; std=max(g*sqrt(-dsigma),min_std)',
        bridge_fields_unchanged_and_unused_in_rigorous=True,unfloored_std_ratio=level/.1,
        mean_correction_ratio=(level/.1)**2,native_recipe_unmodified=True)


def single_sde_reusing_baseline(actor,condition,manifest,token):
    """Exactly one new SDE8 call; reuse sealed ODE8 and canonical B1 evidence."""
    baseline=manifest['baseline_probe']
    import numpy as np
    import torch
    from starVLA.rl.simwam_recipe.contracts import seed_for
    folder=Path(baseline['output']);path=folder/'scenes'/f'{token}.json'
    if sha(path)!=baseline['scene_file_sha256'][token]:raise ValueError('Baseline probe scene changed')
    row=read_sealed(path)
    if row['identity']!=baseline['identity_sha256']:raise ValueError('Baseline probe identity changed')
    if len(row.get('ODE8',[]))!=8 or not row.get('canonical_single_ODE'):
        raise ValueError('Lower-noise probe requires sealed ODE8 feedback and canonical B1; never regenerate them')
    arrays=folder/row['arrays']['path']
    if sha(arrays)!=row['arrays']['sha256']:raise ValueError('Baseline probe arrays changed')
    with np.load(arrays,allow_pickle=False) as old:
        ode=torch.from_numpy(old['ODE_chain'].copy())
        prior=torch.from_numpy(old['prior'].copy())
        innovations=torch.from_numpy(old['innovations'].copy())
    seed=seed_for('TRAIN_fixed_theta_ODE_SDE_probe',42,token)
    generator=torch.Generator(device='cpu').manual_seed(seed);initial=generator.get_state().clone()
    original=actor._randn;observed=[]
    def capture(shape,device,dtype,generator):
        value=original(shape,device,dtype,generator);observed.append(value.detach().cpu().clone());return value
    try:
        actor._randn=capture
        sde,times,deltas,indices=actor.sample_action_chain(condition,8,deterministic=False,generator=generator)
    finally:del actor._randn
    if (sde.shape!=(8,11,8,3) or len(observed)!=4 or indices.tolist()!=[7,8,9]
            or not torch.equal(observed[0],prior) or not torch.equal(torch.stack(observed[1:]),innovations)
            or not torch.equal(sde[:,0].cpu(),prior) or not torch.equal(sde[:,:8].cpu(),ode[:,:8])):
        raise ValueError('Single new SDE call changed roots/innovations/pre7 relative to sealed baseline')
    audit=dict(seed=seed,CPU_generator_state_sha256=tensor_hash(initial)['sha256'],
        root_bit_identical=True,pre_SDE_prefix_bit_identical=True,
        prior=tensor_hash(observed[0]),innovations=[dict(step=k,raw=tensor_hash(v),
            clipped=tensor_hash(v.clamp(-actor.grpo_randn_clip,actor.grpo_randn_clip)))
            for k,v in zip([7,8,9],observed[1:])])
    if (row['pairing']['prior']!=audit['prior']
            or row['pairing']['CPU_generator_state_sha256']!=audit['CPU_generator_state_sha256']
            or row['pairing']['innovations']!=audit['innovations']):
        raise ValueError('Lower-noise call must preserve all8 roots and all raw/clipped innovations')
    cross=dict(baseline_identity_sha256=baseline['identity_sha256'],baseline_scene_file_sha256=sha(path),
        same8_roots=True,same_raw_clipped_innovations=True,pre7_prefix_bit_identical=True,
        ODE_chain_reused_without_forward=True,ODE_chain_bit_identical=True,
        new_generated_candidates=8,new_scored_candidates=8,reused_ODE_candidates=8)
    return ode,sde,dict(SDE8=observed),audit,cross,row


def paired_chains(actor,condition,token):
    """Use the unchanged source sampler, restore private CPU RNG for each call."""
    import torch
    from starVLA.rl.simwam_recipe.contracts import seed_for
    seed=seed_for('TRAIN_fixed_theta_ODE_SDE_probe',42,token)
    gen=torch.Generator(device='cpu').manual_seed(seed);initial=gen.get_state().clone();draws={}
    original=actor._randn
    try:
        chains=[]
        for name,det in [('ODE8',True),('SDE8',False)]:
            gen.set_state(initial.clone());observed=[]
            def capture(shape,device,dtype,generator):
                value=original(shape,device,dtype,generator);observed.append(value.detach().cpu().clone());return value
            actor._randn=capture
            chain,times,deltas,indices=actor.sample_action_chain(condition,8,deterministic=det,generator=gen)
            draws[name]=observed;chains.append((chain,times,deltas,indices))
        ode,sde=chains
        if (ode[0].shape!=(8,11,8,3) or sde[0].shape!=(8,11,8,3)
                or len(draws['ODE8'])!=1 or len(draws['SDE8'])!=4
                or ode[3].numel()!=0 or sde[3].tolist()!=[7,8,9]
                or not torch.equal(ode[0][:,0],sde[0][:,0])
                or not torch.equal(draws['ODE8'][0],draws['SDE8'][0])
                or not torch.equal(ode[0][:,:8],sde[0][:,:8])
                or not torch.equal(ode[1],sde[1]) or not torch.equal(ode[2],sde[2])):
            raise ValueError('Fixed-root / pre-stochastic-prefix parity failed')
        audit=dict(seed=seed,CPU_generator_state_sha256=tensor_hash(initial)['sha256'],
            root_bit_identical=True,pre_SDE_prefix_bit_identical=True,
            prior=tensor_hash(draws['SDE8'][0]),innovations=[dict(step=k,raw=tensor_hash(v),
                clipped=tensor_hash(v.clamp(-actor.grpo_randn_clip,actor.grpo_randn_clip)))
                for k,v in zip([7,8,9],draws['SDE8'][1:])])
        return ode[0],sde[0],draws,audit
    finally:del actor._randn


def canonical_cached(m,token,protocol_sha):
    cache=m.get('canonical_ODE_cache')
    if not cache:return None
    folder=Path(cache['output']);identity=read_sealed(folder/'identity.json')
    if (identity['sha256']!=cache['identity_sha256']
            or Path(identity['native_RL_checkpoint']).resolve()!=Path(m['rl_checkpoint']).resolve()
            or identity['protocol']['sha256']!=protocol_sha
            or identity['inference']!=dict(sampler='ODE',steps=10,shift=5.,seed=2101,
                 scene_seed_namespace='native_qwen_sourceflow_ODE',candidates=1)):
        raise ValueError('Canonical cached model/sampler/protocol binding changed')
    p=folder/'scenes'/f'{token}.json'
    if sha(p)!=cache['scene_file_sha256'][token]:raise ValueError('Canonical cached scene changed')
    from tools.qwen_world_postcot.evaluate import validate_scene_row
    row=validate_scene_row(json.loads(p.read_text()),token=token,identity_sha256=identity['sha256'],
        protocol_sha256=protocol_sha,expected_log=m['token_logs'][token])
    return dict(source='reused_exact_checkpoint_single_candidate_ODE',row=row,
        evaluation_identity_sha256=identity['sha256'],scene_file_sha256=sha(p))


def run(args):
    # All model/scorer imports are intentionally inside the authorized run mode.
    import fcntl
    import numpy as np
    import torch
    from starVLA.rl.qwen_world_postcot.contracts import require_dispatch,boundary_stop_due
    from starVLA.rl.qwen_world_postcot.loading import load_deployment_checkpoint
    from starVLA.rl.qwen_world_postcot.scope_actor import ScopeRecipeActor
    from starVLA.rl.qwen_world_postcot.scope_conditions import MemoryLearningConditions
    from starVLA.rl.qwen_world_research_data import ResearchSceneData
    from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
    from starVLA.rl.simwam_recipe.reward_readouts import RewardService
    from starVLA.rl.flow_grpo.reproducibility import configure_numerics
    from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
    from scripts.analysis.qwen_sourceflow.evaluate_policy import evaluation_caches,preflight_caches,json_feedback
    from tools.qwen_world_postcot.evaluate import validate_feedback
    host=require_dispatch();m=read_sealed(args.manifest);pop=json.loads(POPULATION.read_text())
    if m['schema']!=SCHEMA:raise ValueError('Unknown finite probe protocol')
    if (args.diagnostic_noise_level not in DIAGNOSTIC_NOISE_LEVELS
            or args.diagnostic_noise_level!=m['sampling']['noise_level']
            or args.diagnostic_noise_level!=m.get('diagnostic_noise_level',.1)):
        raise ValueError('Manifest and CLI diagnostic noise level must agree')
    if m.get('required_output') and Path(args.output).resolve()!=Path(m['required_output']).resolve():
        raise ValueError('Lower-noise output namespace is immutable and distinct from baseline')
    validate_train(m['tokens'],pop)
    if not 0<=args.rank<args.world<=m['count']:raise ValueError('Invalid probe partition')
    if (sha(POPULATION)!=m['population_file_sha256'] or sha(Path(m['data'])/'manifest.json')!=m['data_manifest_file_sha256']):raise ValueError('Population/current observations changed')
    cp=Path(m['rl_checkpoint'])
    if any(sha(cp/n)!=v for n,v in m['checkpoint_receipts'].items()):raise ValueError('Pinned checkpoint metadata changed')
    native=json.loads((cp/'identity.json').read_text());cfg=native['recipe'];sample=cfg['grpo']['sample']
    if (native['purpose']!='formal' or native['schema']!='world_postcot_cleanx_scope_RL_v1'
            or sample['sde_mode']!='rigorous' or sample['noise_level']!=.1 or sample['num_inference_steps']!=10
            or cfg['action_infer_shift']!=5. or sample['train_sde_steps']!=3 or sample['merge_sde_window']):
        raise ValueError('Probe must retain the native registered sampler recipe')
    data=ResearchSceneData(m['data'],role='NAVTRAIN');roots=args.cache_root or evaluation_caches('TRAIN')
    tokens=m['tokens'][args.rank::args.world];preflight_caches(data,tokens,roots)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'scenes').mkdir(exist_ok=True)
    lock=(out/f'rank{args.rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('FLASH_ATTENTION_DETERMINISTIC','1')
    configure_numerics();torch.set_num_threads(args.cpu_threads);torch.cuda.set_device(0)
    model,parent=load_deployment_checkpoint(m['parent_checkpoint'],'cuda')
    if parent['model_sha256']!=native['IL_parent']['model_sha256']:raise ValueError('IL parent changed')
    actor=ScopeRecipeActor(model.actor,cfg,scene_provider=model.scene_provider,predictive_provider=model.predictive_provider)
    if sha(cp/'optimizer.pt')!=m['checkpoint_optimizer_sha256']:raise ValueError('Native adapter payload changed')
    payload=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)['adapters']
    if set(payload)!=set(native['parameter_contract']['trainable_names']):raise ValueError('Incomplete native trainable closure')
    restore_adapters(actor,payload);del payload
    noise_audit=configure_diagnostic_noise(actor,args.diagnostic_noise_level)
    conditions=MemoryLearningConditions(model,data,parent['model_sha256'],actor)
    object.__setattr__(actor,'conditions',conditions);actor.eval().requires_grad_(False);model.eval().requires_grad_(False)
    service=RewardService(Path.cwd(),workers=1,cache_roots=roots)
    if m.get('baseline_probe') and m['baseline_probe']['canonical_protocol_sha256']!=service.protocol['sha256']:
        raise ValueError('Sealed reused feedback must retain the exact current canonical protocol')
    source_files=[Path(__file__),Path('starVLA/rl/qwen_world_postcot/scope_actor.py'),Path('starVLA/rl/qwen_world_postcot/rl_actor.py'),
        Path('starVLA/rl/qwen_world_postcot/scope_conditions.py'),Path('starVLA/rl/simwam_recipe/vendor/action_math.py'),
        Path('starVLA/model/modules/qwen_world_postcot/flow.py')]
    identity=sealed(dict(schema=SCHEMA,manifest_sha256=m['sha256'],checkpoint=m['rl_checkpoint'],updates=m['updates'],
        diagnostic_noise_level=args.diagnostic_noise_level,actor_noise_override=noise_audit,
        new_generated_candidates_per_scene=8 if args.diagnostic_noise_level==.03 else 16,
        new_scored_candidates_per_scene=8 if args.diagnostic_noise_level==.03 else 16,
        reused_ODE_candidates_per_scene=8 if args.diagnostic_noise_level==.03 else 0,
        no_training=True,qualification=False,protocol=service.protocol,source={str(p):sha(p) for p in source_files},
        actor_and_deployment_share_current_head_memory=True,Qwen_features_once_per_scene=True,
        prior_pairing='restore private CPU generator before source sampler calls',
        canonical_deployment='single_ODE_seed2101; never best-of8',tokens=m['tokens'],count=m['count'],world=args.world,
        native_scientific_source_match={str(p):sha(p)==native['source_files'].get(str(p)) for p in source_files[1:]}))
    if not all(identity['native_scientific_source_match'].values()):raise ValueError('Native model/sampler scientific source changed')
    with (out/'identity.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if (out/'identity.json').exists() and read_sealed(out/'identity.json')!=identity:raise ValueError('Probe identity changed')
        atomic(out/'identity.json',identity)
    state=dict(status='RUNNING',rank=args.rank,world=args.world,identity=identity['sha256'],new=0,reused=0,host=host,start_unix=time.time())
    statepath=out/f'rank{args.rank}.json';atomic(statepath,state)
    try:
        with torch.no_grad():
            for token in tokens:
                if boundary_stop_due(host):state['status']='STOPPED_AT_SCENE_BOUNDARY';break
                dest=out/'scenes'/f'{token}.json'
                if dest.exists():
                    row=read_sealed(dest)
                    if row['identity']!=identity['sha256'] or row['token']!=token:raise ValueError('Changed cached probe row')
                    if sha(out/row['arrays']['path'])!=row['arrays']['sha256']:raise ValueError('Changed cached probe arrays')
                    state['reused']+=1;continue
                cond=conditions.get(token,actor.device);tick=time.monotonic()
                if args.diagnostic_noise_level==.03:
                    ode,sde,draws,audit,cross_level,baseline_row=single_sde_reusing_baseline(actor,cond,m,token)
                    requested=ActionCodec.denorm_odo(sde[:,-1]).cpu().numpy()
                else:
                    ode,sde,draws,audit=paired_chains(actor,cond,token);cross_level=None;baseline_row=None
                    requested=ActionCodec.denorm_odo(torch.cat([ode[:,-1],sde[:,-1]])).cpu().numpy()
                if not np.isfinite(requested).all():raise FloatingPointError('Nonfinite diagnostic trajectory')
                records=service.score((token,m['token_logs'][token],requested))
                if len(records)!=len(requested):raise ValueError('Partial scorer response')
                for i,r in enumerate(records):validate_feedback(r,token=token,log=m['token_logs'][token],requested=requested[i],protocol_sha256=service.protocol['sha256'])
                canonical=baseline_row['canonical_single_ODE'] if baseline_row else canonical_cached(m,token,service.protocol['sha256'])
                if canonical is None:
                    # Exactly model.predict's native B1 body, reuse already encoded current features.
                    from starVLA.rl.simwam_recipe.contracts import seed_for
                    c=conditions.materialize(cond,actor.device)
                    noise=torch.randn((1,8,3),generator=torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',2101,token)),dtype=torch.float32).cuda()
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        mem=model.predictive_provider(model.scene_provider(c['frozen_features'],c['base_ego']))
                        prepared=model.actor.prepare_context(mem);raw=model.policy.sample_ode(prepared,noise)
                    req=ActionCodec.denorm_odo(raw.float()).cpu().numpy();rec=service.score((token,m['token_logs'][token],req))[0]
                    validate_feedback(rec,token=token,log=m['token_logs'][token],requested=req[0],protocol_sha256=service.protocol['sha256'])
                    canonical=dict(source='native_B1_body_reusing_same_frozen_Qwen_features',feedback=json_feedback(rec),prior=tensor_hash(noise),normalized_action=raw[0].cpu().tolist())
                filename=token+'.npz';np.savez_compressed(out/'scenes'/filename,ODE_chain=ode.cpu().numpy(),SDE_chain=sde.cpu().numpy(),prior=draws['SDE8'][0].numpy(),innovations=np.stack([x.numpy() for x in draws['SDE8'][1:]]))
                row=sealed(dict(token=token,log=m['token_logs'][token],identity=identity['sha256'],status='OK',
                    selection_labels=m['selection_labels'].get(token,{}),pairing=audit,
                    cross_level_pairing=cross_level,
                    ODE8=baseline_row['ODE8'] if baseline_row else [json_feedback(x) for x in records[:8]],
                    SDE8=[json_feedback(x) for x in (records if baseline_row else records[8:])],
                    canonical_single_ODE=canonical,arrays=dict(path='scenes/'+filename,sha256=sha(out/'scenes'/filename)),seconds=time.monotonic()-tick))
                atomic(dest,row);state['new']+=1;atomic(statepath,state)
            else:state['status']='COMPLETE'
    except BaseException:
        state.update(status='FAILED_REQUIRES_REVIEW',error=traceback.format_exc());raise
    finally:
        state.update(end_unix=time.time(),Qwen_prefill_calls=conditions.prefill_calls,CUDA_peak_bytes=torch.cuda.max_memory_allocated())
        atomic(statepath,state);service.close()
    return state


def summarize(output):
    out=Path(output);identity=read_sealed(out/'identity.json')
    if identity['schema']!=SCHEMA:raise ValueError('Unknown probe')
    if (out/'SUMMARY.json').exists():
        previous=read_sealed(out/'SUMMARY.json')
        if previous['identity_sha256']!=identity['sha256']:raise ValueError('Existing probe summary identity changed')
        return previous  # Completed .1 evidence is never rewritten by the new tool.
    for rank in range(identity['world']):
        state=json.loads((out/f'rank{rank}.json').read_text())
        if state['status']!='COMPLETE' or state['identity']!=identity['sha256']:raise ValueError('All fixed shards must complete')
    rows=[read_sealed(p) for p in sorted((out/'scenes').glob('*.json'))]
    if set(x['token'] for x in rows)!=set(identity['tokens']) or len(rows)!=identity['count'] or any(x['identity']!=identity['sha256'] for x in rows):raise ValueError('Missing/foreign probe scene')
    def metric(r):return r['metrics']
    def feasible(r):return all(metric(r)[k]==1 for k in ['no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound'])
    def describe(rs):
        return dict(candidate_count=len(rs),mean_PDMS=100*sum(metric(r)['score'] for r in rs)/len(rs),
            canonical_zero=sum(metric(r)['score']<=1e-12 for r in rs),feasible=sum(feasible(r) for r in rs))
    comparisons=[]
    for row in rows:
        ode,sde=row['ODE8'],row['SDE8'];base=row['canonical_single_ODE'];fb=base['row']['feedback'] if 'row' in base else base['feedback']
        comparisons.append(dict(token=row['token'],log=row['log'],canonical_single_ODE_PDMS=100*fb['metrics']['score'],
            ODE8=describe(ode),SDE8=describe(sde),paired_candidate_PDMS_delta=[100*(metric(s)['score']-metric(o)['score']) for o,s in zip(ode,sde)],
            SDE_feasible_ODE_not=sum(feasible(s) and not feasible(o) for o,s in zip(ode,sde)),
            ODE_feasible_SDE_not=sum(feasible(o) and not feasible(s) for o,s in zip(ode,sde))))
    result=sealed(dict(schema=SCHEMA+'_summary',status='COMPLETE',identity_sha256=identity['sha256'],checkpoint=identity['checkpoint'],updates=identity['updates'],
        diagnostic_noise_level=identity.get('diagnostic_noise_level',.1),actor_noise_override=identity.get('actor_noise_override'),
        new_scored_candidates=identity.get('new_scored_candidates_per_scene',16)*len(rows),
        new_generated_candidates=identity.get('new_generated_candidates_per_scene',16)*len(rows),
        reused_ODE_candidates=identity.get('reused_ODE_candidates_per_scene',0)*len(rows),
        cross_level_pairing_verified=(all(r.get('cross_level_pairing',{}).get('ODE_chain_bit_identical',False) for r in rows)
                                     if identity.get('diagnostic_noise_level',.1)==.03 else None),
        count=len(rows),ODE8=describe([r for row in rows for r in row['ODE8']]),SDE8=describe([r for row in rows for r in row['SDE8']]),
        scenes=comparisons,interpretation='selected TRAIN diagnostic cohort; fixed theta and identical initial roots; G8 mean/feasibility are not deployment best-of8 or heldout score; no training update'))
    atomic(out/'SUMMARY.json',result);return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['prepare','run','summarize'])
    p.add_argument('--root',default=str(DEFAULT_ROOT));p.add_argument('--manifest');p.add_argument('--output',required=True)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
    p.add_argument('--cache-root',action='append');p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--diagnostic-noise-level',type=float,choices=DIAGNOSTIC_NOISE_LEVELS,default=.1)
    p.add_argument('--baseline-manifests',help='Completed .1 probe folder; .03 prepare clones its exact cohort/checkpoints/cache')
    a=p.parse_args()
    if a.mode=='prepare':result=prepare(a.root,a.output,a.diagnostic_noise_level,a.baseline_manifests)
    elif a.mode=='summarize':result=summarize(a.output)
    else:
        if not a.manifest:p.error('--manifest required for run')
        result=run(a)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

"""Finite TRAIN-only fixed-Safety10k support diagnosis; never dispatch or train.

prepare freezes the full historical failure union and physical-log matched high
controls. run executes one immutable rank::8 shard. summarize requires all rows.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

from tools.qwen_world_research.probe_zero_rollouts import sha, fingerprint, sealed, read_sealed, atomic, tensor_hash

SCHEMA='TRAIN_Safety10k_support_four_strategies_G16_v1'
STRATEGIES=('A_late','B_mixed','C_shared_root','D_ODE')
STEPS={'A_late':[7,8,9],'B_mixed':[3,6,8],'C_shared_root':[7,8,9],'D_ODE':[]}
QUALITY=('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort')
CORE_FILES=('starVLA/rl/qwen_world_postcot/scope_actor.py','starVLA/rl/qwen_world_postcot/rl_actor.py',
 'starVLA/rl/simwam_recipe/vendor/action_math.py','starVLA/model/modules/qwen_world_postcot/flow.py')


def physical(log):
    import re
    return re.sub(r'_\d+_\d+$','',log)


def good(metrics):
    return all(float(metrics[k])==1 for k in QUALITY) and float(metrics['score'])>=.9


def describe(records):
    ms=[r['metrics'] for r in records]
    safe=[all(float(m[k])==1 for k in QUALITY[:3]) for m in ms]
    quality=[good(m) for m in ms]
    return dict(candidates=len(ms),strict_quality_count=sum(quality),found_strict_quality=any(quality),
        feasible_count=sum(safe),found_feasible=any(safe),unsafe_count=len(ms)-sum(safe),
        canonical_zero=sum(float(m['score'])<=1e-12 for m in ms),
        mean_PDMS=100*sum(float(m['score']) for m in ms)/len(ms),best_PDMS=100*max(float(m['score']) for m in ms),
        best_safe_PDMS=100*max(float(m['score']) for m,s in zip(ms,safe) if s) if any(safe) else None,
        best_strict_quality_PDMS=100*max(float(m['score']) for m,q in zip(ms,quality) if q) if any(quality) else None,
        DDC_mean=sum(float(m['driving_direction_compliance']) for m in ms)/len(ms),
        DDC_all_pass=all(float(m['driving_direction_compliance'])==1 for m in ms),
        high_quality_definition='NC=DAC=TTC=Comfort1 and score>=.9; DDC reported separately')


def group_credit(records):
    import torch
    from starVLA.rl.qwen_world_safety_exploration.credit import raw_advantage_parts
    names=QUALITY[:3]+('ego_progress',)
    values=torch.tensor([[[r['metrics'][k] for k in names] for r in records]],dtype=torch.float64)
    parts=raw_advantage_parts(values)
    return dict(Safety_raw_PG_active=bool(parts['A_raw'].ne(0).any()),
        Safety_raw_A_std=float(parts['A_raw'].std(unbiased=False)),
        allfail_cost_tie=not bool(parts['feasible'].any()) and bool(parts['cost'].eq(parts['cost'][:,:1]).all()),
        unsafe_positive_candidates=int(((parts['A_raw']>0)&~parts['feasible']).sum()))


def validate_train(tokens,pop):
    train=set(pop['train_tokens'])|set(pop['dev_tokens']);test=set(pop['navtest_tokens'])
    test_logs={physical(pop['token_logs'][t]) for t in test}
    if (not tokens or len(set(tokens))!=len(tokens) or not set(tokens)<=train or set(tokens)&test
            or any(physical(pop['token_logs'][t]) in test_logs for t in tokens)):
        raise ValueError('Only unique explicit TRAIN population, physical-log disjoint from NAVTEST')


def select_cases(ledger,pop):
    """One compact-ledger pass, then only selected high-control payloads.

    Controls have all8 safety-pass, historical mean>=.9 and a real strict4/high
    candidate. Exact scorer fields are checked from that group's sealed payload.
    """
    failures={};controls={}
    with Path(ledger).open() as stream:
        for line in stream:
            row=json.loads(line);token=row['token']
            if row.get('canonical_allzero') or row.get('allfail_cost_tie'):
                failures[token]=row
            if row.get('n_feasible')==8 and sum(row['canonical_scores'])/8>=.9:
                controls.setdefault(physical(row['log']),[]).append(row)
    if not failures:raise ValueError('Full historical failure union is empty')
    tokens=sorted(failures,key=lambda t:fingerprint(['failure',42,t]));validate_train(tokens,pop)
    chosen=[];used=set(tokens);shortages=[]
    for token in tokens:
        f=failures[token];pool=sorted(controls.get(physical(f['log']),[]),key=lambda r:fingerprint(['control',42,token,r['token']]))
        selected=None
        for row in pool:
            if row['token'] in used:continue
            import torch
            path=Path(row['behavior_path'])
            if sha(path)!=row['behavior_sha256']:raise ValueError('Historical control behavior changed')
            raw=torch.load(path,map_location='cpu',weights_only=False)
            if raw.get('token')!=row['token'] or raw.get('log')!=row['log']:
                raise ValueError('Historical control token/log binding changed')
            metrics=[r['metrics'] for r in raw['records']]
            if (len(metrics)!=8 or any(not all(float(m[k])==1 for k in QUALITY[:3]) for m in metrics)
                    or sum(float(m['score']) for m in metrics)/8<.9 or not any(good(m) for m in metrics)):
                continue
            selected=row;break
        if selected is None:shortages.append(dict(token=token,physical_log=physical(f['log'])));continue
        used.add(selected['token'])
        provenance=lambda r:dict(behavior_update=r['behavior_update'],collection=r['collection'],rank=r['rank'],
            behavior_path=r['behavior_path'],behavior_sha256=r['behavior_sha256'],canonical_allzero=bool(r.get('canonical_allzero')),
            allfail_cost_tie=bool(r.get('allfail_cost_tie')),historical_mean_score=sum(r['canonical_scores'])/8)
        chosen.append(dict(token=token,log=f['log'],role='historical_failure',pair_token=selected['token'],historical=provenance(f)))
        chosen.append(dict(token=selected['token'],log=selected['log'],role='matched_high_control',pair_token=token,historical=provenance(selected)))
    if shortages:raise ValueError('No strict high-quality unused same-physical-log control: '+json.dumps(shortages))
    validate_train([r['token'] for r in chosen],pop)
    return chosen,dict(failure_union=len(failures),controls=len(failures),cases=len(chosen),
        canonical_allzero_unique=sum(bool(r.get('canonical_allzero')) for r in failures.values()),
        additional_costtie_unique=sum(not bool(r.get('canonical_allzero')) for r in failures.values()),
        selection='all canonical-allzero OR allfail-costtie TRAIN groups; no cap; one high same-physical-log control per failure',
        control='all8 NC/DAC/TTC pass, historical group mean>=.9, at least one verified NC/DAC/TTC/Comfort1 score>=.9; no replacement',
        historical='full-run TRAIN history may occur after fixed10k; never transfer historical scores to new trajectories')


def prepare(args):
    import yaml
    from starVLA.rl.qwen_world_safety_exploration.group_runtime import validate_recipe,IL70_SHA256
    pop=json.loads(Path(args.population).read_text());cfg=validate_recipe(yaml.safe_load(Path(args.config).read_text()))
    if cfg['grpo']['sample']['group_size']!=16 or cfg['research']['data_policy']!='uniform_full':
        raise ValueError('Supply the separately registered uniform G16 baseline config')
    cases,selection=select_cases(args.ledger,pop)
    cp=Path(args.rl_checkpoint);native=json.loads((cp/'identity.json').read_text());complete=json.loads((cp/'COMPLETE').read_text())
    if (complete['updates']!=10000 or complete['next_inner']!=0 or complete['world']!=8
            or native['SFT_sha256']!=IL70_SHA256 or native['recipe']['research']['arm']!='PPO_SAFETY_EP'
            or native.get('purpose')!='formal' or sha(cp/'files.json')!=complete['files_sha256']):
        raise ValueError('Frozen complete formal original SafetyEP10k checkpoint required')
    out=Path(args.output).resolve()
    inputs={k:dict(path=str(Path(getattr(args,k)).resolve()),sha256=sha(getattr(args,k))) for k in
            ('ledger','population','config','il_checkpoint','parent_selection','manifest') if k!='il_checkpoint'}
    inputs['il_checkpoint']=dict(path=str(Path(args.il_checkpoint).resolve()),model_sha256=IL70_SHA256)
    inputs['data']=dict(path=str(Path(args.data).resolve()),manifest_sha256=sha(Path(args.data)/'manifest.json'))
    files=json.loads((cp/'files.json').read_text())
    m=sealed(dict(schema=SCHEMA,purpose='finite support diagnosis, no formal initialization or actor update',
        inputs=inputs,rl_checkpoint=str(cp.resolve()),checkpoint_receipts={n:sha(cp/n) for n in ['COMPLETE','identity.json','files.json']},
        checkpoint_optimizer_sha256=files['optimizer.pt'],cases=cases,count=len(cases),selection=selection,
        strategies={k:dict(candidates=16,stochastic_steps=STEPS[k],eta=.1,shared_initial_root=k=='C_shared_root') for k in STRATEGIES},
        seed=42,seed_namespace='TRAIN_Safety10k_support_probe',world=8,partition='ordered cases[rank::8]',
        scoring='independent official reference-candidate pair for every trajectory, raw canonical5:5:2',
        diagnostic_B1='D_ODE first candidate reused, seed42 training-private root; not old seed2101 canonical evaluation',
        first8='nested subset of the G16 draw; not an independently executed G8 batch/RNG stream',
        required_output=str(out/'results'),NAVTEST_used=False,original_DEV_independent=False,optimizer_used=False))
    path=out/'MANIFEST.json'
    if path.exists() and read_sealed(path)!=m:raise FileExistsError('Immutable support probe manifest already exists')
    atomic(path,m)
    atomic(out/'LAUNCH_PLAN.json',sealed(dict(schema=SCHEMA+'_launch',status='PREPARED_NOT_DISPATCHED',manifest=str(path),manifest_sha256=m['sha256'],
        commands=[['python','-m','tools.qwen_world_safety_exploration.probe_support','run','--manifest',str(path),'--output',m['required_output'],'--rank',str(rank),'--world','8'] for rank in range(8)],
        generated_candidates=64*len(cases),scored_candidates=64*len(cases),ODE_B1_additional_candidates=0,no_GPU_dispatched=True)))
    return m


def matched_chains(actor,condition,token):
    """Instance-local override, private generator restoration, no method patching."""
    import torch
    from starVLA.rl.simwam_recipe.contracts import seed_for
    generator=torch.Generator(device='cpu').manual_seed(seed_for('TRAIN_Safety10k_support_probe',42,token))
    initial=generator.get_state().clone();original=actor._randn
    own_randn=actor.__dict__.get('_randn');had_randn='_randn' in actor.__dict__
    attrs=('grpo_sde_step_indices','grpo_train_sde_steps','grpo_sde_exclude_last_step','record_sampling_terms')
    saved={k:copy.deepcopy(getattr(actor,k)) for k in attrs}
    chains={};draws={};audit={};times0=deltas0=None
    try:
        actor.grpo_train_sde_steps=3;actor.grpo_sde_exclude_last_step=False;actor.record_sampling_terms=False
        for name in STRATEGIES:
            actor.grpo_sde_step_indices=STEPS[name] if name!='D_ODE' else None
            generator.set_state(initial.clone());observed=[];effective=[];actor._noise_audit=[]
            def capture(shape,device,dtype,generator):
                value=original(shape,device,dtype,generator)
                observed.append(value.detach().cpu().clone())
                result=value[:1].expand_as(value).clone() if name=='C_shared_root' and len(observed)==1 else value
                effective.append(result.detach().cpu().clone());return result
            actor._randn=capture
            chain,times,deltas,indices=actor.sample_action_chain(condition,8,deterministic=name=='D_ODE',generator=generator)
            if (chain.shape!=(16,11,8,3) or indices.tolist()!=STEPS[name] or len(observed)!=(1 if name=='D_ODE' else 4)
                    or not torch.isfinite(chain).all()):raise ValueError('Support probe chain/window/draw count changed')
            if times0 is None:times0,deltas0=times.detach().cpu(),deltas.detach().cpu()
            elif not torch.equal(times.cpu(),times0) or not torch.equal(deltas.cpu(),deltas0):raise ValueError('Strategy timestep changed')
            chains[name]=chain;draws[name]=observed
            audit[name]=dict(raw_root=tensor_hash(observed[0]),effective_root=tensor_hash(effective[0]),
                innovations=[dict(step=k,raw=tensor_hash(v),clipped=tensor_hash(v.clamp(-actor.grpo_randn_clip,actor.grpo_randn_clip)))
                             for k,v in zip(STEPS[name],observed[1:])],generator_final=tensor_hash(generator.get_state()),
                draw_count=len(observed),stochastic_steps=STEPS[name],noise_audit=copy.deepcopy(actor._noise_audit))
            audit[name]['transition_noise_std']={}
            for k in STEPS[name]:
                sigma=float(times[k])/float(actor.infer_action_scheduler.num_train_timesteps)
                sigma=actor.grpo_sigma_max_guard if sigma>=1 else sigma
                std=max(actor.grpo_noise_level*math.sqrt(sigma/(1-sigma))*math.sqrt(max(-float(deltas[k]),0.)),actor.grpo_min_std)
                audit[name]['transition_noise_std'][str(k)]=std
        baseline=draws['A_late']
        if any(not torch.equal(v[0],baseline[0]) for v in draws.values()):raise ValueError('Raw16 roots not paired')
        if any(not torch.equal(a,b) for name in ('B_mixed','C_shared_root') for a,b in zip(draws[name][1:],baseline[1:])):
            raise ValueError('Ordered raw innovations not paired')
        if any(audit[k]['generator_final']!=audit['A_late']['generator_final'] for k in ('B_mixed','C_shared_root')):
            raise ValueError('SDE strategies consumed different private RNG budgets')
        if (not torch.equal(chains['A_late'][:,0],chains['B_mixed'][:,0])
                or not torch.equal(chains['A_late'][:,0],chains['D_ODE'][:,0])
                or not torch.equal(chains['C_shared_root'][:,0].cpu(),baseline[0][:1].expand_as(baseline[0]))):
            raise ValueError('Effective root protocol changed')
        pairing=dict(seed=seed_for('TRAIN_Safety10k_support_probe',42,token),initial_generator=tensor_hash(initial),
            raw_roots_equal_all4=True,ordered_raw_innovations_equal_ABC=True,clipped_innovations_equal_ABC=True,
            generator_poststate_equal_ABC=True,shared_root_consumes_full16draw=True,
            mixed_innovations_match_draw_order_not_time=True,
            SDE_vs_ODE_pre7_prefix_equal=bool(torch.equal(chains['A_late'][:,:8],chains['D_ODE'][:,:8])),
            A_C_candidate0_chain_max_abs=float((chains['A_late'][0]-chains['C_shared_root'][0]).abs().max()),
            per_strategy=audit,timesteps=times0.tolist(),deltas=deltas0.tolist())
        return chains,draws,pairing
    finally:
        if had_randn:actor._randn=own_randn
        else:del actor._randn
        for k,v in saved.items():setattr(actor,k,v)


def run(args):
    import numpy as np
    import torch
    from starVLA.rl.qwen_world_safety_exploration.contracts import require_dispatch,boundary_stop_due
    from starVLA.rl.qwen_world_safety_exploration.group_runtime import GroupRuntime
    from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters
    from starVLA.rl.flow_grpo.reproducibility import configure_numerics
    from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
    from scripts.analysis.qwen_sourceflow.evaluate_policy import preflight_caches,json_feedback
    from tools.qwen_world_postcot.evaluate import validate_feedback
    host=require_dispatch();m=read_sealed(args.manifest)
    if m['schema']!=SCHEMA or args.world!=8 or not 0<=args.rank<8 or str(Path(args.output).resolve())!=m['required_output']:
        raise ValueError('Wrong immutable support-probe shard/output')
    for key,row in m['inputs'].items():
        if key in ('il_checkpoint','ledger'):continue  # sealed selection already materialized; no per-GPU full-ledger rescan
        target=Path(row['path'])/'manifest.json' if key=='data' else Path(row['path'])
        if sha(target)!=row.get('sha256',row.get('manifest_sha256')):raise ValueError('Pinned input changed: '+key)
    cp=Path(m['rl_checkpoint'])
    if any(sha(cp/n)!=v for n,v in m['checkpoint_receipts'].items()):raise ValueError('Fixed Safety10k metadata changed')
    native=json.loads((cp/'identity.json').read_text())
    if not all(sha(p)==native['source_files'].get(p) for p in CORE_FILES):raise ValueError('Native actor/sampler scientific source changed')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'scenes').mkdir(exist_ok=True)
    guard=(out/f'rank{args.rank}.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cases=m['cases'][args.rank::8]
    configure_numerics();torch.set_num_threads(args.cpu_threads);torch.cuda.set_device(0)
    runtime=GroupRuntime();i=m['inputs']
    a=SimpleNamespace(config=i['config']['path'],il_checkpoint=i['il_checkpoint']['path'],parent_selection=i['parent_selection']['path'],
        manifest=i['manifest']['path'],data=i['data']['path'],qualification=False,output=args.output)
    cfg=runtime.load_recipe(a);actor,_,_=runtime.make_actor(cfg,a,torch.device('cuda',0))
    preflight_caches(runtime.conditions.data,[r['token'] for r in cases],cfg['research']['scoring_cache_roots'])
    if sha(cp/'optimizer.pt')!=m['checkpoint_optimizer_sha256']:raise ValueError('Fixed native adapter payload changed')
    payload=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)
    weights=payload['adapters']
    if set(weights)!=set(native['parameter_contract']['trainable_names']):raise ValueError('Native trainable weights incomplete')
    restore_adapters(actor,weights);del weights,payload
    actor.eval().requires_grad_(False);runtime.conditions.qwen.eval().requires_grad_(False)
    service=runtime.make_reward_service()
    identity=sealed(dict(schema=SCHEMA,manifest_sha256=m['sha256'],cases=m['cases'],count=m['count'],world=8,
        checkpoint=m['rl_checkpoint'],checkpoint_optimizer_sha256=m['checkpoint_optimizer_sha256'],updates=10000,
        protocol=service.protocol,source_files=runtime.source_files(),tool_file_sha256=sha(__file__),
        no_training_or_formal_init=True,IL_reference_unchanged=True,seed=42,strategies=m['strategies'],
        diagnostic_B1=m['diagnostic_B1'],first8=m['first8']))
    with (out/'identity.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (out/'identity.json').exists() and read_sealed(out/'identity.json')!=identity:raise ValueError('Support identity changed')
        atomic(out/'identity.json',identity)
    state=dict(status='RUNNING',rank=args.rank,world=8,identity=identity['sha256'],host=host,
        expected_tokens=[r['token'] for r in cases],completed_tokens=[],new=0,reused=0,start_unix=time.time())
    try:
        with torch.no_grad():
            for case in cases:
                if boundary_stop_due(host):state['status']='STOPPED_AT_SCENE_BOUNDARY';break
                token=case['token'];path=out/'scenes'/f'{token}.json'
                if path.exists():
                    old=read_sealed(path)
                    if old['identity']!=identity['sha256'] or sha(out/old['arrays']['path'])!=old['arrays']['sha256']:raise ValueError('Changed existing support scene')
                    state['completed_tokens'].append(token);state['reused']+=1;continue
                tick=time.monotonic();prefill=runtime.conditions.prefill_calls;calls=actor._velocity_calls
                chains,draws,pairing=matched_chains(actor,runtime.conditions.get(token,actor.device),token)
                requested=ActionCodec.denorm_odo(torch.cat([chains[k][:,-1] for k in STRATEGIES])).cpu().numpy()
                feedback=service.score((token,case['log'],requested))
                if len(feedback)!=64:raise ValueError('Partial support scoring is not valid evidence')
                for j,r in enumerate(feedback):validate_feedback(r,token=token,log=case['log'],requested=requested[j],protocol_sha256=service.protocol['sha256'])
                arrays=out/'scenes'/f'{token}.npz'
                np.savez_compressed(arrays,**{k+'_chain':v.cpu().numpy() for k,v in chains.items()},
                    roots=draws['A_late'][0].numpy(),innovations=np.stack([v.numpy() for v in draws['A_late'][1:]]),requested=requested)
                strategies={k:dict(records=[json_feedback(r) for r in feedback[j*16:(j+1)*16]]) for j,k in enumerate(STRATEGIES)}
                frames=sum(p is not None for p in runtime.conditions.data.records[token]['images'].values())
                row=sealed(dict(schema=SCHEMA,identity=identity['sha256'],status='OK',**case,pairing=pairing,
                    strategies=strategies,diagnostic_B1_from_ODE16_first=strategies['D_ODE']['records'][0],
                    arrays=dict(path='scenes/'+arrays.name,sha256=sha(arrays)),
                    costs=dict(new_generated_candidates=64,new_scored_candidates=64,extra_B1_candidates=0,
                        Qwen_prefills=runtime.conditions.prefill_calls-prefill,current_camera_frames=frames,
                        actor_velocity_batch_calls=actor._velocity_calls-calls),seconds=time.monotonic()-tick))
                atomic(path,row);state['new']+=1;state['completed_tokens'].append(token)
                atomic(out/f'rank{args.rank}.json',state)
            else:state['status']='COMPLETE'
    except BaseException:
        state.update(status='FAILED_REQUIRES_REVIEW',error=traceback.format_exc());raise
    finally:
        state.update(end_unix=time.time(),Qwen_prefill_calls=runtime.conditions.prefill_calls)
        atomic(out/f'rank{args.rank}.json',state);service.close()
    return state


def validate_results(output):
    import numpy as np
    out=Path(output);identity=read_sealed(out/'identity.json')
    if identity['schema']!=SCHEMA or identity['world']!=8:raise ValueError('Unknown support identity')
    cases=identity['cases'];tokens=[r['token'] for r in cases]
    if len(tokens)!=len(set(tokens)) or len(tokens)!=identity['count']:raise ValueError('Duplicate support population')
    for rank in range(8):
        state=json.loads((out/f'rank{rank}.json').read_text());expected=tokens[rank::8]
        if (state['status']!='COMPLETE' or state['identity']!=identity['sha256']
                or state['expected_tokens']!=expected or state['completed_tokens']!=expected):raise ValueError('Incomplete or changed support shard')
    if {p.stem for p in (out/'scenes').glob('*.json')}!=set(tokens):raise ValueError('Missing/foreign support scenes')
    rows=[]
    for case in cases:
        row=read_sealed(out/'scenes'/f'{case["token"]}.json')
        if (row['identity']!=identity['sha256'] or row.get('status')!='OK'
                or any(row[k]!=case[k] for k in ('token','log','role','pair_token'))
                or set(row['strategies'])!=set(STRATEGIES)
                or sha(out/row['arrays']['path'])!=row['arrays']['sha256']):raise ValueError('Changed support scene join')
        pairing=row['pairing']
        if any(pairing.get(k) is not True for k in ('raw_roots_equal_all4','ordered_raw_innovations_equal_ABC',
                'clipped_innovations_equal_ABC','generator_poststate_equal_ABC','shared_root_consumes_full16draw')):
            raise ValueError('Unverified random-root/innovation pairing')
        with np.load(out/row['arrays']['path'],allow_pickle=False) as arrays:
            requested=arrays['requested'].copy();roots=arrays['roots'].copy()
            if (requested.shape!=(64,8,3) or roots.shape!=(16,8,3) or not np.isfinite(requested).all()
                    or arrays['innovations'].shape!=(3,16,8,3)):
                raise ValueError('Invalid full support arrays')
            for name in STRATEGIES:
                chain=arrays[name+'_chain']
                expected_root=np.broadcast_to(roots[:1],roots.shape) if name=='C_shared_root' else roots
                if chain.shape!=(16,11,8,3) or not np.isfinite(chain).all() or not np.array_equal(chain[:,0],expected_root):
                    raise ValueError('Stored chain/root contract changed')
        for name in STRATEGIES:
            rs=row['strategies'][name]['records']
            if len(rs)!=16:raise ValueError('Need all16 independent candidate scores')
            for j,r in enumerate(rs):
                metric=r['metrics']
                if (r.get('status')!='OK' or r.get('token')!=case['token'] or r.get('log')!=case['log']
                        or r.get('protocol_sha256')!=identity['protocol']['sha256'] or not r.get('cache',{}).get('sha256')):
                    raise ValueError('Scoring protocol/cache/token changed')
                if not np.array_equal(np.asarray(r['requested_trajectory']),requested[STRATEGIES.index(name)*16+j]):
                    raise ValueError('Scorer trajectory/array join changed')
                if any(not math.isfinite(float(v)) or not 0<=float(v)<=1 for v in metric.values()):raise ValueError('Invalid scoring metric')
                expected=metric['no_at_fault_collisions']*metric['drivable_area_compliance']*(5*metric['ego_progress']+5*metric['time_to_collision_within_bound']+2*metric['comfort'])/12
                if abs(expected-metric['score'])>5e-15:raise ValueError('Canonical score changed')
        rows.append(row)
    return identity,rows


def summarize(output):
    out=Path(output);identity,rows=validate_results(out)
    scores=[];cohorts={}
    for row in rows:
        item=dict(token=row['token'],log=row['log'],role=row['role'],pair_token=row['pair_token'],
            strategies={k:dict(full16=describe(row['strategies'][k]['records']),nested_first8=describe(row['strategies'][k]['records'][:8]),
                Safety_credit16=group_credit(row['strategies'][k]['records']),Safety_credit_first8=group_credit(row['strategies'][k]['records'][:8])) for k in STRATEGIES},
            diagnostic_B1=describe([row['diagnostic_B1_from_ODE16_first']]))
        scores.append(item)
    for role in ('historical_failure','matched_high_control'):
        selected=[r for r in rows if r['role']==role]
        cohorts[role]={k:dict(**describe([v for r in selected for v in r['strategies'][k]['records']]),
            scenes_with_quality=sum(describe(r['strategies'][k]['records'])['found_strict_quality'] for r in selected),
            scenes_with_feasible=sum(describe(r['strategies'][k]['records'])['found_feasible'] for r in selected),
            quality_when_B1_not_quality=sum(not good(r['diagnostic_B1_from_ODE16_first']['metrics'])
                and describe(r['strategies'][k]['records'])['found_strict_quality'] for r in selected),
            quality_lost_vs_B1=sum(good(r['diagnostic_B1_from_ODE16_first']['metrics'])
                and not describe(r['strategies'][k]['records'])['found_strict_quality'] for r in selected)) for k in STRATEGIES}
    contrasts={}
    for role in ('historical_failure','matched_high_control'):
        selected=[r for r in rows if r['role']==role]
        flags=lambda r,k,n=16:describe(r['strategies'][k]['records'][:n])['found_strict_quality']
        contrasts[role]=dict(A16_good_A_first8_not=sum(flags(r,'A_late') and not flags(r,'A_late',8) for r in selected),
            **{k:dict(quality_found_A_not=sum(flags(r,k) and not flags(r,'A_late') for r in selected),
                       quality_A_found_strategy_not=sum(flags(r,'A_late') and not flags(r,k) for r in selected))
               for k in ('B_mixed','C_shared_root','D_ODE')})
    result=sealed(dict(schema=SCHEMA+'_summary',status='COMPLETE',identity_sha256=identity['sha256'],count=len(rows),
        checkpoint=identity['checkpoint'],updates=10000,cohorts=cohorts,paired_support_contrasts=contrasts,scenes=scores,
        costs={k:sum(r['costs'][k] for r in rows) for k in ('new_generated_candidates','new_scored_candidates','extra_B1_candidates','Qwen_prefills','current_camera_frames','actor_velocity_batch_calls')},
        interpretation='Selected TRAIN fixed-theta support diagnostics, no actor update/no NAV evidence; best-of16 is not deployment. Nested first8 is not standalone G8. D_ODE[0] uses private42, not evaluation2101.'))
    path=out/'SUMMARY.json'
    if path.exists() and read_sealed(path)!=result:raise ValueError('Immutable complete support summary differs')
    atomic(path,result);return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['prepare','run','summarize'])
    p.add_argument('--output',required=True);p.add_argument('--manifest');p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=8)
    p.add_argument('--ledger');p.add_argument('--population');p.add_argument('--config');p.add_argument('--il-checkpoint');p.add_argument('--parent-selection');p.add_argument('--rl-checkpoint');p.add_argument('--data');p.add_argument('--cpu-threads',type=int,default=2)
    args=p.parse_args()
    required=('ledger','population','config','il_checkpoint','parent_selection','rl_checkpoint','manifest','data') if args.mode=='prepare' else ('manifest',) if args.mode=='run' else ()
    if any(getattr(args,k) is None for k in required):p.error('Missing inputs: '+','.join(k for k in required if getattr(args,k) is None))
    value=prepare(args) if args.mode=='prepare' else run(args) if args.mode=='run' else summarize(args.output)
    print(json.dumps(dict(status=value.get('status','PREPARED_NOT_DISPATCHED'),output=args.output,
        count=value.get('count'),rank=value.get('rank'),sha256=value.get('sha256')),
        ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':main()

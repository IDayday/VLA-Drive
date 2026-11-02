"""Strict B/G/M native state checks, including pending multi-scene rollouts."""
import math
import torch
import torch.distributed as dist
from starVLA.rl.qwen_world_safety_exploration.credit import canonical_components, GroupCredit
from starVLA.rl.qwen_world_safety_exploration.retention_material import digest as tensor_digest
from .contracts import validate_recipe, SCHEMA


def validate_boundary(state, window, identity, optimizer=None, scheduler=None, *, legacy=False,expected_rank=None):
    if legacy:raise ValueError('No legacy reconstruction for independent performance runs')
    cfg=validate_recipe(identity['recipe']);spec=cfg['research']['performance'];u=state['updates'];i=state['next_inner']
    pending=window is not None;ledger=state.get('collection_ledger',[])
    if state['collections']!=len(ledger) or not 0<=u<=cfg['max_steps']:
        raise ValueError('Collection ledger/update clock changed')
    expected_updates=sum(x['M'] for x in ledger)
    if pending:expected_updates-=ledger[-1]['M']-i
    if u!=expected_updates or not 0<=i<spec['inner_updates'] or (not pending and i):
        raise ValueError('Adam reuse differs from committed scene collections')
    expected_phase=('COMPLETE' if u==cfg['max_steps'] else 'INNER_COMMITTED' if pending and i else 'ROLLOUT_COMMITTED' if pending else 'READY')
    if state['phase']!=expected_phase:raise ValueError('Not a committed native boundary')
    slots=sum(x['B'] for x in ledger);valid=sum(x['valid'] for x in ledger)
    if any(state[k]!=slots for k in ('scene_cursor','raw_scene_exposures')) or state.get('valid_scene_exposures',0)!=valid:
        raise ValueError('Actual nonpadding exposure clock changed')
    if state['new_candidates']!=sum(x['B']*x['G'] for x in ledger) or state.get('padded_scene_exposures',0)!=slots-valid:
        raise ValueError('Candidate or padding counts changed')
    if (optimizer is None)!=(scheduler is None):raise ValueError('Adam/scheduler must be checked together')
    if optimizer is not None:
        opt=optimizer.state_dict() if hasattr(optimizer,'state_dict') else optimizer
        steps=[float(v['step']) for v in opt['state'].values()]
        if not u and opt['state']:raise ValueError('Fresh IL70 requires empty Adam state, not imported zero-step moments')
        if (u and len(steps)!=sum(len(g['params']) for g in opt['param_groups'])) or any(s!=u for s in steps):
            raise ValueError('Adam state reset or missing tensors')
        sch=scheduler.state_dict() if hasattr(scheduler,'state_dict') else scheduler
        if sch['last_epoch']!=u or sch['exposures']!=valid or sch['spec']!=spec['schedule']:
            raise ValueError('Scheduler update or exposure definition changed')
        stage=next(s for s in spec['stages'] if s['name']==state['current_stage'])
        peaks=[stage['action_lr'],stage['memory_lr']]
        if sch['stage']!=stage['name'] or sch['base_lrs']!=peaks or sch['population']!=spec['population']:
            raise ValueError('Native stage or per-group peak learning rates changed')
        groups=opt['param_groups']
        if (len(groups)!=2 or [g['name'] for g in groups]!=['action_lora','scene_predictive_memory']
                or [g['lr'] for g in groups]!=sch['_last_lr']):
            raise ValueError('Restored Adam learning rates must equal the registered scheduler')
        from .stage_scheduler import lr_multiplier
        if any(abs(a-p*lr_multiplier(valid/spec['population'],spec['schedule']))>1e-15 for a,p in zip(sch['_last_lr'],sch['base_lrs'])):
            raise ValueError('Saved LR is inconsistent with exposure clock')
    summary=dict(schema=SCHEMA,phase=state['phase'],updates=u,next_inner=i,collections=len(ledger),
        scene_cursor=slots,valid_scene_exposures=valid,new_candidates=state['new_candidates'],
        run=identity['sha256'],pending=pending,ledger_sha=tensor_digest(ledger))
    rank=expected_rank if expected_rank is not None else dist.get_rank() if dist.is_initialized() else state.get('performance_rank',window['rank'] if window is not None else None)
    if rank is None or state.get('performance_rank')!=rank or not 0<=rank<8:
        raise ValueError('A native rank-private state cannot validate itself as a different rank')
    summary['rank']=rank
    from .sampling_priority import stream_state_validation
    summary.update(stream_state_validation(state,run_identity=identity['sha256'],stream_identity=identity['data_stream_identity']))
    auxiliary=state.get('performance_auxiliary',{})
    if spec['algorithm'] in ('nominal_rescue','temporal_rescue','performance_max') and 'nominal_improvement' not in auxiliary:
        raise ValueError('Native nominal auxiliary calibration state is mandatory')
    if spec['algorithm']=='performance_max' and 'history' not in auxiliary:
        raise ValueError('Native S5 persistent bank is mandatory, including its initial empty state')
    from .feedback import feedback_state_validation
    summary.update(feedback_state_validation(state))
    if 'nominal_improvement' in auxiliary:
        from .nominal_improvement import validate_calibration_state
        validate_calibration_state(auxiliary['nominal_improvement'])
    if 'history' in auxiliary:
        from .persistent_history import validate_history_state
        validate_history_state(auxiliary['history'],identity['sha256'],rank,maximum_update=u-i)
    if not pending:return summary
    if 'performance_history' in window:
        from .persistent_history import validate_history_material
        validate_history_material(window['performance_history'],auxiliary['history'])
    b,g,m=window['B'],window['group_size'],window['M'];samples=window['samples']
    if window['schema']!=SCHEMA or len(samples)*identity['world']!=b or ledger[-1]['B']!=b or ledger[-1]['G']!=g or ledger[-1]['M']!=m:
        raise ValueError('Pending B/G/M changed')
    if window['behavior_version']['run']!=identity['sha256'] or window['behavior_version']['update']!=u-i:
        raise ValueError('Pending trajectories do not belong to collection-start policy')
    material=[];raw_adv=[];tokens=[];validmask=[]
    for leaf in samples:
        if leaf['behavior_version']!=window['behavior_version'] or leaf['group_size']!=g:
            raise ValueError('Multiple behavior versions in one window')
        for key,shape in dict(chain=(g,11,8,3),old_logp=(g,3),rewards=(1,g),advantages=(g,),adv_mask=(g,)).items():
            value=leaf[key]
            if value.shape!=shape or value.requires_grad or not torch.isfinite(value).all():raise ValueError('Invalid saved leaf '+key)
        if leaf['stoch_idx'].tolist()!=[7,8,9]:raise ValueError('W3 source indices changed')
        audit=leaf['old_probability']
        if not torch.allclose(audit['source_clamped_logprob'].mean((-1,-2)).cpu(),leaf['old_logp'].cpu(),atol=5e-7,rtol=0):
            raise ValueError('Stored behavior source probability changed')
        mat=canonical_components(leaf['records'],g)
        if not torch.equal(mat['canonical_rewards'],leaf['rewards'].cpu()):raise ValueError('Canonical reward changed')
        parts=GroupCredit(g).parts(mat['components'])
        if spec['algorithm']=='awfm':
            from .awfm import validate_awfm_material
            validate_awfm_material(leaf,rank=window['rank'])
        elif spec['algorithm']=='fpo':
            from .fpo_adapter import validate_fpo_material
            validate_fpo_material(leaf,rank=window['rank'])
        if 'nominal_improvement' in leaf:
            from .nominal_improvement import validate_material
            validate_material(leaf,rank=window['rank'])
        if cfg['research']['performance']['algorithm']=='temporal_rescue':
            from .temporal_credit import temporal_credit_parts
            parts=temporal_credit_parts(mat['components'],leaf['records'],base_parts=parts)
        raw=parts['A_raw'].reshape(-1)
        scale=leaf['normalization_std'].cpu()
        want=(raw/(scale+1e-8)).float()
        if not torch.allclose(leaf['advantages'].cpu(),want,atol=1e-7,rtol=1e-6):raise ValueError('Saved advantages changed')
        expected_mask=raw.ne(0).any().expand(g).float()*int(leaf['valid_scene'])
        if not torch.equal(expected_mask,leaf['adv_mask'].cpu()):raise ValueError('Padding/zero-group mask changed')
        material.append(torch.cat((mat['canonical_rewards'].double().unsqueeze(-1),mat['components']),-1))
        raw_adv.append(raw.tolist());tokens.append(leaf['token']);validmask.append(bool(leaf['valid_scene']))
    if len({t for t,v in zip(tokens,validmask) if v})!=sum(validmask):raise ValueError('Duplicate valid scenes in collection')
    summary.update(B=b,G=g,M=m,behavior=window['behavior_version'],tokens=tokens,validmask=validmask,
        material=torch.cat(material).tolist(),raw_adv=raw_adv,normalization_std=float(window['normalization_std']),
        digests=tensor_digest(window))
    return summary


def validate_ranks(summaries):
    if len(summaries)!=8:raise ValueError('Exactly8 ranks required')
    common=('schema','phase','updates','next_inner','collections','scene_cursor','valid_scene_exposures','new_candidates','run','pending','ledger_sha','stream_state_sha256')
    first={k:summaries[0][k] for k in common}
    if [s['rank'] for s in summaries]!=list(range(8)):raise ValueError('Rank-private states copied, duplicated or reordered')
    for key in ('priority_feedback_present','priority_feedback_last_bucket','priority_feedback_snapshot_sha256'):
        if any(s.get(key)!=summaries[0].get(key) for s in summaries):raise ValueError('TRAIN priority snapshot differs across ranks')
    if any({k:s[k] for k in common}!=first for s in summaries):raise ValueError('Rank state clocks disagree')
    if not first['pending']:return summaries
    tokens=[t for s in summaries for t,v in zip(s['tokens'],s['validmask']) if v]
    if len(tokens)!=len(set(tokens)):raise ValueError('B must contain different scene tokens globally')
    raw=torch.tensor([a for s in summaries for a,v in zip(s['raw_adv'],s['validmask']) if v],dtype=torch.float64)
    std=float(raw.std(unbiased=False))
    for s in summaries:
        if s['behavior']!=summaries[0]['behavior'] or abs(s['normalization_std']-std)>1e-12:
            raise ValueError('Different policy version or incorrect B*G global population standard deviation')
    return summaries

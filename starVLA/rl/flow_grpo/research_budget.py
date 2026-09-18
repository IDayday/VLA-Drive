"""An explicit 64-update F-only research budget, not production acceptance.

User-authorized longer exploration keeps the historical diagnostic/production
budgets intact. A native two-update pilot must finish on the same source,
assets and numerical profile before this larger research run is permitted.
"""
import json
from pathlib import Path
from .contracts import digest
from .loading import file_sha


def artifact(pointer, *, lines=False):
    if not isinstance(pointer,dict) or not pointer.get('path'):
        raise ValueError('research evidence pointer missing')
    path=Path(pointer['path'])
    if file_sha(path)!=pointer.get('sha256'):
        raise ValueError('research evidence changed: '+str(path))
    raw=path.read_text()
    return [json.loads(s) for s in raw.splitlines()] if lines else json.loads(raw)


def enforce_research(cfg, context=None):
    from .acceptance import validate_observed_dtypes
    import os
    runtime=cfg['runtime']
    if not 1 <= runtime['max_updates'] <= 64:
        raise ValueError('bounded research is capped at 64 updates')
    if int(os.getenv('WORLD_SIZE','1')) != 8 or runtime['accumulation_steps'] != 2:
        raise ValueError('bounded research requires tested world8/global16')
    required={
        'sampling':{'group_size':16,'num_steps':10,'noise_level':.1,
                    'temporal_noise_correlation':.8,'candidate_chunk_size':1,'transition_chunk_size':1},
        'runtime':{'scene_microbatch':1,'seed':42,'deepspeed_stage':2,
                   'numerical_profile':'bf16_zero2_fp32_partition_v2',
                   'noise_seed_schedule':'global_scene_v1'},
        'algorithm':{'inner_epochs':2,'ppo_clip_range':.02,'reference_kl_coefficient':.01,
                     'logprob_reduction':'flow_grpo_dimension_mean','advantage_normalization':'group'},
        'retention':{'original_sft_enabled':True,'original_sft_coefficient':.1},
    }
    for section,fields in required.items():
        if any(cfg[section].get(k)!=v for k,v in fields.items()):
            raise ValueError('bounded research recipe mismatch: '+section)
    if (cfg['checkpoint_contract']['variant']!='frozen_visual'
        or cfg['checkpoint_contract']['sha256']!='9a26685aa3838a2e1b89ab2d92997664afde4b259fed6683851f42781ad16eb4'
        or cfg['sampling'].get('transition_mode','flow_sde')!='flow_sde'
        or cfg['algorithm'].get('denoising_discount',1.) not in (1.,.6)
        or cfg['algorithm'].get('denoising_credit_normalization','raw_discount')!='raw_discount'
        or cfg.get('lora')!='disabled' or cfg.get('trainable_policy')!='inherit_sft'
        or cfg.get('rl_freeze_modules')!=['qwen_vl_interface.model.visual']):
        raise ValueError('bounded research is restricted to the registered F Flow comparison')
    evidence_path=runtime.get('research_evidence')
    if not evidence_path or not Path(evidence_path).is_file():
        raise ValueError('bounded research requires completed native pilot evidence')
    record=json.loads(Path(evidence_path).read_text())
    if record.get('schema_version')!=1 or record.get('status')!='RESEARCH_ONLY' or record.get('budget')!=64:
        raise ValueError('not a valid bounded research registration')
    if context is None:return  # Trainer re-enters with measured assets before loading a model.
    if context not in record.get('contexts',[]):
        raise ValueError('research source/config/asset/world context changed')
    control=artifact(record['pilot_control'])
    if control.get('status')!='PASS' or control.get('exit_codes')!=[0]:
        raise ValueError('native pilot did not finish')
    observed=artifact(record['pilot_context'])
    for key in ('executable_sha256','checkpoint_sha256','world_size'):
        if observed.get(key)!=context.get(key):raise ValueError('pilot '+key+' mismatch')
    if observed.get('resume_identity')!=context.get('resume_identity'):
        raise ValueError('pilot numerical/assets identity mismatch')
    pilot=artifact(record['pilot_config'])
    # Only the predeclared temporal weighting differs across the two research arms.
    for section in ('sampling','retention','optimizer','checkpoint_contract','paths'):
        if pilot.get(section)!=cfg.get(section):raise ValueError('pilot recipe differs: '+section)
    a,b=dict(pilot['algorithm']),dict(cfg['algorithm'])
    for value in (a,b):
        value.pop('denoising_discount',None);value.pop('denoising_credit_normalization',None)
    if a!=b:raise ValueError('pilot algorithm differs beyond temporal weighting')
    rows=artifact(record['pilot_training'],lines=True)
    if [r.get('update') for r in rows]!=[1,2] or [r.get('inner_epoch') for r in rows]!=[0,1]:
        raise ValueError('pilot must contain two completed inner epochs')
    for row in rows:
        ranks=row.get('ranks',[])
        if sorted(r.get('rank',-1) for r in ranks)!=list(range(8)):
            raise ValueError('incomplete pilot ranks')
        if row.get('scene_count')!=16 or row.get('candidate_count')!=256:
            raise ValueError('pilot scene/candidate scope differs')
        for rank in ranks:
            if rank.get('device',{}).get('type')!='cuda' or rank.get('gradient_tensors')!=672:
                raise ValueError('pilot is not actual full-parameter CUDA training')
    if rows[0].get('pre_update_ratio_min',0)<.999 or rows[0].get('pre_update_ratio_max',2)>1.001:
        raise ValueError('pilot initial behavior ratio mismatch')
    for a,b in zip(rows[0]['ranks'],rows[1]['ranks']):
        if any(a[k]!=b[k] for k in ('behavior_sha256','scene_tokens','replay_tokens')):
            raise ValueError('pilot did not reuse fixed behavior')
    validate_observed_dtypes(record['pilot_dtype'],8)
    dtype=artifact(record['pilot_dtype'])
    if digest(dtype.get('ranks'))!=digest(rows[-1]['ranks']):
        raise ValueError('dtype evidence is detached from native training rows')
    immutables=record.get('pilot_immutable',[])
    if len(immutables)!=8:raise ValueError('missing pilot immutable ranks')
    for rank,pointer in enumerate(immutables):
        if Path(pointer['path']).name != f'immutable_update000002_rank{rank}.json':
            raise ValueError('pilot immutable evidence rank mismatch')
        proof=artifact(pointer)
        if proof.get('status')!='TESTED' or proof.get('end_update')!=2 or any(proof['changed'].values()):
            raise ValueError('pilot reference/own visual changed')
        if proof['before']!=proof['after']:raise ValueError('immutable content contradicts status')
    if not runtime['activation_checkpointing']:
        comparison=artifact(record.get('checkpointing_comparison',{}))
        if comparison.get('status')!='PASS' or not comparison.get('initial_bank',{}).get('advantages_equal'):
            raise ValueError('unchecked faster checkpointing profile')
        for step in ('1','2'):
            grads=comparison.get('gradients',{}).get(step,{})
            states=comparison.get('boundaries',{}).get(step,{})
            if len(grads)!=672 or len(states)!=9:
                raise ValueError('checkpointing comparison scope incomplete')
            if not all(v.get('equal') and v.get('finite') for v in grads.values()):
                raise ValueError('checkpointing gradient comparison failed')
            if not all(v.get('equal') for f in states.values() for v in f.values()):
                raise ValueError('checkpointing optimizer comparison failed')
        if comparison['runs']['off']['training_sha256']!=record['pilot_training']['sha256']:
            raise ValueError('checkpointing comparison points to another pilot')

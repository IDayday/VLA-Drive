"""Validate original-input GRPO arm and compare with prior two matched arms."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from diversity import summarize

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/attempt01'


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(line) for line in p.open()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(p,value):
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def interval(values):
    a=np.asarray(values);rng=np.random.default_rng(20261019)
    means=a[rng.integers(0,len(a),size=(5000,len(a)))].mean(axis=1)
    return {'mean':float(a.mean()),'bootstrap95':np.quantile(means,[.025,.975]).tolist(),'n_source_drives':len(a)}


def main():
    result=read(RUN/'RESULT.json');config=read(RUN/'CONFIG.json');data=config['data']
    assert result['status']=='COMPLETE' and not result['final_test_used']
    assert sha(Path(data['source']))==data['source_sha256']
    for name,h in config['code_sha256'].items():assert sha(Path(name))==h,name
    groups=rows(RUN/'groups.jsonl');rollouts=rows(RUN/'rollouts.jsonl');updates=rows(RUN/'updates.jsonl')
    assert len(groups)==160 and len(rollouts)==1280 and len(updates)==16
    by_group=defaultdict(list)
    for row in rollouts:
        assert all(np.isfinite(list(row['metrics'].values()))) and all(0<=v<=1 for v in row['metrics'].values())
        by_group[row['group_id']].append(row)
    assert len(by_group)==len(groups)==len({r['group_id'] for r in groups})
    selected={r['token']:r for r in data['train']+data['dev']}
    train_tokens=[r['token'] for r in data['train']]
    dev_tokens={r['token'] for r in data['dev']}
    monitor_tokens={r['token'] for r in data['train_monitor']}
    assert len(train_tokens)==len(set(train_tokens))==64 and len(dev_tokens)==32 and len(monitor_tokens)==16
    assert monitor_tokens<=set(train_tokens)
    assert not {r['source_group'] for r in data['train']}&{r['source_group'] for r in data['dev']}
    sides=read(ROOT.parent/'splits/log_reason_90_10_v1/ASSIGNMENTS.json')['logs']
    assert all(sides[r['log']]=='train' for r in selected.values())
    eval_keys=defaultdict(set);train_keys=defaultdict(set)
    for group in groups:
        entries=sorted(by_group[group['group_id']],key=lambda r:r['candidate'])
        assert [r['candidate'] for r in entries]==list(range(8))
        assert all(r['token']==group['token'] and r['log']==selected[r['token']]['log'] for r in entries)
        computed=summarize([r['trajectory'] for r in entries],[r['metrics'] for r in entries])
        for key,val in computed.items():
            if isinstance(val,dict):
                for metric,x in val.items():assert abs(x-group[key][metric])<1e-10
            else:assert abs(float(val)-float(group[key]))<1e-10,(key,group['group_id'])
        assert group['rewards']==[r['metrics']['PDMS'] for r in entries]
        if group['stage']=='train':train_keys[group['arm']].add(group['token'])
        else:eval_keys[(group['stage'],group['arm'],group['role'])].add(group['token'])
    for arm in ['plain']:
        assert train_keys[arm]==set(train_tokens)
        arm_updates=[r for r in updates if r['arm']==arm]
        assert [r['window'] for r in arm_updates]==list(range(16))
        assert [t for r in arm_updates for t in r['tokens']]==train_tokens
        for r in arm_updates:
            assert r['scenes']==4 and r['candidates']==32 and r['lr_vlm']==2e-7 and r['lr_action']==1e-6
            assert r['old_new_logprob_max_abs_error']<1e-3
            assert all(np.isfinite(v) and v>0 for v in r['gradient_norms'].values())
            assert all(v>0 for v in r['max_parameter_changes'].values())
        checkpoint=result['arms'][arm];assert sha(Path(checkpoint['checkpoint']))==checkpoint['sha256']
    for stage,arm in [('initial','plain'),('final','plain')]:
        for role,expected in [('dev',dev_tokens),('train_monitor',monitor_tokens)]:assert eval_keys[(stage,arm,role)]==expected
    for token in dev_tokens|monitor_tokens:
        assert len({g['seed'] for g in groups if g['token']==token and g['stage']!='train'})==1
    # Inspect saved weights on CPU, without initializing a PPU model or optimizer.
    import torch
    base_cfg=read(ROOT.parent/'mining_v3r2/CONFIG.json')
    base_weights=torch.load(base_cfg['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    witness_keys=['qwen_vl_interface.model.model.visual.patch_embed.proj.weight','qwen_vl_interface.model.model.language_model.layers.0.self_attn.q_proj.weight','action_model.action_decoder.layer2.weight']
    saved_weight_checks={}
    for arm in ['plain']:
        weights=torch.load(result['arms'][arm]['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
        assert weights.keys()==base_weights.keys()
        assert all(weights[k].shape==base_weights[k].shape for k in weights)
        assert all(v.dtype==torch.float32 for v in weights.values() if v.is_floating_point())
        differences={k:float((weights[k]-base_weights[k].float()).abs().max()) for k in witness_keys}
        assert all(np.isfinite(v) and v>0 for v in differences.values())
        saved_weight_checks[arm]={'keys':len(weights),'fp32_floating_weights':True,'max_abs_change_from_base':differences}
        del weights
    del base_weights
    metrics=['pdms_first','pdms_mean','pdms_max','rule_pass_fraction','pairwise_ADE_m','pairwise_FDE_m','endpoint_std_x_m','endpoint_std_y_m','pairwise_final_yaw_deg','pdms_std']
    aggregate=[]
    for role in ['dev','train_monitor','train']:
        for stage,arm in [('initial','plain'),('final','plain'),('train','plain')]:
            chosen=[g for g in groups if (g['role'],g['stage'],g['arm'])==(role,stage,arm)]
            if not chosen:continue
            aggregate.append(dict(role=role,stage=stage,arm=arm,scenes=len(chosen),**{k:float(np.mean([g[k] for g in chosen])) for k in metrics},zero_variance_groups=sum(g['zero_reward_variance'] for g in chosen),all_zero_groups=sum(g['all_zero'] for g in chosen),all_one_groups=sum(g['all_one'] for g in chosen),mean_metrics={k:float(np.mean([g['mean_metrics'][k] for g in chosen])) for k in chosen[0]['mean_metrics']}))
    lookup={(g['stage'],g['arm'],g['token']):g for g in groups if g['stage']!='train'}
    paired={}
    for role,tokens in [('dev',dev_tokens),('train_monitor',monitor_tokens)]:
        tokens=sorted(tokens);paired[role]={}
        for metric in ['pdms_first','pdms_mean','pdms_max','rule_pass_fraction','pairwise_ADE_m','pairwise_FDE_m']:
            delta={arm:np.asarray([lookup[('final',arm,t)][metric]-lookup[('initial',arm,t)][metric] for t in tokens]) for arm in ['plain']}
            paired[role][metric]={arm:interval(v) for arm,v in delta.items()}
    previous_root=ROOT.parent/'grpo_probe_v2'
    assert (ROOT/'DATA.json').read_bytes()==(previous_root/'DATA.json').read_bytes()
    prior_validation=read(previous_root/'VALIDATION.json')
    assert prior_validation['status']=='ACCEPTED'
    previous=rows(previous_root/'runs/attempt01/groups.jsonl')
    old_lookup={(r['stage'],r['arm'],r['token']):r for r in previous}
    previous_rollouts={(r['token'],r['candidate']):r for r in rows(previous_root/'runs/attempt01/rollouts.jsonl') if r['stage']=='initial' and r['arm']=='plain'}
    assert sha(previous_root/'runs/attempt01/rollouts.jsonl')==prior_validation['evidence_sha256']['rollouts.jsonl']
    replay=read(RUN/'BASE_REPLAY.json')
    assert replay['status']=='PASS' and replay['candidate_count']==384
    for row in rollouts:
        if row['stage']!='initial':continue
        before=previous_rollouts[row['token'],row['candidate']]
        assert np.max(np.abs(np.asarray(row['trajectory'])-np.asarray(before['trajectory'])))<1e-6
        assert all(abs(row['metrics'][k]-v)<1e-9 for k,v in before['metrics'].items())
    inputs=rows(RUN/'inputs.jsonl')
    assert len(inputs)==160 and len({(r['stage'],r['token']) for r in inputs})==160
    navigation=['turn left','keep straight','turn right','unknown']
    allowed={f'You are an autonomous driving agent. The navigation command for the current timestep is {nav}. Your task is to plan future actions based on the understanding of the driving scene.' for nav in navigation}
    for row in inputs:
        assert not row['privileged_input'] and row['instruction'] in allowed
        assert row['instruction_sha256']==hashlib.sha256(row['instruction'].encode()).hexdigest()
        assert row['state_shape']==[1,4]
        assert row['token'] in selected
    for token in dev_tokens|monitor_tokens:
        assert len({r['input_ids_sha256'] for r in inputs if r['token']==token})==1
    comparison={}
    for role,tokens in [('dev',dev_tokens),('train_monitor',monitor_tokens)]:
        tokens=sorted(tokens);comparison[role]={}
        for metric in ['pdms_first','pdms_mean','pdms_max','rule_pass_fraction']:
            comparison[role][metric]={}
            for arm in ['without','with']:
                final_delta=[];gain_delta=[]
                for token in tokens:
                    new0,new1=lookup['initial','plain',token],lookup['final','plain',token]
                    old0,old1=old_lookup['initial',arm,token],old_lookup['final',arm,token]
                    assert new0['seed']==new1['seed']==old0['seed']==old1['seed']
                    final_delta.append(new1[metric]-old1[metric])
                    gain_delta.append((new1[metric]-new0[metric])-(old1[metric]-old0[metric]))
                comparison[role][metric][arm]={'plain_minus_other_final':interval(final_delta),'plain_minus_other_training_gain':interval(gain_delta)}
    diversity_file=ROOT/'GROUP_DIVERSITY.csv' 
    with diversity_file.open('x') as f:
        fields=['group_id','role','stage','arm','token','log']+metrics+['pdms_min','zero_reward_variance','all_zero','all_one']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(groups)
    analysis=dict(status='ACCEPTED',checked_utc=datetime.now(timezone.utc).isoformat(),aggregate=aggregate,paired=paired,counts=result['counts'],saved_weight_checks=saved_weight_checks,comparison=comparison,base_replay=replay,previous_analysis_sha256=sha(previous_root/'ANALYSIS.json'),wall_seconds=result['elapsed_s'],peak_gpu_allocated_gib=result['peak_gpu_allocated_gib'],limitations=['One seed, one pass (16 updates); not evidence of stable strong expert.', 'Internal dev drawn from Base training population; no fixed final-test use.', 'Training monitor has enriched rare reasons and is not population representative.', 'Geometric diversity is not safety or repair quality.', 'Bootstrap is exploratory across source drives; it does not capture training-seed uncertainty.', 'Follow-up third arm after examining previous results; not independent confirmation.', 'Cross-arm final scores use different input conditions; they do not isolate training-only effects.'])
    save(ROOT/'ANALYSIS.json',analysis)
    save(ROOT/'VALIDATION.json',dict(status='ACCEPTED',source_drive_overlap=0,final_test_used=False,privileged_model_inputs=False,all_original_instructions_checked=True,baseline_replay_verified=True,same_prior_data_manifest=True,complete_groups=160,complete_candidates=1280,updates_per_arm=16,all_diversity_recomputed=True,score_bounds_and_finite_trajectories=True,equal_scene_order_and_budget=True,matched_evaluation_seeds=True,gradient_and_parameter_change_receipts_checked=True,checkpoint_hashes_verified=True,code_hashes_verified=True,evidence_sha256={p.name:sha(p) for p in [RUN/'CONFIG.json',RUN/'RESULT.json',RUN/'rollouts.jsonl',RUN/'groups.jsonl',RUN/'updates.jsonl',RUN/'inputs.jsonl',RUN/'BASE_REPLAY.json',ROOT/'DATA.json',ROOT/'ANALYSIS.json',diversity_file]}))
    print(json.dumps({'wall_seconds':analysis['wall_seconds'],'peak_gpu_allocated_gib':analysis['peak_gpu_allocated_gib'],'dev':[r for r in aggregate if r['role']=='dev'],'dev_pdms_gain':paired['dev']['pdms_mean'],'comparison':comparison['dev']['pdms_mean']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

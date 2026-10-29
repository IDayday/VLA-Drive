"""Finite qualification bookkeeping; CPU fixtures are not GPU acceptance."""
import copy
import json
from pathlib import Path

import pytest
import torch

from tools.qwen_world_compare import qualify_rl as q
from starVLA.rl.qwen_world_compare.contracts import file_sha, fingerprint


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def sample():
    generator=torch.Generator().manual_seed(42)
    chain=torch.randn(8,11,8,3,generator=generator);terms=[];means=[];raws=[];clamps=[];stds=[]
    for index,step in enumerate((7,8,9)):
        mean=torch.randn(8,8,3,generator=generator);std=[.071,.043,.027][index]
        # Include a clipped tail; tests preserve the source density proxy.
        chain[:,step+1]=mean+std*torch.randn(8,8,3,generator=generator)
        if index==0:chain[0,step+1,0,0]=mean[0,0,0]+std*4.
        raw=torch.distributions.Normal(mean,torch.tensor(std)).log_prob(chain[:,step+1])
        peak=-__import__('math').log(std)-.5*__import__('math').log(2*__import__('math').pi)
        terms.append((mean,std));means.append(mean);raws.append(raw);clamps.append(raw.clamp(peak-7,peak));stds.append(std)
    raw=torch.stack(raws,1);clamped=torch.stack(clamps,1)
    return dict(chain=chain,stoch_idx=torch.tensor([7,8,9]),qualification_sampling_check=dict(status='PASS'),
        qualification_sample_terms=terms,old_logp=clamped.mean((-1,-2)),
        old_probability=dict(mean=torch.stack(means,1),std=torch.tensor(stds),
            raw_gaussian_logprob=raw,source_clamped_logprob=clamped))


def test_independent_actual_sampling_density_includes_clipped_tail():
    result=q.sampling_evidence(sample())
    assert result['status']=='PASS'and [r['step']for r in result['steps']]==[7,8,9]
    assert all(row['source_mean_logp_max_abs']==0 for row in result['steps'])


@pytest.mark.parametrize('damage',['missing_check','mean','std','raw','clamped','mean_logp','wrong_window'])
def test_actual_sampling_gate_rejects_self_consistent_ratio_without_matching_sampler(damage):
    rollout=sample()
    if damage=='missing_check':rollout['qualification_sampling_check']=None
    elif damage=='mean':rollout['qualification_sample_terms'][0][0][0,0,0]+=.01
    elif damage=='std':rollout['qualification_sample_terms'][0]=(rollout['qualification_sample_terms'][0][0],0.)
    elif damage=='raw':rollout['old_probability']['raw_gaussian_logprob'][0,0,0,0]+=.1
    elif damage=='clamped':rollout['old_probability']['source_clamped_logprob'][0,0,0,0]+=.1
    elif damage=='mean_logp':rollout['old_logp'][0,0]+=.1
    else:rollout['stoch_idx']=torch.tensor([6,7,8])
    with pytest.raises(ValueError):q.sampling_evidence(rollout)


def test_disposable_resume_preserves_original_history_and_pending_chain(tmp_path,monkeypatch):
    baseline=tmp_path/'baseline';out=tmp_path/'resumed'
    write(baseline/'identity.json',dict(sha256='native_identity'));(baseline/'resolved_recipe.yaml').write_text('fixed recipe')
    write(baseline/'source/small.py',dict(original=True))
    mid=tmp_path/'baseline/checkpoints/update000002_inner2';write(mid/'files.json',dict(original='native digest'))
    info=dict(path=str(mid),seal=dict(next_inner=2,phase='INNER_COMMITTED'),
              identity=dict(world=8,purpose='qualification_not_formal_initialization',sha256='native_identity'))
    monkeypatch.setattr(q,'checkpoint_at',lambda run,step:info)
    for rank in range(8):
        (baseline/f'metrics_rank{rank}.jsonl').write_text(''.join(json.dumps(dict(updates=i,rank=rank))+'\n'for i in range(1,5)))
        p=baseline/'behavior/collection000001'/f'rank{rank}.pt';p.parent.mkdir(parents=True,exist_ok=True);torch.save(sample(),p)
    before=file_sha(baseline/'metrics_rank0.jsonl')
    receipt=q.prepare_resume(baseline,out)
    assert receipt['formal_initialization']is False
    assert [json.loads(line)['updates']for line in(out/'metrics_rank0.jsonl').read_text().splitlines()]==[1,2]
    assert file_sha(baseline/'metrics_rank0.jsonl')==before
    assert file_sha(out/'behavior/collection000001/rank0.pt')==file_sha(baseline/'behavior/collection000001/rank0.pt')
    assert q.prepare_resume(baseline,out)==receipt
    info['seal']['next_inner']=0
    with pytest.raises(ValueError,match='inner2'):q.prepare_resume(baseline,tmp_path/'invalid')


def test_native_compare_checks_full_Adam_and_rng_instead_of_just_adapter_weights():
    left=dict(adapters=dict(A=torch.ones(2)),optimizer=dict(exp_avg=torch.ones(2)),rng=torch.tensor([1,2],dtype=torch.uint8))
    q.tree_equal(left,copy.deepcopy(left))
    right=copy.deepcopy(left);right['optimizer']['exp_avg'][0]+=1
    with pytest.raises(ValueError,match='optimizer'):q.tree_equal(left,right)
    right=copy.deepcopy(left);right['rng'][0]+=1
    with pytest.raises(ValueError,match='rng'):q.tree_equal(left,right)


def migration(tmp_path):
    before={'scientific_actor.py':'same','tools/qwen_world_compare/campaign.py':'old'}
    after={'scientific_actor.py':'same','tools/qwen_world_compare/campaign.py':'new','tools/qwen_world_compare/qualify_rl.py':'helper'}
    proof=tmp_path/'actual_tests.json';write(proof,dict(status='PASS',source_files=after))
    body=dict(schema=q.OPS_SCHEMA,status='PASS',training_source=str(tmp_path/'training'),controller_source=str(tmp_path/'controller'),
        training_source_files=before,controller_source_files=after,changes=q._ops_diff(before,after),patches={'patch':'actual'},
        patch_sha256=fingerprint({'patch':'actual'}),evidence=[dict(path=str(proof),sha256=file_sha(proof))],
        scientific_dependencies_unchanged=True,original_IL_acceptance_preserved=True,
        training_jobs_execute_training_source=True,helper_jobs_execute_controller_source=True)
    path=tmp_path/'MIGRATION.json';q.publish(path,body)
    return before,after,path,proof


def test_controller_only_lineage_preserves_exact_scientific_source_and_original_acceptance(tmp_path):
    before,after,path,_=migration(tmp_path)
    result=q.validate_execution_migration(path,tmp_path/'training',tmp_path/'controller',before,after)
    assert result['original_IL_acceptance_preserved']is True
    bad=dict(after,scientific_actor='changed')
    with pytest.raises(ValueError):q._ops_diff(before,bad)
    with pytest.raises(ValueError):q.validate_execution_migration(path,tmp_path/'different_training',tmp_path/'controller',before,after)


def test_controller_lineage_rejects_stale_or_rewritten_test_proof(tmp_path):
    before,after,path,proof=migration(tmp_path)
    write(proof,dict(status='PASS',source_files=after,rewritten=True))
    with pytest.raises(ValueError,match='evidence changed'):
        q.validate_execution_migration(path,tmp_path/'training',tmp_path/'controller',before,after)


def test_controller_migration_writer_binds_actual_diff_and_passed_source_bound_tests(tmp_path,monkeypatch):
    from tools.qwen_world_compare import campaign
    old,new=tmp_path/'old',tmp_path/'new';(old/'tools/qwen_world_compare').mkdir(parents=True);(new/'tools/qwen_world_compare').mkdir(parents=True)
    (old/'tools/qwen_world_compare/campaign.py').write_text('old controller\n')
    (new/'tools/qwen_world_compare/campaign.py').write_text('new controller\n')
    before={'tools/qwen_world_compare/campaign.py':file_sha(old/'tools/qwen_world_compare/campaign.py')}
    after={'tools/qwen_world_compare/campaign.py':file_sha(new/'tools/qwen_world_compare/campaign.py')}
    monkeypatch.setattr(campaign,'expected_source_files',lambda source:before if Path(source)==old else after)
    monkeypatch.setattr(q,'_git',lambda source:'frozen')
    evidence=tmp_path/'tests.json';write(evidence,dict(status='PASS',source_files=after))
    receipt=q.create_execution_migration(old,new,[evidence],tmp_path/'ops.json')
    assert '-old controller'in receipt['patches']['tools/qwen_world_compare/campaign.py']
    assert '+new controller'in receipt['patches']['tools/qwen_world_compare/campaign.py']
    write(evidence,dict(status='PASS',source_files=before))
    with pytest.raises(ValueError,match='exact new source'):
        q.create_execution_migration(old,new,[evidence],tmp_path/'bad.json')


def test_real_evidence_verifier_checks_eight_rank_full_native_continuation(tmp_path,monkeypatch):
    """Exercise the full reader on CPU-created files; never use its PASS as GPU evidence."""
    from tools.qwen_world_compare import campaign
    from starVLA.rl.qwen_world_compare import loading,parent_selection,runtime
    from starVLA.rl.qwen_world_compare.recovery_state import rollout_digest
    source=tmp_path/'frozen';arch_cfg=source/'configs/qwen_world_compare/arch_a.yaml'
    arch_cfg.parent.mkdir(parents=True);arch_cfg.write_text('A locked IL recipe')
    config=tmp_path/'rl.yaml';config.write_text('max_steps: 50000\naction_shape: [8, 3]\nprediction_type: x0\n')
    parent=dict(path=str((tmp_path/'own_IL').resolve()),model_sha256='ownmodel',
        identity=dict(architecture_id='A',qualification=False),seal=dict(updates=5000))
    selected=dict(architecture_id='A',selected=dict(checkpoint=parent['path'],model_sha256='ownmodel'),sha256='ownselection')
    select_path=tmp_path/'IL_selection.json';write(select_path,selected)
    files={'scientific.py':'same'};acceptance=tmp_path/'original_IL_acceptance.json'
    write(acceptance,dict(status='PASS',architecture_id='A',source_files=files,
        formal_initialized_from_qualification=False,config_sha256=file_sha(arch_cfg)))
    recipe=dict(max_steps=50000,action_shape=[8,3],prediction_type='x0')
    identity=dict(schema='qwen_world_compare_RL_v1',architecture_id='A',world=8,recipe=recipe,
        purpose='qualification_not_formal_initialization',IL_parent=parent,
        projected_condition_identity='frozeninputs',source_files=files)
    identity['sha256']=fingerprint(identity)
    monkeypatch.setattr(campaign,'expected_source_files',lambda source:files)
    monkeypatch.setattr(loading,'inspect_IL_checkpoint',lambda *a,**k:parent)
    monkeypatch.setattr(parent_selection,'validate_il_parent_selection',lambda *a:selected)
    monkeypatch.setattr(runtime,'load_recipe',lambda *a:recipe)
    baseline,resumed=tmp_path/'baseline',tmp_path/'resumed'
    checkpoints={}
    optimizer=dict(adapters={'attention.lora_A':torch.ones(2)},optimizer=dict(state={0:dict(step=torch.tensor(4.),exp_avg=torch.ones(2))},param_groups=[dict(params=[0])]),scheduler=dict(last_epoch=4))
    for run,updates in ((baseline,2),(baseline,4),(resumed,4)):
        point=run/'checkpoints'/f'update{updates:06d}_inner{updates%4}';point.mkdir(parents=True)
        write(point/'files.json',dict(native='sealed'))
        torch.save(optimizer,point/'optimizer.pt')
        checkpoints[(str(run),updates)]=dict(path=str(point),identity=identity,
            seal=dict(updates=updates,next_inner=updates%4,phase='INNER_COMMITTED'if updates==2 else'READY'))
    monkeypatch.setattr(q,'checkpoint_at',lambda run,update:checkpoints[(str(run),update)])
    for run in (baseline,resumed):
        run.mkdir(exist_ok=True);(run/'resolved_recipe.yaml').write_bytes(config.read_bytes())
    for rank in range(8):
        rollout=sample();rewards=torch.arange(8).float()[None]/10
        mean=rewards.double().mean();std=rewards.double().std(unbiased=False)
        rollout.update(cond_bg=dict(schema='qwen_world_compare_current_descriptor_v1',token=f'scene{rank}',identity='frozeninputs',repeats=8),
            timesteps=torch.linspace(1000,100,10),deltas=torch.full((10,),-.1),token=f'scene{rank}',log='TRAIN_log',
            rewards=rewards,advantages=((rewards.double()[0]-mean)/(std+1e-8)).float(),adv_mask=torch.ones(8),
            raw_group_std=float(std),global_std=float(std),advantage_statistics={},records=[],
            behavior_version=dict(run=identity['sha256'],update=0,adapter_sha256='fixedold'))
        digest=rollout_digest(rollout)
        for run,updates in ((baseline,2),(baseline,4),(resumed,4)):
            point=Path(checkpoints[(str(run),updates)]['path']);pending=updates==2
            state=dict(updates=updates,collections=1,next_inner=2 if pending else 0,
                phase='INNER_COMMITTED'if pending else'READY',scene_cursor=8,raw_scene_exposures=8,new_candidates=64,
                RL_updates_local=updates,anchor_only_updates_local=0,seconds_update=1. if run==baseline else 2.)
            torch.save(dict(state=state,rollout=rollout if pending else None,rng=dict(torch=torch.tensor([rank,42],dtype=torch.uint8))),point/f'rank{rank}.pt')
        for run in (baseline,resumed):
            rows=[dict(updates=u,collection=1,inner_completed=u,run=identity['sha256'],chain_identity=digest,
                raw_scene_exposures=8,new_candidates=64,token=f'scene{rank}',behavior=rollout['behavior_version'],
                lr=5e-7,loss=.4,gradient_norm=.01,ratio_max_abs_from_one=0. if u==1 else.002,
                source_clamp_hit_rate=0.,metrics=dict(pg=.2,anchor=.2),exploration=dict(zero_PG=False))for u in range(1,5)]
            (run/f'metrics_rank{rank}.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in rows))
            write(run/f'result_rank{rank}.json',dict(qualification=True,updates=4,
                frozen_SHA_before='frozen',frozen_SHA_after='frozen',frozen_parameters_before='frozen',frozen_parameters_after='frozen',
                changed_adapters={'attention.lora_B':True}))
        p=baseline/'behavior/collection000001'/f'rank{rank}.pt';p.parent.mkdir(parents=True,exist_ok=True);torch.save(rollout,p)
    output=tmp_path/'CPU_FIXTURE_ONLY_ACCEPTANCE.json'
    result=q.verify_real_RL('A',baseline,resumed,parent['path'],select_path,source,source,config,acceptance,output)
    assert len(result['rank_checks'])==8 and result['qualification_physical_updates']==6
    assert q.validate_acceptance(output,'A',selected,select_path,files,files,config,acceptance)['status']=='PASS'
    wrong=copy.deepcopy(selected);wrong['selected']['model_sha256']='other_IL_parent'
    with pytest.raises(ValueError,match='exact own IL'):q.validate_acceptance(output,'A',wrong,select_path,files,files,config,acceptance)
    # Corrupt one rank's recovered RNG with all adapter tensors unchanged.
    cp=Path(checkpoints[(str(resumed),4)]['path'])/'rank7.pt';value=torch.load(cp,weights_only=False);value['rng']['torch'][0]+=1;torch.save(value,cp)
    with pytest.raises(ValueError,match='RNG'):
        q.verify_real_RL('A',baseline,resumed,parent['path'],select_path,source,source,config,acceptance,tmp_path/'must_not_publish.json')


@pytest.mark.parametrize('architecture',list('ABCD'))
def test_actual_runtime_sourcecopy_is_strict_and_identity_resolves_own_architecture(tmp_path,architecture):
    import yaml
    from starVLA.rl.qwen_world_compare.runtime import load_recipe
    source=Path(q.__file__).resolve().parents[2]/'configs/qwen_world_compare/rl.yaml'
    assert yaml.safe_load(source.read_text())['architecture_id']is None
    expected=load_recipe(source,architecture)
    assert expected['architecture_id']==architecture
    first,second=tmp_path/'baseline.yaml',tmp_path/'resume.yaml'
    first.write_bytes(source.read_bytes());second.write_bytes(source.read_bytes())
    copied=q.validate_saved_recipes(first,second,source,architecture)
    assert 'exact_common_source_copy'in copied['saved_representation']
    assert copied['resolved_recipe_sha256']==fingerprint(expected)
    # Semantically equivalent serialization is not the source behavior and is
    # intentionally not accepted under the existing qualification contract.
    first.write_text(yaml.safe_dump(expected));second.write_bytes(first.read_bytes())
    with pytest.raises(ValueError,match='exact audited common source copy'):
        q.validate_saved_recipes(first,second,source,architecture)
    altered=copy.deepcopy(expected);altered['learning_rate']*=2
    first.write_text(yaml.safe_dump(altered))
    with pytest.raises(ValueError,match='exact audited common source copy'):
        q.validate_saved_recipes(first,second,source,architecture)

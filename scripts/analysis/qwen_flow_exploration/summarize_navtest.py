"""TEST-only full-population effect summary; no inference, training or dispatch.

Inputs may be complete aggregates or a parent directory of complete shards.
Shard publication reuses aggregate_sharded_evaluation and writes only derived
symlinks/statistics beside the requested analysis output. DEV/TRAIN are refused.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol
from starVLA.rl.qwen_flow_exploration import common_analysis as common
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from scripts.analysis.qwen_flow_exploration import aggregate_sharded_evaluation as sharded

EXPECTED_SCENES=12146
EXPECTED_LOGS=136
SEEDS=(2101,2102,2103,2104,2105)
ARMS={'BASE','L0-S','L1','L2','L3','L1_CENTER_V1','S1','S2','S3','P0','P1','P2','P3','P4','P5'}


def header(path,protocol,*,reference=False):
    identity=common.checked_seal(json.loads((Path(path)/'identity.json').read_text()),'TEST identity')
    if (identity.get('split')!='NAVTEST' or identity.get('scenes')!=EXPECTED_SCENES or
        identity.get('qualification') is not False or identity.get('policy_qualification') is not False):
        raise ValueError('Effect evidence requires the complete formal NAVTEST population; no DEV/TRAIN/qualification/subset')
    if (tuple(identity.get('evaluation_seeds',[]))!=SEEDS or tuple(protocol['evaluation_seeds'])!=SEEDS or
        len(protocol['splits']['NAVTEST'])!=EXPECTED_SCENES or
        identity.get('tokens_sha256')!=fingerprint(protocol['splits']['NAVTEST'])):
        raise ValueError('NAVTEST requires all registered12146 scenes and exactly five fixed seeds2101..2105')
    if identity.get('schema')!='three_route_partitioned_single_deployment_v1':
        raise ValueError('Use the locked, qualified common partitioned deployment evaluator')
    common.deployment_contract(identity,protocol)
    if common.readout_family(identity)!='stochastic_root_single':raise ValueError('TEST primary requires the registered five stochastic-root outputs, not a mean-root readout')
    if (identity.get('protocol_sha256')!=protocol['sha256'] or identity.get('base_model_sha256')!=protocol['base_model_sha256'] or
        identity.get('scorer_config_hash')!=protocol['scorer']['sha256'] or
        identity.get('sampler_config_hash')!=fingerprint(protocol['samplers'])):
        raise ValueError('TEST scorer/sampler/common parent mismatch')
    arm=identity.get('arm');family=identity.get('family')
    if arm not in ARMS or identity.get('training_seed')!=42:raise ValueError('Unknown or unregistered final method/training seed')
    expected_family=('base' if arm in ('BASE','L0-S') else 'latent' if arm in ('L1','L2','L3','L1_CENTER_V1') else 'adapters')
    if family!=expected_family:raise ValueError('Method name and deployed policy family differ')
    if reference and (arm!='BASE' or family!='base' or identity.get('root_scale')!=1. or identity.get('checkpoint_state') is not None):
        raise ValueError('Paired TEST reference must be the unchanged BASE with standard root')
    if family=='base':
        if arm not in ('BASE','L0-S') or identity.get('root_scale') not in ((1.,) if arm=='BASE' else (.7,1.3)):
            raise ValueError('BASE/fixed-scale contract mismatch')
        if any(identity.get(key) is not None for key in ['checkpoint_state','root_policy_sha256','adapter_sha256','package_identity_sha256']):
            raise ValueError('BASE/fixed-scale must not silently load a trained policy')
    elif family in ('latent','adapters'):
        expected='latent' if arm in ('L1','L2','L3','L1_CENTER_V1') else 'adapters'
        state=identity.get('checkpoint_state') or {}
        if family!=expected or state.get('phase')!='COMPLETE' or state.get('updates')!=256:
            raise ValueError('Learned methods must use their registered complete256-update endpoint')
    else:raise ValueError('Unknown deployment family')
    return identity


def resolve(path,protocol,derived,*,reference=False):
    path=Path(path)
    if (path/'COMPLETE.json').is_file():
        header(path,protocol,reference=reference)
        return path
    candidates=sorted({p.parent for root in [path,path/'shards'] for p in root.glob('*/execution.json')})
    if not candidates:raise ValueError('No complete aggregate or completed shard set: '+str(path))
    for candidate in candidates:
        header(candidate,protocol,reference=reference)
        if not (candidate/'SHARD_COMPLETE.json').is_file():raise ValueError('An inference shard is not complete')
    # Existing helper checks disjoint ranks, global scene/seed count, records,
    # complete invocations and full official feedback before atomic publication.
    sharded.aggregate(protocol,candidates,derived)
    return Path(derived)


def paired_headers(current,reference):
    for key in ['source_files','execution_equivalence_sha256','protocol_sha256','base_model_sha256',
                'scorer_config_hash','sampler_config_hash','deployment_contract','root_innovations']:
        if current.get(key)!=reference.get(key):raise ValueError('Unmatched TEST source/execution/score/parent contract: '+key)


def evidence(delta,interval):
    low,high=interval
    if delta>0 and low<=0:
        return '测试点估计为正，但配对区间包含零，稳定性不足；不能据此认定方法提升成立。'
    if delta>0:
        return '完整NAVTEST的测试均值提高，固定推理种子下配对log区间为正；仅一次训练，训练稳定性尚未验证。'
    if delta<0 and high<0:
        return '完整NAVTEST的测试均值下降，固定推理种子下配对log区间为负。'
    if delta<0:
        return '测试点估计为负，配对区间包含零；差异方向尚不确定。'
    return '当前完整NAVTEST点估计无净差异；不能据此认定两方法等价。'


def single2101(current,reference,ci,ri,scene_rows,*,repeats,seed):
    names=common.REQUIRED_METRICS;current_values=[];reference_values=[];logs=[]
    for scene in scene_rows:
        token=scene['scene_token'];logs.append(scene['log_id']);pair=[]
        for path,identity in [(current,ci),(reference,ri)]:
            row=common.checked_seal(json.loads((path/'scenes'/f'{token}_seed2101.json').read_text()),'TEST2101 row')
            if (row['scene_token']!=token or row['seed']!=2101 or row['status']!='OK' or
                row['evaluation_identity']!=identity['sha256'] or row['log_id']!=scene['log_id']):
                raise ValueError('Single-seed record changed after full-population verification')
            pair.append([row['native_subscores'][name] for name in names])
        current_values.append(pair[0]);reference_values.append(pair[1])
    c,r=np.asarray(current_values),np.asarray(reference_values);delta=c[:,0]-r[:,0]
    bootstrap=common.bootstrap(delta,logs,repeats=repeats,seed=seed,kind='score')
    zero=(r[:,0]>0)&(c[:,0]==0);repair=(r[:,0]==0)&(c[:,0]>0)
    safety={}
    for name in common.SAFETY:
        index=names.index(name);cv,rv=c[:,index]<1,r[:,index]<1
        safety[name]=dict(current_mean_percent=float(100*c[:,index].mean()),reference_mean_percent=float(100*r[:,index].mean()),
                          new_violations=int(((~rv)&cv).sum()),recoveries=int((rv&(~cv)).sum()),
                          violation_rate_difference=common.bootstrap(cv.astype(float)-rv.astype(float),logs,repeats=repeats,seed=seed))
    return dict(inference_seed=2101,readout='existing single output seed2101; not another inference job or best-of',
                scenes=len(scene_rows),logs=len(set(logs)),current_PDMS=float(100*c[:,0].mean()),reference_PDMS=float(100*r[:,0].mean()),
                delta_PDMS_points=float(100*delta.mean()),paired_PDMS_log_cluster_bootstrap=bootstrap,
                G_plus_PDMS_points=float(100*np.maximum(delta,0).mean()),G_minus_PDMS_points=float(-100*np.minimum(delta,0).mean()),
                new_zero_count=int(zero.sum()),zero_recoveries=int(repair.sum()),safety=safety,
                current_native_mean_percent={name:float(100*c[:,i].mean()) for i,name in enumerate(names)},
                reference_native_mean_percent={name:float(100*r[:,i].mean()) for i,name in enumerate(names)},
                rare_event_note='No observed new failures, including a zero-width bootstrap, does not establish zero true failure probability.',
                evidence=evidence(float(100*delta.mean()),bootstrap['CI95_PDMS_points']))


def summarize(evaluation,reference,protocol,output,*,repeats=2000,bootstrap_seed=42):
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve earlier analysis output: '+str(output))
    derived=output.with_name(output.stem+'_aggregates')
    current=resolve(evaluation,protocol,derived/'evaluation')
    base=resolve(reference,protocol,derived/'BASE',reference=True)
    ci,ri=header(current,protocol),header(base,protocol,reference=True);paired_headers(ci,ri)
    full,rows=common.analyze_pair(current,base,protocol,reference_strata=None,repeats=repeats,bootstrap_seed=bootstrap_seed)
    if full['scenes']!=EXPECTED_SCENES or full['logs']!=EXPECTED_LOGS or len(rows)!=EXPECTED_SCENES:
        raise ValueError('Incomplete/mismatched NAVTEST12146 scene/136log denominator')
    one=single2101(current,base,ci,ri,rows,repeats=repeats,seed=bootstrap_seed)
    csv_path=output.with_name(output.stem+'_paired_scenes.csv')
    if csv_path.exists():raise FileExistsError(csv_path)
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    with csv_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    complete={label:json.loads((path/'COMPLETE.json').read_text()) for label,path in [('current',current),('BASE',base)]}
    result=dict(schema='full_NAVTEST_only_effect_v1',status='COMPLETE',arm=ci['arm'],root_scale=ci.get('root_scale'),
                effect_basis='Full NAVTEST only; no DEV/VAL/TRAIN scores or labels used',
                protocol_sha256=protocol['sha256'],evaluation_identity=ci['sha256'],reference_identity=ri['sha256'],
                five_seed_mean=full,seed2101=one,
                evidence=evidence(full['delta_PDMS_points'],full['paired_PDMS_log_cluster_bootstrap']['CI95_PDMS_points']),
                practical_scale=dict(threshold_raw=.005,threshold_PDMS_points=.5,
                                     point_estimate_reaches_scale=full['delta_PDMS_points']>=.5,
                                     note='Independent engineering scale, not a reason to hide smaller positive/negative results'),
                training_repetitions=0 if ci['family']=='base' else 1,training_seeds=[ci['training_seed']],inference_seeds=list(SEEDS),
                uncertainty='Learned methods have one training seed42 only. BASE/fixed scales add no training and inherit the common base. Five inference seeds are not five training repetitions; log bootstrap excludes training/inference-seed uncertainty.',
                reference_strata=None,BASE_safe_cohort='Not computed. TRAIN512/DEV256 independent reference strata are inapplicable to NAVTEST.',
                source_files=dict(entrypoint=file_sha(__file__),common_analysis=file_sha(common.__file__),aggregator=file_sha(sharded.__file__)),
                paired_scene_csv=str(csv_path.resolve()),paired_scene_csv_sha256=file_sha(csv_path),
                cost_and_denominators=complete,new_inference_or_scoring_calls=0)
    result['sha256']=fingerprint(result);atomic_json(output,result)
    return result


def self_test():
    """Small explicit CPU fixtures; production12146/136 gate is never bypassable by CLI."""
    import sys
    import tempfile
    from unittest.mock import patch
    import torch
    from starVLA.rl.safe_flow_improvement.contracts import seed_for
    from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
    tokens=['CPU_A','CPU_B','CPU_C','CPU_D']
    protocol=dict(base_model_sha256='CPU_FIXTURE_NOT_A_MODEL',evaluation_seeds=list(SEEDS),splits={'NAVTEST':tokens,'DEV_SELECT':tokens},diagnostic={},
                  scorer=dict(sha256='CPU_FIXTURE_NOT_A_SCORER'),samplers=dict(deployment='single source shift5 ODE10, original standard Gaussian root'),
                  practical=dict(mean_gain_raw=.005,noninferiority_raw=.002,new_failure_rate_absolute=.002))
    protocol['sha256']=fingerprint(protocol);checks=[]
    def fixture(path,arm,split='NAVTEST',changed=False):
        path.mkdir();(path/'scenes').mkdir();base=arm=='BASE'
        contract=dict(root_namespace='native_qwen_sourceflow_ODE',root_rng_device='cpu',root_innovation_dtype='float32',steps=10,time_grid='source_shift5',horizon=8,action_dim=3,base_model_sha256=protocol['base_model_sha256'])
        identity=dict(schema='three_route_partitioned_single_deployment_v1',protocol_sha256=protocol['sha256'],base_model_sha256=protocol['base_model_sha256'],
                      split=split,scenes=4,tokens_sha256=fingerprint(tokens),evaluation_seeds=list(SEEDS),family='base' if base else 'latent',arm=arm,training_seed=42,
                      qualification=False,policy_qualification=False,checkpoint_state=None if base else dict(phase='COMPLETE',round=64,updates=256),
                      root_scale=1.,readout='single_candidate_source_shift5_ODE_K10' if base else 'q_sampled_single',
                      root_policy_sha256=None if base else 'a'*64,root_innovations='native_qwen_sourceflow_ODE scene-bound CPU FP32 namespace',deployment_contract=contract,
                      source_files={'CPU_fixture':'no real evaluation source'},execution_equivalence_sha256='e'*64,
                      scorer_config_hash=protocol['scorer']['sha256'],sampler_config_hash=fingerprint(protocol['samplers']))
        identity['sha256']=fingerprint(identity);atomic_json(path/'identity.json',identity);scores=[]
        for i,token in enumerate(tokens):
            for seed in SEEDS:
                score=(.9 if i==3 else .75 if seed==2101 else .85) if changed else .8;scores.append(score)
                metrics=dict(score=score,no_at_fault_collisions=1.,drivable_area_compliance=1.,time_to_collision_within_bound=1.,ego_progress=(12*score-7)/5,comfort=1.,driving_direction_compliance=1.)
                normalized=torch.zeros(8,3);planned=ActionCodec.denorm_odo(normalized).tolist();sim=torch.zeros(41,11).tolist()
                rng=torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',seed,token));root=torch.randn(1,8,3,generator=rng)[0].tolist()
                params=dict(root=root,single_candidate=True)
                if not base:params.update(mean=torch.zeros(8,3).tolist(),log_std=torch.zeros(8,3).tolist())
                row=dict(scene_token=token,log_id='CPU_log_A' if i<3 else 'CPU_log_B',split=split,seed=seed,status='OK',evaluation_identity=identity['sha256'],
                         raw_reward=score,native_subscores=metrics,normalized_action=normalized.tolist(),planned_trajectory=planned,simulated_trajectory=sim,
                         scorer_config_hash=identity['scorer_config_hash'],sampler_config_hash=identity['sampler_config_hash'],latent_or_proposal_parameters=params,
                         feedback=dict(status='OK',token=token,log='CPU_log_A' if i<3 else 'CPU_log_B',protocol_sha256=identity['scorer_config_hash'],metrics=metrics,requested_trajectory=planned,simulated_states=sim))
                row['sha256']=fingerprint(row);atomic_json(path/'scenes'/f'{token}_seed{seed}.json',row)
        atomic_json(path/'COMPLETE.json',dict(status='COMPLETE',identity_sha256=identity['sha256'],scenes=4,PDMS=100*float(np.mean(scores)),logical_single_output_evaluations=20))
    with tempfile.TemporaryDirectory() as temp:
        temp=Path(temp);fixture(temp/'base','BASE');fixture(temp/'current','L1',changed=True);fixture(temp/'dev','L1',split='DEV_SELECT')
        try:header(temp/'current',protocol)
        except ValueError:pass
        else:raise AssertionError('Tiny fixture accepted by production12146 gate')
        with patch.dict(globals(),EXPECTED_SCENES=4,EXPECTED_LOGS=2):
            try:summarize(temp/'dev',temp/'base',protocol,temp/'badDEV.json',repeats=100)
            except ValueError:pass
            else:raise AssertionError('DEV accepted as effect evidence')
            identity=json.loads((temp/'current/identity.json').read_text())
            foreign=dict(identity,source_files={'CPU_fixture':'different evaluation source'})
            try:paired_headers(foreign,json.loads((temp/'base/identity.json').read_text()))
            except ValueError:pass
            else:raise AssertionError('Changed actor evaluation source accepted')
            badqual=temp/'qualification';badqual.mkdir();q=dict(identity,qualification=True);q.pop('sha256');q['sha256']=fingerprint(q);atomic_json(badqual/'identity.json',q)
            try:header(badqual,protocol)
            except ValueError:pass
            else:raise AssertionError('Qualification became formal TEST evidence')
            result=summarize(temp/'current',temp/'base',protocol,temp/'valid.json',repeats=100)
            assert abs(result['five_seed_mean']['delta_PDMS_points']-4.75)<1e-10
            assert abs(result['seed2101']['delta_PDMS_points']+1.25)<1e-10
            assert result['five_seed_mean']['fixed_reference_strata']['status']=='NOT_AVAILABLE'
            assert 'reference_strata_content_sha256' not in result['five_seed_mean']
            assert result['training_repetitions']==1 and result['inference_seeds']==list(SEEDS)
            # Exercise the real sharded aggregator, not a mock or new scorer.
            import shutil
            shards=temp/'shards';shards.mkdir()
            identity=json.loads((temp/'current/identity.json').read_text())
            for rank in range(2):
                folder=shards/f'rank{rank}';folder.mkdir();(folder/'scenes').mkdir()
                local=sharded.partition(tokens,rank,2)
                atomic_json(folder/'identity.json',identity)
                execution=dict(rank=rank,world=2,identity_sha256=identity['sha256'],local_tokens_sha256=fingerprint(local))
                atomic_json(folder/'execution.json',execution)
                complete=dict(status='COMPLETE',execution=execution);complete['sha256']=fingerprint(complete)
                atomic_json(folder/'SHARD_COMPLETE.json',complete)
                for token in local:
                    for seed in SEEDS:shutil.copy2(temp/'current/scenes'/f'{token}_seed{seed}.json',folder/'scenes')
            shard_result=summarize(shards,temp/'base',protocol,temp/'fromShards.json',repeats=100)
            assert abs(shard_result['five_seed_mean']['delta_PDMS_points']-4.75)<1e-10
            assert (temp/'fromShards_aggregates/evaluation/COMPLETE.json').is_file()
            (temp/'current/scenes/CPU_A_seed2105.json').unlink()
            try:summarize(temp/'current',temp/'base',protocol,temp/'badMissing.json',repeats=100)
            except ValueError:pass
            else:raise AssertionError('Incomplete TEST denominator accepted')
        checks+=['Production12146/136 gate rejects the four-scene synthetic fixture','DEV and qualification are rejected before any effectiveness summary','Changed actor evaluation source is rejected','Actual common analyze_pair: five-seedmean+4.75points versus seed2101−1.25points remain separate','Real two-shard aggregate helper publishes only complete disjoint records and matches direct full-population arithmetic','Missing scene-seed output is rejected','Reference strata explicitlyNone; one learned-method training repetition stated']
    assert EXPECTED_SCENES==12146 and EXPECTED_LOGS==136
    assert '稳定性不足' in evidence(.1,[-.2,.3])
    checks.append('Positive TEST estimate with interval crossingzero states insufficient stability, not success')
    return dict(status='PASS',qualification_only=True,fixture_scenes=4,fixture_logs=2,production_scenes=EXPECTED_SCENES,production_logs=EXPECTED_LOGS,
                new_GPU_tasks=0,new_inference_or_scoring_calls=0,checks=checks,real_TEST_effects_not_evaluated=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--evaluation');parser.add_argument('--reference');parser.add_argument('--protocol')
    parser.add_argument('--output',required=True);parser.add_argument('--bootstrap-repeats',type=int,default=2000);parser.add_argument('--bootstrap-seed',type=int,default=42)
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:
        if Path(args.output).exists():raise FileExistsError(args.output)
        result=self_test();result['entrypoint_sha256']=file_sha(__file__);atomic_json(args.output,result);print(json.dumps(result));return
    if not all([args.evaluation,args.reference,args.protocol]):parser.error('--evaluation --reference --protocol are required')
    result=summarize(args.evaluation,args.reference,read_protocol(args.protocol),args.output,repeats=args.bootstrap_repeats,bootstrap_seed=args.bootstrap_seed)
    print(json.dumps(dict(status=result['status'],arm=result['arm'],TEST_PDMS=result['five_seed_mean']['current_PDMS'],delta=result['five_seed_mean']['delta_PDMS_points'],evidence=result['evidence'])))


if __name__=='__main__':main()

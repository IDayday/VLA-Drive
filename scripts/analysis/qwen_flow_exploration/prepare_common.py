"""Seal the common mature base and log-disjoint finite study population."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import pickle
import itertools
import numpy as np
from starVLA.rl.qwen_sourceflow.contracts import file_sha, fingerprint, POPULATION
from starVLA.rl.safe_flow_improvement.v1_reward import protocol_identity
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_flow_exploration.protocol import ROOT,TASK


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',default=str(ROOT/'common_protocol.json'));a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError('Never replace a registered protocol')
    ROOT.mkdir(parents=True,exist_ok=True)
    data_root=Path('/mnt/project/qwen-sourceflow-native-20261018/data')
    paths=dict(TRAIN=str(data_root/'TRAIN_v3'),DEV=str(data_root/'DEV_v2'),NAVTEST=str(data_root/'NAVTEST'))
    manifests={key:json.loads((Path(path)/'manifest.json').read_text()) for key,path in paths.items()}
    population=json.loads(POPULATION.read_text());logs=population['token_logs']
    logsets={key:{logs[t] for t in value['tokens']} for key,value in manifests.items()}
    assert not(logsets['TRAIN']&logsets['DEV'] or logsets['TRAIN']&logsets['NAVTEST'] or logsets['DEV']&logsets['NAVTEST'])
    rng=np.random.default_rng(20261024)
    devlogs=sorted(logsets['DEV']);rng.shuffle(devlogs)
    sizes={log:sum(logs[t]==log for t in manifests['DEV']['tokens']) for log in devlogs}
    selectlogs=set(min(itertools.combinations(devlogs,len(devlogs)//2),
        key=lambda group:(abs(2*sum(sizes[log] for log in group)-len(manifests['DEV']['tokens'])),group)))
    auditlogs=set(devlogs)-selectlogs
    select=[t for t in manifests['DEV']['tokens'] if logs[t] in selectlogs]
    audit=[t for t in manifests['DEV']['tokens'] if logs[t] in auditlogs]
    train=rng.choice(manifests['TRAIN']['tokens'],512,replace=False).tolist()
    devdiag=rng.choice(select,256,replace=False).tolist()
    base=Path('/mnt/project/qwen-sourceflow-native-20261018/latest_NAVTEST_20261024T152508Z/exports/canonical')
    export=json.loads((base/'export.json').read_text());seal=json.loads((base/'COMPLETE').read_text())
    assert seal['export_sha256']==file_sha(base/'export.json') and export['updates']==50000 and not export['qualification']
    caches=['/mnt/project/flow-safe-improvement-20261013/metric_cache_v1',
            '/mnt/project/constrained-flow-noise-20261014/metric_cache_v1',
            '/mnt/project/DriveDreamer-Policy-paired/runs/paired_dev_pdms_v1/metric_cache',
            '/mnt/project/DriveDreamer-Policy-full-navtrain/runs/full_navtrain_epoch1/metric_cache_navtest_v1']
    missing=[t for t in train+devdiag if not any((Path(cache)/logs[t]/'unknown'/t/'metric_cache.pkl').is_file() for cache in caches)]
    record=dict(schema='qwen_flow_three_routes_common_v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        base_export=str(base),base_model_sha256=export['model_sha256'],base_native_updates=50000,
        base_full_NAVTEST_PDMS=90.513164102699,base_selected_without_new_route_results=True,
        base_identity='full merged mature canonical50k; not weaker IL initialization',
        starting_code='aced3cf01ab9b3adea6e4cf6374364df25ebb776',data=paths,
        data_manifest_sha256={key:file_sha(Path(path)/'manifest.json') for key,path in paths.items()},
        full_log_counts={key:len(value) for key,value in logsets.items()},
        splits=dict(TRAIN=train,DEV_SELECT=select,DEV_AUDIT=audit,NAVTEST=manifests['NAVTEST']['tokens']),
        diagnostic=dict(TRAIN=train,DEV_SELECT=devdiag,K=32,reference_seeds=[11001,11002,11003,11004,11005]),
        smoke=dict(TRAIN=train[:32],DEV_SELECT=devdiag[:16],G=4),
        DEV_SELECT_logs=sorted(selectlogs),DEV_AUDIT_logs=sorted(auditlogs),
        audit_history='Both DEV halves have historical baseline observations. Sealed only for this new study; not a newly blind test.',
        train_sampling='Fixed uniform random512 TRAIN scenes; strata are measured, not selected from method outcomes',
        model_contract='native Qwen3VL2B full current left/front/right tokens; source 30layer1024 ActionDiT, native source absolute8x3',
        action='rear axle local x-forward y-left heading rad, 0.5...4 seconds, exact source min/max normalizer',
        samplers=dict(deployment='single source shift5 ODE10, original standard Gaussian root',
                      latent='learn root24D joint diagonal Gaussian; generator frozen; no internal ODE xi',
                      policy_training='source rigorous SDE, stochastic[7,8,9], noise.1, minstd.001, innovation clamp5, relative per-dimension logp clamp7 then H/D mean'),
        precision='FP32 weights/action/latents; BF16 frozenQwen/provider prefill; TF32 off',
        freeze=dict(latent='all generator parameters/buffers',others='frozenQwen/provider/base; source action self/cross qkvo attentionLoRA r16alpha32dropout0'),
        scorer=protocol_identity(Path(__file__).resolve().parents[3]),metric_cache_roots=caches,
        missing_cache_count=len(missing),missing_cache_tokens=missing,
        safety_metrics=['no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound'],
        preservation_costs='1-NC,1-DAC,1-TTC; TRAIN initial pool expectation plus epsilon0; frozen budget',
        advantages='FP64 group centered / max(fresh raw global population std,0.01); no post scalarization whitening; adv_clip=None',
        active_changes_from_historical_recipe=['G16/B16 serial accumulation versus oldG8/world8',
             'all-scene representative fixed512 versus original public hard17037', 'common rawstd floor0.01','disableadvclip5 in all new P arms'],
        training_seed=42,confirmation_training_seeds=[42,43,44],evaluation_seeds=[2101,2102,2103,2104,2105],
        seed_override='Latest user explicitly requests selective multi-training/inference-seed confirmation; old50k runs unchanged seed42.',
        screen=dict(B=16,G=16,rollout_rounds=64,inner_epochs=4,new_candidates_per_online_arm=16384,optimizer_updates_per_online_arm=256),
        confirmation=dict(max_TRAIN=2048,new_candidates_per_arm_per_seed=32768,only_promoted_routes=True),
        execution=dict(candidate_chunk=4,model_processes_per_GPU_max=1,CPU_LRU_scenes=32),
        practical=dict(mean_gain_raw=.005,noninferiority_raw=.002,new_failure_rate_absolute=.002),
        stage='SMOKE_AND_PROFILE_ONLY; formal finite budgets freeze after measured profile',
        no_unlimited_extensions=True,no_archive_or_feedback_from_DEV_or_NAVTEST=True)
    record['sha256']=fingerprint(record);atomic_json(out,record)
    authorization=dict(task=TASK,user_instruction='Three independent exploration routes with exactly three child agents; actual controlled experiments',
        allowed_hosts=['training-vla-zt','training-vlawm-tsinghua-zt','training-vlawm-tsinghua-zt3','training-vlawm-tsinghua-zt4','training-vlawm-tsinghua-zt5'],
        preserve_all_existing_training=True,no_new_training_on_Navtest_only_hosts=True,
        allowed_stage='smoke and common profile; finite formal budget registered separately')
    atomic_json(ROOT/'RESOURCE_AUTHORIZATION.json',authorization)
    policy_path=Path('/mnt/project/server_dispatch_policy.json')
    with Path(str(policy_path)+'.lock').open('a') as lk:
        fcntl.flock(lk,fcntl.LOCK_EX);policy=json.loads(policy_path.read_text())
        assert TASK not in policy['task_authorizations']
        policy['task_authorizations'][TASK]=authorization;atomic_json(policy_path,policy)
    print(json.dumps(dict(status='PREPARED',protocol=str(out),TRAIN=512,DEV_SELECT=len(select),DEV_AUDIT=len(audit),
        select_logs=len(selectlogs),audit_logs=len(auditlogs),missing_caches=len(missing))))


if __name__=='__main__':main()

"""Register explicit matched research arms before their first GPU execution."""
import argparse
import copy
import json
from pathlib import Path
import time

import yaml

from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from starVLA.rl.qwen_world_postcot.contracts import file_sha, fingerprint
from starVLA.rl.qwen_world_postcot.scope_contracts import MEMORY, expected_scope_recipe
from starVLA.rl.qwen_world_research.runtime import CAMPAIGN, SCHEMA, validate_recipe
from starVLA.rl.qwen_world_research_data.manifest import write_stream_manifest
from starVLA.rl.qwen_world_research_data.adaptive import prepare_stage


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',default=str(CAMPAIGN));a=p.parse_args()
    root=Path(a.root)
    if root != CAMPAIGN:
        raise ValueError('This registration is bound to its independent user-authorized campaign')
    if (root/'REGISTRATION.json').exists():
        raise FileExistsError('Do not replace an already registered campaign')
    data=root/'data/NAVTRAIN';manifests=root/'data/manifests'
    full=json.loads((manifests/'FULL_103288.json').read_text())
    adaptive=manifests/'ADAPTIVE_FULL_103288.json'
    write_stream_manifest(adaptive,data_root=data,tokens=full['tokens'],name='adaptive_FULL_103288',
                          mode='adaptive',stage_exposures=2000,stage_registry=root/'adaptive_stages')
    prepare_stage(adaptive,0)
    specifications=[
        ('PPO_HARD','ppo','hard','WORLD_HARD_15171.json',1,'canonical Memory control; new IL70 parent'),
        ('PPO_FULL','ppo','uniform_full','FULL_103288.json',1,'full103288 population-only comparison'),
        ('PPO_ADAPTIVE','ppo','adaptive',adaptive.name,2,'stagewise low-score/regression/coverage selection on NAVTRAIN'),
        ('CISPO_FULL','cispo','uniform_full','FULL_103288.json',2,'objective-only control against PPO_FULL'),
        ('NFT_FULL','nft','uniform_full','FULL_103288.json',2,'pure ODE collection plus positive/negative forward objective'),
        ('PPO_HELDOUT','ppo','uniform_full','FULL_MINUS_RL_DEV.json',3,'97965 training scenes; fixed5323 physical-log RL-heldout'),
    ]
    runs=[]
    (root/'configs').mkdir(exist_ok=True)
    for name,algorithm,policy,manifest_name,priority,hypothesis in specifications:
        manifest=manifests/manifest_name
        cfg=copy.deepcopy(expected_scope_recipe(MEMORY))
        cfg.update(evaluate_updates=[0,1000,5000,10000,20000,35000,50000],
                   save_every_optimizer_updates=500,
                   reference_observation='permanently frozen pinned complete World IL70k Qwen and memory',
                   data_order='explicit registered NAVTRAIN scene stream; exact source sampler seed42 or sealed adaptive stages')
        cfg['research']=dict(schema=SCHEMA,campaign_root=str(root),arm=name,
            policy_objective=algorithm,cispo_upper=1.01,data_policy=policy,
            manifest_sha256=file_sha(manifest),observation_data=str(data),
            scoring_cache_roots=[
                '/mnt/project/flow-safe-improvement-20261013/metric_cache_v1',
                '/mnt/project/constrained-flow-noise-20261014/metric_cache_v1',
                '/mnt/project/DriveDreamer-Policy-paired/runs/paired_dev_pdms_v1/metric_cache'],
            stage_gate_updates=[1000,5000,10000,20000,35000,50000],
            training_seed=42,primary_evaluation_seed=2101,
            NAVTEST_used_for_training_selection=False,original_DEV_independent=False)
        if algorithm=='nft':
            cfg['grpo']['train']['kl_beta']=0
            cfg.update(learning_rate=1e-5,memory_learning_rate=1e-6)
            cfg['research'].update(nft_beta=1.,nft_ema_decay=.9,nft_sigma_min=.05,nft_loss='velocity_mse')
        validate_recipe(cfg)
        path=root/'configs'/f'{name}.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        count=json.loads(manifest.read_text())['count']
        runs.append(dict(id=name,algorithm=algorithm,data_policy=policy,root=str(root/'runs'/name),
            status='REGISTERED',target_updates=50000,stage_budget_updates=1000,priority=priority,
            data_scenes=count,config=str(path),config_sha256=file_sha(path),manifest=str(manifest),
            parent_updates=70000,parent_PDMS=88.79290990699972,host=None,
            qualification_root=str(root/'qualification'/name),hypothesis=hypothesis,
            old_DEV_role='TRAIN_DIAGNOSTIC; not held-out for full population arms',
            module='starVLA.rl.qwen_world_research_algorithms.train_nft' if algorithm=='nft' else 'tools.qwen_world_research.train'))
    registration=dict(schema='world_rl_optimization_campaign_v1',registered_unix=time.time(),
        parent_selection=str(root/'PARENT_SELECTION.json'),parent_selection_sha256=file_sha(root/'PARENT_SELECTION.json'),
        resource_authorization=str(root/'resources/RESOURCE_AUTHORIZATION.json'),
        source_base='9dcac22f37cfa8b3c6d2ff75b6aad8952b4af3c3',
        data_audit=str(data/'POPULATION_AUDIT.json'),data_audit_sha256=file_sha(data/'POPULATION_AUDIT.json'),
        report=str(root/'report.html'),runs=runs,
        invariants=dict(parent='same pinned complete IL70k',scope='52,707,840 Memory+attention LoRA parameters; Qwen frozen',
            candidates_per_scene=8,scenes_per_fresh_global_collection=8,updates_per_collection=4,
            original_cancelled_IL70_run_resumed=False,canonical='official NAVSIM v1.1 5:5:2',
            deployment='single clean-x K10 shift5 ODE; no trajectory selection',
            observed_Test=True,claim_of_untouched_final_test=False),
        stage_policy=dict(first_gate_updates=1000,subsequent=[5000,10000,20000,35000,50000],
            adaptive_gate_every_updates=1000,
            comparisons='report both fresh scored candidates and actual time; stochastic TRAIN and ODE evaluation separate',
            promotion='continue promising/non-inferior complete trajectories; retain all failures and negative arms; never publish partial Test mean'))
    registration['sha256']=fingerprint(registration)
    atomic_json(root/'REGISTRATION.json',registration)
    atomic_json(root/'RUNS.json',dict(runs=runs))
    atomic_json(root/'STATE.json',dict(status='REGISTERED_QUALIFICATION_PENDING',updated_unix=time.time(),runs=runs,results=[],events=[dict(at=time.time(),kind='registration',message='Six explicit independent research arms; full103288 current observations and canonical cache coverage verified.')]))
    print(json.dumps(dict(root=str(root),arms=[r['id'] for r in runs],report=str(root/'report.html'))))

if __name__=='__main__':main()

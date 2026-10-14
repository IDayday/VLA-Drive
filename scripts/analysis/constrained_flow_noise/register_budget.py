"""Lock the common finite budget from measured full-head profiles before formal RL."""
import argparse
import datetime
import json
import math
from pathlib import Path
from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run-root',required=True);p.add_argument('--population',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();host=require_dispatch()
    root=Path(a.run_root);pop=json.loads(Path(a.population).read_text());total=2*len(pop['train_tokens'])
    if Path(a.output).exists():raise FileExistsError('Budget is immutable before formal outcomes')
    profiles={}
    for variant,name in [('C_SDE','profile_C_v2'),('N_FLOW_NOISE','profile_N_v2')]:
        run=root/name;reports=[json.loads((run/f'result_rank{i}.json').read_text()) for i in range(8)]
        if any(d['status']!='COMPLETE_FIXED_RAW_BUDGET' or not d['qualification_only'] or d['raw_cursor']!=128 for d in reports):raise ValueError('Complete real-model profile required')
        d=reports[0];updates=max(x['seconds_updates'] for x in reports)
        idealized=((d['seconds_rollout']+d['seconds_scoring']+d['seconds_reference'])/8+updates)/128
        observed=max(x['wall_seconds_this_invocation'] for x in reports)/128
        profile=dict(identity_sha256=d['identity_sha256'],raw_scenes=128,candidates=2048,updates=d['updates'],accepted_exposures=d['accepted_exposures'],
            seconds_rollout_sum=d['seconds_rollout'],seconds_scoring_worker_wall_sum=d['seconds_scoring'],
            seconds_scoring_CPU_sum=d['seconds_scoring_cpu'],seconds_reference_sum=d['seconds_reference'],
            optimizer_seconds_max_rank=updates,seconds_per_raw_perfect_rank_balance=idealized,
            seconds_per_raw_observed_with_end_IO=observed,
            extrapolated_2_pass_hours_per_host=[total*idealized/3600,total*observed/3600],
            extrapolated_allocated_GPU_hours=[8*total*idealized/3600,8*total*observed/3600],
            profile_receipt_sha256=file_sha(run/'result_rank0.json'))
        profiles[variant]=profile
    result=dict(schema='cfn_common_formal_budget_v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=host,
        variants=['C_SDE','N_FLOW_NOISE'],population_identity=pop['identity_sha256'],training_seed=42,evaluation_seed=2101,
        train_scenes=len(pop['train_tokens']),raw_passes=2,raw_exposures_per_variant=total,
        training_candidates_per_variant=16*total,shared_budget_clock='unfiltered raw scenes including refill and discarded groups',
        milestones=[int(len(pop['train_tokens'])*x) for x in (.25,.5,1.,2.)],
        effective_groups_target=16,raw_groups_cap=32,G=16,K=10,inner_epochs=2,
        save_every_optimizer_updates=1024,checkpoint_retention='Keep all saved recovery points, milestones and failure evidence; no automatic deletion',
        monitor=dict(scenes=128,candidates_per_scene=16,points=[0,.25,.5,1.,2.],candidates_per_variant=128*16*5,
                     population='precalibration_registration.monitor_tokens',updates_or_dual_use=False),
        DEV_evaluation=dict(baseline_scores=str(root/'evaluation/SFT_DEV_original/scores'),override=str(root/'evaluation_override.json'),
            cache_root='/mnt/project/DriveDreamer-Policy-paired/runs/paired_dev_pdms_v1/metric_cache',CPU_workers=16,
            points='All four fixed raw-exposure milestones, full1696 scenes, original single-candidate ODE',
            baseline_report_sha256=file_sha(root/'evaluation/SFT_DEV_original/report.json')),
        severe_safety_rule=json.loads((root/'precalibration_registration.json').read_text())['severe_safety_pause'],
        numerical_saturation_pause=dict(one_log_std_bound_fraction=.95,relative_log_range_edge=.01,consecutive_collections=20),
        initialization='Original SFT only; all qualification/profile weights excluded',profiles=profiles,
        profile_caveats='Measured under active authorized shared GPU/cache workloads. Bounds describe measured balance and IO overhead, not a guaranteed runtime interval. No paper speed-up multiplier assumed.',
        authorized_resources='At most two independent full-eight-GPU training jobs on capacity-qualified allowed Tsinghua hosts; preserve all other genuine jobs',
        time_window='No explicit expiry or GPU-hour cap was supplied; retain requested finite two-pass budget',
        navtest='No access until common endpoint/selection lock; one scene-bound inference seed2101',
        source_sha256=file_sha(__file__))
    result['sha256']=fingerprint(result);atomic_json(Path(a.output),result)
    print(json.dumps(dict(status='LOCKED',raw_exposures_per_variant=total,training_candidates_per_variant=16*total,
                    hours_by_variant={k:v['extrapolated_2_pass_hours_per_host'] for k,v in profiles.items()})),flush=True)


if __name__=='__main__':main()

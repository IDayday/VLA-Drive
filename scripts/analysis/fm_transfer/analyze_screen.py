"""Summarize a verified full own-SFT screen, without changing hard membership.

This is one fixed ODE sample per TRAIN scene, not G8 group statistics and not
raw-observation DEV/Navtest. Physical invocation costs are separate from the
unique logical population; incomplete failure ledgers remain explicitly unknown.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import POPULATION,file_sha,fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json

METRICS=('score','no_at_fault_collisions','drivable_area_compliance',
    'time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance')


def summarize(screen, hard_manifest):
    screen=Path(screen);hard_manifest=Path(hard_manifest)
    identity=json.loads((screen/'manifest.json').read_text())
    hard=json.loads(hard_manifest.read_text());body=dict(hard);sha=body.pop('sha256')
    proof_path=hard_manifest.with_suffix('.verification.json');proof=json.loads(proof_path.read_text())
    pop=json.loads(Path(POPULATION).read_text());tokens=pop['train_tokens']
    if (hard['status']!='COMPLETE_FIXED_SFT_SINGLE_NATIVE_ODE_HARD_SUBSET' or fingerprint(body)!=sha or
            hard['screen_identity']!=identity or identity['qualification_only'] or
            proof['status']!='COMPLETE_ALL_PAYLOAD_HASHES_VERIFIED' or proof['checked']!=len(tokens) or
            proof['manifest_sha256']!=file_sha(hard_manifest) or hard['screened_scene_count']!=len(tokens)):
        raise ValueError('Full population and frozen membership must be verified first')
    costs=[]
    for rank in range(identity['shards']):
        state=json.loads((screen/f'rank{rank}.json').read_text())
        if state['status']!='ASSIGNED_COMPLETE' or state['identity_sha256']!=identity['sha256']:
            raise ValueError('A global screen shard is incomplete')
    selected=set(hard['tokens']);observed_hard=set();counts=Counter();sums=Counter();hard_sums=Counter()
    for token in tokens:
        p=screen/'scenes'/token[:2]/(token+'.json');row=json.loads(p.read_text());m=row['metrics']
        if row['token']!=token or row['log']!=pop['token_logs'][token] or row['identity_sha256']!=identity['sha256']:
            raise ValueError('Foreign scene or score identity')
        if any(not math.isfinite(m[k]) or not 0<=m[k]<=1 for k in METRICS):raise ValueError('Invalid official metric')
        for k in METRICS:sums[k]+=m[k]
        safe=all(m[k]==1 for k in METRICS[1:4]);score=m['score']
        counts['joint_NC_DAC_TTC_pass']+=safe
        for k in METRICS[1:4]:counts[k+'_not_fully_pass']+=m[k]<1
        counts['score_exact_zero']+=score==0;counts['score_exact_one']+=score==1
        counts['score_strictly_between_0_and_09']+=0<score<.9
        counts['score_ge09_lt1']+=.9<=score<1
        if score<.9:
            observed_hard.add(token);counts['hard_safe' if safe else 'hard_not_fully_safe']+=1
            for k in METRICS:hard_sums[k]+=m[k]
    if observed_hard!=selected:raise ValueError('Published hard membership differs from the frozen .9 rule')
    missing=[]
    for p in sorted(screen.glob('invocations/*/rank*.json')):
        r=json.loads(p.read_text())
        if r.get('identity_sha256') not in (None,identity['sha256']):raise ValueError('Foreign screen invocation')
        complete=(r['status'] in ('ASSIGNED_COMPLETE','STOPPED_AT_SCENE_BOUNDARY','FAILED') and
            all(k in r for k in ('end_unix','total_worker_wall_seconds','actual_new_inferences')))
        if not complete:missing.append(str(p))
        costs.append(dict(record=str(p),sha256=file_sha(p),host=r['host'],rank=r['rank'],status=r['status'],
            cost_record_terminal=complete,**{k:r[k] for k in ('start_unix','end_unix','new','reused',
                'actual_new_inferences','model_load_seconds','generation_seconds','scoring_wait_seconds',
                'worker_scoring_seconds','total_worker_wall_seconds','unpublished_scoring_requests') if k in r}))
    if not costs:raise ValueError('Missing physical invocation ledger')
    duration_complete=not missing
    unpublished=sum(len(x.get('unpublished_scoring_requests',[])) for x in costs)
    component_keys=('model_load_seconds','generation_seconds','scoring_wait_seconds','worker_scoring_seconds')
    missing_components={k:[x['record'] for x in costs if k not in x] for k in component_keys}
    if duration_complete and (sum(x.get('new',0) for x in costs)!=len(tokens) or
            sum(x['actual_new_inferences'] for x in costs)<len(tokens)):
        raise ValueError('Terminal ledger cannot account for all unique committed scenes')
    return dict(schema='fm_transfer_complete_TRAIN_screen_analysis_v1',status='COMPLETE_POPULATION',
        identity=identity,scenes=len(tokens),logs=len({pop['token_logs'][t] for t in tokens}),
        seed=2101,grid='native',statistic='one fixed ODE sample per scene; never a G8 tied-group census',
        scores_mean_raw={k:sums[k]/len(tokens) for k in METRICS},counts=dict(counts),hard_scenes=len(selected),
        hard_fraction=len(selected)/len(tokens),hard_scores_mean_raw={k:hard_sums[k]/len(selected) for k in METRICS},
        hard_types=hard['hard_types'],hard_manifest_sha256=file_sha(hard_manifest),
        prior_all_payload_verification_sha256=file_sha(proof_path),analysis_sha256=file_sha(__file__),
        cost=dict(logical_unique_single_candidate_scores=len(tokens),worker_invocations=len(costs),
            complete_duration_ledger=duration_complete,incomplete_invocations=missing,
            recorded_physical_predictions=sum(x.get('actual_new_inferences',0) for x in costs),
            recorded_new_committed_scenes=sum(x.get('new',0) for x in costs),recorded_reused_scenes=sum(x.get('reused',0) for x in costs),
            physical_count_is_lower_bound_if_missing_invocations=not duration_complete,
            cumulative_worker_wall_seconds=sum(x.get('total_worker_wall_seconds',0) for x in costs) if duration_complete else None,
            job_wall_seconds=max(x['end_unix'] for x in costs)-min(x['start_unix'] for x in costs) if duration_complete else None,
            recorded_component_seconds={k:None if missing_components[k] else sum(x[k] for x in costs) for k in component_keys},
            missing_component_records=missing_components,unpublished_scoring_requests=unpublished,
            scoring_time_may_be_incomplete=bool(unpublished or missing),shared_GPU_not_exclusive_compute=True,
            generation_includes_condition_read_and_CUDA_synchronization=True,
            population_scores_unchanged=True,invocations=costs))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--screen',required=True);p.add_argument('--hard-manifest',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();require_dispatch();output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve old summaries')
    result=summarize(a.screen,a.hard_manifest);output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,result)
    print(json.dumps(dict(status=result['status'],scenes=result['scenes'],hard_scenes=result['hard_scenes'],
        hard_fraction=result['hard_fraction'],canonical_PDMS=100*result['scores_mean_raw']['score'],counts=result['counts'])))


if __name__=='__main__':main()

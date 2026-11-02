"""Read-only audit of every logged integrated training group through a fixed step.

The optional G8 prefix is a within-G16 discovery counterfactual, not a separate
G8 policy, an equal-compute comparison, or test-time best-of sampling.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path('/mnt/project/qwen-world-safety-exploration-20261031/integrated_training_20261102')
ARM = 'INTEGRATED_SAFETY_EP_B64_G16_RETENTION_WKC'
STRICT = ('no_at_fault_collisions', 'drivable_area_compliance',
          'time_to_collision_within_bound', 'comfort')


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=True, indent=2)+'\n')
    tmp.replace(path)


def rows(path, limit):
    result = []
    with path.open() as stream:
        for line in stream:
            if not line.endswith('\n'):
                continue  # An in-progress final append is not evidence yet.
            value = json.loads(line)
            if value['updates'] <= limit:
                result.append(value)
    return result


def summarize(rank_rows):
    counts = Counter()
    sums = Counter()
    unique = set()
    for rank, values in enumerate(rank_rows):
        by_collection = {r['collection']: r for r in values}
        for row in by_collection.values():
            ex = row['exploration']
            groups = ex['groups']
            counts['rank_windows'] += 1
            for name in ('fresh_B1_queries','history_B1_queries','history_slots_selected','history_gates'):
                counts[name] += ex[name]
            for token, group in zip(ex['tokens'], groups):
                unique.add(token)
                counts['groups'] += 1
                counts['fresh_PG_candidates'] += 16
                for name in ('all_zero','all_full','tied','zero_PG','safety_differing'):
                    counts[name] += int(group[name])
                counts['all_zero_with_PG'] += int(group['all_zero'] and not group['zero_PG'])
                counts['all_zero_without_PG'] += int(group['all_zero'] and group['zero_PG'])
                counts['no_feasible_candidate'] += int(group['safe_fraction'] == 0)
                counts['one_feasible_candidate'] += int(group['safe_fraction'] == 1/16)
                sums['random_canonical_score'] += group['canonical_random_mean']
            # Quality fields are means over the eight scenes on this rank.
            m = row['metrics']
            sums['groups_with_strict_high'] += 8*m['quality_has_strict_high']
            sums['strict_high_candidates'] += 8*m['quality_strict_high_count']
            sums['strict_high_with_positive_A'] += 8*m['quality_positive_high_count']
            sums['strict_high_with_negative_A'] += 8*m['quality_negative_high_count']
            sums['bank_size_per_rank_window'] += m['retention_bank_size']
        for row in values:
            m=row['metrics']
            counts['rank_optimizer_records'] += 1
            for name in ('retention_effective_RL_share','quality_high_positive_PPO_blocked',
                         'clipfrac','approx_kl','quality_active_positive_high_weight',
                         'quality_active_negative_high_weight','quality_active_positive_unsafe_weight'):
                sums[name] += m[name]
            sums['lr'] += row['lr']
    groups=counts['groups']; slots=counts['history_slots_selected'];records=counts['rank_optimizer_records']
    return dict(counts=dict(counts), unique_fresh_tokens=len(unique),
        group_percent={name:100*counts[name]/groups for name in
            ('all_zero','all_full','tied','zero_PG','all_zero_with_PG','all_zero_without_PG',
             'no_feasible_candidate','one_feasible_candidate')},
        history_gate_percent_of_selected=100*counts['history_gates']/slots if slots else None,
        nominal_history_anchor_share_percent_of_all_scene_slots=25*counts['history_gates']/groups,
        extra_B1_queries=counts['fresh_B1_queries']+counts['history_B1_queries'],
        extra_B1_queries_percent_of_fresh_PG=100*(counts['fresh_B1_queries']+counts['history_B1_queries'])/counts['fresh_PG_candidates'],
        groups_with_strict_high_percent=100*sums['groups_with_strict_high']/groups,
        strict_high_candidate_count=round(sums['strict_high_candidates']),
        strict_high_negative_advantage_percent=100*sums['strict_high_with_negative_A']/sums['strict_high_candidates'],
        mean_random_canonical_PDMS=100*sums['random_canonical_score']/groups,
        mean_optimizer_record_metrics={k:sums[k]/records for k in
            ('retention_effective_RL_share','quality_high_positive_PPO_blocked','clipfrac','approx_kl',
             'quality_active_positive_high_weight','quality_active_negative_high_weight',
             'quality_active_positive_unsafe_weight','lr')},
        cautions=['Mean scalar losses or surrogate coefficients do not measure parameter-gradient competition.',
                  'Counts are deduplicated by rank and collection; the same rollout reused for four updates is counted once.',
                  'This is the active valid lineage; archived failed-attempt and recovery replay costs remain separate.',
                  'Admission starts after update250; early history counts are zero by design.',
                  'TRAIN sampling quality is distinct from deployed single-B1 NAVTEST performance.'])


def good(record):
    m=record['metrics']
    return all(m[k] == 1 for k in STRICT) and m['score'] >= .9


def feasible(record):
    m=record['metrics']
    return all(m[k] == 1 for k in STRICT[:3])


def has_signal(records):
    flags=[feasible(r) for r in records]
    if any(flags):
        if not all(flags):
            return True
        return len({r['metrics']['ego_progress'] for r in records}) > 1
    costs=[sum(1-r['metrics'][k] for k in STRICT[:3])/3 for r in records]
    return max(costs) > min(costs)


def prefix_audit(run_root, collections):
    import torch
    counts=Counter();bins={};failure_examples=[];files=0
    for collection in collections:
        for rank in range(8):
            path=run_root/f'behavior/collection{collection:06}/rank{rank}.pt'
            window=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            assert len(window['samples']) == 8
            for sample in window['samples']:
                records=sample['records'];assert len(records)==16
                h8=any(good(r) for r in records[:8]);h16=any(good(r) for r in records)
                f8=any(feasible(r) for r in records[:8]);f16=any(feasible(r) for r in records)
                z16=all(r['metrics']['score']==0 for r in records)
                signal=has_signal(records)
                counts['groups']+=1
                for name,val in [('strict_high_first8',h8),('strict_high_full16',h16),
                                 ('strict_high_rescued_by_second8',h16 and not h8),
                                 ('feasible_first8',f8),('feasible_full16',f16),
                                 ('feasible_rescued_by_second8',f16 and not f8),
                                 ('all_zero16',z16),('all_zero16_with_PG',z16 and signal),
                                 ('all_zero16_without_PG',z16 and not signal)]:counts[name]+=int(val)
                n=sum(good(r) for r in records[:8]);key=str(n)
                b=bins.setdefault(key,Counter(groups=0,second8_adds_first_strict_high=0))
                b['groups']+=1;b['second8_adds_first_strict_high']+=int(h16 and not h8)
                if z16 and not signal:
                    failure_examples.append(dict(token=sample['token'],log=sample['log'],rank=rank,
                        collection=collection,NC_DAC_TTC=list({tuple(r['metrics'][k] for k in STRICT[:3])for r in records})))
            files+=1
            if files%128==0:print(json.dumps(dict(prefix_files_processed=files,groups=counts['groups'])),flush=True)
    return dict(counts=dict(counts),percent_of_all_groups={k:100*v/counts['groups'] for k,v in counts.items() if k!='groups'},
        first8_strict_count_bins={k:dict(v) for k,v in bins.items()},all_zero_no_signal_examples=failure_examples,
        source='All sealed TRAIN behavior files for all common logged collections/ranks, fixed policy per window.',
        interpretation='First8 is a prefix of the same G16 draw: measures marginal discovery only; not an independent G8 training ablation.',
        candidate_order_not_adaptive=True,noncausal=True)


def nominal_audit(root, collections):
    """Compare the actual same-version B1 witness to that scene's G16 samples."""
    import torch
    counts=Counter();score_sum=0.;pairs=[];files=0
    for collection in collections:
        if 4*(collection-1)<250:
            continue
        for rank in range(8):
            window=torch.load(root/f'behavior/collection{collection:06}/rank{rank}.pt',
                              map_location='cpu',weights_only=False,mmap=True)
            for sample,event in zip(window['samples'],window['integrated_retention']['events']):
                witness=event['fresh_witness'];assert witness is not None
                assert witness['token']==sample['token'] and witness['source_version']==window['behavior_version']
                nominal=witness['feedback'];m=nominal['metrics'];records=sample['records']
                candidates=[(i,r) for i,r in enumerate(records) if good(r)]
                has_good=bool(candidates);nominal_good=good(nominal);nominal_safe=feasible(nominal)
                counts['groups_with_actual_B1']+=1;score_sum+=m['score']
                for k,val in [('nominal_strict_high',nominal_good),('nominal_feasible',nominal_safe),
                              ('nominal_zero',m['score']==0),('sampled_strict_high',has_good),
                              ('sampled_high_but_nominal_not_high',has_good and not nominal_good),
                              ('sampled_high_but_nominal_unsafe',has_good and not nominal_safe),
                              ('sampled_high_but_nominal_zero',has_good and m['score']==0),
                              ('nominal_high_but_no_sampled_high',nominal_good and not has_good)]:counts[k]+=int(val)
                for k in STRICT:
                    counts['nominal_fail_'+k]+=int(m[k]!=1)
                if has_good and not nominal_good:
                    index,teacher=max(candidates,key=lambda v:(v[1]['metrics']['score'],v[1]['metrics']['driving_direction_compliance']))
                    margin=teacher['metrics']['score']-m['score']
                    if margin>=.02:
                        counts['rescue_pair_with_margin_0p02']+=1
                        pairs.append(dict(token=sample['token'],log=sample['log'],rank=rank,collection=collection,
                            behavior_update=window['behavior_version']['update'],candidate_index=index,
                            nominal_metrics=m,strict_candidate_metrics=teacher['metrics'],canonical_margin=margin,
                            behavior_file=str(root/f'behavior/collection{collection:06}/rank{rank}.pt'),
                            reason='nominal_unsafe'if not nominal_safe else'nominal_below_strict_quality'))
            files+=1
            if files%128==0:print(json.dumps(dict(nominal_files_processed=files,groups=counts['groups_with_actual_B1'])),flush=True)
    total=counts['groups_with_actual_B1'];found=counts['sampled_strict_high']
    return dict(schema='world_integrated64_same_version_nominal_vs_sampled_train_v1',
        counts=dict(counts),percent_of_all_B1_groups={k:100*v/total for k,v in counts.items() if k!='groups_with_actual_B1'},
        nominal_mean_PDMS=100*score_sum/total,
        sampled_good_but_nominal_not_good_percent_of_sampled_good=100*counts['sampled_high_but_nominal_not_high']/found,
        rescue_pairs=pairs,TRAIN_only=True,NAVTEST_labels_used=False,live_training_data_changed=False,
        interpretation='Same scene and behavior version, true deployment B1 versus stochastic G16; discovery does not establish policy improvement.',
        proposed_use='TRAIN-only candidate rescue / self-imitation curriculum proposal; not registered or dispatched.')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--through-update',type=int,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--prefix-audit',action='store_true')
    p.add_argument('--nominal-audit-only',action='store_true');a=p.parse_args()
    root=ROOT/'runs'/ARM
    values=[rows(root/f'metrics_rank{rank}.jsonl',a.through_update)for rank in range(8)]
    assert all(v and v[-1]['updates']==a.through_update for v in values)
    sets=[{r['collection'] for r in v} for v in values]
    assert all(s==sets[0] for s in sets)
    if a.nominal_audit_only:
        result=nominal_audit(root,sorted(sets[0]));result['through_optimizer_update']=a.through_update
        save(a.out,result);print(json.dumps(dict(status='NOMINAL_AUDIT_COMPLETE',counts=result['counts'],output=str(a.out))),flush=True)
        return
    result=dict(schema='world_integrated64_all_logged_groups_trick_review_v1',
        as_of_UTC=datetime.now(timezone.utc).isoformat(),through_optimizer_update=a.through_update,
        all8_ranks=True,full_summary=summarize(values),intervals={},live_scientific_training_modified=False)
    for label,lo,hi in [('before_history',0,249),('history_early',250,499),('history_latest',500,a.through_update)]:
        subset=[[r for r in v if lo<=r['behavior']['update']<=hi] for v in values]
        if all(subset):result['intervals'][label]=dict(behavior_updates=[lo,hi],**summarize(subset))
    save(a.out,result);print(json.dumps(dict(metrics_audit_complete=str(a.out),counts=result['full_summary']['counts'])),flush=True)
    if a.prefix_audit:
        result['within_G16_first8_counterfactual']=prefix_audit(root,sorted(sets[0]))
        save(a.out,result);print(json.dumps(dict(status='COMPLETE',output=str(a.out),prefix_counts=result['within_G16_first8_counterfactual']['counts'])),flush=True)


if __name__=='__main__':main()

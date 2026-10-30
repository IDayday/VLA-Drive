"""CPU-only coverage/learning-signal accounting and truly paired ODE regression.

Fresh groups are counted once (inner1), never four times. Stochastic rollout
means are descriptive and cannot be compared with ODE baseline scores as if
they were same-sampler paired regression measurements.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from .adaptive import validate_bank
from .manifest import atomic_json, file_sha, fingerprint, physical, read_sealed, sealed

def read_rank(path):
    """Read a fixed byte-length snapshot, excluding an active partial last line."""
    path=Path(path);snapshot_bytes=path.stat().st_size
    digest=hashlib.sha256();fresh=[];n=0;last=0;partial=False;run_ids=set()
    with path.open('rb') as stream:
        while stream.tell()<snapshot_bytes:
            raw=stream.readline()
            if stream.tell()>snapshot_bytes or not raw.endswith(b'\n'):
                partial=True;break
            row=json.loads(raw)
            u=row['updates'];inner=row['inner_completed']
            if u!=last+1 or inner!=(u-1)%4+1:
                raise ValueError(f'Noncontiguous native update/inner journal: {path}: {last}->{u},inner{inner}')
            if any(not math.isfinite(float(row[k])) for k in ('loss','gradient_norm')):
                raise ValueError('Nonfinite completed training journal row')
            run_ids.add(row.get('run'))
            digest.update(raw);n+=1;last=u
            if inner==1:
                fresh.append(dict(update=u,collection=row['collection'],token=row['token'],
                                  exploration=row['exploration'],behavior=row.get('behavior',{})))
    return fresh,dict(path=str(path.resolve()),snapshot_bytes=snapshot_bytes,
                      complete_rows=n,last_update=last,valid_prefix_sha256=digest.hexdigest(),
                      partial_tail_excluded=partial,run_ids=sorted(v for v in run_ids if v is not None))

def paired_regressions(stream_manifest,current_bank,baseline_bank):
    m=read_sealed(stream_manifest);legal=set(m['tokens'])
    current=validate_bank(current_bank,manifest=m,legal_tokens=legal)
    baseline=validate_bank(baseline_bank,manifest=m,legal_tokens=legal)
    if (current['inference']!=baseline['inference']
            or set(current['records'])!=set(baseline['records'])):
        raise ValueError('Regressions require identical scene census and sampler/seed/scorer protocol')
    changes={t:current['records'][t]['score']-baseline['records'][t]['score'] for t in current['records']}
    n=len(changes)
    safety={}
    for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound'):
        if not all(k in x.get('metrics',{}) for bank in (current,baseline) for x in bank['records'].values()):
            continue
        pairs=[(baseline['records'][t]['metrics'][k],current['records'][t]['metrics'][k]) for t in changes]
        safety[k]=dict(new_failure=sum(a==1 and b<1 for a,b in pairs),repaired_failure=sum(a<1 and b==1 for a,b in pairs))
    return dict(scope='matched current and fixed-IL TRAIN/NAVTRAIN diagnostic banks only',
                sampler=current['inference'],scene_count=n,
                mean_delta_PDMS=100*sum(changes.values())/n,
                G_plus_PDMS=100*sum(max(d,0) for d in changes.values())/n,
                G_minus_PDMS=100*sum(min(d,0) for d in changes.values())/n,
                regressed_scenes=sum(d<0 for d in changes.values()),improved_scenes=sum(d>0 for d in changes.values()),
                new_zero=sum(baseline['records'][t]['score']>0 and current['records'][t]['score']==0 for t in changes),
                repaired_zero=sum(baseline['records'][t]['score']==0 and current['records'][t]['score']>0 for t in changes),
                low_score_below_08=sum(x['score']<.8 for x in current['records'].values()),
                safety_transitions=safety,per_scene_delta=changes,
                current_bank_file_sha256=file_sha(current_bank),baseline_bank_file_sha256=file_sha(baseline_bank),
                uncertainty='descriptive paired scene counts; no training-seed confidence interval')

def analyze_journals(run_root,stream_manifest,*,world_size=8,window_updates=1000,
                     current_bank=None,baseline_bank=None):
    run_root=Path(run_root);m=read_sealed(stream_manifest)
    if world_size<=0 or window_updates<=0:raise ValueError('Positive world and window sizes required')
    if bool(current_bank)!=bool(baseline_bank):raise ValueError('Provide both diagnostic banks or neither')
    paths=[run_root/f'metrics_rank{r}.jsonl' for r in range(world_size)]
    missing=[str(p) for p in paths if not p.exists()]
    if missing:return dict(schema='world_rl_research_scene_coverage_v1',status='WAITING_FOR_RANK_JOURNALS',missing=missing)
    identity_path=run_root/'identity.json'
    if not identity_path.exists():return dict(schema='world_rl_research_scene_coverage_v1',status='WAITING_FOR_RUN_IDENTITY')
    identity=read_sealed(identity_path)
    if (identity.get('world')!=world_size
            or identity.get('data_stream_identity',{}).get('manifest_sha256')!=file_sha(stream_manifest)):
        raise ValueError('Journal analysis stream/world does not identify the actual training run')
    from .data import SceneStream
    SceneStream(stream_manifest,world_size)  # Validate legal population and source data identity.
    ranks=[];snapshots=[]
    for rank,path in enumerate(paths):
        fresh,snapshot=read_rank(path);snapshots.append(dict(snapshot,rank=rank));ranks.append(fresh)
        if snapshot['run_ids'] and snapshot['run_ids']!=[identity['sha256']]:
            raise ValueError('Journal rows have a foreign run identity')
    common=min(v['last_update'] for v in snapshots)
    selected=set(m['tokens'])
    data=json.loads((Path(m['data_root'])/'manifest.json').read_text())
    navtrain=set(data['tokens']);old_dev=set(data['old_DEV_tokens']);logs=data['token_logs']
    rows=sorted([dict(row,rank=rank) for rank,fresh in enumerate(ranks) for row in fresh if row['update']<=common],
                key=lambda row:(row['update'],row['rank']))
    if any(row['token'] not in selected for row in rows):
        raise ValueError('Training journal contains a scene outside the registered scene pool')
    groups_by_collection=Counter(row['collection'] for row in rows)
    if any(n!=world_size for n in groups_by_collection.values()):
        raise ValueError('Fresh collection is not present exactly once on each rank at common boundary')
    seen=Counter();per_scene={};buckets={}
    group_size=identity.get('recipe',{}).get('grpo',{}).get('sample',{}).get('group_size',8)
    for row in rows:
        token=row['token'];e=row['exploration'];window=(row['update']-1)//window_updates
        if any(not math.isfinite(float(e[k])) or not 0<=e[k]<=1
               for k in ('canonical_random_mean','safe_fraction','NC','DAC','TTC')):
            raise ValueError('Invalid stochastic group diagnostics')
        if window not in buckets:
            buckets[window]=dict(from_update=window*window_updates+1,to_update=min((window+1)*window_updates,common),
                                 fresh_scene_groups=0,new_unique_scenes=0,repeated_scene_exposures=0,
                                 canonical_random_score_sum=0.,safe_fraction_sum=0.,all_zero_groups=0,
                                 zero_PG_groups=0,group_has_NC_failure=0,group_has_DAC_failure=0,group_has_TTC_failure=0)
        b=buckets[window];is_new=seen[token]==0;seen[token]+=1
        b['fresh_scene_groups']+=1;b['new_unique_scenes']+=int(is_new);b['repeated_scene_exposures']+=int(not is_new)
        for dest,key in [('canonical_random_score_sum','canonical_random_mean'),('safe_fraction_sum','safe_fraction')]:
            b[dest]+=float(e[key])
        b['all_zero_groups']+=int(e['all_zero']);b['zero_PG_groups']+=int(e['zero_PG'])
        for name,key in [('NC','NC'),('DAC','DAC'),('TTC','TTC')]:b[f'group_has_{name}_failure']+=int(e[key]<1)
        s=per_scene.setdefault(token,dict(fresh_groups=0,random_score_sum=0.,safe_fraction_sum=0.,
                                         all_zero_groups=0,first_update=row['update'],last_update=row['update']))
        s['fresh_groups']+=1;s['random_score_sum']+=float(e['canonical_random_mean']);s['safe_fraction_sum']+=float(e['safe_fraction'])
        s['all_zero_groups']+=int(e['all_zero']);s['last_update']=row['update']
    for b in buckets.values():
        n=b['fresh_scene_groups'];b['new_candidates']=n*group_size
        b['mean_stochastic_PDMS']=100*b.pop('canonical_random_score_sum')/n
        b['mean_stochastic_safe_fraction']=b.pop('safe_fraction_sum')/n
    for s in per_scene.values():
        n=s['fresh_groups'];s['mean_stochastic_PDMS']=100*s.pop('random_score_sum')/n;s['mean_stochastic_safe_fraction']=s.pop('safe_fraction_sum')/n
    total=sum(seen.values());unique=len(seen)
    repeat_hist=dict(sorted(Counter(seen.values()).items()))
    result=dict(schema='world_rl_research_scene_coverage_v1',status='COMPLETE_JOURNAL_PREFIX',
                completed_at_utc=datetime.now(timezone.utc).isoformat(),run_root=str(run_root.resolve()),
                run_identity_sha256=identity['sha256'],training_purpose=identity.get('purpose'),
                common_optimizer_update=common,file_snapshots=snapshots,
                scene_manifest=str(Path(stream_manifest).resolve()),scene_manifest_sha256=file_sha(stream_manifest),
                mode=m['mode'],scene_pool_name=m['name'],selected_scene_pool=len(selected),
                full_NAVTRAIN_population=len(navtrain),fresh_global_collections=len(groups_by_collection),
                fresh_scene_groups=total,new_candidates=total*group_size,unique_seen_scenes=unique,
                repeated_scene_exposures=total-unique,coverage_of_selected_pool=unique/len(selected),
                coverage_of_NAVTRAIN=unique/len(navtrain),seen_physical_logs=len({physical(logs[t]) for t in seen}),
                old_DEV_scene_exposures=sum(v for t,v in seen.items() if t in old_dev),
                repeat_histogram={str(k):v for k,v in repeat_hist.items()},
                max_scene_exposures=max(seen.values()) if seen else 0,
                effective_unique_scenes=(total**2/sum(v*v for v in seen.values())) if total else 0,
                windows=list(buckets.values()),per_scene=per_scene,
                stochastic_means_are_not_paired_ODE_regressions=True,
                future_training_seeds_not_inferred=True,new_model_forwards=0,new_scores=0)
    if current_bank:
        result['paired_ODE_regressions']=paired_regressions(stream_manifest,current_bank,baseline_bank)
    else:
        result['paired_ODE_regressions']=dict(status='NOT_MEASURED',reason='Matched same-sampler fixed-census current/IL banks not supplied; stochastic rollouts do not establish paired regression')
    return result

def save_analysis(path,**kwargs):
    result=analyze_journals(**kwargs);atomic_json(path,sealed(result));return result

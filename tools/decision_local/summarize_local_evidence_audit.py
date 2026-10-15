"""Local support funnel and canonical/checker cross-table with scene clustering."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,read,digest,source_sha,check_host


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('evidence','current-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--natural-scores',type=Path);p.add_argument('--recovery-scores',type=Path)
    a=p.parse_args();check_host();start=time.monotonic();cpu=time.process_time()
    manifest=read(a.evidence/'identity.json');complete=read(a.evidence/'COMPLETE.json')
    if complete['identity']!=manifest['identity'] or complete['completed']!=len(manifest['index']):raise ValueError('Full fixed audit population required')
    scores={}
    for name,path in [('NATURAL',a.natural_scores),('RECOVERY',a.recovery_scores)]:
        if path:
            with path.open() as stream:scores[name]={row['token']:row for row in csv.DictReader(stream)}
            if set(scores[name])!={r['token'] for r in manifest['index']}:raise ValueError('Canonical population mismatch')
    counts=Counter();scenes=defaultdict(set);logs=defaultdict(set);actors=defaultdict(set);absolute_events=defaultdict(set)
    cross=Counter();score_rows=[];reason_sets=defaultdict(set)
    def add(group,key,token,log,n=1):
        counts[(group,key)]+=n;scenes[(group,key)].add(token);logs[(group,key)].add(log)
    for row in manifest['index']:
        token,log=row['token'],row['log']
        with gzip.open(a.evidence/'records'/(token+'.json.gz'),'rt') as f:record=json.load(f)
        if record['identity']!=manifest['identity']:raise ValueError('Foreign evidence record')
        current=read(a.current_root/'current'/(token+'.json'));timestamp=current['timestamp']
        recovery_plan=1+int(hashlib.sha256((token+':fixed-recovery-v1').encode()).hexdigest()[:8],16)%4
        for case in record['cases']:
            group=case['distribution'];add(group,'population',token,log)
            for key in ('local_road_positive','local_overlap_positive','strict_target_valid'):
                if case[key]:add(group,key,token,log)
            if (case['local_road_positive'] or case['local_overlap_positive']) and not case['strict_target_valid']:
                add(group,'known_local_fact_without_full_target',token,log)
            for kind,values in [('at',case['per_time_support']),('prefix',case['prefix_support'])]:
                for t,channels in values.items():
                    for ch,yes in enumerate(channels):
                        if yes:add(group,f'{kind}_{t}_channel{ch}_supported',token,log)
            for reason,yes in case['unknown_reasons'].items():
                if yes:add(group,'unknown_'+reason,token,log)
            distribution='NATURAL' if case['plan_index']==0 else 'RECOVERY' if case['plan_index']==recovery_plan else None
            if case['anchor_index']!=0 or distribution not in scores:continue
            s=scores[distribution][token];score=float(s['score'])
            category='zero' if score==0 else 'low_0_to_0.5' if score<=.5 else 'normal_above_0.5'
            known=case['old_input_check']['known_violation'];eligible=case['strict_target_valid']
            cross[(distribution,category,str(known),str(eligible),case['strict_status'])]+=1
            score_rows.append(dict(token=token,log=log,distribution=distribution,score=score,canonical_group=category,
                known_violation=known,strict_target_valid=eligible,strict_status=case['strict_status'],
                local_road_positive=case['local_road_positive'],local_overlap_positive=case['local_overlap_positive'],
                **{k:float(s[k]) for k in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound',
                                          'ego_progress','comfort','driving_direction_compliance')}))
        for event in record['events']:
            group='NATURAL' if event['plan_index']==0 else 'SYNTHETIC';key=event['constraint_type']
            add(group,'event_'+key,token,log)
            actors[(group,key)].add(event['actor'])
            absolute_events[(group,key)].add((event['actor'],int(timestamp+round(event['time']*1e6))))
    result=dict(schema='natural_error_local_evidence_summary_v1',source_sha=source_sha(ROOT),evidence_identity=manifest['identity'],
        source_complete_sha256=digest(a.evidence/'COMPLETE.json'),split=manifest['split'],scenes=len(manifest['index']),
        counts={g:{key:dict(plan_or_time_records=n,unique_scenes=len(scenes[(g,key)]),unique_logs=len(logs[(g,key)]))
                   for (gg,key),n in counts.items() if gg==g} for g in ('NATURAL','SYNTHETIC')},
        unique_events={g:{key:dict(actors=len(actors[(g,key)]),actor_absolute_time_pairs=len(absolute_events[(g,key)]))
                          for gg,key in actors if gg==g} for g in ('NATURAL','SYNTHETIC')},
        canonical_cross_table=[dict(zip(('distribution','canonical_group','known_violation','target_eligible','strict_status'),k),scenes=v)
                               for k,v in sorted(cross.items())],
        canonical_available_distributions=list(scores),
        distinction='Geometric annotated overlap is not official responsibility/TTC; support and full target eligibility are separate',
        repeated_plans_not_independent=True,CPU_process_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start)
    a.output.mkdir(parents=True,exist_ok=True);atomic_json(a.output/'SUMMARY.json',result)
    if score_rows:
        with gzip.open(a.output/'canonical_cross_rows.csv.gz','wt') as stream:
            writer=csv.DictWriter(stream,list(score_rows[0]));writer.writeheader();writer.writerows(score_rows)
    print(json.dumps({g:{k:v for k,v in result['counts'][g].items() if k in ('population','local_road_positive','local_overlap_positive','known_local_fact_without_full_target')} for g in result['counts']}))


if __name__=='__main__':main()

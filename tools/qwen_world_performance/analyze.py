"""Consume existing sealed official NAVTEST specs/results only; no scoring."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT=Path('/mnt/project/qwen-world-performance-rl-20261103')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read(path, inputs):
    raw=Path(path).read_bytes();inputs.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
    return json.loads(raw)


def sealed(value):
    body=dict(value);saved=body.pop('sha256',None)
    return isinstance(saved,str)and fingerprint(body)==saved


def official_result(spec_path, registry_runs, root, inputs):
    spec=read(spec_path,inputs)
    if spec.get('cohort')!='NAVTEST':return None,'OTHER_COHORT'
    if not sealed(spec) or spec.get('count')!=12146 or spec.get('seed')!=2101 or spec.get('inference')!='single_candidate_cleanx_K10_shift5':
        raise ValueError('Unsealed/noncanonical official NAVTEST spec')
    arm=spec.get('arm');run=registry_runs.get(arm)
    if run is None:raise ValueError('Spec arm is not registered; qualification cannot become a formal score')
    checkpoint=Path(spec['checkpoint']['path']).resolve()
    if Path(run['root']).resolve()/'checkpoints' not in checkpoint.parents:raise ValueError('Foreign/qualification native checkpoint')
    output=Path(spec.get('output')or spec_path.parent).resolve()
    if root.resolve()not in output.parents:raise ValueError('Foreign evaluation output')
    summary_path=output/'SUMMARY.json'
    if not summary_path.is_file():return None,'SUMMARY_PENDING'
    value=read(summary_path,inputs)
    if value.get('status')!='COMPLETE':return None,'SUMMARY_NOT_COMPLETE'
    if (not sealed(value)or value.get('schema')!='world_rl_research_evaluation_summary_v1'
            or value.get('cohort')!='NAVTEST'or value.get('scenes')!=12146 or value.get('expected_scenes')!=12146
            or value.get('invalid_scenes')!=0 or value.get('seed')!=2101 or value.get('IL_updates')!=70000
            or value.get('RL_updates')!=spec.get('updates')):
        raise ValueError('Only sealed full12146/invalid0 official summaries are publishable')
    score=value.get('PDMS');mean=value.get('means',{}).get('score')
    if not isinstance(score,(int,float))or not math.isfinite(score)or not 0<=score<=100 or not isinstance(mean,(int,float))or abs(score-100*mean)>1e-8:
        raise ValueError('Nonfinite or inconsistent canonical PDMS')
    identity_path=output/'identity.json'
    identity=read(identity_path,inputs)
    if not sealed(identity)or value.get('evaluation_identity_sha256')!=identity.get('sha256'):
        raise ValueError('Summary is not bound to its complete evaluation identity')
    # Do not scan/rescore12146 scenes or invent a paired repair statistic.
    return dict(arm=arm,updates=spec['updates'],PDMS=score,scenes=12146,invalid_scenes=0,
                means=value.get('means'),physical_logs=value.get('physical_logs'),seed=2101,
                checkpoint=str(checkpoint),spec=str(spec_path),summary=str(summary_path),summary_sha256=value['sha256'],
                model_sha256=value.get('model_sha256'),protocol_sha256=value.get('protocol_sha256'),
                token_sha256=value.get('token_sha256'),
                zero_count=None,paired_zero_repaired=None,paired_new_zero=None,paired_PDMS_CI95=None,
                comparison_status='NO_QUALIFIED_PAIRED_ARTIFACT_CONSUMED'),None


def collect_results(root=ROOT):
    root=Path(root);inputs=[];runs={};pending=[];rejected=[];points=[]
    registry_path=root/'RUNS.json'
    if registry_path.is_file():
        registry=read(registry_path,inputs)
        source=registry.get('runs',[])
        source=list(source.values())if isinstance(source,dict)else source
        runs={row['id']:row for row in source}
    paths=set((root/'evaluations').glob('**/SPEC.json'))|set((root/'evaluation_specs').glob('*.json'))
    for path in sorted(paths):
        try:
            point,reason=official_result(path,runs,root,inputs)
            if point:points.append(point)
            elif reason!='OTHER_COHORT':pending.append(dict(spec=str(path),status=reason))
        except (OSError,ValueError,KeyError,TypeError)as error:
            rejected.append(dict(spec=str(path),status='REVIEW_REQUIRED_NOT_PUBLISHED',reason=str(error)))
    # Duplicate indexes may refer to the same official summary; publish once.
    points=list({row['summary']:row for row in points}.values())
    points.sort(key=lambda row:(row['arm'],row['updates']))
    holds=[]
    for arm,run in sorted(runs.items()):
        observed=None;progress_path=Path(run['root'])/'progress.json'
        if progress_path.is_file():
            try:
                progress=read(progress_path,inputs);observed=progress.get('valid_scene_exposures')
            except (OSError,ValueError):pass
        actual=run.get('long_training_hold')
        holds.append(dict(arm=arm,observed_valid_scene_exposures=observed,
                          minimum_epochs=run.get('minimum_epochs'),initial_target_epochs=run.get('target_epochs'),
                          hold_status=actual.get('status')if isinstance(actual,dict)else None,
                          hold_reason=actual.get('reason')if isinstance(actual,dict)else None,
                          evidence=actual.get('evidence')if isinstance(actual,dict)else None,
                          interpretation='Unknown hold fields are null; not an instruction to stop or extend.'))
    best=max(points,key=lambda row:row['PDMS'],default=None)
    return dict(schema='world_performance_existing_official_NAVTEST_analysis_v1',
                status='COMPLETE_AVAILABLE_OFFICIAL_RESULTS'if points else'NO_COMPLETE_OFFICIAL_NAVTEST',
                generated_UTC=datetime.now(timezone.utc).isoformat(),complete_results=points,complete_count=len(points),
                best=best,new_global_best_claim=False,formal_gain_vs_baseline=None,
                paired_comparisons=[],paired_repairs_status='NOT_COMPUTED_NO_SEALED_PAIRED_INPUT',
                paired_zero_repaired=None,paired_new_zero=None,longtrainholds=holds,
                longtrainhold_template=dict(minimum_epochs=None,target_epochs=None,observed_epochs=None,
                    hold_status=None,hold_reason=None,decision_evidence=None),
                pending=pending,rejected=rejected,input_files=inputs,
                limitations=['Only registered formal arms and sealed canonical official specs with full12146/invalid0 are included.',
                    'No partial mean, qualification result, TRAIN panel, oracle or raw candidate score is a deployment result.',
                    'Missing zeros/paired CIs/repair counts/hold decisions remain null, not numerical zero.',
                    'This CLI does not create bootstrap estimates, new GPU jobs, watchers, training data or lifecycle decisions.'])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path)
    args=p.parse_args();value=collect_results(args.root);path=args.output or args.root/'analysis/performance_results.json'
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name('.'+path.name+'.analyze.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
    print(json.dumps(dict(status=value['status'],complete_count=value['complete_count'],pending=len(value['pending']),rejected=len(value['rejected']),output=str(path))))


if __name__=='__main__':main()

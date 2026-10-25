"""Read-only explicit native-v1 baseline registration and matched-K comparison.

Only completed own native IL/CANONICAL rows are accepted, using their original
integrity verifier. This adapter never relaxes normal new-method comparisons or
creates an evaluation. Maximum completed update, never observed score, selects
one fixed native IL and one fixed canonical-RL reference before new results.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from scripts.analysis.qwen_sourceflow.evaluation_integrity import verify_identity,verify_scene
from starVLA.rl.qwen_sourceflow.contracts import fingerprint as native_fingerprint
from starVLA.rl.qwen_world_postcot.contracts import ROOT,file_sha,fingerprint
from starVLA.rl.qwen_world_postcot.data import POPULATION
from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from tools.qwen_world_postcot.evaluate import validate_feedback

OLD_ROOT=Path('/mnt/project/qwen-sourceflow-native-20261018')
BINDING_SCHEMA='world_postcot_pinned_legacy_native_baselines_v1'


def _summary(path):
    value=json.loads(Path(path).read_text());body=dict(value);seal=body.pop('sha256',None)
    if native_fingerprint(body)!=seal or value.get('status')!='COMPLETE'or value.get('qualification')is not False:
        raise ValueError('Legacy native summary is not a complete sealed formal observation')
    if value.get('cohort')!='NAVTEST'or value.get('full_population')is not True or value.get('seed')!=2101:
        raise ValueError('Legacy adapter only accepts full fixed seed2101 Navtest')
    return value


def _updates(summary):
    if summary.get('rl_export')is not None:return int(summary['rl_export']['updates'])
    values=re.findall(r'update(\d{5,6})',str(summary['checkpoint']))
    if not values:raise ValueError('Legacy IL checkpoint update is not explicit')
    return int(values[-1])


def load_legacy(entry,*,population=None):
    summary_path=Path(entry['summary']);summary=_summary(summary_path)
    if file_sha(summary_path)!=entry['summary_file_sha256']:raise ValueError('Pinned legacy summary bytes changed')
    root=Path(summary['evaluation']);identity=verify_identity(json.loads((root/'identity.json').read_text()))
    if identity['sha256']!=entry['evaluation_identity_sha256']or identity['cohort']!='NAVTEST'or identity['qualification']:
        raise ValueError('Pinned legacy evaluation identity changed')
    source_model=identity['rl_export']['model_sha256']if identity.get('rl_export')else identity['parent_model_sha256']
    if (summary.get('source_identity')!=identity['sha256']or summary['model_sha256']!=source_model
            or entry['model_sha256']!=source_model or summary['IL_parent_checkpoint']!=identity['checkpoint']
            or summary.get('rl_export')!=identity.get('rl_export')):
        raise ValueError('Legacy summary/source model, parent or export lineage changed')
    population=json.loads(POPULATION.read_text())if population is None else population
    tokens=population['navtest_tokens']
    if (identity['count']!=len(tokens)or summary['scenes']!=len(tokens)
            or identity['token_sha256']!=fingerprint(tokens)
            or {path.stem for path in(root/'scenes').glob('*.json')}!=set(tokens)):
        raise ValueError('Legacy baseline must contain every actual fixed scene; no favorable subset')
    rows={}
    for token in tokens:
        value=verify_scene(json.loads((root/'scenes'/f'{token}.json').read_text()),identity,
            token=token,log=population['token_logs'][token])
        validate_feedback(value['feedback'],token=token,log=value['log'],requested=value['requested_trajectory'],
            protocol_sha256=identity['protocol']['sha256'])
        rows[token]=dict(token=token,log=value['log'],feedback=dict(metrics=value['feedback']['metrics'],cache=value['feedback']['cache']))
    means={name:float(np.mean([value['feedback']['metrics'][name]for value in rows.values()]))for name in summary['means']}
    if any(abs(means[name]-summary['means'][name])>1e-12 for name in means):
        raise ValueError('Legacy aggregate does not reconstruct from its sealed actual per-scene rows')
    return identity,rows


def register(output,*,old_root=OLD_ROOT,population=None):
    output=Path(output)
    if output.exists():raise FileExistsError('Baseline selection is pinned once before new-method results; keep previous binding')
    population=json.loads(POPULATION.read_text())if population is None else population
    canonical_ids=set()
    for path in(Path(old_root)/'runs').glob('RL_CANONICAL*/identity.json'):
        value=json.loads(path.read_text())
        if value.get('schema')=='native_qwen_sourceflow_RL_v1'and value.get('purpose')=='formal'and value['recipe']['variant']=='R_CANONICAL':
            canonical_ids.add(value['sha256'])
    candidates={'native_IL':[],'native_CANONICAL_RL':[]}
    rejected=[]
    for filename,label in [('IL_summary.json','native_IL'),('canonical_summary.json','native_CANONICAL_RL')]:
        for path in sorted(Path(old_root).glob('latest_NAVTEST_*/analysis/'+filename)):
            try:
                summary=_summary(path);root=Path(summary['evaluation']);identity=verify_identity(json.loads((root/'identity.json').read_text()))
                if(identity['count']!=len(population['navtest_tokens'])or identity['token_sha256']!=fingerprint(population['navtest_tokens'])
                        or {p.stem for p in(root/'scenes').glob('*.json')}!=set(population['navtest_tokens'])):
                    raise ValueError('Actual complete rows unavailable')
                if label=='native_IL'and summary['policy_type']!='IL':raise ValueError('Not a plain native IL reference')
                if label=='native_CANONICAL_RL'and(summary['policy_type']!='RL_MERGED'or summary['rl_export']['run_identity']not in canonical_ids):
                    raise ValueError('Not the actual own original canonical-RL lineage')
                candidates[label].append((_updates(summary),str(path),identity,summary))
            except(ValueError,OSError,KeyError)as error:rejected.append(dict(path=str(path),reason=str(error)))
    entries=[]
    for label,values in candidates.items():
        if not values:raise ValueError('No actual complete eligible legacy baseline: '+label)
        # Score is deliberately absent from selection/tie keys.
        updates,path,identity,summary=max(values,key=lambda row:(row[0],row[1]))
        entry=dict(name=label,updates=updates,summary=path,summary_file_sha256=file_sha(path),
            evaluation=summary['evaluation'],evaluation_identity_sha256=identity['sha256'],
            model_sha256=summary['model_sha256'],IL_parent_checkpoint=summary['IL_parent_checkpoint'],
            source_policy_type=summary['policy_type'],precision=identity['precision'],
            architecture=dict(family='original native QwenSourceFlowVLA',prediction_type='velocity',
                action_config=None if summary['rl_export']is None else summary['rl_export']['action_config']),
            selection='maximum actual complete optimizer update, exact tie lexical registered run path; never PDMS',
            data_exposure='Original native training population101592; includes15physicalDEV logs subsequently embargoed for new90890TRAIN',
            budget_and_parent_caveat='Native canonicalRL has its own53kIL parent; never subtract newer77kIL to call that RL training gain')
        _,rows=load_legacy(entry,population=population)
        entry.update(actual_scene_count=len(rows),actual_physical_logs=len({re.sub(r'_\d+_\d+$','',row['log'])for row in rows.values()}),
            PDMS_reconstructed=100*float(np.mean([row['feedback']['metrics']['score']for row in rows.values()])),
            full_actual_row_integrity_verified=True)
        entries.append(entry)
    body=dict(schema=BINDING_SCHEMA,entries=entries,rejected_unavailable_observations=rejected,
        registration_rule='highest completed step per explicitly eligible native method before any new-method Navtest',
        comparison_protocol=dict(cohort='NAVTEST',seed=2101,candidates=1,ODE_steps=10,shift=5,
            poses=8,interval_seconds=.5,coordinate='rear-axle ego x/y/yaw-radians',metric='canonical NAVSIMv1.1 independent reference+one candidate'),
        no_new_baseline_evaluation=True,no_score_based_selection=True)
    body['sha256']=fingerprint(body);output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,body)
    return body


def compare_registered(identity,rows,binding_path,*,population=None):
    from tools.qwen_world_postcot.summarize import paired_statistics
    value=json.loads(Path(binding_path).read_text());body=dict(value);seal=body.pop('sha256',None)
    if body['schema']!=BINDING_SCHEMA or fingerprint(body)!=seal:raise ValueError('Legacy baseline binding changed')
    if(identity.get('schema')!='world_postcot_evaluation_v1'or identity['cohort']!='NAVTEST'or identity['qualification']
            or identity['seed']!=2101 or identity['inference']!='single_candidate_cleanx_K10_shift5'):
        raise ValueError('Explicit legacy comparison is only the new primary single-cleanx K10shift5 seed2101 Navtest')
    results={}
    for entry in body['entries']:
        old,base=load_legacy(entry,population=population)
        if old['protocol']['sha256']!=identity['protocol']['sha256']or set(base)!=set(rows):
            raise ValueError('Different methods must still use the exact official protocol and full matched physical scenes')
        result=paired_statistics(rows,base)
        result.update(baseline=entry['name'],baseline_updates=entry['updates'],baseline_PDMS=entry['PDMS_reconstructed'],
            new_method_PDMS=100*float(np.mean([row['feedback']['metrics']['score']for row in rows.values()])),
            explicitly_allowed_method_differences=dict(legacy_architecture=entry['architecture'],new_policy_lineage=identity.get('policy_lineage'),
                legacy_precision=entry['precision'],legacy_data_manifest_sha256=old['data_sha256'],new_data_manifest_sha256=identity['data_sha256'],
                training_exposure=entry['data_exposure'],parent_budget=entry['budget_and_parent_caveat']),
            interpretation='Matched-population method comparison; architecture,precision,trainingbudget/parents differ; not isolated auxiliary-module causal gain')
        results[entry['name']]=result
    return results


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',default=str(ROOT/'campaign/BASELINE_BINDING.json'))
    parser.add_argument('--legacy-root',default=str(OLD_ROOT));args=parser.parse_args()
    value=register(args.output,old_root=args.legacy_root)
    print(json.dumps({entry['name']:dict(updates=entry['updates'],PDMS=entry['PDMS_reconstructed'],scenes=entry['actual_scene_count'],physical_logs=entry['actual_physical_logs'])for entry in value['entries']}))


if __name__=='__main__':main()

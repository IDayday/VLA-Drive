"""Thin P0-P5 comparisons over the shared verified scene/seed/log analysis."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from starVLA.rl.qwen_flow_exploration.protocol import read_protocol
from starVLA.rl.qwen_flow_exploration.common_analysis import analyze_pair, load_evaluation, bootstrap, checked_seal
from starVLA.rl.qwen_sourceflow.contracts import fingerprint, file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


COMPARISONS = dict(P3_minus_P2=('P3','P2'), P5_minus_P1=('P5','P1'),
                   P3_minus_P0=('P3','P0'), P4_minus_P0=('P4','P0'),
                   P5_minus_P4=('P5','P4'), P5_minus_P3=('P5','P3'))


def factorial_interaction(evaluations, *, repeats=2000, seed=42):
    """Use the common loader's verified scene-first arrays; no new bootstrap."""
    if not all(arm in evaluations for arm in ('P0','P3','P4','P5')):
        return dict(status='NOT_AVAILABLE',requires=['P0','P3','P4','P5'])
    base=evaluations['P0']
    for arm in ('P3','P4','P5'):
        value=evaluations[arm]
        if any(getattr(value,key)!=getattr(base,key) for key in
               ('tokens','seeds','logs','metric_names','scorer_sha256','sampler_sha256')):
            raise ValueError('Unmatched archive-by-constraint factorial population')
        if value.identity['training_seed']!=base.identity['training_seed']:
            raise ValueError('Do not silently combine different training seeds')
    index=base.metric_names.index('score')
    means={arm: value.values[:,:,index].mean(1) for arm,value in evaluations.items()}
    interaction=means['P5']-means['P3']-means['P4']+means['P0']
    return dict(status='COMPLETE',contrast='P5-P3-P4+P0',
                meaning='Additional combined mean benefit beyond separate archive and cost effects; not a mechanistic proof',
                bootstrap=bootstrap(interaction,base.logs,repeats=repeats,seed=seed,kind='score'))


def training_cost(path, protocol):
    """Read committed metrics and all physical ledgers, never load the actor."""
    path=Path(path);identity=checked_seal(json.loads((path/'identity.json').read_text()),str(path/'identity.json'))
    if identity['common_protocol_sha256']!=protocol['sha256'] or identity['base_model_sha256']!=protocol['base_model_sha256']:
        raise ValueError('Training cost belongs to another population/base')
    complete_path=path/'COMPLETE.json'
    if not complete_path.exists():
        return dict(status='NOT_COMPLETE',path=str(path),identity_sha256=identity['sha256'],
                    note='No final throughput/efficacy estimate from a partially committed run')
    complete=json.loads(complete_path.read_text())
    if complete['status']!='COMPLETE' or complete['identity']!=identity['sha256'] or identity['qualification']:
        raise ValueError('Wrong/qualification final training receipt')
    updates=complete['updates'];B,G=identity['B'],identity['G']
    if updates!=4*identity['rounds'] or complete['fresh_collections']!=identity['rounds'] or (
            complete['scene_exposures']!=B*identity['rounds'] or complete['new_candidates']!=B*G*identity['rounds']):
        raise ValueError('Four inner steps cannot become four fresh candidate batches')
    rows=[]
    for update in range(1,updates+1):
        row=json.loads((path/'metrics'/f'update{update:06d}.json').read_text())
        if (row['identity']!=identity['sha256'] or row['collection']!=(update-1)//4 or
                row['inner_completed']!=(update-1)%4+1 or row['true_optimizer_steps']!=1):
            raise ValueError('Discontinuous/foreign committed metric sequence')
        rows.append(row)
    physical=[json.loads(file.read_text()) for file in (path/'physical_invocations').glob('*.json')]
    physical_totals={}
    for ledger in physical:
        for name,value in ledger.items():
            if isinstance(value,(int,float)) and not isinstance(value,bool):
                physical_totals[name]=physical_totals.get(name,0)+value
    policy_keys=sorted(set().union(*(row['policy_metrics'].keys() for row in rows)))
    return dict(status='COMPLETE',path=str(path),identity_sha256=identity['sha256'],arm=identity['arm'],
        training_seed=identity['seed'],logical_optimizer_updates=updates,
        logical_fresh_collections=complete['fresh_collections'],logical_scene_exposures=complete['scene_exposures'],
        logical_new_candidates=complete['new_candidates'],physical_ledger_totals=physical_totals,
        physical_invocations=len(physical),committed_inner_update_seconds=sum(row['seconds'] for row in rows),
        effective_PG_scenes=sum(row['effective_PG_scenes'] for row in rows if row['inner_completed']==1),
        supervised_scene_exposures=sum(row['supervised_scene_exposures'] for row in rows),
        velocity_forward_current=sum(row['velocity_forward_current'] for row in rows),
        velocity_forward_reference=sum(row['velocity_forward_reference'] for row in rows),
        backward_calls=sum(row['backward_calls'] for row in rows),
        mean_policy_metrics={key:float(np.mean([row['policy_metrics'][key] for row in rows if key in row['policy_metrics']])) for key in policy_keys},
        last_dual=complete['preservation_custom']['dual'],
        cost_semantics='All available physical invocation ledgers include retries; summed worker occupancy is not exclusive GPU time or end-to-end wall',
        untracked_failure_warning='A hard kill before ledger flush may leave unrecoverable physical time; never infer zero overhead from missing data')


def parse_paths(items, allowed):
    result={}
    for item in items:
        arm,path=item.split('=',1)
        if arm not in allowed or arm in result:raise ValueError('Unknown/duplicate comparison arm')
        result[arm]=Path(path)
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--protocol',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--evaluation',action='append',default=[],help='P0=/complete/plain-evaluation, repeat by arm')
    p.add_argument('--training-run',action='append',default=[],help='P0=/run, optional physical cost linkage')
    p.add_argument('--reference-strata',required=True);p.add_argument('--calibration');p.add_argument('--archive')
    p.add_argument('--output',required=True);p.add_argument('--bootstrap-repeats',type=int,default=2000)
    p.add_argument('--bootstrap-seed',type=int,default=42)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Preserve earlier analyses; use a new output identity')
    protocol=read_protocol(a.protocol);paths=parse_paths(a.evaluation,('P0','P1','P2','P3','P4','P5'))
    if not paths:raise ValueError('At least one complete deployment evaluation is required; qualification is insufficient')
    runs=parse_paths(a.training_run,paths)
    out.mkdir(parents=True)
    summaries={};paired_scenes={};evaluations={}
    for arm,path in paths.items():
        value=load_evaluation(path,protocol)
        if value.identity['arm']!=arm:raise ValueError('CLI arm does not match merged evaluation identity')
        evaluations[arm]=value
        summary,rows=analyze_pair(path,a.reference,protocol,a.reference_strata,repeats=a.bootstrap_repeats,bootstrap_seed=a.bootstrap_seed)
        summaries[arm]=summary;paired_scenes[arm]=rows
    contrasts={}
    for name,(current,reference) in COMPARISONS.items():
        if current not in paths or reference not in paths:
            contrasts[name]=dict(status='NOT_AVAILABLE',requires=[current,reference]);continue
        summary,rows=analyze_pair(paths[current],paths[reference],protocol,a.reference_strata,
                                  repeats=a.bootstrap_repeats,bootstrap_seed=a.bootstrap_seed)
        contrasts[name]=summary;paired_scenes[name]=rows
    calibration=checked_seal(json.loads(Path(a.calibration).read_text()),'TRAIN calibration') if a.calibration else None
    if calibration and calibration['protocol_sha256']!=protocol['sha256']:raise ValueError('Foreign P1 calibration')
    archive=None
    if a.archive:
        from starVLA.rl.qwen_flow_exploration.preservation.archive import FrozenArchive
        value=FrozenArchive.from_state_dict(json.loads(Path(a.archive).read_text()))
        if not set(value.scenes)<=set(protocol['splits']['TRAIN']):raise ValueError('Archive contains nontraining scenes')
        archive=dict(state_sha256=value.digest(),provenance=value.provenance,
                     replay_population_coverage=len(value.scenes)/len(protocol['splits']['TRAIN']),
                     population_bias='Rehearsal samples only qualified fixed TRAIN archive scenes; PG samples the entire registered population')
    costs={arm:training_cost(path,protocol) for arm,path in runs.items()}
    for name,rows in paired_scenes.items():
        with (out/(name+'_paired_scenes.csv')).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(schema='preservation_shared_analysis_v1',status='COMPLETE',protocol_sha256=protocol['sha256'],
        implementation='Only common_analysis performs deployment loading, seed-within-scene aggregation and log bootstrap',
        source_sha256=file_sha(__file__),per_arm_vs_base=summaries,mechanism_contrasts=contrasts,
        archive_by_constraint_interaction=factorial_interaction(evaluations,repeats=a.bootstrap_repeats,seed=a.bootstrap_seed),
        training_costs=costs,archive=archive,P1_matching=calibration['P1_matching'] if calibration else None,
        P1_interpretation=('Only TRAIN drift-matched, never true output KL; inspect MATCHED status before excluding slower learning'
                           if calibration else 'No calibration attached; slower-learning explanation not excluded'),
        research_label='证据不足',
        classification_note='Numerical summaries do not automatically confer research support; review all controls, rarity resolution and independent training seed evidence',
        remaining_arms=[arm for arm in ('P0','P1','P2','P3','P4','P5') if arm not in paths])
    result['sha256']=fingerprint(result);atomic_json(out/'summary.json',result)
    lines=['# P0–P5 共同部署口径分析','','所有分数先按scene平均固定推理seed，再按完整log作配对bootstrap。',
           '固定候选oracle与存档不进入部署输出选择。','','|臂|PDMS|相对base净变化|G+|G−|', '|---|---:|---:|---:|---:|']
    for arm,row in summaries.items():
        lines.append(f"|{arm}|{row['current_PDMS']:.4f}|{row['delta_PDMS_points']:+.4f}|{row['G_plus_PDMS_points']:.4f}|{row['G_minus_PDMS_points']:.4f}|")
    lines+=['','此自动汇总的研究标签为“证据不足”，须结合完整机制对照、训练seed和稀有事件分辨率判断，不能仅凭均分自动宣告支持。',
            'P3−P2检查多轨存档；P5−P1检查幅度匹配替代解释；P0/P3/P4/P5检查存档与约束的独立及联合效果。',
            '若P1未达到TRAIN漂移匹配，不排除只是更新更保守；若P0没有明显退化，本轮对能力保持问题的判别力不足。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='COMPLETE',arms=list(paths),missing=result['remaining_arms'],output=str(out))))


if __name__=='__main__':main()

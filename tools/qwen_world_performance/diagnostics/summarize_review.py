"""One CPU-only review of sealed actual TRAIN19 results; no rescoring/training."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/mnt/project/qwen-world-performance-rl-20261103/runtime/formal_v1')
sys.path.insert(0,str(SOURCE))
from starVLA.rl.qwen_world_research_data.manifest import read_sealed,file_sha,fingerprint,atomic_json,physical


def high(m):
    return float(m['score'])>=.9 and strict4(m)


def strict4(m):
    return all(float(m[k])==1 for k in ('no_at_fault_collisions','drivable_area_compliance',
        'time_to_collision_within_bound','comfort'))


def feasible(m):
    return all(float(m[k])==1 for k in ('no_at_fault_collisions','drivable_area_compliance',
        'time_to_collision_within_bound'))


def sealed(x):return dict(x,sha256=fingerprint(x))


def main():
    import numpy as np
    result=ROOT/'results'
    manifest=read_sealed(ROOT/'MANIFEST.json');identity=read_sealed(result/'identity.json')
    original=read_sealed(result/'SUMMARY.json');complete=read_sealed(result/'COMPLETE.json')
    job=Path('/mnt/project/qwen-world-performance-rl-20261103/jobs/PROBE_IL70_TRAIN19_G16_G32')
    transport=json.loads((job/'RESULT.json').read_text())
    ledger=json.loads((result/'BUDGET_LEDGER.json').read_text())
    if (transport['returncode']!=0 or complete['status']!='COMPLETE' or original['status']!='COMPLETE'
            or complete['summary_sha256']!=original['sha256']
            or original['manifest_sha256']!=manifest['sha256']
            or identity['manifest_sha256']!=manifest['sha256']
            or manifest['script_sha256']!=file_sha(ROOT/'probe_train19.py')
            or ledger['reserved_queries']!=912 or ledger['completed_queries']!=912
            or len(ledger['groups'])!=38 or identity['Adam_updates']!=0
            or identity['scope']['trainable_numel']!=52707840):
        raise ValueError('Actual complete probe/source/budget/IL70 evidence differs')
    cases=[];group_evidence={};array_evidence={};all_groups=[]
    for token in manifest['tokens']:
        registered=manifest['scenes'][token]
        b1=read_sealed(registered['B1_scene_path']);gt=read_sealed(registered['GT_scene_path'])
        if b1['sha256']!=registered['B1_scene_sha256'] or gt['sha256']!=registered['GT_scene_sha256']:
            raise ValueError('Reused B1/GT identity changed')
        case=dict(token=token,log=registered['log'],physical_log=physical(registered['log']),
            role=registered['role'],B1=registered['B1_metrics'],GT=registered['GT_metrics'],
            GT_strict4=strict4(registered['GT_metrics']),GT_strict_high=high(registered['GT_metrics']),groups={})
        for g in (16,32):
            key=token+'_G'+str(g);path=result/'groups'/(key+'.json');row=read_sealed(path)
            if (row['identity_sha256']!=identity['sha256'] or row['G']!=g
                    or row['token']!=token or row['log']!=registered['log']
                    or row['chain_shape']!=[g,11,8,3] or row['probability_shape']!=[g,3]
                    or row['one_actual_batch_call'] is not True or row['normalization_applied'] is not False
                    or row['fresh_ratio_error']>1e-4 or row['sampling_check']['status']!='PASS'
                    or ledger['groups'][key]['status']!='COMPLETE'
                    or ledger['groups'][key]['output_sha256']!=row['sha256']
                    or len(row['records'])!=g):
                raise ValueError('Wrong actual G/shape/source probability/group mapping')
            ap=Path(row['arrays']['path'])
            if file_sha(ap)!=row['arrays']['sha256']:raise ValueError('Actual chain/rawA material changed')
            with np.load(ap,allow_pickle=False) as arr:
                a=arr['A_raw'][0].copy();requests=arr['requested_trajectory'].copy()
                if arr['chain'].shape!=(g,11,8,3) or arr['stoch_idx'].tolist()!=[7,8,9]:
                    raise ValueError('G32 cannot be reconstructed from two smaller groups')
            ms=[r['metrics'] for r in row['records']]
            for i,record in enumerate(row['records']):
                if (record['token']!=token or record['log']!=registered['log']
                        or record['cache']['sha256']!=registered['cache']['sha256']
                        or record['protocol_sha256']!=identity['protocol']['sha256']
                        or not np.array_equal(requests[i],np.asarray(record['requested_trajectory'],dtype=np.float32))):
                    raise ValueError('Scored actual request/cache/protocol changed')
            stats=dict(row['statistics'])
            pg0=bool(np.all(a==0));allzero=all(float(m['score'])==0 for m in ms)
            if pg0!=stats['PG0'] or allzero!=stats['all_canonical_zero']:
                raise ValueError('Stored raw-credit/canonical-zero distinction differs')
            stats.update(strict4_found=bool(stats['strict4_candidates']),
                strict_high_found=bool(stats['strict_high_candidates']),
                strict_high_zero_rawA=sum(high(m) and float(x)==0 for m,x in zip(ms,a)),
                source_clamp_hit_rate=row['source_clamp_hit_rate'],fresh_ratio_error=row['fresh_ratio_error'],
                sampled_recomputed_W3_mean_std_exact=True,
                nominal_B1_bad_but_group_high=(not high(case['B1'])) and bool(stats['strict_high_candidates']),
                records_path=str(path),records_sha256=row['sha256'],arrays_sha256=row['arrays']['sha256'])
            case['groups'][str(g)]=stats
            all_groups.append(dict(token=token,role=registered['role'],G=g,statistics=stats,seconds=row['seconds']))
            group_evidence[str(path)]=file_sha(path);array_evidence[str(ap)]=row['arrays']['sha256']
        a=case['groups']['16'];b=case['groups']['32'];fa=a['strict_high_found'];fb=b['strict_high_found']
        case.update(high_witness_pattern=('both' if fa and fb else 'G16_only' if fa else 'G32_only' if fb else 'neither'),
            both_no_high_but_GT_high=(not fa and not fb and case['GT_strict_high']),
            both_no_high_but_GT_strict4=(not fa and not fb and case['GT_strict4']))
        cases.append(case)

    cohorts={}
    for role in ('B1_zero','fixed_strict_high_control'):
        cc=[c for c in cases if c['role']==role];groups={}
        for g in (16,32):
            vv=[c['groups'][str(g)] for c in cc];n=len(vv)*g
            totals={k:sum(v[k] for v in vv) for k in ('feasible_candidates','strict4_candidates',
                'strict_high_candidates','strict_high_positive_rawA','strict_high_negative_rawA',
                'strict_high_zero_rawA','unsafe_candidates','unsafe_positive_rawA','canonical_zero_candidates',
                'raw_A_positive','raw_A_negative')}
            groups[str(g)]=dict(scenes=len(vv),actual_PDM_queries=n,
                scenes_with_strict4=sum(v['strict4_found'] for v in vv),
                scenes_with_strict_high=sum(v['strict_high_found'] for v in vv),
                scenes_with_unsafe_candidates=sum(v['unsafe_candidates']>0 for v in vv),
                all_canonical_zero_groups=sum(v['all_canonical_zero'] for v in vv),
                canonical_tie_groups=sum(v['canonical_tie'] for v in vv),
                raw_credit_PG0_groups=sum(v['PG0'] for v in vv),**totals,
                unsafe_fraction=totals['unsafe_candidates']/n,
                strict_high_fraction=totals['strict_high_candidates']/n)
        cohorts[role]=dict(scenes=len(cc),physical_logs=len({c['physical_log'] for c in cc}),groups=groups,
            high_found_both=sum(c['high_witness_pattern']=='both' for c in cc),
            high_found_G16_only=sum(c['high_witness_pattern']=='G16_only' for c in cc),
            high_found_G32_only=sum(c['high_witness_pattern']=='G32_only' for c in cc),
            high_found_either=sum(c['high_witness_pattern']!='neither' for c in cc),
            high_found_neither=sum(c['high_witness_pattern']=='neither' for c in cc),
            both_no_high_but_GT_high=sum(c['both_no_high_but_GT_high'] for c in cc),
            both_no_high_but_GT_strict4=sum(c['both_no_high_but_GT_strict4'] for c in cc))
    neither=[c for c in cases if c['role']=='B1_zero' and c['high_witness_pattern']=='neither']
    report=dict(schema='world_IL70_TRAIN19_exploration_research_review_v1',status='COMPLETE_CPU_REVIEW_OF_REAL_GPU_PROBE',
        created_at=datetime.now(timezone.utc).isoformat(),
        title='IL70真实TRAIN零分：采样支持、SafetyEP信用与单轨B1差距',
        headline_zh=[
            '固定15个B1零分中，G16和G32各12个场景采到strict-high，两组并集13个；11个GT达到strict-high的场景，两组全部找到同门槛候选。',
            '两组都无>=.9高见证的2个场景，其GT均strict4但score只有.857625和.890610。因此按strict-high门槛，GT独有可解而两组找不到的场景为0；不能据此说GT对全部训练场景无用。',
            '全部38组均非canonical全0，均有非零SafetyEP rawA。15个零分场景的91条采样strict-high全部rawA>0，未见好教师被本次原始优势压成负数。',
            '候选分布仍含大量风险：零分cohort中G16有172/240个strict3未通过候选，G32有323/480个；4个高分B1对照中也有8/64和22/128个。',
            '这更直接支持当前theta下采样好轨迹未体现在单轨B1的现象；本次0Adam，不能称训练后无法吸收，也没有证明仅SDE改ODE就是因果。'],
        quality_definitions=dict(strict3_feasible='NC=DAC=TTC=1',strict4='strict3 AND Comfort=1',
            strict_high='strict4 AND canonical score>=.9',canonical_all_zero='all G original canonical scores exactly 0',
            PG0='all G saved native GPU SafetyEP rawA exactly 0; not an actual network-gradient measurement',
            credit_normalization='not applied: no fabricated world8/std from this targeted probe'),
        cohorts=cohorts,cases=cases,
        neither_high_cases=[dict(token=c['token'],log=c['log'],GT_score=c['GT']['score'],GT_strict4=c['GT_strict4'],
            GT_EP=c['GT']['ego_progress'],G16_best_strict4=c['groups']['16']['best_strict4_score'],
            G32_best_strict4=c['groups']['32']['best_strict4_score'],
            G16_best_feasible=c['groups']['16']['best_feasible_score'],G32_best_feasible=c['groups']['32']['best_feasible_score']) for c in neither],
        interpretations=[
            dict(status='SUPPORTED_WITHIN_FIXED19',finding='当前IL70采样分布里已有安全高分见证，B1零分不能一律归于无好解或无优势。',
                mechanism='源W3采样候选与单轨ODE使用不同先验seed、候选数、随机步骤、BF16 batch shape；本实验没有逐项隔离。'),
            dict(status='NO_SUPPORT_IN_THIS_PANEL',finding='15个B1零分并未形成G组全零/原始优势饥饿的观察。',
                limitation='未覆盖训练全量的persistent-allfail群体；也未测参数实际PG梯度、跨场景抵消或anchor方向。'),
            dict(status='RESEARCH_DIRECTION_NOT_NEW_DEPLOYMENT',finding='优先研究如何将已评分strict-high采样见证迁移到真实B1，同时保高分。',
                limitation='本结果不授权并入S6、不新增teacher bank、不证明某种辅助损失能提分。'),
            dict(status='GT_THRESHOLD_LIMITATION',finding='本panel不存在GT>=.9独有、G16与G32都没有>=.9见证的例子。',
                limitation='2个无高见证case的GT仍可安全修零；.9门槛排除了这种较低EP成功轨迹。')],
        protocol=dict(manifest_sha256=manifest['sha256'],identity_sha256=identity['sha256'],
            actual_source=identity['source'],actual_source_commit=identity['source_commit'],
            probe_script_sha256=manifest['script_sha256'],parent_native_model_sha256=identity['parent_model_sha256'],
            baseline_export_model_sha256=manifest['baseline_export_model_sha256'],
            actual_scope_numel=identity['scope']['trainable_numel'],Adam_updates=0,
            G16_single_batch=[16,11,8,3],G32_single_batch=[32,11,8,3],probability_shape='[G,3]',
            sampler=manifest['sampling'],source_probability='clamped dimension-mean W3 proxy, not full endpoint density',
            B1_reused=manifest['reused_B1'],GT_new_queries=0,
            fresh_ratio_max=max(g['statistics']['fresh_ratio_error'] for g in all_groups),
            all_actual_sample_vs_recomputed_terms_exact=True),
        actual_budget=dict(complete_groups=38,unique_TRAIN_scenes=19,new_PDM_candidate_queries=912,hard_cap=1024,
            G16_queries=304,G32_queries=608,baseline_and_GT_new_queries=0,CPU_workers_maximum=2,
            GPU_training_updates=0,transport_returncode=0,host=transport['host'],visible_GPU=identity['visible_devices'],
            actual_job_wall_seconds=transport['end_unix']-transport['start_unix'],
            actual_model_loading_seconds=identity['model_loading_seconds'],
            recorded_forward_seconds=sum(g['seconds']['rollout'] for g in all_groups),
            recorded_scoring_wait_seconds=sum(g['seconds']['scoring_wait'] for g in all_groups)),
        limits=[
            '15 failures selected from fixed TRAIN2048 plus4 deterministic high controls; not random全TRAIN or NAVTEST evaluation.',
            'One independent private seed per G/scene; no repeated sampling or training seeds, no population uncertainty estimate.',
            'G32 costs2x G16 candidate queries and uses a different BF16 batch kernel; neither CRN nor same-compute causal comparison.',
            'At fixed IL70/Adam0, good sampled but bad B1 describes a present support/deployment gap, not a post-training absorption failure.',
            'RawA signs and PG0 do not determine actual network-gradient magnitude or source-clamp/cross-scene/anchor effects.',
            'SafetyEP can give strict-high negative relative EP credit in successful controls: this occurred here and is not unsafe positive credit.',
            'No new model score, no NAVTEST labels, no training/official recipe/run/owner changes.'],
        input_evidence=dict(job_RESULT=str(job/'RESULT.json'),actual_summary=str(result/'SUMMARY.json'),
            actual_complete=str(result/'COMPLETE.json'),identity=str(result/'identity.json'),
            manifest=str(ROOT/'MANIFEST.json'),GT_audit=str(Path('/mnt/project/qwen-world-performance-rl-20261103/analysis/VERIFIED_TRAIN_GT_TEACHER_AUDIT'))))
    atomic_json(ROOT/'RESEARCH_REVIEW.json',sealed(report))
    with (ROOT/'CASE_TABLE.csv').open('w',newline='') as stream:
        fields=['token','role','B1_score','GT_score','GT_strict4','GT_strict_high','witness_pattern',
                'G16_strict_high','G32_strict_high','G16_unsafe','G32_unsafe','G16_PG0','G32_PG0']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for c in cases:
            writer.writerow(dict(token=c['token'],role=c['role'],B1_score=c['B1']['score'],GT_score=c['GT']['score'],
                GT_strict4=c['GT_strict4'],GT_strict_high=c['GT_strict_high'],witness_pattern=c['high_witness_pattern'],
                **{f'G{g}_{suffix}':c['groups'][str(g)][key] for g in (16,32)
                   for suffix,key in [('strict_high','strict_high_candidates'),('unsafe','unsafe_candidates'),('PG0','PG0')]}))
    evidence=dict(schema='world_IL70_TRAIN19_exploration_evidence_seal_v1',status='COMPLETE',
        reviewer_script_sha256=file_sha(__file__),
        immutable_actual_outputs_unchanged=True,
        source_execution_vs_CPU_review='formal_v1 source and original probe script frozen; this file is independent CPU-only postprocessing',
        file_sha256={str(p):file_sha(p) for p in [ROOT/'RESEARCH_REVIEW.json',ROOT/'CASE_TABLE.csv',
            ROOT/'MANIFEST.json',result/'identity.json',result/'SUMMARY.json',result/'COMPLETE.json',
            result/'BUDGET_LEDGER.json',job/'RESULT.json']},
        group_file_sha256=group_evidence,array_file_sha256=array_evidence)
    atomic_json(ROOT/'EVIDENCE_SEAL.json',sealed(evidence))
    print(json.dumps(dict(status='COMPLETE',cohorts=cohorts,review=str(ROOT/'RESEARCH_REVIEW.json'),
        review_file_sha256=file_sha(ROOT/'RESEARCH_REVIEW.json'),evidence=str(ROOT/'EVIDENCE_SEAL.json'),
        evidence_file_sha256=file_sha(ROOT/'EVIDENCE_SEAL.json')),sort_keys=True))


if __name__=='__main__':main()

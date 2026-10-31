"""Finite, existing-evidence-only normalization/credit attribution readout.

No population scene reread, evaluation, training, model/weight load, or live edits.
The missing historical BATCHSTD1k/5k paired CIs are deliberately not fabricated.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import sys
import numpy as np

A = Path(__file__).resolve().parent
R = A.parent
sys.path.insert(0, str(A))
from safety_evidence_reviewer import complete_input, load_metrics, atomic, sha
from lr_study_matched_analyzer import context_equal, bootstrap_delta, failure_changes

OUT = A/'SAFETY_NORMALIZATION_EXISTING_ABLATION_20261031.json'
ARMS = ('PPO_FULL', 'PPO_BATCHSTD', 'PPO_SAFETY_EP')
UPDATES = (1000, 5000, 10000)
METRICS = ('score', 'no_at_fault_collisions', 'drivable_area_compliance',
           'time_to_collision_within_bound', 'ego_progress', 'comfort',
           'driving_direction_compliance')


def read(p):
    return json.loads(Path(p).read_text())


def fingerprint(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def differences(a, b, prefix=''):
    out = []
    for k in sorted(set(a)|set(b)):
        x, y = a.get(k), b.get(k)
        if isinstance(x, dict) and isinstance(y, dict):
            out.extend(differences(x, y, prefix+k+'.'))
        elif x != y:
            out.append(dict(field=prefix+k, FULL=x, BATCHSTD=y))
    return out


info = {(arm,u):complete_input(R,arm,u) for arm in ARMS for u in UPDATES}
points = []
native = {}
for (arm,u),v in info.items():
    if v is None:
        points.append(dict(arm=arm, updates=u, status='NO_COMPLETE_EVALUATION', not_imputed=True))
        continue
    points.append(dict(arm=arm, updates=u, status='COMPLETE_EXISTING_NAVTEST',
        PDMS=v['summary']['PDMS'], means=v['summary']['means'],
        summary=str(v['summarypath']), summary_seal=v['summary']['sha256'],
        summary_file_sha256=sha(v['summarypath']), identity_seal=v['identity']['sha256'],
        scenes=v['summary']['scenes']))
    if u == 10000:
        path=Path(v['identity']['native_RL_checkpoint'])/'identity.json'
        n=read(path)
        native[arm]=dict(path=str(path), file_sha256=sha(path), value=n)

full, batch = native['PPO_FULL']['value'], native['PPO_BATCHSTD']['value']
recipe_diffs=differences(full['recipe'],batch['recipe'])
if {x['field'] for x in recipe_diffs} != {'research.arm','research.advantage_normalization'}:
    raise ValueError('BATCHSTD is no longer the expected one-factor recipe')
equal_fields=('IL_parent','SFT_sha256','trainable_scope','parameter_contract',
              'actual_optimizer','rng_protocol','data_stream_identity',
              'reference_observation','policy_ratio','deployment','x_to_v')
checks={k:full[k]==batch[k] for k in equal_fields}
if not all(checks.values()):
    raise ValueError('Historical BATCHSTD controlled factor changed')
science_paths=(
    'starVLA/rl/qwen_world_postcot/rl_actor.py',
    'starVLA/rl/qwen_world_postcot/scope_actor.py',
    'starVLA/rl/qwen_world_postcot/rl_conditions.py',
    'starVLA/rl/simwam_recipe/collection.py',
    'starVLA/rl/simwam_recipe/source_advantages.py',
    'starVLA/rl/simwam_recipe/trainer_adapter.py',
    'starVLA/rl/simwam_recipe/vendor/action_math.py',
    'starVLA/rl/qwen_world_research/runtime.py',
    'starVLA/rl/qwen_world_research_data/data.py',
    'starVLA/model/modules/qwen_sourceflow/action_codec.py',
)
source_checks={p:dict(FULL=full['source_files'][p],BATCHSTD=batch['source_files'][p],
                     same=full['source_files'][p]==batch['source_files'][p]) for p in science_paths}
if not all(x['same'] for x in source_checks.values()):
    raise ValueError('Expected unchanged collection/actor/runtime scientific source changed')
changed_source=[dict(path=p,FULL=full['source_files'][p],BATCHSTD=batch['source_files'][p])
    for p in sorted(full['source_files'].keys() & batch['source_files'].keys())
    if full['source_files'][p]!=batch['source_files'][p]]

mechanism_path=A/'SAFETY_EP_VS_PPO_MECHANISM_20261031.json'
mechanism=read(mechanism_path)
safety_pairs={v['updates']:v for v in mechanism['NAVTEST_matched'] if v.get('status')=='COMPLETE'}
portfolio_path=A/'PORTFOLIO_10K_PAIRED.json'
portfolio=read(portfolio_path)
contrasts={}

def point_contrast(target, baseline, u):
    a,b=info[(target,u)],info[(baseline,u)]
    if a is None or b is None:
        return dict(status='MISSING_COMPLETE_RESULT_NOT_IMPUTED',updates=u,target=target,baseline=baseline)
    for k in ('tokens','token_sha256','token_logs','data_sha256','population_sha256','seed','numerics','inference','parent_model_sha256'):
        if a['identity'][k]!=b['identity'][k]:
            raise ValueError('Historical matched evaluation mismatch '+k)
    if a['identity']['protocol']['sha256']!=b['identity']['protocol']['sha256']:
        raise ValueError('Changed canonical protocol')
    return dict(status='COMPLETE_POINT_ESTIMATE_PAIRED_CI_UNAVAILABLE',updates=u,
        target=a['label'],baseline=b['label'],target_PDMS=a['summary']['PDMS'],baseline_PDMS=b['summary']['PDMS'],
        delta_PDMS=a['summary']['PDMS']-b['summary']['PDMS'],CI95_PDMS=None,
        metrics={k:dict(delta_pp=100*(a['summary']['means'][k]-b['summary']['means'][k]),CI95_pp=None) for k in METRICS},
        missing='No preexisting exact pair bootstrap/compact BATCHSTD cache for this point; raw scene populations deliberately not reread.',
        summary_complete=True,paired_token_log_protocol_identity_matched=True,
        provenance=dict(target_summary=str(a['summarypath']),baseline_summary=str(b['summarypath']),
                        target_seal=a['summary']['sha256'],baseline_seal=b['summary']['sha256']))

for u in UPDATES:
    for t,b in [('PPO_BATCHSTD','PPO_FULL'),('PPO_SAFETY_EP','PPO_BATCHSTD')]:
        contrasts[f'{t}_minus_{b}_u{u:06}']=point_contrast(t,b,u)
    if u in safety_pairs:
        v=copy.deepcopy(safety_pairs[u])
        a,b=info[('PPO_SAFETY_EP',u)],info[('PPO_FULL',u)]
        if v['target']!=a['label'] or v['baseline']!=b['label'] or abs(v['delta_PDMS']-(a['summary']['PDMS']-b['summary']['PDMS']))>1e-10:
            raise ValueError('Changed existing Safety pair point')
        v.update(existing_pair_source=dict(path=str(mechanism_path),sha256=sha(mechanism_path),updates=u),paired_CI_reused=True)
        contrasts[f'PPO_SAFETY_EP_minus_PPO_FULL_u{u:06}']=v

# The original10k BATCH/FULL cluster bootstrap is reused verbatim.
k='PPO_BATCHSTD_minus_PPO_FULL_u010000'
prior=portfolio['contrasts']['PPO_BATCHSTD_u010000_NAVTEST_minus_PPO_FULL_u010000_NAVTEST']
contrasts[k].update(status='COMPLETE_EXISTING_PAIRED_CI_REUSED',delta_PDMS=prior['delta_PDMS'],
    CI95_PDMS=prior['CI95_PDMS'],paired_CI_reused=True,
    existing_pair_source=dict(path=str(portfolio_path),sha256=sha(portfolio_path),key='PPO_BATCHSTD_u010000_NAVTEST_minus_PPO_FULL_u010000_NAVTEST'))
contrasts[k].pop('missing',None)

# Only one newly requested pair; all three10k compact caches already exist.
cached={}
for arm in ARMS:
    v=info[(arm,10000)]
    if not (A/'evaluation_metric_cache'/(v['label']+'.npz')).exists():
        raise ValueError('Refuse original scene population reread')
    cached[arm]=load_metrics(R,v)
a,b=cached['PPO_SAFETY_EP'],cached['PPO_BATCHSTD']
context_equal(a,b)
delta=a['values']-b['values'];mask=np.ones(len(delta),bool)
m,n=bootstrap_delta(delta,a['logs'],mask)
contrasts['PPO_SAFETY_EP_minus_PPO_BATCHSTD_u010000'].update(status='COMPLETE',
    delta_PDMS=m['score']['delta_pp'],CI95_PDMS=m['score']['CI95_pp'],metrics=m,
    zeros=failure_changes(a,b,mask),G_plus_PDMS=float(np.maximum(delta[:,0],0).mean()*100),
    G_minus_PDMS=float(-np.minimum(delta[:,0],0).mean()*100),
    paired_CI_reused=False,bootstrap_source='existing exact compact metric caches only',valid_bootstrap_draws=n,
    provenance=dict(target=a['provenance'],baseline=b['provenance']))
contrasts['PPO_SAFETY_EP_minus_PPO_BATCHSTD_u010000'].pop('missing',None)

out=dict(schema='world_SafetyEP_existing_normalization_credit_ablation_v1',
    status='COMPLETE_FINITE_EXISTING_EVIDENCE',created_at=datetime.now(timezone.utc).isoformat(),
    observed_points=points,contrasts=contrasts,
    mathematical_factors=dict(
        PPO_FULL=dict(numerator='r_gi-mean_i(r_gi)',denominator='population_std(all64 raw canonical rewards)+1e-8'),
        PPO_BATCHSTD=dict(numerator='r_gi-mean_i(r_gi)',denominator='population_std(all64 group-centered canonical rewards)+1e-8'),
        PPO_SAFETY_EP=dict(numerator='A_safe + A_EP (candidate metric-conditioned credit)',
            safety='if any feasible f: f-mean(f); else: mean(c)-c, c=mean NC/DAC/TTC deficit',
            efficiency='f*(EP-mean_feasible EP) if >=2 feasible, else0',
            denominator='population_std(all64 structured A_raw)+1e-8'),
        common='one global8rank xG8 population; finalized once/4inners; sourceAclip5; PPOclip.005; beta50physicalmeananchor',
        scaling_limit='Within same canonical batch, BATCHSTD scales all nonzero A by a positive common factor: signs/ranking unchanged before clamp5. Effective PG/anchor balance changes; Adam, Aclamp and gradclip prevent strict optimizer equivalence.'),
    controlled_recipe_evidence=dict(recipe_differences=recipe_diffs,equal_native_identity_fields=checks,
        unchanged_scientific_source_hashes=source_checks,changed_common_source_files=changed_source,
        source_review=dict(policy_change='new normalization schema guards and scale logging only; PPO ratio/clip and physical anchor equations unchanged',
            train_change='explicit finalizer/boundary/rank/checkpoint validation hooks only',
            adaptive_change='diagnostics/adaptive support; uniform_full data.py and native stream identities identical',
            checkpoint_worker_changes='recovery validation/entry routing support; not claimed byte-identical entire source'),
        actual_optimizer=full['actual_optimizer'],data_stream_identity=full['data_stream_identity'],
        native_identity_inputs={k:dict(path=v['path'],file_sha256=v['file_sha256'],seal=v['value']['sha256']) for k,v in native.items()},
        registration=dict(path=str(R/'EXPERIMENT_PPO_BATCHSTD_REGISTRATION.json'),sha256=sha(R/'EXPERIMENT_PPO_BATCHSTD_REGISTRATION.json'))),
    conclusions_zh=[
        'SafetyEP同时改变候选信用定义和归一化尺度；不能把相对FULL的全部收益归于安全gate。BATCHSTD是已有归一化单因素对照。',
        '仅换canonical centered denominator：1k较FULL+0.100676，5k−0.000616，10k+0.134370且既有44logCI[+0.037132,+0.220919]；收益随预算变化，不能称归一化恒优。1/5k现成配对CI缺失，本次不补扫场景。',
        '10k Safety较FULL+0.330404，其中BATCHSTD较FULL+0.134370；这是一条可观测的顺序对照路径，不是把总收益按比例分配给gate和std的因果分摊。Safety较BATCHSTD的独立配对区间见contrasts。',
        'Safety较BATCHSTD仍包含安全/可行EP信用改写、删除canonical Comfort差异、allfail fallback及其诱导的新尺度/行为分布；不能把该残差命名为纯gate收益。',
        '三臂无法识别信用定义×归一化的完整2×2交互：缺structured numerator/raw canonical reward std一格；不同后续behavior的rewardstd也不可跨臂直接相除。',
        '对FPO可把canonical-centered作为未来独立信用对照以检验交互，但现有连续transition-PPO结果不能证明它比Safety更值得优先，也不能推断FPO收益可加；当前FPO SafetyA配方保持不变。',
    ],
    FPO_followup=dict(recommendation='NO_EXISTING_PROOF_TO_REPLACE_CURRENT_SAFETY_CREDIT',
        conditional_candidate='After actual FPO gate and gradient/ratio evidence, optional same freshIL70/FPO objective comparison: canonical-centeredA vs structured-centeredA, allother settingsfixed.',
        avoid='Do not alter currentliveFPO, add normalizationgain toFPO, assume continuous energy-ratio and transition-ratio objectives share an interaction, or revive stoppedBATCHSTD.',
        attribution_gap='A fourth structuredA/rawrewardstd arm would isolate denominator under that numerator; not registered/recommended for immediate GPU solely for attribution.'),
    uncertainty=dict(training_seeds=[42],evaluation_seeds=[2101],scenes=12146,physical_logs=44,
        bootstrap=2000,seed=42,estimator='scene-weighted ratio-of-sums under paired44physical-log resampling',
        note='CI describes physical-log variability at fixed training/evaluation seed; exploratory unadjusted comparisons, not across-training-seed certainty.',
        missing_CIs=['BATCHSTD-FULL1k','BATCHSTD-FULL5k','Safety-BATCHSTD1k','Safety-BATCHSTD5k'],
        evaluation_missing=False,CI_missing_does_not_mean_evaluation_incomplete=True),
    provenance=dict(script=str(Path(__file__).resolve()),script_sha256=sha(__file__),
        previous_Safety_pair_review=str(mechanism_path),previous_Safety_pair_review_sha256=sha(mechanism_path),
        previous_10k_pair_review=str(portfolio_path),previous_10k_pair_review_sha256=sha(portfolio_path)),
    scope=dict(CPU_only=True,new_GPU=0,new_scores=0,new_evaluations=0,new_seeds=0,
        original_scene_populations_reread=0,weights_or_official_cache_payloads_loaded=False,
        live_source_owner_renderer_modified=False,new_registration=False,git_stage=False,git_commit=False))
out['sha256']=fingerprint(out)
atomic(OUT,out)
print(json.dumps(dict(path=str(OUT),status=out['status'],new10kpair={k:contrasts['PPO_SAFETY_EP_minus_PPO_BATCHSTD_u010000'][k] for k in ('delta_PDMS','CI95_PDMS','zeros')},missing_CI=out['uncertainty']['missing_CIs']),ensure_ascii=False))

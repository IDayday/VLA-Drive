"""Common single-output, seed-within-scene, paired log-cluster analysis.

No model inference or scoring happens here. Candidate oracle banks cannot be
loaded as deployment evaluations: duplicate scene/seed records are rejected.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import numpy as np
from starVLA.rl.qwen_sourceflow.contracts import fingerprint, file_sha
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference


SAFETY = ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')
REQUIRED_METRICS = ('score',*SAFETY,'ego_progress','comfort','driving_direction_compliance')
STRATA = ('stable_high','stable_zero','low_nonzero','other_or_unstable')


def checked_seal(value, description):
    body = dict(value)
    key = 'sha256' if 'sha256' in body else 'hash' if 'hash' in body else None
    if key is None or body.pop(key) != fingerprint(body):
        raise ValueError('Missing/changed seal: '+description)
    return value


def finite_tree(value, description):
    if isinstance(value,float) and not math.isfinite(value):
        raise ValueError('Nonfinite record field: '+description)
    if isinstance(value,dict):
        for key,item in value.items():
            finite_tree(item,description+'/'+key)
    elif isinstance(value,list):
        for index,item in enumerate(value):
            finite_tree(item,description+'/'+str(index))


def readout_family(identity):
    name = identity.get('readout',identity.get('inference'))
    if name in ('sample','q_sampled_single','source_ODE','native_shift5_ODE',
                'single_candidate_source_shift5_ODE_K10','ODE','single_ODE'):
        return 'stochastic_root_single'
    if name in ('mean','q_mean_root_single','deterministic_mean_root','mean_root_single'):
        return 'deterministic_mean_root_single'
    raise ValueError('Explicit single-output readout identity required: '+str(name))


def deployment_contract(identity, protocol):
    """Normalize two audited evaluator schemas, never a loose readout alias.

    The common sampler hash also contains training settings; by itself it does
    not establish which root RNG or deployment grid an evaluator actually used.
    These exact legacy descriptions bind the two inspected evaluator entrypoints.
    Unknown namespaces (including feedback's earlier seed97 auxiliary evaluation)
    require a separately named analysis instead of entering this paired table.
    """
    expected = dict(root_namespace='native_qwen_sourceflow_ODE',root_rng_device='cpu',
                    root_innovation_dtype='float32',steps=10,time_grid='source_shift5',
                    horizon=8,action_dim=3,base_model_sha256=protocol['base_model_sha256'])
    schema = identity.get('schema')
    descriptions = {
        'frozen_root_single_deployment_evaluation_v1':
            'original native_qwen_sourceflow_ODE CPU scene-bound namespace',
        'native_plain_merged_preservation_evaluation_v1':
            'native_qwen_sourceflow_ODE scene-bound CPU FP32 namespace',
        'three_route_partitioned_single_deployment_v1':
            'native_qwen_sourceflow_ODE scene-bound CPU FP32 namespace',
    }
    if schema not in descriptions or identity.get('root_innovations') != descriptions[schema]:
        raise ValueError('Unaudited deployment schema/root namespace; auxiliary evaluations cannot enter the primary comparison')
    if schema=='three_route_partitioned_single_deployment_v1':
        if identity.get('deployment_contract')!=expected or not isinstance(identity.get('source_files'),dict):
            raise ValueError('Partitioned evaluation lacks its explicit unchanged inference contract/source')
        if not identity.get('qualification') and len(identity.get('execution_equivalence_sha256') or '')!=64:
            raise ValueError('Formal partitioned evaluation lacks executed numerical equivalence evidence')
    if identity.get('deployment_contract',expected) != expected:
        raise ValueError('Changed deployment grid, root RNG, precision, action space or source base')
    if protocol.get('samplers',{}).get('deployment') != 'single source shift5 ODE10, original standard Gaussian root':
        raise ValueError('Common protocol does not declare the audited K10/shift5 deployment')
    if identity.get('base_model_sha256') != expected['base_model_sha256']:
        raise ValueError('Changed source base in deployment identity')
    return dict(expected,readout_family=readout_family(identity))


def validate_root_record(row, identity):
    """Independently reproduce the CPU innovation, including learned affine q.

    This checks saved root values; it performs no actor forward and no scoring.
    CPU/GPU affine operations may differ by float32 rounding, hence a small
    absolute tolerance. Mean-only is a separate deterministic readout.
    """
    import torch
    from starVLA.rl.safe_flow_improvement.contracts import seed_for
    parameters = row.get('latent_or_proposal_parameters',{})
    if not isinstance(parameters,dict) or parameters.get('single_candidate') is not True:
        raise ValueError('Missing explicit single-candidate/root provenance')
    root = np.asarray(parameters.get('root'),dtype=np.float64)
    if root.shape != (8,3) or not np.isfinite(root).all():
        raise ValueError('Missing or invalid saved deployment root')
    mean = np.asarray(parameters.get('mean',np.zeros((8,3))),dtype=np.float32)
    if mean.shape != (8,3) or not np.isfinite(mean).all():
        raise ValueError('Invalid latent policy mean')
    if readout_family(identity) == 'deterministic_mean_root_single':
        expected = mean
    else:
        generator = torch.Generator(device='cpu').manual_seed(seed_for('native_qwen_sourceflow_ODE',row['seed'],row['scene_token']))
        innovation = torch.randn((1,8,3),generator=generator,dtype=torch.float32)[0].numpy()
        if identity.get('root_policy_sha256') is not None:
            if not {'mean','log_std'}.issubset(parameters):
                raise ValueError('Learned q root lacks its actual Gaussian parameters')
            log_std = np.asarray(parameters['log_std'],dtype=np.float32)
            if log_std.shape != (8,3) or not np.isfinite(log_std).all():
                raise ValueError('Invalid learned root standard deviation')
            expected = mean+np.exp(log_std)*innovation
        else:
            expected = float(identity.get('root_scale',1.))*innovation
    if not np.allclose(root,expected,atol=3e-6,rtol=1e-6):
        raise ValueError('Saved root disagrees with native scene-bound CPU RNG/q parameters')


@dataclass
class Evaluation:
    path: str
    identity: dict
    tokens: tuple
    seeds: tuple
    logs: tuple
    metric_names: tuple
    values: np.ndarray  # [scene, inference_seed, native_metric]
    scorer_sha256: str
    sampler_sha256: str


def load_evaluation(path, protocol, *, allow_qualification=False):
    path = Path(path)
    identity = checked_seal(json.loads((path/'identity.json').read_text()),str(path/'identity.json'))
    if identity.get('protocol_sha256') != protocol['sha256'] or identity.get('base_model_sha256') != protocol['base_model_sha256']:
        raise ValueError('Evaluation does not bind the shared protocol/base')
    if not isinstance(identity.get('qualification'),bool):
        raise ValueError('Missing formal/qualification identity')
    if identity['qualification'] and not allow_qualification:
        raise ValueError('Qualification cannot enter formal policy comparison')
    split = identity['split']
    if split == 'DEV_AUDIT':
        from .protocol import ROOT
        if not (ROOT/'FINAL_CONFIGURATION_LOCK.json').exists():
            raise PermissionError('New-study audit logs remain sealed')
    seeds = tuple(identity.get('evaluation_seeds',identity.get('seeds',[])))
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(seed,int) or isinstance(seed,bool) for seed in seeds):
        raise ValueError('Fixed unique inference seed set is required')
    family = readout_family(identity)
    deployment_contract(identity,protocol)
    expected_seeds = tuple(protocol['evaluation_seeds'] if family == 'stochastic_root_single' else protocol['evaluation_seeds'][:1])
    if set(seeds) != set(expected_seeds):
        raise ValueError('Missing/changed inference seed population, or repeated deterministic outputs')
    seeds = tuple(sorted(seeds))
    if split not in protocol['splits']:
        raise ValueError('Unknown evaluation split')
    population = set(protocol['splits'][split])
    rows,logs,metric_names = {},{},None
    scorer,sampler = None,None
    for file in sorted((path/'scenes').glob('*.json')):
        row = checked_seal(json.loads(file.read_text()),str(file))
        finite_tree(row,str(file))
        if row.get('status') != 'OK' or row.get('evaluation_identity') != identity['sha256']:
            raise ValueError('Failed or foreign scene cannot vanish from denominator: '+str(file))
        token,seed = row['scene_token'],row['seed']
        key = (token,seed)
        if key in rows:
            raise ValueError('Duplicate scene/seed; candidate pools are not single deployment outputs')
        if token not in population or seed not in seeds:
            raise ValueError('Unexpected scene/seed in evaluation')
        validate_root_record(row,identity)
        if token in logs and logs[token] != row['log_id']:
            raise ValueError('Log identity changes across seeds')
        logs[token] = row['log_id']
        values = row['native_subscores']
        if not set(REQUIRED_METRICS).issubset(values):
            raise ValueError('Missing native canonical submetrics')
        names = tuple(sorted(values))
        if metric_names is None:
            metric_names = names
        if names != metric_names or any(not isinstance(values[name],(int,float)) or not 0 <= values[name] <= 1 for name in names):
            raise ValueError('Mixed metric versions, nonnumeric or out-of-range native scores')
        if not isinstance(row['raw_reward'],(int,float)) or abs(row['raw_reward']-values['score']) > 1e-9:
            raise ValueError('Training/readout scalar is not the canonical raw score')
        normalized = np.asarray(row['normalized_action'],dtype=np.float64)
        planned = np.asarray(row['planned_trajectory'],dtype=np.float64)
        simulated = np.asarray(row['simulated_trajectory'],dtype=np.float64)
        if normalized.shape != (8,3) or planned.shape != (8,3) or simulated.ndim != 2 or simulated.shape[0] < 2 or simulated.shape[1] < 3:
            raise ValueError('Invalid request or simulated trajectory contract')
        if not np.isfinite(planned).all() or not np.isfinite(simulated).all():
            raise ValueError('Nonfinite physical trajectory')
        feedback = row['feedback']
        if (feedback.get('status') != 'OK' or feedback.get('token') != token or feedback.get('log') != row['log_id'] or
            feedback.get('protocol_sha256') != row['scorer_config_hash']):
            raise ValueError('Canonical feedback does not match scene/log/scorer')
        if feedback['metrics'] != values:
            raise ValueError('Stored native metrics differ from actual feedback')
        if not np.allclose(np.asarray(feedback['requested_trajectory']),planned,atol=1e-7,rtol=0):
            raise ValueError('Scored request differs from deployed request')
        if not np.array_equal(np.asarray(feedback['simulated_states']),simulated):
            raise ValueError('Stored consequences differ from canonical simulator')
        if row['scorer_config_hash'] != protocol['scorer']['sha256']:
            raise ValueError('Wrong scoring protocol')
        if 'samplers' in protocol and row['sampler_config_hash'] != fingerprint(protocol['samplers']):
            raise ValueError('Sampling configuration does not bind the common protocol')
        if scorer is None:
            scorer,sampler = row['scorer_config_hash'],row['sampler_config_hash']
        if row['scorer_config_hash'] != scorer or row['sampler_config_hash'] != sampler:
            raise ValueError('Mixed scoring or sampling contract within evaluation')
        rows[key] = [float(values[name]) for name in metric_names]
    tokens = tuple(sorted(logs))
    expected_count = int(identity['scenes'])
    if expected_count <= 0 or metric_names is None or len(tokens) != expected_count or len(rows) != expected_count*len(seeds):
        raise ValueError('Incomplete scene/seed population; missing outputs remain missing')
    valid_token_orders = [protocol['splits'][split]]
    if split in protocol.get('diagnostic',{}):
        valid_token_orders.append(protocol['diagnostic'][split])
    if identity['qualification']:
        valid_token_orders += [order[:expected_count] for order in valid_token_orders]
    matches = [order for order in valid_token_orders if len(order) == expected_count and set(order) == set(tokens)
               and fingerprint(order) == identity['tokens_sha256']]
    if not matches:
        raise ValueError('Wrong/unregistered scene membership or ordering identity')
    for token in tokens:
        if any((token,seed) not in rows for seed in seeds):
            raise ValueError('A scene is missing an inference seed')
    value = np.asarray([[rows[(token,seed)] for seed in seeds] for token in tokens],dtype=np.float64)
    complete = json.loads((path/'COMPLETE.json').read_text())
    if complete.get('status') != 'COMPLETE' or complete.get('identity_sha256') != identity['sha256']:
        raise ValueError('Full evaluation publication is missing or belongs to another identity')
    if (complete.get('scenes') != len(tokens) or not math.isfinite(float(complete['PDMS'])) or
        abs(float(complete['PDMS'])-100*value[:,:,metric_names.index('score')].mean()) > 1e-7):
        raise ValueError('Published aggregate disagrees with actual single outputs')
    if ('logical_single_output_evaluations' in complete and complete['logical_single_output_evaluations'] != len(rows)):
        raise ValueError('Published logical denominator differs from actual records')
    return Evaluation(str(path.resolve()),identity,tokens,seeds,tuple(logs[token] for token in tokens),metric_names,value,scorer,sampler)


def bootstrap(values, logs, *, repeats, seed, kind='rate'):
    result = log_bootstrap_difference(values,logs,repeats=repeats,seed=seed)
    result['quantity'] = 'raw_PDMS_score_difference' if kind == 'score' else 'event_rate_or_rate_difference'
    suffix = 'PDMS_points' if kind == 'score' else 'percentage_points'
    result['CI95_'+suffix] = [100*value for value in result['CI95']]
    result['mean_'+suffix] = 100*result['mean']
    result['inference_seed_uncertainty_included'] = False
    result['limited_log_resolution'] = len(set(logs)) < 8
    result['one_log_interval_is_degenerate'] = len(set(logs)) < 2
    return result


def event_summary(per_scene_event_rate, logs, *, repeats, seed):
    values = np.asarray(per_scene_event_rate,dtype=np.float64)
    result = bootstrap(values,logs,repeats=repeats,seed=seed)
    observed = bool(np.any(values > 0))
    result.update(observed_any_event=observed,rate=float(values.mean()),rate_percent=100*float(values.mean()),
                  zero_observed_event_does_not_prove_zero_true_probability=not observed,
                  interpretation=('No event observed in these finite scenes/seeds; a zero-width ordinary bootstrap is not a safety guarantee'
                                  if not observed else 'Finite-sample scene-weighted event rate'))
    return result


def zero_summary(scores):
    exact = scores == 0
    return dict(any_seed_zero_scenes=int(exact.any(1).sum()),all_fixed_seeds_zero_scenes=int(exact.all(1).sum()),
                scene_mean_exact_zero_scenes=int((scores.mean(1) == 0).sum()),
                scene_mean_near_zero_le1e6_scenes=int((scores.mean(1) <= 1e-6).sum()),
                single_draw_zero_rate=float(exact.mean()),
                semantics='Different events explicitly named; equal counts do not make their definitions interchangeable')


def analyze_pair(current_path, reference_path, protocol, reference_strata=None, *, repeats=2000, bootstrap_seed=42):
    current = load_evaluation(current_path,protocol)
    reference = load_evaluation(reference_path,protocol)
    for key in ['tokens','seeds','logs','metric_names','scorer_sha256','sampler_sha256']:
        if getattr(current,key) != getattr(reference,key):
            raise ValueError('Unmatched paired deployment population/configuration: '+key)
    if current.identity['split'] != reference.identity['split'] or readout_family(current.identity) != readout_family(reference.identity):
        raise ValueError('Match split/readout; deterministic mean cannot silently replace sampled deployment')
    current_contract = deployment_contract(current.identity,protocol)
    if current_contract != deployment_contract(reference.identity,protocol):
        raise ValueError('Readout aliases do not share the same root RNG/grid/source base')
    metrics = current.metric_names
    cm,rm = current.values.mean(1),reference.values.mean(1)
    score_index = metrics.index('score')
    cs,rs = current.values[:,:,score_index],reference.values[:,:,score_index]
    delta = cm[:,score_index]-rm[:,score_index]
    useful = protocol['practical']['mean_gain_raw']
    tolerance = protocol['practical']['noninferiority_raw']
    ci = bootstrap(delta,current.logs,repeats=repeats,seed=bootstrap_seed,kind='score')
    new_zero = (rs > 0)&(cs == 0)
    repaired_zero = (rs == 0)&(cs > 0)
    safety_indices = [metrics.index(key) for key in SAFETY]
    cv = current.values[:,:,safety_indices] < 1
    rv = reference.values[:,:,safety_indices] < 1
    new_joint = (~rv.any(-1))&cv.any(-1)
    repair_joint = rv.any(-1)&(~cv.any(-1))
    result = dict(schema='three_routes_common_paired_deployment_v1',status='COMPLETE',
        evaluation=current.path,reference=reference.path,protocol_sha256=protocol['sha256'],
        evaluation_identity=current.identity['sha256'],reference_identity=reference.identity['sha256'],
        analysis_source_sha256=file_sha(__file__),scorer_config_hash=current.scorer_sha256,
        sampler_config_hash=current.sampler_sha256,
        matched_deployment_contract=current_contract,
        data_history=protocol.get('audit_history'),
        budget_comparison=dict(current_checkpoint_progress=current.identity.get('checkpoint_state'),
            reference_checkpoint_progress=reference.identity.get('checkpoint_state'),
            note='Optimizer updates, actual scorer calls and total compute are distinct budgets; use recorded run ledgers, not this comparison alone'),
        scenes=len(current.tokens),logs=len(set(current.logs)),inference_seeds=list(current.seeds),
        scene_seed_records=len(current.tokens)*len(current.seeds),readout_family=readout_family(current.identity),
        current_training_seed=current.identity.get('training_seed'),reference_training_seed=reference.identity.get('training_seed'),
        training_seed_pair_mismatch=(current.identity.get('training_seed') != reference.identity.get('training_seed')
                                     and reference.identity.get('arm') not in ('L0','L0-S')),
        reward_scale='native0-1; PDMS points and percent are explicitly multiplied100',
        current_PDMS=float(100*cm[:,score_index].mean()),reference_PDMS=float(100*rm[:,score_index].mean()),
        delta_PDMS_points=float(100*delta.mean()),paired_PDMS_log_cluster_bootstrap=ci,
        uncertainty='Inference seeds first averaged within each scene; scene-weighted log resampling only; no training/inference seed uncertainty',
        G_plus_raw_score_sum=float(np.maximum(delta,0).sum()),G_minus_raw_score_sum=float(-np.minimum(delta,0).sum()),
        G_plus_PDMS_points=float(100*np.maximum(delta,0).mean()),G_minus_PDMS_points=float(-100*np.minimum(delta,0).mean()),
        net_raw_score_sum=float(delta.sum()),any_improved_scenes=int((delta > 0).sum()),any_degraded_scenes=int((delta < 0).sum()),
        practical_improved_scenes=int((delta >= useful).sum()),practical_degraded_scenes=int((delta <= -useful).sum()),
        current_mean_percent={name:float(100*cm[:,index].mean()) for index,name in enumerate(metrics)},
        reference_mean_percent={name:float(100*rm[:,index].mean()) for index,name in enumerate(metrics)},
        native_mean_delta_points={name:float(100*(cm[:,index]-rm[:,index]).mean()) for index,name in enumerate(metrics)},
        current_zero=zero_summary(cs),reference_zero=zero_summary(rs),
        new_zero=dict(scene_any_seed_count=int(new_zero.any(1).sum()),paired_seed_event_count=int(new_zero.sum()),
                      scene_all_fixed_seeds_count=int(new_zero.all(1).sum()),
                      per_draw_rate=event_summary(new_zero.mean(1),current.logs,repeats=repeats,seed=bootstrap_seed)),
        repaired_zero=dict(scene_any_seed_count=int(repaired_zero.any(1).sum()),paired_seed_event_count=int(repaired_zero.sum())),
        new_joint_violation=dict(scene_any_seed_count=int(new_joint.any(1).sum()),paired_seed_event_count=int(new_joint.sum()),
                      per_draw_rate=event_summary(new_joint.mean(1),current.logs,repeats=repeats,seed=bootstrap_seed)),
        repaired_joint_violation=dict(scene_any_seed_count=int(repair_joint.any(1).sum()),paired_seed_event_count=int(repair_joint.sum())),
        current_score_tail=dict(zip(['p01','p05','p10','p50'],np.quantile(cm[:,score_index],[.01,.05,.1,.5]).tolist())),
        reference_score_tail=dict(zip(['p01','p05','p10','p50'],np.quantile(rm[:,score_index],[.01,.05,.1,.5]).tolist())),
        safety={},practical=dict(protocol['practical'],mean_gain_reaches_scale=bool(delta.mean() >= useful),
            CI_excludes_zero_on_positive_side=ci['CI95'][0] > 0,
            bootstrap_noninferiority_criterion_met=ci['CI95'][0] >= -tolerance,
            research_noninferiority_conclusion='Not automatic: assess log/seed resolution, rare-event evidence and matching controls',
            no_difference_is_not_equivalence=True),fixed_reference_strata=None)
    for index,name in enumerate(SAFETY):
        new = (~rv[:,:,index])&cv[:,:,index]
        repair = rv[:,:,index]&(~cv[:,:,index])
        rate_delta = cv[:,:,index].mean(1)-rv[:,:,index].mean(1)
        result['safety'][name] = dict(current_violation_rate=float(cv[:,:,index].mean()),
            reference_violation_rate=float(rv[:,:,index].mean()),violation_delta_percentage_points=float(100*rate_delta.mean()),
            new_scene_any_seed_count=int(new.any(1).sum()),new_paired_seed_event_count=int(new.sum()),
            recovered_scene_any_seed_count=int(repair.any(1).sum()),recovered_paired_seed_event_count=int(repair.sum()),
            new_per_draw_rate=event_summary(new.mean(1),current.logs,repeats=repeats,seed=bootstrap_seed),
            violation_delta_log_cluster_bootstrap=bootstrap(rate_delta,current.logs,repeats=repeats,seed=bootstrap_seed))
    strata_rows = None
    if reference_strata is not None:
        strata_value = json.loads(Path(reference_strata).read_text()) if isinstance(reference_strata,(str,Path)) else reference_strata
        if 'sha256' in strata_value:
            checked_seal(strata_value,'independent fixed reference strata')
        if strata_value['protocol_sha256'] != protocol['sha256']:
            raise ValueError('Reference strata changed common identity')
        strata_rows = strata_value['by_split'][current.identity['split']]
        if any(token not in strata_rows or strata_rows[token]['stratum'] not in STRATA for token in current.tokens):
            raise ValueError('Fixed reference strata missing scenes; never reclassify after learning')
        if any(not isinstance(strata_rows[token]['finite_reference_joint_safe'],bool) or
               not isinstance(strata_rows[token]['reference_mean'],(int,float)) or
               not 0 <= strata_rows[token]['reference_mean'] <= 1 for token in current.tokens):
            raise ValueError('Invalid independent reference strata values')
        result['reference_strata_content_sha256'] = fingerprint(strata_value)
        fixed = {}
        for label in STRATA:
            mask = np.asarray([strata_rows[token]['stratum'] == label for token in current.tokens])
            n = int(mask.sum())
            if not n:
                fixed[label] = dict(scenes=0,evidence='No registered reference scenes in this stratum')
                continue
            ref_mean = np.asarray([strata_rows[token]['reference_mean'] for token in current.tokens])[mask]
            fixed[label] = dict(scenes=n,current_PDMS=float(100*cm[mask,score_index].mean()),
                control_PDMS=float(100*rm[mask,score_index].mean()),paired_delta_PDMS_points=float(100*delta[mask].mean()),
                current_any_seed_zero_scenes=int((cs[mask] == 0).any(1).sum()),
                current_all_seed_zero_scenes=int((cs[mask] == 0).all(1).sum()),
                current_mean_zero_scenes=int((cm[mask,score_index] == 0).sum()),
                current_mean_drop_ge_useful_from_independent_reference=int(((cm[mask,score_index]-ref_mean) <= -useful).sum()),
                paired_control_to_current_drop_ge_useful=int((delta[mask] <= -useful).sum()),
                current_mean_improved_ge_useful_from_independent_reference=int(((cm[mask,score_index]-ref_mean) >= useful).sum()),
                reference_noise_comparison='Stratum/reference_mean are frozen independent reference seeds, not the current evaluation noise',
                paired_bootstrap=bootstrap(delta[mask],np.asarray(current.logs)[mask],repeats=repeats,seed=bootstrap_seed,kind='score'))
        safe = np.asarray([strata_rows[token]['finite_reference_joint_safe'] for token in current.tokens],dtype=bool)
        fixed['finite_reference_joint_safe'] = dict(scenes=int(safe.sum()),
             current_any_seed_joint_violation_scenes=int(cv[safe].any(-1).any(1).sum()),
             paired_new_joint_violation_scene_count=int(new_joint[safe].any(1).sum()),
             caveat='Safe in a finite independent reference sample, not all noise or true-road safety')
        result['fixed_reference_strata'] = fixed
    else:
        result['fixed_reference_strata'] = dict(status='NOT_AVAILABLE',
            interpretation='Do not rename current evaluation-high scenes as a preregistered stable-high population')
    scene_rows = []
    for index,token in enumerate(current.tokens):
        scene_rows.append(dict(scene_token=token,log_id=current.logs[index],inference_seed_count=len(current.seeds),
            reference_mean=float(rm[index,score_index]),current_mean=float(cm[index,score_index]),
            delta_raw=float(delta[index]),new_zero_seed_events=int(new_zero[index].sum()),
            recovered_zero_seed_events=int(repaired_zero[index].sum()),new_joint_violation_seed_events=int(new_joint[index].sum()),
            stratum=strata_rows[token]['stratum'] if strata_rows is not None else None))
    result['sha256'] = fingerprint(result)
    return result,scene_rows


def cpu_self_test():
    """Arithmetic fixtures only; no model/scorer result or research efficacy."""
    import tempfile
    import torch
    from starVLA.rl.safe_flow_improvement.contracts import seed_for
    def make(root,protocol,token_scores,metrics_override=None,qualification=False,readout='sample',identity_extra=None):
        tokens = list(token_scores)
        identity = dict(protocol_sha256=protocol['sha256'],base_model_sha256=protocol['base_model_sha256'],
            split='DEV_SELECT',tokens_sha256=fingerprint(tokens),scenes=len(tokens),
            evaluation_seeds=[1,2,3,4,5] if readout=='sample' else [1],
            readout=readout,qualification=qualification,training_seed=42,
            schema='frozen_root_single_deployment_evaluation_v1',
            root_innovations='original native_qwen_sourceflow_ODE CPU scene-bound namespace')
        identity.update(identity_extra or {})
        identity['sha256'] = fingerprint(identity)
        root.mkdir();(root/'scenes').mkdir()
        (root/'identity.json').write_text(json.dumps(identity))
        for i,(token,scores) in enumerate(token_scores.items()):
            for seed,score in zip(identity['evaluation_seeds'],scores):
                values = dict(score=score,no_at_fault_collisions=1.,drivable_area_compliance=1.,
                    time_to_collision_within_bound=1.,ego_progress=score,comfort=1.,driving_direction_compliance=1.)
                if metrics_override:
                    values.update(metrics_override(token,seed))
                planned = np.zeros((8,3)).tolist();simulated=np.zeros((41,11)).tolist()
                generator=torch.Generator(device='cpu').manual_seed(seed_for('native_qwen_sourceflow_ODE',seed,token))
                root_value=(torch.randn((1,8,3),generator=generator)[0] if readout_family(identity)=='stochastic_root_single'
                            else torch.zeros(8,3)).tolist()
                log = 'log_a' if i < 3 else 'log_b'
                row = dict(scene_token=token,seed=seed,log_id=log,status='OK',raw_reward=score,native_subscores=values,
                    evaluation_identity=identity['sha256'],scorer_config_hash='cpu_mock_scorer',sampler_config_hash=fingerprint(protocol['samplers']),
                    normalized_action=planned,planned_trajectory=planned,simulated_trajectory=simulated,
                    latent_or_proposal_parameters=dict(root=root_value,single_candidate=True),
                    feedback=dict(status='OK',token=token,log=log,protocol_sha256='cpu_mock_scorer',metrics=values,
                                  requested_trajectory=planned,simulated_states=simulated))
                row['sha256']=fingerprint(row)
                (root/'scenes'/f'{token}_seed{seed}.json').write_text(json.dumps(row))
        pdms=100*float(np.mean([np.mean(value) for value in token_scores.values()]))
        (root/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',identity_sha256=identity['sha256'],scenes=len(tokens),PDMS=pdms)))
    checks=[]
    tokens=['A','B','C','D']
    protocol=dict(base_model_sha256='CPU_MOCK',evaluation_seeds=[1,2,3,4,5],splits=dict(DEV_SELECT=tokens),
                  diagnostic={},scorer=dict(sha256='cpu_mock_scorer'),
                  samplers=dict(deployment='single source shift5 ODE10, original standard Gaussian root'),
                  practical=dict(mean_gain_raw=.005,noninferiority_raw=.002,new_failure_rate_absolute=.002))
    protocol['sha256']=fingerprint(protocol)
    with tempfile.TemporaryDirectory() as folder:
        folder=Path(folder)
        baseline={token:[.4]*5 for token in tokens}
        changed={token:([.5,.55,.6,.65,.7] if token!='D' else [.1,.15,.2,.25,.3]) for token in tokens}
        make(folder/'base',protocol,baseline);make(folder/'current',protocol,changed)
        result,paired=analyze_pair(folder/'current',folder/'base',protocol,repeats=200)
        assert abs(result['delta_PDMS_points']-10)<1e-12
        assert result['scenes']==4 and result['logs']==2 and result['scene_seed_records']==20
        assert abs(result['paired_PDMS_log_cluster_bootstrap']['mean']-.1)<1e-12
        assert abs(sum(row['delta_raw'] for row in paired)/4-.1)<1e-12
        assert result['new_zero']['per_draw_rate']['zero_observed_event_does_not_prove_zero_true_probability']
        checks.append('Average5 seeds inside each scene; unequal3:1 log sizes retain scene weights (delta.1, not equal-log0)')
        checks.append('Zero observed new failures are labelled finite-sample non-observation, not true probability0')
        zero={token:[0,.4,.4,.4,.4] if token=='A' else [0]*5 if token=='B' else [.4]*5 for token in tokens}
        make(folder/'zero',protocol,zero)
        value,_=analyze_pair(folder/'zero',folder/'base',protocol,repeats=100)
        assert value['current_zero']['any_seed_zero_scenes']==2
        assert value['current_zero']['all_fixed_seeds_zero_scenes']==1
        assert value['current_zero']['scene_mean_exact_zero_scenes']==1
        checks.append('Any-seed zero, all-seed zero and scene-mean zero retain distinct definitions')
        make(folder/'halfNC',protocol,changed,metrics_override=lambda token,seed:
             dict(no_at_fault_collisions=.5) if token=='A' and seed==1 else {})
        half,_=analyze_pair(folder/'halfNC',folder/'base',protocol,repeats=100)
        assert half['safety'][SAFETY[0]]['new_paired_seed_event_count']==1
        assert abs(half['safety'][SAFETY[0]]['current_violation_rate']-.05)<1e-12
        checks.append('NC=.5 counts as a violation; metric mean and binary event rate stay distinct')
        strata=dict(protocol_sha256=protocol['sha256'],by_split=dict(DEV_SELECT={
            token:dict(stratum=STRATA[index],finite_reference_joint_safe=index in (0,2),
                       reference_mean=[.95,0.,.4,.5][index]) for index,token in enumerate(tokens)}))
        fixed,_=analyze_pair(folder/'current',folder/'base',protocol,strata,repeats=100)
        assert all(fixed['fixed_reference_strata'][label]['scenes']==1 for label in STRATA)
        checks.append('Reference strata remain fixed even when current/control evaluation scores suggest a different class')
        make(folder/'plain_alias',protocol,changed,identity_extra=dict(
            schema='native_plain_merged_preservation_evaluation_v1',
            root_innovations='native_qwen_sourceflow_ODE scene-bound CPU FP32 namespace',
            readout='single_candidate_source_shift5_ODE_K10'))
        alias,_=analyze_pair(folder/'plain_alias',folder/'base',protocol,repeats=100)
        assert abs(alias['delta_PDMS_points']-10)<1e-12
        checks.append('Latent and merged-adapter readout aliases compare only through the audited shared root RNG/K10/shift5/source base')
        wrong_namespace=dict(json.loads((folder/'plain_alias/identity.json').read_text()),
                             root_innovations='feedback_student_eval97')
        try:deployment_contract(wrong_namespace,protocol)
        except ValueError:pass
        else:raise AssertionError('Auxiliary seed97 namespace accepted as primary evaluation')
        changed_grid=dict(json.loads((folder/'plain_alias/identity.json').read_text()))
        changed_grid['deployment_contract']=deployment_contract(changed_grid,protocol)
        changed_grid['deployment_contract'].pop('readout_family')
        changed_grid['deployment_contract']['time_grid']='uniform10'
        try:deployment_contract(changed_grid,protocol)
        except ValueError:pass
        else:raise AssertionError('Grid changed behind a readout alias')
        checks.append('Different root namespace or deployment grid is rejected even with the same readout alias')
        altered=json.loads((folder/'plain_alias/scenes/A_seed1.json').read_text())
        altered['latent_or_proposal_parameters']['root'][0][0]+=.01
        try:validate_root_record(altered,json.loads((folder/'plain_alias/identity.json').read_text()))
        except ValueError:pass
        else:raise AssertionError('Saved root with different RNG was accepted')
        checks.append('Saved root independently reproduces the native CPU scene-bound innovation; changed roots are rejected')
        make(folder/'mean',protocol,baseline,readout='mean')
        try:analyze_pair(folder/'mean',folder/'base',protocol,repeats=100)
        except ValueError:pass
        else:raise AssertionError('Mean-root and5seed random policy mixed')
        checks.append('Deterministic mean-root cannot silently replace the5seed sampled readout')
        make(folder/'qualification',protocol,changed,qualification=True)
        try:load_evaluation(folder/'qualification',protocol)
        except ValueError:pass
        else:raise AssertionError('Qualification accepted as formal')
        # Duplicate records cannot masquerade as additional deployment seeds.
        one=folder/'current/scenes/A_seed1.json'
        (folder/'current/scenes/duplicate.json').write_bytes(one.read_bytes())
        try:load_evaluation(folder/'current',protocol)
        except ValueError:pass
        else:raise AssertionError('Oracle candidate pool accepted')
        checks.append('Qualification and duplicate scene/seed pools are rejected')
        (folder/'current/scenes/duplicate.json').unlink()
        (folder/'current/scenes/A_seed5.json').unlink()
        try:load_evaluation(folder/'current',protocol)
        except ValueError:pass
        else:raise AssertionError('Missing seed disappeared from denominator')
        checks.append('Missing seed record invalidates the full evaluation denominator')
        invalid={token:([float('nan'),.4,.4,.4,.4] if token=='A' else [.4]*5) for token in tokens}
        make(folder/'invalid',protocol,invalid)
        try:load_evaluation(folder/'invalid',protocol)
        except ValueError:pass
        else:raise AssertionError('Nonfinite score was accepted or dropped')
        checks.append('Nonfinite feedback invalidates evaluation instead of becoming zero or disappearing')
    return dict(status='PASS',qualification_only=True,checks=checks,driving_efficacy=False)

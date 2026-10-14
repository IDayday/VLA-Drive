"""Read-only direct specialist learning evidence on its fixed training support."""
from pathlib import Path
import numpy as np
import torch
from iqe.contracts import require
from iqe.io import read_json, atomic_json, file_hash
from iqe.losses import expert_il_terms
from .binding import scenes_by_role
from .models import make_specialist
from .data import ReusedContext, verified_base_scores
from .training import candidate_labels, matrix_manifest
from .experiments import endpoint, load_endpoint
from .metrics import metrics
from iqe.internal_baseline.bank import COMPONENTS


def scorer_fit(config, cell, device='cuda:0'):
    """Read the fixed D_S after that cell's endpoint is closed, with no updates."""
    from .bank import load_bank
    from .scorer_training import selected_scorer, predict
    from .objectives import pair_population
    from .metrics import report_metrics, select
    from iqe.io import lock
    from .resource_guard import check
    check(config, 'diagnose-completed-scorer-fit')
    folder=config.root/'scorers'/cell
    result=read_json(folder/'result.json')
    require(result['status']=='COMPLETE' and result['optimizer_steps']==6000,
            'fit diagnosis needs a completed fixed-budget cell')
    meta,bank=load_bank(config,'deploy','scorer_fit')
    bindings={'cell':cell,'Base_binding_hash':file_hash(config.root/'BASE_BINDING.json'),
        'bank_hash':meta['hash'],'checkpoint_hash':file_hash(folder/'best.pt')}
    require(bindings['checkpoint_hash']==result['best_checkpoint_hash'],'fit endpoint changed')
    output=folder/'FULL_FIT_DIAGNOSTIC.json'
    with lock(folder/'FIT_DIAGNOSTIC.lock'):
        if output.exists():
            previous=read_json(output)
            require(previous['bindings']==bindings,'fit diagnosis dependencies changed')
            return previous
        model=selected_scorer(config,cell,device)
        _,_,pred=predict(config,model,'scorer_fit',device)
        stats=report_metrics(bank['scores'],bank['components'],
            select(pred,bank['valid'],expert_ids=meta['provenance']['sources']),meta['source_logs'],
            seed=config.bootstrap_seed,repetitions=config.bootstrap_repetitions)
        pair,weight=pair_population(bank['scores'],bank['comparison_valid'])
        pair_fit={'valid_pairs':len(pair),'gap_weighted_accuracy':None}
        if len(pair):
            x,i,j=pair.T
            correct=(pred[x,i]-pred[x,j])*(bank['scores'][x,i]-bank['scores'][x,j])>0
            pair_fit['gap_weighted_accuracy']=float(np.dot(weight,correct)/weight.sum())
        error=np.abs(pred-bank['scores'])
        smooth=np.where(error<.1,error**2/.2,error-.05)
        natural_value=float(((smooth*bank['valid']).sum(1)/bank['valid'].sum(1)).mean())
        report={'bindings':bindings,'scope':'entire_fixed_D_S','metrics':stats,'pair_fit':pair_fit,
            'natural_scene_SmoothL1_beta_0_1':natural_value,'optimizer_updates':0,
            'new_training_run':False,'used_for_checkpoint_or_recipe_selection':False,
            'external_scorer_weights_loaded':False,'stage_val_dev_cal_test_read':False}
        atomic_json(output,report,immutable=True)
        return report


@torch.no_grad()
def target_il(config, module, scenes, cache, target_kind, device):
    optimized = None
    if target_kind == 'offline_optimized':
        from .target_rules import verify_target_assets
        verify_target_assets(config)
        ids = {s.scene_id for s in scenes}
        with np.load(Path(config.optimized_root) / 'optimized_trajectories.npz') as source:
            optimized = {str(i): torch.from_numpy(t.copy()).float()
                         for i, t in zip(source['tokens'], source['trajectories']) if str(i) in ids}
        require(set(optimized) == ids, 'diagnostic optimized targets missing')
    weight = read_json(config.root / 'OFFLINE_BASE_CONTRACT.json')['query_architecture']['prev_weight']
    numerator = denominator = 0.
    for start in range(0, len(scenes), 32):
        batch = scenes[start:start + 32]
        x = cache.batch(batch, device)
        target = x['target_raw'] if optimized is None else module.trajectory_contract.raw(
            torch.stack([optimized[s.scene_id] for s in batch]).to(device))
        term = expert_il_terms(module(x), target, prev_weight=weight)['il']
        numerator += float(term.numerator)
        denominator += float(term.denominator)
    require(denominator == len(scenes), 'diagnostic IL population changed')
    return numerator / denominator


def specialist_learning(config, cell, device='cuda:0'):
    """Fixed 5000 raw/EMA direct evidence; never selects using fit outcomes."""
    output = config.root / 'experts' / cell / 'SPECIALIZATION_DIAGNOSTIC.json'
    manifest, spec = matrix_manifest(config, cell)
    points = [endpoint(config, cell, 5000, kind) for kind in ('raw', 'ema')]
    binding = {'manifest': manifest['hash'], 'Base': file_hash(config.root / 'BASE_BINDING.json'),
               'endpoints': [p['checksum'] for p in points]}
    if output.exists():
        existing = read_json(output)
        require(existing['bindings'] == binding, 'specialization diagnostic dependencies changed')
        return existing
    role = scenes_by_role(config)['expert_fit']
    lookup = {s.scene_id: s for s in role}
    require(set(manifest['scene_ids']) <= set(lookup), 'non D_E specialist diagnostic')
    scenes = [lookup[i] for i in manifest['scene_ids']]
    cache = ReusedContext(config, capacity=min(len(scenes), 1024))
    base_labels = verified_base_scores(config, scenes)
    base = make_specialist(config, False).to(device).eval().requires_grad_(False)
    base_il = target_il(config, base, scenes, cache, spec['target'], device)
    reports = []
    for point in points:
        module = load_endpoint(config, point, device)
        _, labels = candidate_labels(config, module, scenes, cell + '_' + point['endpoint'] + '_train', device, cache)
        truth = np.asarray([[base_labels[s.scene_id]['total_score_01'], label.total_score_01]
                            for s, label in zip(scenes, labels)])
        components = np.asarray([[[base_labels[s.scene_id]['named_components'][c]
            if base_labels[s.scene_id]['component_valid_masks'].get(c, False) else np.nan for c in COMPONENTS],
            [label.named_components[c] if label.component_valid_masks.get(c, False) else np.nan for c in COMPONENTS]]
            for s, label in zip(scenes, labels)])
        report = metrics(truth, components, np.ones(len(scenes), dtype=int),
                         [s.source_log_id for s in scenes], seed=config.bootstrap_seed,
                         repetitions=config.bootstrap_repetitions)
        validation = read_json(config.root / 'experts' / cell / 'validation_005000.json')['endpoints'][point['endpoint']]
        report.update(endpoint=point['endpoint'], step=5000, H_train_direct_PDMS_gain=report['selected_gain_01'],
                      H_val_direct_PDMS_gain=validation['hard_val_direct_gain'],
                      H_val_count=validation['hard_val_count'], full_stage_val_Oracle_gain=validation['oracle_gain_01'],
                      Base_eval_IL=base_il, specialist_eval_IL=target_il(config, module, scenes, cache, spec['target'], device),
                      target_source=spec['target'], dropout_mode='eval for diagnostic only; production train dropout unchanged')
        reports.append(report)
        del module
    result = {'bindings': binding, 'cell': cell, 'training_population': 'entire fixed H_train',
              'scene_ids': manifest['scene_ids'], 'reports': reports, 'selection_uses_these_training_scores': False}
    atomic_json(output, result, immutable=True)
    return result

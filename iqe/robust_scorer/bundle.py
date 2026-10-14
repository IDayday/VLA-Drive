"""Research-only export using the existing single-VLA deployment wrapper."""
from pathlib import Path
import torch
from iqe.contracts import require, TrajectoryContract
from iqe.io import atomic_torch, atomic_json, read_json, file_hash, digest
from iqe.specialist_study.bank import pool
from iqe.specialist_study.evaluation import ResearchSystem
from .model import RobustScorer
from .config import CELLS
from .evaluation import chosen_endpoint


def export(config):
    choice = read_json(config.root/'MODEL_CHOICE.json')
    source = config.source()
    cell = choice['selected']['cell']
    model, _ = chosen_endpoint(config, cell)
    selected = pool(source)
    states = []
    for expert in selected['experts']:
        require(file_hash(expert['checkpoint']) == expert['checksum'], 'expert checkpoint changed')
        states.append(torch.load(expert['checkpoint'], map_location='cpu', weights_only=True))
    payload = dict(kind='iqe_robust_scorer_research_v1', Base_binding=read_json(source.root/'BASE_BINDING.json'),
                   contract=read_json(source.root/'OFFLINE_BASE_CONTRACT.json'), pool=selected,
                   expert_states=states, scorer=model.state_dict(), scorer_spec=CELLS[cell], scorer_cell=cell,
                   initialization=read_json(config.root/'scorers'/cell/'SCORER_INITIALIZATION.json'),
                   choice_hash=file_hash(config.root/'MODEL_CHOICE.json'), margin=read_json(config.root/'SELECTOR_MARGIN.json'),
                   selection_rule_default='raw_argmax_Base_then_stable_expert_id', GT_required=False,
                   metric_cache_required=False, ACTIVE=False)
    folder = config.root/'research_bundle'
    atomic_torch(folder/'bundle.pt', payload, immutable=True)
    atomic_json(folder/'BUNDLE.json', dict(checksum=file_hash(folder/'bundle.pt'), pool_hash=selected['hash'],
                                         ACTIVE=False, Base_external_dependency=payload['Base_binding']['checkpoint']), immutable=True)
    return str(folder/'bundle.pt')


def scorer_from_payload(payload):
    require(payload['kind'] == 'iqe_robust_scorer_research_v1' and payload['scorer_spec'] == CELLS[payload['scorer_cell']],
            'unknown Scorer definition')
    require(not payload['initialization']['external_scorer_weights_loaded'] and not payload['initialization']['loaded_scorer_weights'],
            'external Scorer initializer forbidden')
    state = payload['scorer']
    model = RobustScorer(state['world_projection.1.weight'].shape[1], state['action_projection.1.weight'].shape[1],
                         TrajectoryContract(**payload['contract']['trajectory']), **payload['scorer_spec'])
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def load(path, device='cuda:0'):
    from iqe.s0_adapter import S0Adapter
    from iqe.specialist_study.models import Specialist
    from iqe.specialist_study.training import runtime
    # This version's training, cached inference and online reference all use
    # this locked backend. The public loader must restore it itself: callers
    # must not need the qualification CLI's incidental runtime setup.
    # This restores execution flags, not weights, RNG seeds or the Base recipe.
    device = runtime(device)
    path = Path(path)
    receipt = read_json(path.parent/'BUNDLE.json')
    require(file_hash(path) == receipt['checksum'], 'bundle checksum changed')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    require(not payload['GT_required'] and not payload['metric_cache_required'] and not payload['ACTIVE'], 'research-only inference boundary')
    require(file_hash(payload['Base_binding']['checkpoint']) == payload['Base_binding']['checkpoint_hash'], 'Base dependency changed')
    require(digest(payload['contract']) == payload['Base_binding']['contract_hash'], 'Base contract changed')
    require(payload['pool']['hash'] == receipt['pool_hash'] == digest({k: v for k, v in payload['pool'].items() if k != 'hash'}),
            'pool identity mismatch')
    require(len(payload['pool']['experts']) == len(payload['expert_states']), 'incomplete private expert states')
    scorer = scorer_from_payload(payload)
    adapter = S0Adapter.load_and_validate_s0(payload['contract'], device)
    experts = []
    for spec, state in zip(payload['pool']['experts'], payload['expert_states']):
        expert = Specialist(adapter.framework.action_model, private_reader=spec['private_reader'],
                            contract_hash=adapter.contract_hash,
                            planner_mode=payload['contract']['source_config']['foresight']['planner_condition_mode'])
        expert.load_state_dict(state, strict=True)
        experts.append(expert.to(device).eval().requires_grad_(False))
    return ResearchSystem(adapter, experts, scorer.to(device), payload['pool']['expert_ids']).to(device).eval().requires_grad_(False)


def main(argv=None):
    import argparse
    import json
    from .config import load as load_config
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    a = p.parse_args(argv)
    print(json.dumps({'status': 'COMPLETE', 'bundle': export(load_config(a.config)), 'ACTIVE_updated': False}))


if __name__ == '__main__':
    main()

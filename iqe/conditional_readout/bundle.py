"""Research bundle for the single candidate-conditioned readout experiment."""
import argparse
import json
from pathlib import Path
import torch
from iqe.contracts import require, TrajectoryContract
from iqe.io import atomic_json, atomic_torch, read_json, file_hash, digest
from iqe.specialist_study.bank import pool
from iqe.specialist_study.evaluation import ResearchSystem
from iqe.specialist_study.training import runtime
from .evaluation import endpoint
from .model import ConditionalReadoutScorer


def export(config):
    choice=read_json(config.root/'MODEL_CHOICE.json');source=config.source()
    model,_=endpoint(config);selected=pool(source);states=[]
    for expert in selected['experts']:
        require(file_hash(expert['checkpoint'])==expert['checksum'], 'expert checkpoint changed')
        states.append(torch.load(expert['checkpoint'],map_location='cpu',weights_only=True))
    payload=dict(kind='iqe_conditioned_readout_research_v1',Base_binding=read_json(source.root/'BASE_BINDING.json'),
        contract=read_json(source.root/'OFFLINE_BASE_CONTRACT.json'),pool=selected,expert_states=states,
        scorer=model.state_dict(),conditioned=True,source_hash=file_hash(Path(__file__).with_name('model.py')),
        initialization=read_json(config.root/'scorer/SCORER_INITIALIZATION.json'),
        choice_hash=file_hash(config.root/'MODEL_CHOICE.json'),margin=read_json(config.root/'SELECTOR_MARGIN.json'),
        selection_rule_default='raw_argmax_Base_then_stable_expert_id',GT_required=False,metric_cache_required=False,ACTIVE=False)
    folder=config.root/'research_bundle'
    atomic_torch(folder/'bundle.pt',payload,immutable=True)
    atomic_json(folder/'BUNDLE.json',dict(checksum=file_hash(folder/'bundle.pt'),pool_hash=selected['hash'],
        ACTIVE=False,Base_external_dependency=payload['Base_binding']['checkpoint']),immutable=True)
    return str(folder/'bundle.pt')


def scorer_from_payload(payload):
    require(payload['kind']=='iqe_conditioned_readout_research_v1' and payload['conditioned'] is True,
            'wrong readout definition')
    require(file_hash(Path(__file__).with_name('model.py'))==payload['source_hash'], 'readout implementation changed')
    initial=payload['initialization']
    require(not initial['external_scorer_weights_loaded'] and initial['loaded_scorer_weights']==[],
            'external Scorer initialization forbidden')
    state=payload['scorer']
    module=ConditionalReadoutScorer(state['world_projection.1.weight'].shape[1],
        state['action_projection.1.weight'].shape[1],TrajectoryContract(**payload['contract']['trajectory']),
        visual_dim=state['spatial_reader.projection.1.weight'].shape[1],
        views=len(payload['contract']['input']['views']),conditioned=True)
    module.load_state_dict(state,strict=True)
    return module.eval().requires_grad_(False)


def load(path,device='cuda:0'):
    from iqe.s0_adapter import S0Adapter
    from iqe.specialist_study.models import Specialist
    device=runtime(device);path=Path(path);receipt=read_json(path.parent/'BUNDLE.json')
    require(file_hash(path)==receipt['checksum'], 'bundle checksum changed')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    require(not payload['GT_required'] and not payload['metric_cache_required'] and not payload['ACTIVE'],
            'research inference boundary violated')
    binding=payload['Base_binding']
    require(file_hash(binding['checkpoint'])==binding['checkpoint_hash'], 'Base dependency changed')
    require(digest(payload['contract'])==binding['contract_hash'], 'Base contract changed')
    selected=payload['pool']
    require(selected['hash']==receipt['pool_hash']==digest({k:v for k,v in selected.items() if k!='hash'}), 'pool identity mismatch')
    require(len(selected['experts'])==len(payload['expert_states']), 'incomplete expert bundle')
    scorer=scorer_from_payload(payload);adapter=S0Adapter.load_and_validate_s0(payload['contract'],device)
    experts=[]
    for spec,state in zip(selected['experts'],payload['expert_states']):
        expert=Specialist(adapter.framework.action_model,private_reader=spec['private_reader'],
            contract_hash=adapter.contract_hash,planner_mode=payload['contract']['source_config']['foresight']['planner_condition_mode'])
        expert.load_state_dict(state,strict=True);experts.append(expert.to(device).eval().requires_grad_(False))
    return ResearchSystem(adapter,experts,scorer.to(device),selected['expert_ids']).to(device).eval().requires_grad_(False)


def main(argv=None):
    from .config import load as load_config
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args(argv)
    print(json.dumps(dict(status='COMPLETE',bundle=export(load_config(args.config)),ACTIVE_updated=False)))


if __name__=='__main__':main()

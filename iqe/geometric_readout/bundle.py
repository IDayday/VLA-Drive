"""Single-VLA research inference with an explicit current-camera side channel."""
import argparse
import json
from pathlib import Path
import torch
from iqe.contracts import require, TrajectoryContract
from iqe.io import atomic_torch, atomic_json, read_json, file_hash, digest, lock
from iqe.evaluation.retention import module_hash
from iqe.specialist_study.evaluation import ResearchSystem
from iqe.specialist_study.bank import pool
from iqe.specialist_study.training import runtime
from .model import GeometricReadoutScorer

CAMERAS={'camera_intrinsics','camera_to_ego','camera_image_sizes'}


def readout_contract():
    """Weights alone cannot distinguish the same-shaped global/local readers."""
    root=Path(__file__).parents[2]
    paths=('iqe/geometric_readout/model.py','iqe/geometric_readout/geometry.py',
           'iqe/internal_baseline/scorer.py','iqe/specialist_study/models.py')
    return dict(kind='candidate_local_pinhole_v1',sources={p:file_hash(root/p) for p in paths},
                camera_fields=sorted(CAMERAS),height_hypotheses_m=[0.,1.],neighborhood=[3,3])


class GeometricResearchSystem(ResearchSystem):
    @torch.no_grad()
    def shared_forward(self,observations,*,expert_count=None,score=True,calibration=None):
        count=len(self.experts) if expert_count is None else expert_count
        if not score or count==0:
            return super().shared_forward(observations,expert_count=count,score=score)
        require(isinstance(calibration,dict) and set(calibration)==CAMERAS,
                'explicit current camera calibration required; arbitrary side-channel fields forbidden')
        device=next(self.scorer.parameters()).device
        calibration={k:v.detach().to(device=device,dtype=torch.float32) for k,v in calibration.items()}
        require(all(len(v)==len(observations) for v in calibration.values()),'camera/observation batch mismatch')
        # Augment the Scorer-only boundary, never the observation input/prompt or
        # original VLA mask/positions. Base's shared forward is reused verbatim.
        def current_camera(_module,args):
            require(not set(args[0])&CAMERAS,'duplicate camera context')
            return (args[0]|calibration,*args[1:])
        handle=self.scorer.register_forward_pre_hook(current_camera,prepend=True)
        try:return super().shared_forward(observations,expert_count=count,score=True)
        finally:handle.remove()


def scorer_from_payload(payload):
    require(payload['kind']=='iqe_geometric_scorer_research_v1','unknown geometric bundle schema')
    require(payload.get('readout_contract')==readout_contract(),'geometric readout implementation changed')
    require(not payload['initialization']['external_scorer_weights_loaded'] and
            not payload['initialization']['loaded_scorer_weights'],'external Scorer weights forbidden')
    state=payload['scorer']
    model=GeometricReadoutScorer(state['world_projection.1.weight'].shape[1],
        state['action_projection.1.weight'].shape[1],TrajectoryContract(**payload['contract']['trajectory']),
        visual_dim=state['spatial_reader.projection.1.weight'].shape[1])
    model.load_state_dict(state,strict=True)
    return model.eval().requires_grad_(False)


def assemble(payload,device):
    from iqe.s0_adapter import S0Adapter
    from iqe.specialist_study.models import Specialist
    device=runtime(device)
    require(not payload['GT_required'] and not payload['metric_cache_required'] and not payload['ACTIVE'],
            'research-only inference boundary')
    require(file_hash(payload['Base_binding']['checkpoint'])==payload['Base_binding']['checkpoint_hash'],
            'Base dependency changed')
    require(digest(payload['contract'])==payload['Base_binding']['contract_hash'],'Base contract changed')
    require(payload['pool']['hash']==digest({k:v for k,v in payload['pool'].items() if k!='hash'}),
            'candidate pool identity changed')
    require(len(payload['pool']['experts'])==len(payload['expert_states']),'incomplete expert states')
    scorer=scorer_from_payload(payload)
    adapter=S0Adapter.load_and_validate_s0(payload['contract'],device)
    experts=[]
    for spec,state in zip(payload['pool']['experts'],payload['expert_states']):
        expert=Specialist(adapter.framework.action_model,private_reader=spec['private_reader'],
            contract_hash=adapter.contract_hash,planner_mode=payload['contract']['source_config']['foresight']['planner_condition_mode'])
        expert.load_state_dict(state,strict=True);experts.append(expert.to(device).eval().requires_grad_(False))
    return GeometricResearchSystem(adapter,experts,scorer.to(device),payload['pool']['expert_ids']).to(device).eval().requires_grad_(False)


def payload_for(config,model,initial):
    source=config.source();selected=pool(source);states=[]
    for expert in selected['experts']:
        require(file_hash(expert['checkpoint'])==expert['checksum'],'expert changed')
        states.append(torch.load(expert['checkpoint'],map_location='cpu',weights_only=True))
    return dict(kind='iqe_geometric_scorer_research_v1',readout_contract=readout_contract(),
        Base_binding=read_json(source.root/'BASE_BINDING.json'),
        contract=read_json(source.root/'OFFLINE_BASE_CONTRACT.json'),pool=selected,expert_states=states,
        scorer=model.state_dict(),initialization=initial,geometry='current NAVSIM pinhole, z0/1m, 3x3',
        current_camera_input_required=True,selection_rule_default='raw_argmax_Base_then_stable_expert_id',
        GT_required=False,metric_cache_required=False,ACTIVE=False)


def export(config):
    from .evaluation import selected
    from iqe.specialist_study.qualifications import equal_tree
    model,_=selected(config)
    choice=read_json(config.root/'MODEL_CHOICE.json')
    require(module_hash(model)==choice['state_hash'] and
            file_hash(config.root/'scorer/best.pt')==choice['checkpoint_hash'],'selected endpoint changed')
    payload=payload_for(config,model,read_json(config.root/'scorer/SCORER_INITIALIZATION.json'))
    payload.update(choice_hash=file_hash(config.root/'MODEL_CHOICE.json'),
        margin=read_json(config.root/'SELECTOR_MARGIN.json'),diagnostic=False)
    folder=config.root/'research_bundle'
    identity=dict(choice_hash=payload['choice_hash'],margin_hash=file_hash(config.root/'SELECTOR_MARGIN.json'),
        readout_contract=payload['readout_contract'],pool_hash=payload['pool']['hash'],
        scorer_state_hash=choice['state_hash'])
    with lock(folder/'EXPORT.lock'):
        path=folder/'bundle.pt';receipt=folder/'BUNDLE.json'
        if receipt.exists():
            saved=read_json(receipt)
            require(saved.get('identity')==identity and file_hash(path)==saved['checksum'],'stale research bundle')
            return str(path)
        if path.exists():
            # Recover a crash after the atomic payload but before its receipt.
            require(equal_tree(torch.load(path,map_location='cpu',weights_only=False),payload),
                    'uncommitted research bundle belongs to another endpoint')
        else:atomic_torch(path,payload,immutable=True)
        atomic_json(receipt,dict(checksum=file_hash(path),pool_hash=payload['pool']['hash'],identity=identity,
            ACTIVE=False,Base_external_dependency=payload['Base_binding']['checkpoint']),immutable=True)
    return str(folder/'bundle.pt')


def load(path,device='cuda:0'):
    device=runtime(device)
    path=Path(path);receipt=read_json(path.parent/'BUNDLE.json')
    require(file_hash(path)==receipt['checksum'],'bundle checksum mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    require(payload['pool']['hash']==receipt['pool_hash'],'bundle pool receipt mismatch')
    return assemble(payload,device)


def main():
    from .full import load_config,resource_check
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=load_config(args.config);resource_check(config)
    print(json.dumps(dict(status='COMPLETE',bundle=export(config),ACTIVE_updated=False)))


if __name__=='__main__':main()

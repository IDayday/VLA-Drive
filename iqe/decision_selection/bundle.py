"""Research bundle with one own Base, fixed own expert pool and one scratch-trained head."""
from pathlib import Path
import torch
from iqe.contracts import TrajectoryContract,require
from iqe.io import atomic_torch,atomic_json,read_json,file_hash,digest
from iqe.evaluation.retention import module_hash
from .models import SceneRouter,SetSelector
from .evaluation import final_endpoint,_final_lock
from .deployment import DecisionSystem


def export(config,kind):
    from iqe.specialist_study.bank import pool
    frozen=_final_lock(config);model,endpoint=final_endpoint(config,kind)
    source=config.source();selected=pool(source)
    require(selected['expert_ids']==['expert_0']+sorted(x['expert_id'] for x in selected['experts']), 'export canonical ID ordering')
    states={}
    for e in selected['experts']:
        require(file_hash(e['checkpoint'])==e['checksum'],'frozen expert changed')
        states[e['expert_id']]=torch.load(e['checkpoint'],map_location='cpu',weights_only=True)
    init=read_json(config.root/'refit'/kind/'INITIALIZATION.json')
    margin=read_json(config.root/'MARGIN_CALIBRATION.json')
    require(margin['final_lock_hash']==frozen['hash'],'margin belongs to different fixed models')
    payload=dict(schema='iqe_cost_sensitive_'+kind+'_research_v1',kind=kind,
        Base_binding=read_json(source.root/'BASE_BINDING.json'),contract=read_json(source.root/'OFFLINE_BASE_CONTRACT.json'),
        pool=selected,expert_states=states,decision_state=model.state_dict(),initialization=init,endpoint=endpoint,
        final_lock_hash=frozen['hash'],margin=margin['models'][kind],default_rule='raw_argmax_Base_then_stable_ID',
        inference_dependencies=['own_Base','own_private_experts','own_decision_head'],
        GT_required=False,metric_cache_required=False,privileged_current_state_required=False,ACTIVE=False,
        scene_router_executes_only_selected_branch=kind=='M_scene',decision_outputs_are_PDMS=False)
    folder=config.root/'research_bundle'/kind;path=folder/'bundle.pt'
    if path.exists():
        receipt=read_json(folder/'BUNDLE.json')
        require(receipt['endpoint']==endpoint and file_hash(path)==receipt['checksum'],'existing bundle mismatch')
        return str(path)
    atomic_torch(path,payload,immutable=True)
    atomic_json(folder/'BUNDLE.json',dict(checksum=file_hash(path),schema=payload['schema'],kind=kind,
        pool_hash=selected['hash'],endpoint=endpoint,Base_dependency=payload['Base_binding']['checkpoint'],ACTIVE=False),immutable=True)
    return str(path)


def head_from_payload(payload):
    kind=payload['kind'];require(kind in ('M_scene','M_set'),'privileged probe cannot be deployed')
    require(payload['schema']=='iqe_cost_sensitive_'+kind+'_research_v1','wrong router/set bundle schema')
    init=payload['initialization']
    require(not init['external_scorer_weights_loaded'] and not init['own_pretrained_scorer_weights_loaded'],'not a scratch decision lineage')
    state=payload['decision_state'];wd=state['context.world.1.weight'].shape[1];ad=state['context.action.1.weight'].shape[1]
    contract=TrajectoryContract(**payload['contract']['trajectory'])
    model=(SceneRouter(wd,ad,payload['pool']['expert_ids']) if kind=='M_scene' else SetSelector(wd,ad,contract))
    model.load_state_dict(state,strict=True)
    require(module_hash(model)==payload['endpoint']['state_hash'],'decision state identity mismatch')
    return model.eval().requires_grad_(False)


def load(path,device='cuda:0',adapter=None):
    from iqe.s0_adapter import S0Adapter
    from iqe.specialist_study.models import Specialist
    from iqe.specialist_study.training import runtime
    device=runtime(device);path=Path(path);receipt=read_json(path.parent/'BUNDLE.json')
    require(file_hash(path)==receipt['checksum'],'research bundle checksum mismatch')
    p=torch.load(path,map_location='cpu',weights_only=False)
    require(p['kind']==receipt['kind'] and p['schema']==receipt['schema'],'bundle schema mismatch')
    require(not any(p[k] for k in ('GT_required','metric_cache_required','privileged_current_state_required','ACTIVE')),'invalid deployment dependencies')
    require(file_hash(p['Base_binding']['checkpoint'])==p['Base_binding']['checkpoint_hash'],'Base weight dependency changed')
    require(digest(p['contract'])==p['Base_binding']['contract_hash'],'Base numerical contract mismatch')
    require(p['pool']['hash']==receipt['pool_hash']==digest({k:v for k,v in p['pool'].items() if k!='hash'}),'pool changed')
    if adapter is None:adapter=S0Adapter.load_and_validate_s0(p['contract'],device)
    else:require(adapter.contract_hash==p['Base_binding']['contract_hash'],'shared adapter contract mismatch')
    experts={}
    require(set(p['expert_states'])==set(p['pool']['expert_ids'][1:]),'incomplete fixed expert states')
    for spec in p['pool']['experts']:
        m=Specialist(adapter.framework.action_model,private_reader=spec['private_reader'],contract_hash=adapter.contract_hash,
            planner_mode=p['contract']['source_config']['foresight']['planner_condition_mode'])
        m.load_state_dict(p['expert_states'][spec['expert_id']],strict=True)
        require(module_hash(m)==spec['model_hash'],'frozen expert model hash differs')
        experts[spec['expert_id']]=m.to(device).eval().requires_grad_(False)
    return DecisionSystem(adapter,experts,head_from_payload(p).to(device),p['kind'],p['pool']['expert_ids']).to(device)

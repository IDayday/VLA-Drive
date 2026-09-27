"""Version2 local graph cache identities include selection, noise and public foundation."""
import hashlib
import json
from pathlib import Path
import torch
from .local_graph import LocalInteractionGraph,build_local_graph,gather_current

FIELDS={'schema_version','token','identity_sha256','observation_identity','native_actions','full_current','current_prediction','local_graph'}


def signature(identity):return hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def cache_identity(foundation_sha256,public_origin,selector,graph_config):
    root=Path(__file__).resolve().parents[4]
    files=['starVLA/model/modules/joint_world/'+name+'.py' for name in ('observability','local_graph','local_masks','local_cache')]
    if public_origin.get('private_driving_weights_loaded',True):raise ValueError('Formal local cache requires independently public-origin foundation')
    return {'schema_version':2,'kind':'local_current_graph_v2','foundation_sha256':foundation_sha256,'public_origin':public_origin,
            'selector':selector,'graph_config':graph_config,'source_files':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},
            'sensor_contract':{'cameras':['CAM_F0','CAM_L0','CAM_R0'],'time':'current','size':[1024,576],'ego_history_states':4},
            'noise_protocol':'CPU_FP32_full_source_bank_SHA256_local_v2_scene_sample_seed','frozen_upstream':True,'targets_loaded':False}


def current_from_prediction(native,pred):
    b=native.shape[0]
    confidence=1-pred['logits'].softmax(-1)[...,-1]
    return {'actor_features':torch.cat([native.mean(1,keepdim=True),pred['agent_features']],1).float(),
            'context':torch.cat([native,pred['scene_features']],1).float(),
            'current_xy':torch.cat([pred['boxes'].new_zeros(b,1,2),pred['boxes'][...,:2]],1).float(),
            'existence':torch.cat([confidence.new_ones(b,1),confidence],1).float()}


def build_payload(native,pred,observation,identity):
    graph=build_local_graph(pred['boxes'][0].detach().float().cpu().numpy(),pred['logits'][0].detach().float().cpu().numpy(),observation,identity['selector'])
    return {'schema_version':2,'token':observation.token,'identity_sha256':signature(identity),'observation_identity':observation.fingerprint(),
            'native_actions':native.detach().cpu(),'full_current':{k:v.detach().cpu() for k,v in current_from_prediction(native,pred).items()},
            'current_prediction':{k:pred[k].detach().cpu() for k in ('boxes','logits','references')},'local_graph':graph.tensor_state()}


def validate_payload(payload,identity,token):
    if set(payload)!=FIELDS or payload['schema_version']!=2:raise ValueError('Old/fullslot/unknown graph cache format is not local graph v2')
    if payload['identity_sha256']!=signature(identity) or payload['token']!=token:raise ValueError('Local selector/noise/config/foundation identity mismatch')
    if identity['schema_version']!=2 or identity['targets_loaded'] or not identity['frozen_upstream']:raise ValueError('Unsafe current cache provenance')
    graph=LocalInteractionGraph(**payload['local_graph']).validate()
    if graph.graph_provenance[0]['targets_read'] or graph.graph_provenance[0]['observation_identity']!=payload['observation_identity']:
        raise ValueError('Graph observation/provenance differs')
    if graph.graph_provenance[0]['config']!=identity['selector']:raise ValueError('Payload graph selector differs from cache identity')
    return graph


def load_payload(root,record,manifest):
    path=Path(root)/(record['token']+'.pt')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Local cache payload bytes changed')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    validate_payload(payload,manifest['identity'],record['token'])
    return payload


def pack_payload(payload,identity,device='cpu'):
    graph=validate_payload(payload,identity,payload['token']).to(device)
    return gather_current({k:v.to(device) for k,v in payload['full_current'].items()},graph)

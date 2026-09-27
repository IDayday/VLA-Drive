"""Public-origin foundation runtime; supervision is outside the observation boundary."""
import hashlib
import json
from pathlib import Path
import pickle
import random
from dataclasses import fields
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from omegaconf import OmegaConf
from starVLA.model.modules.joint_world.public_baseline import PublicQwenBaseline
from starVLA.model.modules.joint_world.visual_cache import FrozenVisualCache,visual_identity
from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy
from starVLA.model.modules.structured_world.contracts import WorldTargets
from starVLA.model.modules.structured_world.rehab import reference_loss_sums
from tools.local_interaction_mask_v2.data import current_metadata_from_training_pickle,current_example

WORLD_CONFIG={'enabled':True,'provider':'qwen','token_layout':'append_tail','head_location':'post_qwen','reference_heads':True,
              'agent_tokens':64,'scene_tokens':64,'reader_dim':256,'return_world_features':True,'historical_auxiliary_losses':False}
MODULES=('history','DiT','reader','heads')


def modules(world):
    return dict(history=world.baseline.action_input_model,DiT=world.baseline.action_model,reader=world.reader,heads=world.heads)


def create_world(public_path,config,provenance,seed=42,visual_cache=None,code_sha=''):
    base=PublicQwenBaseline(public_path,config,provenance,seed=seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+100);world=StructuredWorldPolicy(base,WORLD_CONFIG).cuda()
    world.heads.motion.requires_grad_(False)  # Unused legacy future head is not silently trained/decayed.
    if visual_cache:base.current_visual_cache=FrozenVisualCache(visual_cache,visual_identity(base,code_sha),write=False)
    set_train_mode(world)
    return world


def set_train_mode(world):
    world.train();world.baseline.qwen_vl_interface.eval()


def module_state(world):return {name:module.state_dict() for name,module in modules(world).items()}


def restore_modules(world,saved):
    if set(saved)!=set(MODULES):raise ValueError('Missing/unexpected foundation modules')
    for name,module in modules(world).items():module.load_state_dict(saved[name],strict=True)


def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])


def to_device(target,device):
    return WorldTargets(**{f.name:getattr(target,f.name).to(device) if torch.is_tensor(getattr(target,f.name)) else getattr(target,f.name) for f in fields(target)})


def ego_label(metadata_path):
    # Exact original normalized action representation, separately loaded as LABELS.
    with Path(metadata_path).open('rb') as f:poses=np.asarray(pickle.load(f)['glo_status']['global_poses'],dtype=np.float64)
    if len(poses)<12:raise ValueError('Missing eight ego future labels')
    yaw=poses[3,2];rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    xy=(poses[4:12,:2]-poses[3,:2])@rot
    angle=(poses[4:12,2]-yaw+np.pi)%(2*np.pi)-np.pi
    xy[:,0]=(xy[:,0]-10.172484)/8.805105;xy[:,1]=(xy[:,1]-.360762)/2.277741
    result=np.concatenate([xy,np.sin(angle)[:,None],np.cos(angle)[:,None]],-1).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite ego labels')
    return torch.from_numpy(result)


class FoundationDataset(torch.utils.data.Dataset):
    def __init__(self,tokens,meta_root,observations,targets):
        self.tokens=list(tokens);self.meta_root=Path(meta_root);self.observations=Path(observations);self.targets=Path(targets)
    def __len__(self):return len(self.tokens)
    def __getitem__(self,index):
        token=self.tokens[index];meta=self.meta_root/(token+'.pkl')
        record=current_metadata_from_training_pickle(meta,token)
        example,observation=current_example(self.observations/(token+'.npz'),record)
        target=WorldTargets(**torch.load(self.targets/'targets'/(token+'.pt'),map_location='cpu',weights_only=True))
        if target.overflow:raise ValueError('Foundation requires complete ROI targets')
        return {'example':example,'target':target,'ego':ego_label(meta)}


def collate(samples):return samples


def epoch_batches(length,batch,rank,world,seed,epoch,start=0):
    if batch<world:raise ValueError('Global batch smaller than number of ranks')
    order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(length).tolist()
    batches=[]
    for offset in range(start,len(order),batch):
        chunk=order[offset:offset+batch]
        if len(chunk)<world:raise ValueError('Uneven tail cannot supply every rank; use a compatible global batch without dropping data')
        batches.append(chunk[rank::world])
    return batches


class FoundationObjective(nn.Module):
    def __init__(self,world):super().__init__();self.world=world
    def forward(self,samples,empty_targets=False):
        predictions=[];conditions=[];targets=[];ego=[]
        for sample in samples:
            target=to_device(sample['target'],'cuda')
            if empty_targets:
                target.annotation_valid_mask=torch.zeros_like(target.annotation_valid_mask)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                condition,prediction=self.world.encode_conditions([sample['example']])
            conditions.append(condition);predictions.append(prediction);targets.append(target);ego.append(sample['ego'])
        # Accumulation keeps the entire effective local batch's sums; ranks reduce
        # denominators BEFORE normalization. DDP averages gradients, hence world factor.
        prediction={k:torch.cat([p[k] for p in predictions]) for k in predictions[0]}
        with torch.autocast('cuda',enabled=False):
            ego_loss=self.world.baseline.action_model(torch.cat(conditions).float(),torch.stack(ego).cuda(),None)
            sums,counts,_=reference_loss_sums(prediction,targets)
            numerator=torch.stack([ego_loss*len(samples),sums['cls'],sums['box']])
        denominator=torch.tensor([len(samples),counts['cls'],counts['box']],device='cuda',dtype=torch.float64)
        world=dist.get_world_size() if dist.is_initialized() else 1
        if world>1:dist.all_reduce(denominator)
        loss=(numerator/denominator.clamp_min(1).to(numerator.dtype)).sum()*world
        detached=numerator.detach().double()
        if world>1:dist.all_reduce(detached)
        return loss,detached,denominator


def parameter_probes(world):
    return {name:float(sum(p.detach().float().square().sum().double() for p in module.parameters()).sqrt()) for name,module in modules(world).items()}

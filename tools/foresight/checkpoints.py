"""Only complete new-campaign student checkpoints; no historical driving loads."""
import hashlib
import json
from pathlib import Path
from starVLA.model.modules.vehicle_joint.initialization import identity_hash,file_sha256


def checkpoint_identity(run,tag):
    run=Path(run);identity=json.loads((run/'identity.json').read_text())
    if identity.get('schema')!='foresight_student_training_v1' or identity_hash({k:v for k,v in identity.items() if k!='identity'})!=identity['identity']:
        raise ValueError('Foreign or changed student identity')
    if Path(tag).name!=tag:raise ValueError('Checkpoint tag must be a directory name')
    folder=run/'checkpoints'/tag;complete=json.loads((folder/'COMPLETE.json').read_text())
    if complete['identity']!=identity['identity'] or complete['tag']!=tag:raise ValueError('Incomplete/foreign checkpoint')
    files={p.name:file_sha256(p) for p in sorted(folder.glob('*.pt'))}
    if not any('model_states' in k for k in files) or not any('optim_states' in k for k in files):raise ValueError('Missing model or FP32 masters')
    if identity['config']['framework']['name']!='DDPForesight':raise ValueError('Wrong checkpoint class')
    record={'run_identity':identity['identity'],'training_source_sha':identity['source_sha'],'tag':tag,
            'completed':complete['completed'],'arm':identity['config']['foresight']['arm'],'scope':identity['scope'],'files':files,
            'training_seed':int(identity['config']['seed']),'model_class':'starVLA.model.framework.DDPForesight.DDPForesight'}
    return identity,{'sha256':identity_hash(record),**record}


def scene_noise(token,seed,device):
    import torch
    hashed=int.from_bytes(hashlib.sha256(f'foresight-action-v1:{seed}:{token}'.encode()).digest()[:8],'little')%(2**63-1)
    return torch.randn((1,8,4),generator=torch.Generator().manual_seed(hashed),dtype=torch.float32).to(device)


def load_student(run,tag,identity,device='cuda',precision='fp32',strip=True):
    from omegaconf import OmegaConf
    from starVLA.model.framework.DDPForesight import DDPForesight
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    config=OmegaConf.create(identity['config']);config.framework.qwenvl.device_map='cpu'
    model=DDPForesight(config)
    # Reconstruct learned FP32 master tensors; do not evaluate rounded BF16 save values.
    state=get_fp32_state_dict_from_zero_checkpoint(str(Path(run)/'checkpoints'),tag=tag)
    model.float();model.load_state_dict(state,strict=True);del state
    if strip:model.strip_auxiliary_heads()
    model.to(device).eval();model.inference_fp32=precision=='fp32'
    if precision=='bf16':model.bfloat16()
    elif precision!='fp32':raise ValueError('Unknown evaluation precision')
    return model

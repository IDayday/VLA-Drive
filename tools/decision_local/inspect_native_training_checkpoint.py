"""Inspect a completed native S0_A recovery point without restarting training.

CPU mmap loads inspect model/client metadata and all optimizer/RNG partitions.
Large tensor payloads are not copied, reconstructed, or numerically replayed.
This is distinct from the already completed native profile save/resume test.
"""
import argparse
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,check_host,digest,read,source_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','registration','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--tag',required=True);a=p.parse_args();resources=check_host()
    if a.output.exists():raise FileExistsError('Preserve prior inspection')
    if Path(a.tag).name!=a.tag:raise ValueError('Require a native checkpoint tag')
    reg=read(a.registration);meta=read(a.run/'identity.json');folder=a.run/'checkpoints'/a.tag
    complete=read(folder/'COMPLETE.json');complete_hash=digest(folder/'COMPLETE.json')
    if (meta['scope']!='formal' or meta['registration_sha256']!=digest(a.registration) or
        meta['source_sha']!=reg['training_source_sha'] or complete['identity']!=meta['identity'] or
        complete['tag']!=a.tag or complete['scheduler']!={**meta['schedule'],'completed':complete['completed']}):
        raise ValueError('Foreign or incomplete registered training state')
    world=meta['world_size'];n=(meta['scene_count']+meta['global_batch']-1)//meta['global_batch']
    epoch,offset=divmod(complete['completed'],n)
    expected_exposure=epoch*meta['scene_count']+min(offset*meta['global_batch'],meta['scene_count'])
    if complete['exposure']!=expected_exposure:raise ValueError('Wrong saved scene exposure')
    # Native checkpoints at an epoch boundary may precede cursor normalization.
    cursor=(complete['epoch'],complete['offset'])
    if cursor!=(epoch,offset) and not (offset==0 and cursor==(epoch-1,n)):
        raise ValueError('Wrong saved data cursor')
    model_path=folder/'mp_rank_00_model_states.pt'
    model=torch.load(model_path,map_location='cpu',weights_only=False,mmap=True)
    if (any(model.get(k)!=v for k,v in complete.items()) or model['global_steps']!=complete['completed'] or
        model['skipped_steps']!=0 or model['dp_world_size']!=world):raise ValueError('Model/native boundary mismatch')
    optimizer_files=sorted(folder.glob('*_optim_states.pt'))
    if len(optimizer_files)!=world or len(list(folder.glob('rng_rank*.pt')))!=world:
        raise ValueError('Missing/extra optimizer or RNG rank')
    ranks=[]
    for rank in range(world):
        optimizer_path=folder/f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'
        state=torch.load(optimizer_path,map_location='cpu',weights_only=False,mmap=True)['optimizer_state_dict']
        masters=state['single_partition_of_fp32_groups'];base=state['base_optimizer_state']
        if state['overflow'] or any(x!=world for x in state['partition_count']):raise ValueError('Invalid optimizer partition')
        if len(base['state'])!=len(masters) or any(x.dtype!=torch.float32 for x in masters):
            raise ValueError('Incomplete FP32 master/optimizer groups')
        for group,master in enumerate(masters):
            value=base['state'][group]
            if int(value['step'])!=complete['completed']:raise ValueError('Optimizer step differs')
            for key in ('exp_avg','exp_avg_sq'):
                if value[key].dtype!=torch.float32 or value[key].shape!=master.shape:
                    raise ValueError('Missing FP32 moment state')
        rng_path=folder/f'rng_rank{rank}.pt';rng=torch.load(rng_path,map_location='cpu',weights_only=False)
        if (set(rng)!= {'python','numpy','torch','cuda','generators'} or
            set(rng['generators'])!= {'noise','time','horizon'} or
            any(not isinstance(x,torch.Tensor) or x.dtype!=torch.uint8 or x.numel()==0
                for x in [rng['torch'],rng['cuda'],*rng['generators'].values()])):
            raise ValueError('Missing native random state')
        ranks.append(dict(rank=rank,optimizer_file=optimizer_path.name,optimizer_bytes=optimizer_path.stat().st_size,
            master_groups=len(masters),master_elements=[x.numel() for x in masters],
            optimizer_steps=[int(value['step']) for value in base['state'].values()],
            actual_group_LR=[group['lr'] for group in base['param_groups']],
            rng_file=rng_path.name,rng_sha256=digest(rng_path),all_FP32_masters_and_moments_present=True))
        del state,masters,base,rng
    if digest(folder/'COMPLETE.json')!=complete_hash:raise ValueError('Checkpoint changed during inspection')
    observed=read(a.run/'status.json')
    atomic_json(a.output,dict(schema='native_A_checkpoint_metadata_inspection_v1',passed=True,
        inspection_source_sha=source_sha(ROOT),training_source_sha=meta['source_sha'],training_identity=meta['identity'],
        checkpoint=str(folder),complete_sha256=complete_hash,complete=complete,rank_states=ranks,
        model_bytes=model_path.stat().st_size,model_tensor_count=len(model['module']),
        observed_training_status=observed,resources=resources,
        scope='CPU mmap metadata, native step/exposure/scheduler and all FP32 master/moment/RNG partitions',
        limitations=['No interruption or optimizer replay of the formal run',
            'Large tensor payloads were not fully hashed or numerically compared; this is not a new numerical acceptance',
            'Rolling checkpoint may later be replaced by a newer complete periodic point under the registered policy']))
    print({'tag':a.tag,'completed':complete['completed'],'ranks':len(ranks),'training_after_save':observed['completed']})


if __name__=='__main__':main()

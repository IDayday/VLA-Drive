"""Four-scene portability check on newly authorized hosts; no training/exports."""
import argparse
import json
import socket
import subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--registration',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    import numpy as np
    import torch
    import transformers
    from tools.ddpolicy_vehicle.run_meter import metered_run
    from tools.foresight.checkpoints import checkpoint_identity,load_student,scene_noise
    from tools.full_foresight.fm_step_sweep import load_registration,sample_encoded,sweep_usage
    from tools.full_foresight.navtest_schedule import atomic,sha,source_identity
    from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
    registration,config=load_registration(a.registration)
    hostname=socket.gethostname()
    if hostname not in ('training-vlawm-zt-worker-0','training-vlawm-zt2-worker-0'):
        raise ValueError('This check is for the explicitly added resource hosts')
    source=Path.cwd();source_sha=source_identity(source)
    pinned=Path(config['source_worktree'])
    # Helper/report source may be newer; actual sampling implementation must
    # remain exactly the registered immutable file, with separate source IDs.
    for file in ('tools/full_foresight/fm_step_sweep.py','tools/foresight/checkpoints.py',
                 'starVLA/model/modules/action_model/GR00T_ActionHeader.py',
                 'starVLA/model/framework/DDPForesight.py',
                 'starVLA/model/framework/ddp_full_foresight.py'):
        if sha(source/file)!=sha(pinned/file):raise ValueError('Sampling implementation changed')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    run_id=f'fm_sweep_{registration["identity"][:12]}_extra_{hostname}'
    with metered_run(config['campaign_root'],run_id,1,{'kind':'additional_host_portability',
            'validation_source':source_sha,'sampling_source':registration['source_sha']}) as (meter,_,save):
        if sweep_usage(registration,config)>=config['gpu_hour_cap']:raise RuntimeError('Sweep resource cap reached')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.5)
        identity,checkpoint=checkpoint_identity(config['training_run'],config['checkpoint_tag'])
        if checkpoint!=registration['checkpoint']:raise ValueError('Wrong checkpoint')
        model=load_student(config['training_run'],config['checkpoint_tag'],identity)
        model.requires_grad_(False)
        data=ForesightCurrentDataset(config['current_root']);rows=[]
        for i in range(4):
            observation=data[i];token=data.index[i]['token']
            noise=scene_noise(token,config['sampling_seed'],'cuda')
            with torch.inference_mode():
                encoded=model.encode_current([observation])
                reused=sample_encoded(model,encoded,noise,10)[0]
                standard=model.predict_action([observation],initial_noise=noise.clone())[0]
            prior=np.load(Path(config['baseline_predictions'])/'predictions'/(token+'.npz'))['trajectory']
            maximum=float(np.max(np.abs(prior-reused.cpu().numpy())))
            if not torch.equal(reused,standard) or maximum!=0:
                raise AssertionError(f'Host original10 parity failed: {maximum}')
            rows.append({'token':token,'max_original10_difference':maximum,'standard_vs_reuse_bitwise_equal':True})
            meter['inference_scenes']=len(rows);save()
        atomic(out/'RESULTS.json',{'status':'COMPLETE','hostname':hostname,'device':torch.cuda.get_device_name(),
               'validation_source':source_sha,'sampling_source':registration['source_sha'],
               'checkpoint':checkpoint['sha256'],'torch':torch.__version__,
               'transformers':transformers.__version__,'cuda':torch.version.cuda,
               'precision':model.deployment_precision,'scenes':rows,'real_optimizer_updates':0,
               'pressure_management':'existing verified pressure controller automatically yields and restores; no signals sent'})


if __name__=='__main__':main()

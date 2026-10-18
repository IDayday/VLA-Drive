"""Execute locked source and DDP decoders to compare RL coordinate units.

No change to normalization, noise, loss or SFT. Intermediate generation states
are not physical trajectories: these are coordinate/penalty comparisons only.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import torch
from omegaconf import OmegaConf

from infer import deal_action_1225
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import file_sha,load_recipe,SFT_EXPORT
from starVLA.rl.simwam_recipe.vendor.action_math import ActionGRPOMixin
from starVLA.rl.simwam_recipe.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler
from starVLA.rl.flow_grpo.transactions import atomic_json


def source_decoder(path):
    tree=ast.parse(Path(path).read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='NavSimVideoDataset')
    fn=copy.deepcopy(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='denorm_odo'))
    fn.decorator_list=[]
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    namespace={'torch':torch}
    exec(compile(module,str(path),'exec'),namespace)
    return namespace['denorm_odo']


def xy_scale_ratio(scales,weights=(.2,1.)):
    c=np.asarray(weights)/np.square(scales)
    return dict(coefficients_before_common_beta_and_HD_denominator=c.tolist(),
        lateral_to_longitudinal_penalty_ratio=float(c[1]/c[0]))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args();require_dispatch()
    if a.output.exists():raise FileExistsError(a.output)
    up=Path('/mnt/project/simwam-recipe-20261015/upstream/SimWAM')
    commit=subprocess.check_output(['git','-C',str(up),'rev-parse','HEAD'],text=True).strip()
    if commit!='1433ee7e06a62bd8b89da74455df0d82bda7b726':raise ValueError('Upstream changed')
    relative='src/simwam/datasets/navsim/navsim_dataset.py';path=up/relative
    subprocess.run(['git','-C',str(up),'diff','--exit-code','HEAD','--',relative],check=True,stdout=subprocess.DEVNULL)
    resolved=OmegaConf.load('reports/ddp_simwam_recipe_20261015/resolved_source_recipe.yaml')
    if not resolved.data.train.normalize_action or resolved.data.train.trajectory_mode!='absolute':
        raise ValueError('The source active normalization branch differs')
    cfg=load_recipe('configs/flow_rl/simwam_source_ddp.yaml')
    act_norm=int(OmegaConf.load(str(Path(SFT_EXPORT)/'config.yaml')).datasets.vla_data.act_norm)
    if act_norm!=1:raise ValueError('DDP action normalization changed')
    source=source_decoder(path)
    zero=torch.zeros((1,8,3),dtype=torch.float64);native=np.zeros((1,8,4));native[:,:,3]=1
    source_scales=[];ddp_scales=[]
    for axis in (0,1):
        x=zero.clone();x[:,:,axis]=1
        source_scales.append(float((source(x)-source(zero))[0,0,axis]))
        y=native.copy();y[:,:,axis]=1
        ddp_scales.append(float((deal_action_1225(y,act_norm=1)-deal_action_1225(native,act_norm=1))[0,0,axis]))
    sampler=cfg['grpo']['sample'];stub=SimpleNamespace(grpo_sigma_max_guard=.999,
        grpo_noise_level=sampler['noise_level'],grpo_min_std=sampler['min_std'])
    scheduler=WanContinuousFlowMatchScheduler(1000,shift=5.)
    times,deltas=scheduler.build_inference_schedule(10,'cpu',torch.float32)
    rows=[]
    for k in (7,8,9):
        sigma=float(times[k])/1000.;dt=float(deltas[k])
        mean,std=ActionGRPOMixin._sde_mean_std(stub,torch.zeros(1),torch.ones(1),sigma,dt)
        rows.append(dict(step=k,sigma=sigma,delta_sigma=dt,normalized_std=std,
            DDP_direct_XY_innovation_std_meters=(np.asarray(ddp_scales)*std).tolist(),
            source_direct_XY_innovation_std_meters=(np.asarray(source_scales)*std).tolist(),
            mean_anchor_velocity_squared_coefficient=float(mean[0])**2))
    result=dict(status='COMPLETE_LOCKED_ACTION_UNIT_COMPARISON',upstream_commit=commit,
        upstream_dataset_sha256=file_sha(path),DDP_decoder_sha256=file_sha('infer.py'),source_sha256=file_sha(__file__),
        active_source_normalization=dict(trajectory_mode='absolute',normalize_action=True),
        DDP_XY_meters_per_normalized_unit=ddp_scales,source_XY_meters_per_normalized_unit=source_scales,
        DDP_anchor=xy_scale_ratio(ddp_scales),source_anchor=xy_scale_ratio(source_scales),random_steps=rows,
        DDP_vs_source_physical_penalty_per_meter_ratio=(np.asarray(source_scales)/np.asarray(ddp_scales))**2,
        new_inference=0,new_scoring=0,optimizer_updates=0,
        limits=['Direct Gaussian XY innovation is before existing simulation/tracking; no collision probability interpretation.',
            'Intermediate latent mean anchor is not final ODE physical trajectory KL or safety guarantee.',
            'DDP is more anisotropic AND stronger per meter in both XY axes; do not claim its absolute x anchor is weaker than source.',
            'Heading angle vs sine/cosine pair is not equated by this XY-only comparison.',
            'Unit mismatch is real, but does not alone prove the cause of RL failure.'])
    result['DDP_vs_source_physical_penalty_per_meter_ratio']=result['DDP_vs_source_physical_penalty_per_meter_ratio'].tolist()
    atomic_json(a.output,result);print(json.dumps(result))


if __name__=='__main__':main()

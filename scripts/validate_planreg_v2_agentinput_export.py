"""Real AgentInput replay with student construction/future/PDM access traps."""
import argparse
import gc
import json
import pickle
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from navsim.common.dataclasses import Scene,SensorConfig
from navsim.agents.EpisodeDrive.planreg_v2.agent import PlanRegV2Agent,file_sha256
from navsim.agents.EpisodeDrive.planreg_v2.checkpoint import export_student,load_student
from navsim.agents.EpisodeDrive.planreg_v2.ema import FP32MasterEMA
from navsim.agents.EpisodeDrive.planreg_v2.predictor import ActionCausalPredictor
from navsim.agents.EpisodeDrive.planreg_v2.runtime import source_fingerprint


def fail(*args,**kwargs):raise AssertionError('Student attempted a training-only operation')


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('training','student','manifest','logs','sensors','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    record=json.loads(Path(a.manifest).read_text())['records'][0]
    with (Path(a.logs)/(record['log']+'.pkl')).open('rb') as stream:raw=pickle.load(stream)
    index=next(i for i,r in enumerate(raw) if r['token']==record['token'])
    # Exactly four historical numeric frames, only the CURRENT image. No future.
    frames=raw[index-3:index+1]
    sensors=SensorConfig(cam_f0=[3],cam_l0=[],cam_l1=[],cam_l2=[],cam_r0=[],cam_r1=[],cam_r2=[],cam_b0=[],lidar_pc=[])
    scene=Scene.from_scene_dict_list(frames,Path(a.sensors),4,0,sensors,load_image_path=False)
    inputs=scene.get_agent_input();del scene,raw
    ckpt=torch.load(a.training,map_location='cpu',weights_only=False)
    config=dict(ckpt['config'],shared_init_path=None)
    original=PlanRegV2Agent(config,a.device)
    original.load_state_dict(ckpt['model'],strict=True);original.eval()
    expected=original.compute_trajectory(inputs).poses
    del original,ckpt;gc.collect();torch.cuda.empty_cache()
    manifest=export_student(a.training,a.student)
    with patch.object(FP32MasterEMA,'__init__',fail),patch.object(ActionCausalPredictor,'__init__',fail),\
         patch.object(PlanRegV2Agent,'encode_teacher',fail),patch.object(PlanRegV2Agent,'compute_metric_targets',fail),\
         patch.object(Scene,'get_future_trajectory',fail),patch('PIL.Image.open',side_effect=fail):
        # AgentInput contains an already decoded real RGB array, so all file
        # image access is forbidden, not just keys starting with future_.
        student=load_student(a.student,a.device)
        actual=student.compute_trajectory(inputs).poses
    difference=float(np.max(np.abs(expected-actual)))
    if difference>1e-5:raise AssertionError('Real AgentInput export replay mismatch')
    result=dict(status='PASS',input_type='real NAVSIM AgentInput/current RGB array',token=record['token'],
        max_abs_diff=difference,teacher_predictor_and_future_access_trapped=True,
        external_dependencies='Original VLM structure/trust_remote_code and tokenizer files are still required; weights are retained in student',
        export=manifest,source_fingerprint=source_fingerprint())
    Path(a.output).write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()

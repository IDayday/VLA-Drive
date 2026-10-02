import json
from pathlib import Path
import torch
import pytest
from tools.action_video_foresight.run_experiments import create_config, ARMS
from starVLA.model.modules.foresight.config import ForesightConfig


def test_all_study_arms_keep_current_and_shared_MAE_and_explicit_future_modes():
    from omegaconf import OmegaConf
    monkey=pytest.MonkeyPatch();monkey.setenv('FULL_LAMBDA_FUT','1');monkey.setenv('FULL_LAMBDA_INT','0.8328945981862067')
    monkey.setenv('FORESIGHT_QWEN','/unopened/generic-Qwen');monkey.setenv('FORESIGHT_SOURCES','/unopened/generic-sources.json')
    base=OmegaConf.to_container(OmegaConf.load(Path(__file__).parents[2]/'configs/foresight_resolution/c1.yaml'),resolve=True)
    for k in ('lambda_cur','lambda_fut','lambda_int'):base['foresight'][k]=float(base['foresight'][k])
    for arm,(target,action,planner) in ARMS.items():
        spans=[[t,t] for t in (.5,1,1.5,2,2.5,3,3.5,4)] if target=='dino_sequence' else [[.5,1],[1.5,2],[2.5,3],[3.5,4]]
        c=create_config(base,arm,{'recipe':{'weight_hash':'verified'},'time_intervals_s':spans},.75,42)
        f=ForesightConfig(**c['foresight']).validate()
        assert f.enable_current_dino and f.num_queries==144
        assert f.enable_interaction==(arm!='S4_NO_MAE')
        assert f.enable_future_dino==(not arm.startswith('C_BASE'))
        assert f.planner_condition_mode==planner and f.future_action_condition==action
        assert c['framework']['name']=='DDPActionVideoForesight'
    monkey.undo()

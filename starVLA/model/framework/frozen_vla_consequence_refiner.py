"""Camera-only single-anchor deployment. Load after the pinned S0 runtime.

The loader extends only the missing decision_local package search path, never
replaces DDPForesight/Qwen/DiT with the report branch's historical approximations.
"""
import importlib
from pathlib import Path
import sys
import types
import numpy as np
import torch
from torch import nn
from starVLA.model.modules.decision_local.consequence_contract import ModelInputs, PredictionQueries, predicted_relations
from starVLA.model.modules.decision_local.trajectory_queries import QueryConfig, trajectory_queries
from starVLA.model.modules.decision_local.consequence_head import ConsequenceQueryHead
from starVLA.model.modules.decision_local.residual_refiner import ResidualRefiner
from starVLA.dataloader.foresight_dataset import decode_ego


def geometry_operators():
    # Namespace isolation permits reusing these exact two mature G operators
    # without substituting any pinned S0 package or initializing a G model.
    name='decision_local_geometry_ops'
    if name not in sys.modules:
        package=types.ModuleType(name)
        package.__path__=[str(Path(__file__).resolve().parents[1]/'modules/structured_world')]
        sys.modules[name]=package
    return importlib.import_module(name+'.grid').GridSpec(), importlib.import_module(name+'.queries').body_points


class FrozenVLAConsequenceRefiner(nn.Module):
    camera_fields=frozenset(('image','lang','state','token'))

    def __init__(self, anchor, refiner, predictor=None, *, mode='R_FUTURE_DIFF', feature_dtype=torch.float16, query_config=None):
        super().__init__()
        if mode not in ('E_NONE','R_CURRENT_PRED','R_FUTURE_ABS','R_FUTURE_DIFF'):
            raise ValueError('Oracle probes are forbidden in the deployable policy')
        if mode!='E_NONE' and predictor is None: raise ValueError('Missing frozen relation predictor')
        self.anchor=anchor.requires_grad_(False).eval()
        self.refiner=refiner.eval();self.predictor=predictor.eval() if predictor is not None else None
        self.mode,self.feature_dtype=mode,feature_dtype
        self.query_config=query_config or QueryConfig()
        self.body={'length':5.176,'width':2.297,'rear_axle_to_center':1.461}
        self.grid,self.body_points=geometry_operators()

    @torch.no_grad()
    def predict(self, observation, *, sampling_seed=42):
        if set(observation)!=self.camera_fields:
            raise ValueError('camera_only accepts only current image/lang/legal state/token; no GT fields')
        if self.anchor.training or any(x.requires_grad for x in self.anchor.parameters()):
            raise RuntimeError('The common mature anchor must stay frozen and eval')
        from tools.foresight.checkpoints import scene_noise
        encoded=self.anchor.encode_current([observation])
        condition=self.anchor.build_planner_condition(encoded)
        qencoded=self.anchor.action_model.predict_action(condition,
            initial_noise=scene_noise(observation['token'],sampling_seed,condition.device)).float()
        q0=decode_ego(qencoded)
        state=torch.as_tensor(np.asarray(observation['state']),device=q0.device,dtype=torch.float32).reshape(1,4)
        velocity=decode_ego(state)[0,:2].cpu().numpy()/.5
        generated=trajectory_queries(q0[0].cpu().numpy(),velocity,self.query_config)
        poses=torch.from_numpy(generated['poses']).to(q0.device)[None]
        footprint=self.body_points(poses,self.body,execution_fast=True)
        # Deterministic physical support only, never the offline target mask.
        _,outside=self.grid.normalized_reference(footprint)
        inside=~outside
        queries=PredictionQueries(poses,torch.from_numpy(generated['times']).to(q0.device),inside.all(-1),
            torch.from_numpy(generated['kinematic_valid']).to(q0.device)[None])
        current=ModelInputs(encoded['action_queries'].to(self.feature_dtype).float(),encoded['W'].to(self.feature_dtype).float(),
            qencoded,q0,state,q0.new_tensor([[self.body[k] for k in ('length','width','rear_axle_to_center')]]))
        relations=predicted_relations(self.predictor(current,queries)) if self.predictor is not None else None
        if self.mode=='R_CURRENT_PRED': relations=torch.cat((relations[...,:3],relations[...,1:3]),-1)
        final_encoded=self.refiner.camera_only(current,queries,relations)
        final=decode_ego(final_encoded)
        if not torch.isfinite(final).all(): raise FloatingPointError('Nonfinite full trajectory; never silently drop scene')
        return {'q0':q0,'q_final':final,'relations':relations,'queries':queries,'inputs':current,
                'q_final_encoded':final_encoded,'base_forward_count':1,'proposal_count':1}

    def strip_auxiliary_heads(self):
        # The original S teacher heads are dispensable. Both new heads execute
        # the camera policy and MUST survive stripping.
        self.anchor.strip_auxiliary_heads()
        return self

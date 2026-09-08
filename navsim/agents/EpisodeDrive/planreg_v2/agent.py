"""Complete V2 current-only policy and separate training-only TF/RO branch."""
import copy
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
import numpy as np
import torch
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig,Trajectory
from ..layers.losses.episode_drive_loss import EpisodeDriveLoss
from ..layers.world_model.future_image_io import encode_path_tensor
from .backbone import V2Backbone
from .normalizers import TrajectoryNormalizer,MotionConditionNormalizer
from .memory import RichSceneMemory,pad_tile_registers
from .action import V2ActionDecoder
from .predictor import ActionCausalPredictor
from .motion import CandidateKinematicsCodec,HORIZONS
from .ema import FP32MasterEMA
from .losses import trajectory_loss,world_model_loss,wm_weight
from .optimizer import build_optimizer
from . import ARCHITECTURE_VERSION
from .initialization import load_shared_bank


def file_sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


class PlanRegV2Agent(AbstractAgent):
    def __init__(self,config,device='cpu',deployment=False):
        super().__init__()
        self.config = copy.deepcopy(dict(config))
        self.deployment = deployment
        if self.config.get('architecture_version') != ARCHITECTURE_VERSION:
            raise ValueError('Explicit V2 architecture required')
        if self.config.get('checkpoint_path') or self.config.get('stage1_checkpoint_path'):
            raise ValueError('V2 VLM-only construction prohibits an M0/full-agent checkpoint')
        self.backbone = V2Backbone(config['vlm_path'],device,config.get('gradient_checkpointing',True),
                                  config.get('register_init_std',.02),config.get('read_only_attention_backend','eager'))
        self.scene_memory = RichSceneMemory(config.get('scene_memory_mode','per_tile_register_memory'))
        if config.get('normalizer_statistics'):
            statistics = config['normalizer_statistics']
            normalizer = TrajectoryNormalizer(statistics['mean'],statistics['std'],statistics['metadata'],
                statistics['metadata'].get('std_floor',.001))
        else:
            normalizer = TrajectoryNormalizer.load(config['normalizer_path'])
        if normalizer.metadata['mode']!=config.get('normalizer_mode','stepwise_zscore'):
            raise ValueError('Normalizer artifact mode does not match explicitly configured control')
        self.action_head = V2ActionDecoder(normalizer,config.get('ego_scales'))
        self.world_model_enabled = bool(config.get('world_model_enabled',True)) and not deployment
        self.wm_predictor = ActionCausalPredictor(layers=config.get('predictor_layers',2)) if self.world_model_enabled else None
        scales=config.get('motion_scales',(30.,10.,1.,1.,15.,15.,8.,8.))
        self.motion_normalizer = MotionConditionNormalizer(scales) if self.world_model_enabled else None
        self.kinematics_codec = CandidateKinematicsCodec(scales) if self.world_model_enabled else None
        self.exact_loss = EpisodeDriveLoss()
        self.ema_teacher = None
        self.register_buffer('optimizer_updates',torch.zeros((),dtype=torch.long))
        self._score_pool = None
        self.to(device=device)
        shared = config.get('shared_init_path')
        if shared and not deployment:
            self.load_shared_initialization(shared)
        # Teacher construction happens after all declared student initialization.
        if self.world_model_enabled:
            self.ema_teacher = FP32MasterEMA(self.backbone,config['total_steps'],config['global_batch'])

    def name(self): return 'PlanReg-WM-V2'

    def initialize(self):
        """Weights already loaded explicitly at construction; never load a legacy agent implicitly."""
        return None

    def get_sensor_config(self):
        return SensorConfig(cam_f0=[3],cam_l0=[],cam_l1=[],cam_l2=[],cam_r0=[],cam_r1=[],cam_r2=[],cam_b0=[],lidar_pc=[])

    def train(self,mode=True):
        super().train(mode)
        if self.ema_teacher is not None: self.ema_teacher.eval()
        return self

    def trainable_state(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}

    def load_shared_initialization(self,path):
        artifact = torch.load(path,map_location='cpu',weights_only=False)
        return load_shared_bank(self,artifact)

    def forward(self,features):
        # This path deliberately never accesses any future, teacher or motion-target key.
        device = next(self.action_head.parameters()).device
        pixels = [p.to(device,non_blocking=True) for p in features['pixel_values']]
        counts = [len(p) for p in pixels]
        metadata = torch.cat(features['tile_metadata']).to(device,non_blocking=True)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            visual = self.backbone(torch.cat(pixels),counts,metadata,features)
        # Planning/scoring stay FP32, preserving the original scorer's numerical operations.
        memory,valid = self.scene_memory(visual['visual_content'].float(),visual['tile_geometry'].float(),
                                        visual['scene_valid_mask'],visual['semantic_queries'].float())
        with self.backbone.step_timing.stage('action_scorer_seconds',device.type=='cuda'):
            output = self.action_head(memory,valid,features['status_feature'].to(device).float())
        output['memory_valid_mask']=valid
        output.update(visual)
        return output

    def encode_teacher(self,features,predictions,include_current=False):
        if self.ema_teacher is None: raise RuntimeError('Deployment has no teacher')
        device = predictions['visual_content'].device
        groups,counts = [],[]
        for current,future in zip(features['pixel_values'],features['future_pixel_values']):
            if len(future) != 3 or any(len(f) != len(current) for f in future):
                raise ValueError('EMA current/future must have identical per-scene tile layout')
            for pixels in ([current]+list(future) if include_current else future):
                groups.append(pixels.to(device,non_blocking=True)); counts.append(len(pixels))
        with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            encoded = self.ema_teacher(torch.cat(groups))  # ONE call; normally only B*3 FUTURE frames.
        split = encoded.split(counts)
        n = predictions['visual_content'].shape[1]
        padded = [torch.nn.functional.pad(x.flatten(0,1),(0,0,0,n-len(x)*16)) for x in split]
        return torch.stack(padded).reshape(len(features['pixel_values']),4 if include_current else 3,n,256)

    def submit_metric_targets(self,targets,proposals):
        """Snapshot detached coordinates; submit the unchanged full-64 CPU PDM jobs."""
        from ..score_module.compute_navsim_score import get_sub_score
        paths = targets['metric_cache_path']
        coords = proposals.detach().float().cpu().numpy().copy()
        if len(paths) != len(coords):
            raise ValueError('Metric cache paths must match the proposal batch')
        workers = int(self.config.get('scorer_processes',4))
        request = dict(device=proposals.device,futures=[],results=None)
        if workers:
            if self._score_pool is None:
                self._score_pool = ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn'))
            try:
                for path,p in zip(paths,coords):
                    request['futures'].append(self._score_pool.submit(get_sub_score,path,p,False))
            except BaseException:
                self.cancel_metric_targets(request)
                raise
        else:
            request['results'] = [get_sub_score(path,p,False) for path,p in zip(paths,coords)]
        return request

    @staticmethod
    def cancel_metric_targets(request):
        for future in request['futures']:
            future.cancel()

    def resolve_metric_targets(self,request):
        try:
            results = request['results']
            if results is None:
                # Submission order, never completion order; no candidate repartitioning.
                results = [f.result() for f in request['futures']]
            return torch.tensor(np.stack([r[0] for r in results]),device=request['device'],dtype=torch.float32)
        except BaseException:
            self.cancel_metric_targets(request)
            raise

    def compute_metric_targets(self,targets,proposals):
        return self.resolve_metric_targets(self.submit_metric_targets(targets,proposals))

    def metric_targets_with_teacher(self,features,targets,predictions,wm_active):
        """Overlap independent CPU labels with no-grad EMA, without changing either function.

        The legacy serial path is retained. Teacher has no dropout/RNG or state
        update in forward; it runs once on the same three future frames. Pending
        labels are cancelled if either branch fails; errors are never masked.
        """
        device = predictions['proposals'].device
        timer = self.backbone.step_timing
        overlap = (self.config.get('overlap_metric_target_with_ema',False) and wm_active
                   and int(self.config.get('scorer_processes',4)) > 0)
        if not overlap:
            with timer.stage('pdm_seconds',cuda=False):
                scores = self.compute_metric_targets(targets,predictions['proposals'])
            return scores,None
        with timer.stage('pdm_submit_seconds',cuda=False):
            request = self.submit_metric_targets(targets,predictions['proposals'])
        try:
            with timer.stage('teacher_seconds',device.type=='cuda'):
                teacher = self.encode_teacher(features,predictions)
            with timer.stage('pdm_seconds',cuda=False):
                scores = self.resolve_metric_targets(request)
            return scores,teacher
        except BaseException:
            self.cancel_metric_targets(request)
            raise

    def compute_loss(self,features,targets,predictions):
        device = predictions['proposals'].device
        targets = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in targets.items()}
        wm_active = self.world_model_enabled and (self.training or getattr(self,'wm_diagnostic_forward',False))
        scores,teacher = self.metric_targets_with_teacher(features,targets,predictions,wm_active)
        self.last_metric_targets=scores.detach()
        count_context=getattr(self,'valid_count_context',None)
        accumulation=count_context['accumulate'] if count_context else 1
        from .runtime import validate_ttc_reduction
        validate_ttc_reduction(scores,accumulation)
        trajectory,stages = trajectory_loss(predictions['stage_proposals'],targets,self.action_head.normalizer.std,valid_counts=count_context)
        trajectory=trajectory*accumulation
        stages=[value*accumulation for value in stages]
        components,scorer,_,_,_ = self.exact_loss.score_loss(predictions['pred_logit'],None,None,None,
                                                          scores.double(),None,None,None,None)
        result = dict(loss=trajectory+scorer,trajectory_loss=trajectory,scorer_loss=scorer)
        result.update({'trajectory_stage_'+str(i+1):value for i,value in enumerate(stages)})
        result.update({name:value for name,value in zip(('dac_loss','ttc_loss','nc_loss','ep_loss','ddc_loss','comfort_loss'),components)})
        selected = scores[torch.arange(len(scores),device=device),predictions['selected_indices'],-1]
        result.update(selected_pdms=selected.mean(),oracle64=scores[:,:,-1].max(-1).values.mean())
        diagnostic_step=int(self.optimizer_updates)%500==0
        if diagnostic_step:
            from .diagnostics import representation_summary,score_summary
            self.last_diagnostics=dict(semantic=representation_summary(predictions['semantic_queries']),
                planning=representation_summary(predictions['visual_content'],predictions['scene_valid_mask']),
                scorer=score_summary(predictions,scores),ttc_label_values=scores[...,3].unique().cpu().tolist())
        if wm_active:
            if teacher is None:
                with self.backbone.step_timing.stage('teacher_seconds',device.type=='cuda'):
                    teacher = self.encode_teacher(features,predictions)
            if self.config.get('motion_mode','gt_log') == 'gt_log':
                motion = self.motion_normalizer(targets['motion_sequence'])
                times,valid = targets['motion_timestamps'],targets['motion_valid']
            else:
                codec = self.kinematics_codec(targets['trajectory'][:,None],features['status_feature'].to(device)[:,4:8],
                    targets['motion_timestamps'][:,None],targets['motion_valid'][:,None])
                motion,times,valid = [codec[n][:,0] for n in ('motion_sequence','timestamps','valid_mask')]
            actions,coverage = self.wm_predictor.motion_encoder(motion,times,valid,HORIZONS)
            tf,ro,_ = self.wm_predictor.branches(predictions['visual_content'].float(),teacher.float(),actions,
                predictions['tile_geometry'],predictions['scene_valid_mask'],predictions['semantic_queries'].float())
            wm = world_model_loss(tf,ro,teacher,predictions['scene_valid_mask'],targets['future_valid_mask'],coverage,count_context)
            if self.config.get('wm_objective','tf_and_ro')=='tf_only':
                wm['wm_loss']=wm['wm_tf_loss']
            wm={name:value*accumulation for name,value in wm.items()}
            weight = wm_weight(int(self.optimizer_updates),self.config['total_steps'])
            result.update(wm)
            result['wm_weight'] = trajectory.new_tensor(weight)
            result['loss'] = result['loss']+weight*wm['wm_loss']
            if diagnostic_step:
                from .predictor import state_norm
                with torch.no_grad():
                    common=(predictions['tile_geometry'],predictions['scene_valid_mask'],predictions['semantic_queries'].float())
                    no_tf,no_ro,_=self.wm_predictor.branches(predictions['visual_content'].float(),teacher.float(),torch.zeros_like(actions),*common)
                    no_loss=world_model_loss(no_tf,no_ro,teacher,common[1],targets['future_valid_mask'],coverage)
                    copied=state_norm(predictions['visual_content'])[:,None].expand_as(ro)
                    copy_loss=world_model_loss(copied,copied,teacher,common[1],targets['future_valid_mask'],coverage)
                    controls=dict(no_action_ro=float(no_loss['wm_ro_loss']),copy_current_ro=float(copy_loss['wm_ro_loss']))
                    if len(actions)>1:
                        sh_tf,sh_ro,_=self.wm_predictor.branches(predictions['visual_content'].float(),teacher.float(),actions.roll(1,0),*common)
                        sh=world_model_loss(sh_tf,sh_ro,teacher,common[1],targets['future_valid_mask'],coverage)
                        controls['mismatched_actions_ro']=float(sh['wm_ro_loss'])
                    else:controls['mismatched_actions_ro']='NOT_AVAILABLE_BATCH1'
                    self.last_diagnostics['wm_controls']=controls
        return result

    def get_optimizers(self,total_optimizer_steps=None):
        total = total_optimizer_steps or self.config['total_steps']
        if total != self.config['total_steps']:
            raise ValueError('Optimizer/EMA schedule total must match the locked step budget')
        optimizer,scheduler,summary = build_optimizer(self,total,self.config.get('learning_rates'))
        # PyTorch invokes post-step hooks only when optimizer.step actually runs.
        # Accumulation batches and GradScaler skipped steps do not invoke this hook.
        def successful_step(_optimizer,_args,_kwargs):
            self.optimizer_updates.add_(1)
            if self.ema_teacher is not None:
                self.ema_teacher.update(self.backbone,diagnostics=int(self.optimizer_updates)%500==0 or int(self.optimizer_updates)<=2)
        optimizer.register_step_post_hook(successful_step)
        self.optimizer_summary = summary
        return optimizer,scheduler

    @torch.no_grad()
    def compute_trajectory(self,agent_input):
        from PIL import Image
        from .data import preprocess_fixed_layout,v2_collate
        from .backbone import V2_SYSTEM_PROMPT
        from ..utils.internvl_tokenize import build_internvl_model_inputs
        from ..utils.utils import build_drivevla_questions
        statuses=agent_input.ego_statuses
        command=torch.tensor(statuses[-1].driving_command,dtype=torch.float32)
        feature=dict(history_trajectory=torch.tensor(np.stack([e.ego_pose for e in statuses]),dtype=torch.float32),
            high_command_one_hot=command,status_feature=torch.cat((command,
                torch.tensor(statuses[-1].ego_velocity),torch.tensor(statuses[-1].ego_acceleration))).float())
        raw=agent_input.cameras[-1].cam_f0.image
        image=Image.fromarray(raw) if isinstance(raw,np.ndarray) else raw
        pixels,metadata=preprocess_fixed_layout(image)
        questions=build_drivevla_questions(feature['history_trajectory'],command)
        inputs=build_internvl_model_inputs(self.backbone.tokenizer,questions,[len(pixels)],V2_SYSTEM_PROMPT)
        mask=inputs['attention_mask'][0].bool()
        feature.update(pixel_values=pixels.bfloat16(),tile_metadata=metadata,input_ids=inputs['input_ids'][0][mask],
                       attention_mask=inputs['attention_mask'][0][mask])
        features,_ = v2_collate([(feature,{})])
        self.eval()
        return Trajectory(self(features)['trajectory'][0].float().cpu().numpy())

"""One current encoding, architecture-specific prefix, identical supervision."""
import copy
import torch
import torch.distributed as dist
from torch import nn
from torch.profiler import record_function
from ..qwen_world_postcot.qwen_encoder import SelectedQwenEncoder
from ..qwen_world_postcot.memory import SceneMemoryProvider
from ..qwen_world_postcot.flow import clean_x_step,shift_sigma
from ..qwen_world_postcot.readouts import valid_scene_mse
from ..qwen_sourceflow.action_codec import ActionCodec
from .contracts import CurrentContext,ArchitecturePrefix,LOSS_WEIGHTS,validate_action
from .heads import CommonReadouts,object_track_loss,evaluation_loss

class FourFrameworkVLA(nn.Module):
    def __init__(self,architecture_id,qwen_path=None,*,config=None,checkpoint_config=None,
                 train_heads=True,qwen_encoder=None):
        super().__init__();self.architecture_id=architecture_id;self.config=copy.deepcopy(config or {})
        self.train_heads=bool(train_heads);self.policy_version='initial';self.interface=False
        if architecture_id=='D':
            from .qwen_flow_adapter import QwenFlowAdapter
            from ..qwen_sourceflow.qwen_encoder import QwenEncoder
            native=(qwen_encoder.model if qwen_encoder is not None else
                    QwenEncoder(qwen_path,checkpoint_config=checkpoint_config).model)
            self.decision=QwenFlowAdapter(native,**self.config.get('d',{}))
            width=self.decision.hidden_dim
        else:
            from .independent_actor import ArchitectureA,ArchitectureC
            from .shared_actor import ArchitectureB
            self.qwen=qwen_encoder if qwen_encoder is not None else SelectedQwenEncoder(qwen_path,checkpoint_config=checkpoint_config)
            self.scene_provider=SceneMemoryProvider(**self.config.get('scene',{}))
            cls={'A':ArchitectureA,'B':ArchitectureB,'C':ArchitectureC}[architecture_id]
            self.decision=cls(**self.config.get('decision',{}))
            width=self.decision.hidden_dim if architecture_id=='B' else self.decision.world_stream.hidden_dim
        if train_heads:self.readouts=CommonReadouts(width)

    def configure_stage(self,interface):
        # D's new interfaces must retain a graph through frozen native layers.
        self.interface=bool(interface)
        if self.architecture_id!='D':self.qwen.eval() if interface else self.qwen.train(self.training)

    def train(self,mode=True):
        super().train(mode)
        if self.interface and self.architecture_id!='D':self.qwen.eval()
        return self

    def parameter_roles(self):
        if self.architecture_id=='D':
            roles=self.decision.named_parameter_roles()
            mapping={id(p):roles[name] for name,p in self.decision.named_parameters()}
        else:
            mapping={}
            for name,p in self.qwen.named_parameters():
                mapping[id(p)]='pretrained_vision' if name.startswith('model.visual.')and'merger'not in name else 'pretrained_language_merger'
        result={}
        for name,p in self.named_parameters():
            role=mapping.get(id(p),'train_only_readouts' if name.startswith('readouts.')else'new_modules')
            if role in ('pretrained_language','pretrained_merger'):role='pretrained_language_merger'
            result[name]=role
        return result

    def encode_current(self,current_inputs,ego,scene_identity=()):
        with record_function('compare/current_Qwen'):
            if self.architecture_id=='D':
                encoded=self.decision.encode_current(current_inputs,ego)
                return CurrentContext(None,None,ego,encoded,tuple(scene_identity),self.policy_version,'FP32params_BF16forward','native_Qwen_positions')
            if self.interface:
                with torch.no_grad():features=self.qwen(current_inputs)
            else:features=self.qwen(current_inputs)
        with record_function('compare/scene_C160'):
            memory=self.scene_provider(features,ego)
            scene=torch.cat((memory.spatial,memory.global_memory),1)
            valid=torch.cat((memory.spatial_valid,memory.global_valid),1)
        if scene.shape[1]!=160:raise ValueError('A/B/C current scene provider must expose C160 only')
        return CurrentContext(scene,valid,ego,None,tuple(scene_identity),self.policy_version,'FP32params_BF16forward','selected8_16_28_C160')

    def prepare_prefix(self,current,*,auxiliary=None):
        if auxiliary is None:auxiliary=self.train_heads
        with record_function('compare/P_W_prefix'):
            if self.architecture_id=='D':value=self.decision.prepare_prefix(current.encoded,policy_version=current.policy_version)
            elif self.architecture_id=='A':value=self.decision.prepare_prefix(current.scene,current.valid,current.ego,need_auxiliary=auxiliary)
            else:value=self.decision.prepare_prefix(current.scene,current.valid,current.ego)
        return ArchitecturePrefix(self.architecture_id,current.scene_identity,current.policy_version,current.execution_identity,current.position_mask_identity,value)

    def denoise_x0(self,action_state_fp32,sigma_fp32,prefix):
        if prefix.architecture_id!=self.architecture_id:raise ValueError('Another architecture owns this prefix')
        return self.decision.denoise_x0(action_state_fp32.float(),sigma_fp32.float(),prefix.tensors).float()

    def pre_reasoning_states(self,prefix):return self.decision.pre_reasoning_states(prefix.tensors)

    def replay_candidates(self,normalized_candidate,candidate_to_scene,prefix):
        return self.decision.replay_candidates(normalized_candidate.float(),candidate_to_scene,prefix.tensors)

    def forward(self,batch,*,denominators=None,weights=None):
        if not self.train_heads:raise RuntimeError('Deployment instance has no supervised readouts')
        den=denominators or {};weights=LOSS_WEIGHTS if weights is None else weights
        if set(weights)!=set(LOSS_WEIGHTS):raise ValueError('All seven fixed losses must be explicit')
        context=self.encode_current(batch['current_inputs'],batch['ego'],batch.get('tokens',()))
        prefix=self.prepare_prefix(context)
        target=batch['target'].detach().float();noise=batch['noise'].float();sigma=batch['sigma'].float()
        target=torch.where(batch['action_valid'][...,None],target,0.)
        with record_function('compare/main_action'):
            state=(1-sigma[:,None,None])*target+sigma[:,None,None]*noise
            prediction=self.denoise_x0(state,sigma,prefix)
            valid=batch['action_valid'][...,None].expand_as(prediction)
            numerator=torch.where(valid,(prediction-target).abs(),0.).sum()
            world=dist.get_world_size()if dist.is_initialized()else 1
            denominator=torch.as_tensor(den.get('action_elements',valid.sum()),device=prediction.device).float().clamp_min(1)
            loss_x=numerator*world/denominator
        with record_function('compare/pre_readouts'):
            p=self.readouts.bridge(self.pre_reasoning_states(prefix))
            dino=valid_scene_mse(self.readouts.dino(p),batch['dino_target'],batch['dino_valid'],scene_denominator=den.get('dino'))
            wa_pre=valid_scene_mse(self.readouts.wa(p,batch['future_queries']),batch['future_target'],batch['future_valid'],scene_denominator=den.get('wa_pre'))
            objects=object_track_loss(self.readouts.objects(p),batch['objects'],den)
        # Every rank has the same logical post budget; empty microchunks keep a
        # zero dependency on heads, and denominators are computed per update.
        selected=batch['aux_scene_indices'];nc=3
        with record_function('compare/clean_candidate_replay'):
            candidates=batch['post_trajectories'].detach().float()
            if len(selected):
                flat=candidates.reshape(-1,8,3)
                r=self.replay_candidates(flat,selected.repeat_interleave(nc),prefix)
                r=self.readouts.bridge(r)
                factual=r.reshape(len(selected),nc,16,-1)[:,0]
                factual_valid=batch['future_valid'][selected]&batch['action_valid'][selected].all(1)[:,None]
                wa_post=valid_scene_mse(self.readouts.wa(factual,batch['future_queries'][selected]),batch['future_target'][selected],factual_valid,scene_denominator=den.get('wa_post'))
                evaluation=evaluation_loss(self.readouts.evaluation(r,flat),batch['post_labels'],batch['post_valid'],batch['post_score'],batch['post_score_valid'],batch['post_events'],batch['post_event_valid'],den)
                loss_eval=evaluation['loss']
            else:
                zero=p.sum()*0.+sum(parameter.sum()*0. for parameter in self.readouts.parameters())
                wa_post=loss_eval=zero
        losses=dict(x=loss_x,dino=dino,wa_pre=wa_pre,object=objects['object'],future_track=objects['future_track'],wa_post=wa_post,evaluation=loss_eval)
        total=sum(float(weights[k])*v for k,v in losses.items())
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite four-framework objective')
        return dict(loss=total,losses={k:v.detach().float()for k,v in losses.items()},sigma_mean=sigma.mean().detach())

    @torch.no_grad()
    def predict(self,current_inputs,ego,initial_noise):
        with torch.autocast(initial_noise.device.type,dtype=torch.bfloat16,enabled=initial_noise.device.type in('cuda','cpu')):
            current=self.encode_current(current_inputs,ego);prefix=self.prepare_prefix(current,auxiliary=False)
            grid=shift_sigma(torch.linspace(1,0,11,dtype=torch.float32),5.).tolist()
            state=initial_noise.float()
            for sigma,nxt in zip(grid[:-1],grid[1:]):
                time=torch.full((len(state),),sigma,device=state.device,dtype=torch.float32)
                prediction=self.denoise_x0(state,time,prefix)
                state=clean_x_step(state,prediction,sigma,nxt,validate=False)
        return dict(normalized_action=state,requested_trajectory=ActionCodec.denorm_odo(state))

    def deployment_components(self):
        return dict(architecture_id=self.architecture_id,post_reasoning=False,readouts=False,
                    world_prefix_retained=self.architecture_id=='C',prediction='single_clean_x_K10_shift5_8x3')

    def deployment_state_dict(self):
        # Architecture implementations supply pruning; export builds a matching
        # deployment topology instead of loading missing keys non-strictly.
        if hasattr(self.decision,'deployment_state_dict'):
            decision=self.decision.deployment_state_dict()
        else:decision=self.decision.state_dict()
        result={k:v for k,v in self.state_dict().items()if not k.startswith(('readouts.','decision.'))}
        result.update({'decision.'+k:v for k,v in decision.items()});return result

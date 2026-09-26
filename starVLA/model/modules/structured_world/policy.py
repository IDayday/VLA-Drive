"""Opt-in QwenOFT composition. Disabled inference calls the original method exactly."""
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from .scene_agent_reader import SceneAgentReader
from .agent_heads import AgentHeads
from .action_adapter import WorldToActionAdapter
from .providers import GeometricBEVProvider
from .tokens import token_positions, insert_world_tokens
from .losses import world_losses


class StructuredWorldPolicy(nn.Module):
    def __init__(self, baseline, config):
        super().__init__()
        self.baseline = baseline
        self.world_config = dict(config)
        self.world_enabled = bool(config.get('enabled',False))
        dim = baseline.qwen_vl_interface.model.get_input_embeddings().embedding_dim
        self.provider = GeometricBEVProvider(channels=config.get('bev_channels',64)) if config.get('provider','qwen') == 'geometric_bev' else None
        self.reader = SceneAgentReader(self.provider.channels if self.provider else dim,dim,
                                      dim=config.get('reader_dim',256),scene_tokens=config.get('scene_tokens',64),
                                      agent_tokens=config.get('agent_tokens',32))
        self.heads = AgentHeads(dim,steps=baseline.config.framework.action_model.action_horizon)
        self.adapter = WorldToActionAdapter(dim) if config.get('action_adapter',False) else None

    def encode_conditions(self, examples, model_inputs=None, include_world=True):
        base = self.baseline
        if any(e.get('qwen_feature_cache') is not None for e in examples):
            raise ValueError('World path requires live Qwen features; legacy cache identity is insufficient')
        hist = base.robot_history_token
        rgb,gs,act,rew = [''.join(x) for x in [base.rgb_query_tokens,base.gs_query_tokens,base.act_query_tokens,base.reward_query_tokens]]
        suffix = f' {hist}{gs}{rgb}{act}{rew}' if base.w_depth else f' {hist}{rgb}{gs}{act}{rew}'
        # Labels are never passed to the processor, Reader, provider or Qwen.
        q = base.qwen_vl_interface.build_qwenvl_inputs(images=[e['image'] for e in examples],instructions=[e['lang']+suffix for e in examples])
        ids,mask = q['input_ids'],q['attention_mask']
        model = base.qwen_vl_interface.model
        tok = base.qwen_vl_interface.processor.tokenizer
        positions = {name:token_positions(ids,tok.convert_tokens_to_ids(tokens)) for name,tokens in
                     [('history',[hist]),('rgb',base.rgb_query_tokens),('gs',base.gs_query_tokens),('action',base.act_query_tokens)]}
        with torch.no_grad():
            rope,_ = model.model.get_rope_index(input_ids=ids,image_grid_thw=q['image_grid_thw'],attention_mask=mask)
        vision_trainable = self.world_config.get('vision_trainable',False)
        with nullcontext() if vision_trainable else torch.no_grad():
            parts,deepstack = model.model.get_image_features(q['pixel_values'],q['image_grid_thw'])
            image = torch.cat(parts,0)
        embeds = model.get_input_embeddings()(ids)
        state = torch.as_tensor(np.array([e['state'] for e in examples]),device=ids.device,dtype=torch.float32)[:,0]
        with torch.autocast('cuda', enabled=False):
            states = base.action_input_model(state).to(embeds.dtype)
        batch = torch.arange(len(examples),device=ids.device)
        embeds[batch,positions['history'][:,0]] = states
        if base.config.datasets.video_data.load_2d_data:
            embeds[batch[:,None],positions['rgb']] = base.rgb_query.to(embeds.dtype)
        if base.config.datasets.gs_data.load_3d_data or base.w_depth:
            embeds[batch[:,None],positions['gs']] = base.gs_query.to(embeds.dtype)
        image_mask = ids == model.config.image_token_id
        if int(image_mask.sum()) != len(image):
            raise ValueError('Native visual token/feature count mismatch')
        embeds = embeds.masked_scatter(image_mask[...,None].expand_as(embeds),image.to(embeds.dtype))
        world_positions = None
        action_positions = positions['action']
        if include_world:
            if self.provider is not None:
                if model_inputs is None:
                    raise ValueError('BEV requires current calibrated ModelInputs')
                f,coords,support,metadata = self.provider(model_inputs)
                memory = self.reader(f,coords,support,metadata)
            else:
                lengths = image_mask.sum(-1).tolist()
                padded = image.new_zeros(len(examples),max(lengths),image.shape[-1])
                support = torch.zeros(padded.shape[:2],device=ids.device,dtype=torch.bool)
                offset = 0
                for b,n in enumerate(lengths):
                    padded[b,:n] = image[offset:offset+n];support[b,:n] = True;offset += n
                memory = self.reader(padded,support=support)
            world = torch.cat([memory.scene_memory,memory.agent_memory],1)
            ids,embeds,mask,rope,world_positions,action_positions = insert_world_tokens(
                ids,embeds,mask,rope,action_positions,world,int(tok.pad_token_id))
            image_mask = ids == model.config.image_token_id
        outputs = model.model.language_model(input_ids=None,inputs_embeds=embeds,attention_mask=mask,position_ids=rope,
                                            visual_pos_masks=image_mask,deepstack_visual_embeds=[x.to(embeds.dtype) for x in deepstack],
                                            use_cache=False,output_hidden_states=True)
        # Released checkpoint consumes the legacy hidden_states[-1] (pre-final-norm).
        hidden = outputs.hidden_states[-1] if self.world_config.get('hidden_mode','prenorm') == 'prenorm' else outputs.last_hidden_state
        actions = hidden.gather(1,action_positions[...,None].expand(-1,-1,hidden.shape[-1]))
        world_prediction = None
        if include_world:
            world_hidden = hidden.gather(1,world_positions[...,None].expand(-1,-1,hidden.shape[-1]))
            world_prediction = self.heads(world_hidden[:,self.reader.scene_tokens:])
            if self.adapter is not None:
                actions = self.adapter(actions,world_hidden)
        return actions,world_prediction

    def forward(self, examples, model_inputs=None, targets=None):
        # A1 intentionally disables historical auxiliary losses while retaining prompt/modules.
        with torch.autocast('cuda',dtype=torch.bfloat16):
            conditions,prediction = self.encode_conditions(examples,model_inputs,include_world=self.world_enabled)
        actions = torch.as_tensor(np.array([e['action'] for e in examples]),device=conditions.device,dtype=torch.float32)
        repeats = self.baseline.config.framework.action_model.get('repeated_diffusion_steps',1)
        loss = self.baseline.action_model(conditions.repeat(repeats,1,1),actions.repeat(repeats,1,1),None)
        losses = {'ego':loss}
        if prediction is not None and targets is not None:
            terms,_ = world_losses(prediction,targets)
            losses.update(terms)
            for name,value in terms.items():
                loss = loss + self.world_config.get('lambda_'+name,0.) * value
        return {'loss':loss,'losses':losses,'world_prediction':prediction}

    @torch.no_grad()
    def predict_action(self, examples, model_inputs=None):
        if not self.world_enabled:
            return self.baseline.predict_action_infer_1d(examples)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            conditions,prediction = self.encode_conditions(examples,model_inputs)
        actions = self.baseline.action_model.predict_action(conditions)
        return {'normalized_actions':actions.cpu().numpy(),'world_prediction':prediction}

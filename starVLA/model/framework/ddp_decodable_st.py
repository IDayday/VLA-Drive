"""One audited Qwen/DiT, appended world queries, and student-only action reading.

This module has no teacher imports and accepts only AllowedObservation. Supervision
is deliberately implemented outside forward. Existing tokenizer rows are untouched:
new query embeddings are appended directly after the original processed sequence.
"""
import numpy as np
import torch
from torch import nn
from starVLA.cache.navsim_feature_cache import ROBOT_HISTORY_TOKEN
from starVLA.model.modules.foresight.tokens import replace_query_embeddings
from starVLA.model.modules.decodable_st.contracts import AllowedObservation, StudentRepresentation
from starVLA.model.modules.decodable_st.attention_masks import world_attention_mask,append_mrope_positions
from starVLA.model.modules.decodable_st.student_queries import QueryBank
from starVLA.model.modules.decodable_st.vigt_readout import SpatialReadout,PatchSpatialReadout
from starVLA.model.modules.decodable_st.sgdrive_readout import TemporalProjector
from starVLA.model.modules.decodable_st.world_to_action_bridge import WorldToActionBridge


class DDPDecodableST(nn.Module):
    def __init__(self,baseline,*,spatial_contract=None,temporal_contract=None,
                 spatial_grid=(16,16),cameras=("CAM_F0","CAM_L0","CAM_R0"),seed=42,
                 query_execution="prefix_cached_v1",spatial_readout_kind="bilinear"):
        super().__init__()
        self.baseline=baseline
        self.cameras=tuple(cameras)
        self.spatial_contract,self.temporal_contract=spatial_contract,temporal_contract
        self.reader_enabled=False
        if query_execution not in ("prefix_cached_v1", "appended_v1"):
            raise ValueError("Unknown Qwen query execution")
        self.query_execution=query_execution
        hidden=int(baseline.qwen_vl_interface.model.config.hidden_size)
        if spatial_contract:
            if spatial_readout_kind not in ("bilinear","patch"):
                raise ValueError("Unknown spatial readout; no implicit fallback")
            self.spatial_queries=QueryBank(hidden,spatial_grid,seed+100)
            cls=SpatialReadout if spatial_readout_kind=="bilinear" else PatchSpatialReadout
            self.spatial_readout=cls(hidden,spatial_contract,spatial_grid,seed+101)
        if temporal_contract:
            side=int(temporal_contract.num_queries**.5)
            if side*side!=temporal_contract.num_queries:raise ValueError("Native temporal query grid")
            self.temporal_queries=QueryBank(hidden,(side,side),seed+102)
            self.temporal_projector=TemporalProjector(hidden,temporal_contract,seed+103)
        self.bridge=WorldToActionBridge(hidden,spatial_contract,temporal_contract,seed+104)

    def train(self,mode=True):
        super().train(mode)
        self.baseline.qwen_vl_interface.model.model.visual.eval()
        return self

    def encode_observation(self,obs:AllowedObservation)->StudentRepresentation:
        if type(obs) is not AllowedObservation:
            raise TypeError("Student accepts only AllowedObservation, never examples/targets/cache dictionaries")
        obs.validate(self.cameras)
        current=[{"image":list(images),"lang":lang,"state":state.detach().cpu().numpy()}
                 for images,lang,state in zip(obs.images,obs.navigation,obs.ego_state)]
        base=self.baseline
        if self.spatial_contract is None and self.temporal_contract is None:
            encoded=base.encode_current(current)
            return StudentRepresentation(encoded["action_queries"],encoded["W"])
        instructions=[e["lang"]+" "+ROBOT_HISTORY_TOKEN+"".join(base.foresight_tokens)+"".join(base.act_query_tokens) for e in current]
        ids,attention,positions,slots,visual,deepstack=base._build_qwen_batch(current,instructions)
        n=ids.shape[1];b=len(current);device=ids.device
        with base.amp():
            embeddings=base.qwen_vl_interface.model.get_input_embeddings()(ids)
            states=obs.ego_state.to(device=device,dtype=torch.float32)
            embeddings=replace_query_embeddings(embeddings,slots["history"],base.action_input_model(states))
            if base.foresight_tokens:
                embeddings=replace_query_embeddings(embeddings,slots["foresight"],base.query_embeddings())
            tail=[];ns=nt=0
            if self.spatial_contract:
                s=self.spatial_queries();ns=len(s);tail.append(s)
            if self.temporal_contract:
                t=self.temporal_queries();nt=len(t);tail.append(t)
            tail=torch.cat(tail,0).to(embeddings.dtype)[None].expand(b,-1,-1)
            mask=world_attention_mask(attention,slots["action"],ns,nt,embeddings.dtype)
            full_positions=append_mrope_positions(positions,attention,ns+nt)
            if self.query_execution=="prefix_cached_v1":
                from starVLA.model.modules.decodable_st.qwen_tail import prefix_cached_forward
                hidden=prefix_cached_forward(base,ids,embeddings,attention,positions,visual,deepstack,
                                              tail,full_positions[:,:,n:],mask[:,:,n:,:])
            else:
                embeddings=torch.cat((embeddings,tail),1)
                pad=int(base.qwen_vl_interface.processor.tokenizer.pad_token_id)
                ids=torch.cat((ids,ids.new_full((b,ns+nt),pad)),-1)
                hidden=base._qwen_language_forward(ids,embeddings,mask,full_positions,visual,deepstack)
        rows=torch.arange(b,device=device)[:,None]
        representation=StudentRepresentation(hidden[rows,slots["action"]],hidden[rows,slots["foresight"]])
        if ns:
            representation.spatial_query_hidden=hidden[:,n:n+ns]
            representation.predicted_spatial_context=self.spatial_readout(representation.spatial_query_hidden.float())
        if nt:
            representation.temporal_query_hidden=hidden[:,n+ns:n+ns+nt]
            representation.predicted_future_latent=self.temporal_projector(representation.temporal_query_hidden.float())
        return representation

    def forward(self,obs:AllowedObservation):
        return self.encode_observation(obs)

    def action_condition(self,representation,intervention=None):
        action=self.bridge(representation,intervention) if self.reader_enabled else representation.original_action_hidden
        return self.baseline.build_planner_condition({"action_queries":action,"W":representation.dino_hidden})

    @torch.no_grad()
    def predict_action(self,obs:AllowedObservation,initial_noise=None):
        if self.training:raise RuntimeError("Deployment requires eval mode")
        cfg=self.baseline.config.framework.action_model
        if int(cfg.num_inference_timesteps)!=10:raise ValueError("This campaign preserves the original 10-step sampler")
        representation=self.encode_observation(obs)
        condition=self.action_condition(representation)
        with self.baseline.amp():
            action=self.baseline.action_model.predict_action(condition,initial_noise=initial_noise)
        from starVLA.dataloader.foresight_dataset import decode_ego
        return decode_ego(action.float())

    def dino_prediction(self,representation):
        cfg=self.baseline.foresight_config
        if not cfg.enable_current_dino or not hasattr(self.baseline,"dino_head"):
            raise ValueError("Verified DINO head required; no automatic replacement")
        return self.baseline.dino_head(representation.dino_hidden,
                representation.dino_hidden.new_zeros(len(representation.dino_hidden)),(cfg.dino_height,cfg.dino_width))

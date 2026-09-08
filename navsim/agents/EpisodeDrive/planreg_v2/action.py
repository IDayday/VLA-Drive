"""V2 shared normalized trajectory head; fixed-source scorer remains untouched."""
from types import SimpleNamespace
import torch
from torch import nn
from .normalizers import EgoStateNormalizer
from ..transformer_decoder import TransformerDecoder, TransformerDecoderScorer
from ..layers.utils.mlp import MLP
from ..score_module.scorer import Scorer, aggregate_drivor_pdm_score


def scorer_config():
    return SimpleNamespace(ref_num=4, scorer_ref_num=4, tf_d_model=256, tf_d_ffn=1024,
        refiner_num_heads=1, refiner_ls_values=0., proposal_num=64, num_poses=8,
        b2d=False, double_score=False, agent_pred=False, area_pred=False, bev_map=False,
        bev_agent=False, one_token_per_traj=True, noc=1., dac=1., ddc=0., ttc=5., ep=5., comfort=2.)


class V2ActionDecoder(nn.Module):
    def __init__(self, normalizer, ego_scales=None):
        super().__init__()
        self.config = scorer_config()
        self.normalizer = normalizer
        self.ego_normalizer = EgoStateNormalizer() if ego_scales is None else EgoStateNormalizer(ego_scales)
        self.hist_encoding = nn.Linear(11,256)
        self.init_feature = nn.Embedding(64,256)
        self.attention = TransformerDecoder(.1,.2,self.config)
        self.trajectory_head = MLP(256,1024,24)  # ONE instance, all four stages.
        self.pos_embed = nn.Sequential(nn.Linear(24,1024),nn.ReLU(),nn.Linear(1024,256))
        self.scorer_attention = TransformerDecoderScorer(4,256,.1,.2,self.config)
        self.scorer = Scorer(self.config)

    def score(self, proposals, memory, valid, ego):
        # Physical 8x3 flatten after detach; no generator coordinates in this graph.
        embedded = self.pos_embed(proposals.detach().flatten(-2))
        # DrivoR@fc6e5aa: ego is injected AFTER the independent scoring decoder.
        padding = None if valid is None else ~valid
        hidden = self.scorer_attention(embedded, memory, memory_key_padding_mask=padding)
        hidden = hidden + ego[:,None]
        result = self.scorer(proposals.detach(),hidden)
        log_score = aggregate_drivor_pdm_score(result[0],self.config)
        return result[0], log_score

    def forward(self, memory, valid, status):
        ego = self.hist_encoding(torch.cat((status.new_zeros(len(status),3),self.ego_normalizer(status)),-1))
        queries = self.init_feature.weight[None].expand(len(status),-1,-1)+ego[:,None]
        hidden = self.attention(queries,memory,memory_key_padding_mask=~valid)
        normalized = self.trajectory_head(hidden).reshape(4,len(status),64,8,3)
        physical = self.normalizer.inverse(normalized)
        logits, scores = self.score(physical[-1],memory,valid,ego)
        index = scores.argmax(-1)
        selected = physical[-1][torch.arange(len(status),device=status.device),index]
        return dict(trajectory=selected, proposals=physical[-1], stage_proposals=physical,
                    normalized_proposals=normalized, pred_logit=logits, log_pdm_score=scores,
                    selected_indices=index, scene_features=memory)

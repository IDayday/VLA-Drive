"""Reproducible public Qwen + RANDOM original DiT. Never loads a driving checkpoint."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf
from starVLA.cache.navsim_feature_cache import ROBOT_HISTORY_TOKEN,RGB_QUERY_TOKENS,GS_QUERY_TOKENS,REWARD_QUERY_TOKENS,action_query_tokens
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead,MLP


PUBLIC_REPO='Qwen/Qwen3-VL-2B-Instruct'
PUBLIC_REVISION='89644892e4d85e24eaac8bacfd4f463576704203'
PUBLIC_WEIGHTS_SHA256='7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verify_public_weights(path,provenance):
    """Check actual bytes before model loading, including processor/tokenizer identity."""
    if provenance['repo_id']!=PUBLIC_REPO or provenance['revision']!=PUBLIC_REVISION:
        raise ValueError('Unexpected public base revision')
    for row in provenance['files']:
        file=Path(path)/row['name']
        if not file.is_file() or sha256(file)!=row['sha256']:
            raise ValueError('Public Qwen file mismatch: '+row['name'])
    weight=next((r for r in provenance['files'] if r['name']=='model.safetensors'),None)
    if weight is None or weight['sha256']!=PUBLIC_WEIGHTS_SHA256:raise ValueError('Unverified public weights')
    return True


class PublicQwenBaseline(nn.Module):
    """Same original 24-layer FM DiT and action contract, public-only initialization.

    Qwen is frozen; new history projector, special embeddings, DiT and optional
    world Reader/heads start from declared seeds. No hidden legacy auxiliary models.
    """
    def __init__(self, public_path, config, provenance, seed=42, initialize_head=True):
        super().__init__()
        verify_public_weights(public_path,provenance)
        from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface
        self.config=OmegaConf.create(OmegaConf.to_container(config,resolve=True))
        self.config.framework.qwenvl.base_vlm=str(public_path)
        self.qwen_vl_interface=_QWen3_VL_Interface(self.config)
        self.qwen_vl_interface.requires_grad_(False)
        self.robot_history_token=ROBOT_HISTORY_TOKEN
        self.rgb_query_tokens=list(RGB_QUERY_TOKENS);self.gs_query_tokens=list(GS_QUERY_TOKENS)
        self.act_query_tokens=list(action_query_tokens(8));self.reward_query_tokens=list(REWARD_QUERY_TOKENS)
        self.w_depth=0;self.act_tok=8
        tokenizer=self.qwen_vl_interface.processor.tokenizer;model=self.qwen_vl_interface.model
        tokens=[self.robot_history_token,*self.rgb_query_tokens,*self.gs_query_tokens,*self.act_query_tokens,*self.reward_query_tokens]
        if any(t in tokenizer.get_vocab() for t in tokens):raise ValueError('Expected pristine public tokenizer without driving tokens')
        tokenizer.add_special_tokens({'additional_special_tokens':tokens},replace_additional_special_tokens=False)
        size=max(len(tokenizer),model.get_input_embeddings().num_embeddings)
        model.resize_token_embeddings(size,mean_resizing=False)
        dim=model.get_input_embeddings().embedding_dim
        ids=tokenizer.convert_tokens_to_ids(tokens)
        if len(set(ids))!=len(ids):raise ValueError('New driving token identities collide')
        generator=torch.Generator(device='cpu').manual_seed(seed+11)
        initial=torch.randn(len(ids),dim,generator=generator)*.02
        with torch.no_grad():
            model.get_input_embeddings().weight[ids]=initial.to(model.device,dtype=model.get_input_embeddings().weight.dtype)
        model.requires_grad_(False)
        # Separate declared initialization stream, independent of public model loading.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+23)
            self.action_input_model=MLP(4,dim,dim)
            self.action_model=FlowmatchingActionHead(self.config) if initialize_head else None
        self.action_input_model=self.action_input_model.cuda().float()
        if self.action_model is not None:self.action_model=self.action_model.cuda().float()
        self.public_origin={'schema_version':2,'public_repo':PUBLIC_REPO,'public_revision':PUBLIC_REVISION,
                            'public_weights_sha256':PUBLIC_WEIGHTS_SHA256,'initialization_seed':seed,
                            'token_ids':dict(zip(tokens,ids)),'added_embedding_seed':seed+11,'new_module_seed':seed+23,
                            'private_driving_weights_loaded':False,'action_head_initialization':'random_original_architecture',
                            'public_source_files':provenance['files']}

    def native_conditions(self,examples):
        from starVLA.model.modules.structured_world.policy import StructuredWorldPolicy
        proxy=SimpleNamespace(baseline=self,world_config={'vision_trainable':False,'hidden_mode':'prenorm'})
        values=[]
        for example in examples:
            current={k:example[k] for k in ('image','lang','state','token') if k in example}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                value,_=StructuredWorldPolicy.encode_conditions(proxy,[current],include_world=False)
            values.append(value)
        return torch.cat(values,0)

    @torch.no_grad()
    def predict_action_infer_1d(self,examples,initial_noise=None):
        conditions=self.native_conditions(examples)
        if initial_noise is None:
            initial_noise=torch.randn(len(examples),8,4,device=conditions.device,dtype=conditions.dtype)
        with torch.autocast('cuda',enabled=False):
            actions=self.action_model.predict_action(conditions.float(),initial_noise=initial_noise.float())
        return {'normalized_actions':actions.cpu().numpy()}


"""Stage1-only extraction. Current/future heads have distinct strict receipts."""
import json
import sys
from pathlib import Path
import torch
from starVLA.model.modules.decodable_st.contracts import TemporalContract
from starVLA.model.modules.decodable_st.frozen_decoders import FrozenSGDriveDecoder
from .source_registry import load_exact_subset

SGDRIVE_SOURCE = "fcda385371f8f40af5b8969ca5090be06f421358"


def load_native_stage1(source_root,weight_dir,device="cuda",dtype=torch.float32):
    from .source_registry import verify_source
    verify_source(source_root,SGDRIVE_SOURCE)
    sys.path.insert(0,str(Path(source_root)/"internvl_chat"))
    from internvl.model.internvl_chat.configuration_internvl_chat_wm import InternVLChatConfig_WM
    from internvl.model.internvl_chat.modeling_internvl_chat_wm import InternVLChatModel_WM
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    config=InternVLChatConfig_WM.from_pretrained(str(weight_dir),local_files_only=True)
    config.llm_config._attn_implementation="sdpa"
    config.llm_config._attn_implementation_internal="sdpa"
    model=InternVLChatModel_WM(config,use_flash_attn=False)
    receipt=model.load_state_dict(load_file(str(Path(weight_dir)/"model.safetensors")),strict=True)
    model.to(device=device,dtype=dtype).eval().requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(str(weight_dir),use_fast=False,trust_remote_code=False,local_files_only=True)
    for field,token in (("world_token_id","<WORLD TOKEN>"),("dream_token_id","<DREAM TOKEN>"),("img_context_token_id","<IMG_CONTEXT>")):
        index=tokenizer.convert_tokens_to_ids(token)
        if index==tokenizer.unk_token_id:raise ValueError(f"Missing native special token {token}")
        setattr(model,field,index)
    return model,tokenizer,{"missing":list(receipt.missing_keys),"unexpected":list(receipt.unexpected_keys),
                           "parameter_dtype":str(dtype),"attention":"native sdpa"}


def current_prefix(model,tokenizer,*,navigation,num_patches,history_text=""):
    """No answer slots, sample IDs, future QA, future trajectory or future states."""
    from internvl.conversation import get_conv_template
    question="".join("<image>" for _ in num_patches)+"\nCurrent front camera observations. Navigation: "+navigation
    if history_text:question+="\nPermitted ego history: "+history_text
    question+="\n"+"<world>"*model.NUM_WORLD_TOKEN+"\n"+"<dream>"*(model.occ_token_number+model.agent_token_number)
    template=get_conv_template(model.template)
    template.system_message=model.system_message
    template.append_message(template.roles[0],question);template.append_message(template.roles[1],None)
    query=template.get_prompt()
    for count in num_patches:
        query=query.replace("<image>","<img>"+"<IMG_CONTEXT>"*(model.num_image_token*count)+"</img>",1)
    query=query.replace("<world>"*model.NUM_WORLD_TOKEN,"<world>"+"<WORLD TOKEN>"*model.NUM_WORLD_TOKEN+"</world>",1)
    n=model.occ_token_number+model.agent_token_number
    query=query.replace("<dream>"*n,"<dream>"+"<DREAM TOKEN>"*n+"</dream>",1)
    encoded=tokenizer(query,return_tensors="pt")
    return encoded,query


def load_sgdrive_decoder(weights, config, *, future):
    from third_party.decodable_st.sgdrive.point_decoder import Decoder2D
    hidden = int(config["llm_config"]["hidden_size"])
    count = int(config.get("occ_token_number",config.get("occ_token_numbers",0)))
    if count != 625:
        raise ValueError("Unverified SGDrive decoder geometry")
    decoder = Decoder2D(llm_hidden_size=hidden,ch=128,out_ch=32,ch_mult=(1,2,4),num_res_blocks=2,
                        attn_resolutions=(50,),dropout=0.,resamp_with_conv=True,in_channels=64,
                        resolution=100,z_channels=256,give_pre_end=False)
    prefix = "_occ_head_dream." if future else "_occ_head."
    receipt = load_exact_subset(decoder,weights,prefix)
    contract = TemporalContract(hidden,count,(100,100),16,None,"native_lidar_frame_unverified",
                                "binary_point_hits_with_GT_trajectory_corridor_zeroing","T-native-v1")
    return FrozenSGDriveDecoder(decoder,contract),receipt


class SGDriveTeacherAdapter:
    """Wrap a strictly loaded native model; prefix must be constructed from current inputs.

    The native cache_forward returns LM hidden states, not a driving future field.
    Unknown horizon stays unknown; callers cannot promote this to T-driving.
    """
    def __init__(self, native_model, current_decoder, future_decoder):
        self.native = native_model.eval().requires_grad_(False)
        self.current_decoder,self.future_decoder = current_decoder,future_decoder

    def decode_hidden(self,current_hidden,future_hidden):
        return {"current_hidden":current_hidden,"future_hidden":future_hidden,
                "current_logits":self.current_decoder(current_hidden),
                "future_logits":self.future_decoder(future_hidden),
                "native_horizon_seconds":None,"native_coordinate_system":"native_lidar_frame_unverified",
                "native_target_semantics":"local binary point-hit occupancy; corridor depends on logged ego future",
                "validity_metadata":{"driving_verified":False,"outside_corridor_is_free":False}}

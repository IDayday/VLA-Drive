"""Stage1-only extraction. Current/future heads have distinct strict receipts."""
import json
import torch
from starVLA.model.modules.decodable_st.contracts import TemporalContract
from starVLA.model.modules.decodable_st.frozen_decoders import FrozenSGDriveDecoder
from .source_registry import load_exact_subset

SGDRIVE_SOURCE = "fcda385371f8f40af5b8969ca5090be06f421358"


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

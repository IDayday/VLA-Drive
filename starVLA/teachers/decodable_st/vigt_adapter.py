from dataclasses import asdict
import sys
from pathlib import Path
import torch
from starVLA.model.modules.decodable_st.contracts import SpatialContract
from starVLA.model.modules.decodable_st.frozen_decoders import FrozenViGTDecoder
from .source_registry import load_exact_subset, verify_source


VIGT_SOURCE = "a0bab2af48c6485774778fec33d68364c70d5618"


def load_vigt_decoder(weights, config):
    from third_party.decodable_st.vigt.modules.implicit_decoder import ImplicitDecoder
    required = ("implicit_f_channels","implicit_width","implicit_num_blocks","implicit_sampling_mode",
                "implicit_n_input_dims","roi_min","roi_max","dpt_output_shape")
    if any(k not in config for k in required):
        raise ValueError("Explicit, source-audited ViGT config required")
    decoder = ImplicitDecoder(f_channels=config["implicit_f_channels"],width=config["implicit_width"],
        num_blocks=config["implicit_num_blocks"],sampling_mode=config["implicit_sampling_mode"],
        n_input_dims=config["implicit_n_input_dims"],roi_min=config["roi_min"],roi_max=config["roi_max"],
        query_encoding=config.get("query_encoding",{"otype":"Identity"}))
    receipt = load_exact_subset(decoder,weights,"model.decoder.")
    contract = SpatialContract(config["implicit_f_channels"],*config["dpt_output_shape"],
                               tuple(config["roi_min"]),tuple(config["roi_max"]))
    return FrozenViGTDecoder(decoder,contract), receipt


class ViGTTeacherAdapter:
    def __init__(self, source_root, weights, config, device="cpu"):
        verify_source(source_root,VIGT_SOURCE)
        sys.path.insert(0,str(Path(source_root)))
        from vigt.vigt_hf import ViGTForInference,ViGTHFConfig
        from safetensors.torch import load_file
        self.native = ViGTForInference(ViGTHFConfig(**config))
        incompatible = self.native.load_state_dict(load_file(str(weights)),strict=True)
        self.native.to(device).eval().requires_grad_(False)
        self.load_receipt = {"missing":list(incompatible.missing_keys),"unexpected":list(incompatible.unexpected_keys),
                             "config":asdict(self.native.config)}

    def encode(self, samples):
        return self.native.model.encode(samples)

    def decode(self, context, xyz):
        decoder = self.native.model.decoder
        good = torch.isfinite(xyz).all(-1) & (xyz >= decoder.roi_min).all(-1) & (xyz < decoder.roi_max).all(-1)
        if not bool(good.all()): raise ValueError("ROI-external query rejected before native clamp")
        return self.native.model.decode(context,xyz)

    def roundtrip(self,samples,xyz):
        with torch.no_grad():
            whole = self.native(samples,xyz)
            context = self.encode(samples)
            split = self.decode(context,xyz)
        torch.testing.assert_close(whole,split,rtol=1e-4,atol=1e-5)
        return context,split,{"max_abs":(whole-split).abs().max().item(),"context_shape":list(context.shape),"logits_shape":list(split.shape)}

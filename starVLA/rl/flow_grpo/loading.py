from pathlib import Path
import hashlib
import torch
from .contracts import inherit_freezing


def weight_path(checkpoint):
    root = Path(checkpoint)
    if root.is_file():
        return root
    for rel in ("pytorch_model.pt", "final_model/pytorch_model.pt"):
        p = root / rel
        if p.is_file():
            return p
    raise FileNotFoundError(f"no complete SFT model at {root}")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_policy(cfg, sft, accelerator=None):
    from starVLA.model.framework.QwenOFT import Qwenvl_OFT

    if accelerator is None:
        from types import SimpleNamespace

        accelerator = SimpleNamespace(
            process_index=0, device=torch.device("cuda", torch.cuda.current_device())
        )
    checkpoint_path = weight_path(cfg["sft_checkpoint"])
    checkpoint_sha = file_sha(checkpoint_path)
    if checkpoint_sha != cfg["checkpoint_contract"]["sha256"]:
        raise ValueError("SFT checkpoint SHA does not match audited contract")
    model = Qwenvl_OFT(sft, accelerator=accelerator)
    model._flow_source_sha256 = checkpoint_sha
    with torch.serialization.safe_globals([set]):
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    state = state.get("module", state)
    model.load_state_dict(state, strict=True)
    inherit_freezing(model, sft)
    return model


def enable_checkpointing(policy, enabled):
    # Non-reentrant wrappers work in eval-with-grad and preserve no-dropout mode.
    # HF's stock language checkpointing is train-mode gated, so use explicit
    # per-block wrappers, retaining every original parameter/state_dict name.
    import types
    from torch.utils.checkpoint import checkpoint

    def install(module):
        if hasattr(module, "_flow_original_forward"):
            module._flow_checkpointing = enabled
            return
        module._flow_original_forward = module.forward
        module._flow_checkpointing = enabled

        def wrapped(self, *args, **kwargs):
            if self._flow_checkpointing and torch.is_grad_enabled():
                return checkpoint(
                    self._flow_original_forward, *args, use_reentrant=False, **kwargs
                )
            return self._flow_original_forward(*args, **kwargs)

        module.forward = types.MethodType(wrapped, module)

    for layer in policy.qwen_vl_interface.model.model.language_model.layers:
        install(layer)
    if hasattr(policy, "rgb_model"):
        policy.rgb_model.transformer3d.enable_gradient_checkpointing() if enabled else policy.rgb_model.transformer3d.disable_gradient_checkpointing()
    if hasattr(policy, "gs_model"):
        install(policy.gs_model.dit)

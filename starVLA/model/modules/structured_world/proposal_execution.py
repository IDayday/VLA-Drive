"""Opt-in, instance-local K/V Linear reuse for ONE native eval proposal.

No attention processor/operator is replaced. No training forward is adapted.
Install disabled by default, outside numerical/profile probes. A used scope
requires outer no_grad and every head submodule in eval; it always restores its
temporary Linear forwards, even on exceptions. GPU acceptance remains pending.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import inspect
from pathlib import Path
import textwrap
from types import MethodType
import threading

import diffusers
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
import torch
from torch import nn
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT

_STATE = "_structured_world_proposal_kv_v1"
_SOURCES = (
    (FlowmatchingActionHead, FlowmatchingActionHead.predict_action,
     "b67edc82a2ad110ef9feec68b5a457ea2371784f54b5ba75e2b32a18dce42999",
     "fffe39195d71da02b9737072b6b8bc1edc08b4ab8e3312006ed7f589a36ff259"),
    (DiT, DiT.forward,
     "3386dbf472a20b6b0e938b5eb5e12d760358f6d7f699d42a01f17e6e7ad612e2",
     "09b6999219e708494517052980917f0f5884c7f891680cfc8ba8eb9308b1d443"),
    (AttnProcessor2_0, AttnProcessor2_0.__call__,
     "d56adb29122171c897568ae0235dba6c5409b9a4176de6196d563c6f56c9df90",
     "bff040a099598b093263b84b78d4c53f83cb1e2be2af1c81209fdef6075fa7bc"),
    (Attention, Attention.forward,
     "d56adb29122171c897568ae0235dba6c5409b9a4176de6196d563c6f56c9df90",
     "b8021dea011a83c3aa739de1cb8c6aad9d991129b32550aed26d40f2127973ec"),
    (nn.Linear, nn.Linear.forward,
     "d7f5cadf773deb5ab4bcc9f57414e8239b2859c70d2aa6f141576de1ec918823",
     "4fed9af0cf2326f84d7fa1fd675941f3c3262378ee94d927c5a4f5695990cf5f"),
)


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _attest_sources():
    if diffusers.__version__ != "0.35.2" or torch.__version__.split("+")[0] != "2.5.1":
        raise ValueError("Proposal K/V candidate requires audited Diffusers 0.35.2 / Torch 2.5.1")
    records = []
    for cls, method, module_hash, function_hash in _SOURCES:
        if getattr(cls, method.__name__) is not method:
            raise ValueError("Proposal class method already has an adapter: " + cls.__name__)
        path = Path(inspect.getsourcefile(inspect.unwrap(method))).resolve()
        actual = _hash(path.read_bytes())
        source_hash = _hash(textwrap.dedent(inspect.getsource(method)).encode())
        if actual != module_hash or source_hash != function_hash:
            raise ValueError("Unaudited proposal source: " + str(path))
        records.append({"class": cls.__module__ + "." + cls.__name__, "source_file": str(path),
                        "source_sha256": actual, "method_sha256": source_hash})
    return records


def _native_linear(layer):
    if (type(layer) is not nn.Linear or "forward" in layer.__dict__ or
            layer.forward.__func__ is not nn.Linear.forward or
            layer._forward_hooks or layer._forward_pre_hooks or layer._backward_hooks):
        raise ValueError("Proposal K/V requires an unwrapped native Linear without hooks")


class _Projection:
    def __init__(self, layer, name, thread_id):
        self.layer, self.name, self.thread_id = layer, name, thread_id
        self.original = layer.forward
        self.input = self.output = self.key = None
        self.output_version = None
        self.hits = self.misses = self.bypasses = 0

    def forward(self, owner, value):
        if (threading.get_ident() != self.thread_id or torch.is_grad_enabled() or
                owner.training or torch.is_inference_mode_enabled()):
            self.bypasses += 1
            return self.original(value)
        # CPU-side identity/version and autocast metadata only: no .item(),
        # tensor hashes, GPU comparisons or synchronizing device transfers.
        device = value.device.type
        key = (value._version, value.dtype, value.device,
               torch.is_autocast_enabled(device), torch.get_autocast_dtype(device),
               tuple((id(p), p._version, p.dtype, p.device) for p in owner.parameters()))
        if (value is self.input and key == self.key and self.output is not None and
                self.output._version == self.output_version):
            self.hits += 1
            return self.output
        self.misses += 1
        result = self.original(value)
        self.input, self.key, self.output = value, key, result
        self.output_version = result._version
        return result

    def clear(self):
        self.input = self.output = self.key = None
        self.output_version = None


def _predict(state, owner, *args, **kwargs):
    reason = ("disabled" if not state["enabled"] else "grad_enabled" if torch.is_grad_enabled()
              else "inference_mode" if torch.is_inference_mode_enabled()
              else "training_submodule" if any(m.training for m in owner.modules()) else None)
    if reason:
        state["last_scope"] = {"used": False, "reason": reason}
        return state["original"](*args, **kwargs)
    if state["active"]:
        raise ValueError("Nested/concurrent proposal K/V scopes are not supported")
    for _, layer in state["targets"]:
        _native_linear(layer)
    state["active"] = True
    projections = [_Projection(layer, name, threading.get_ident()) for name, layer in state["targets"]]
    bindings = []
    succeeded = False
    try:
        for projection in projections:
            # Bind only the native Linear's arithmetic entry; Module.__call__,
            # original Diffusers processor, norms, masks and SDPA remain intact.
            function = projection.forward
            projection.layer.forward = MethodType(function, projection.layer)
            bindings.append((projection.layer, projection.layer.forward))
        result = state["original"](*args, **kwargs)  # exactly one original ten-step call
        succeeded = True
        return result
    finally:
        state["last_scope"] = {
            "used": True, "succeeded": succeeded,
            "projection_calls_saved": sum(p.hits for p in projections),
            "projection_calls_executed": sum(p.misses for p in projections),
            "projection_calls_bypassed": sum(p.bypasses for p in projections),
            "retained_KV_bytes_at_scope_end": sum(p.output.numel()*p.output.element_size()
                for p in projections if p.output is not None),
            "per_projection": [{"name": p.name, "hits": p.hits, "misses": p.misses} for p in projections],
        }
        for layer, bound in bindings:
            if layer.__dict__.get("forward") is bound:
                delattr(layer, "forward")
        for projection in projections: projection.clear()
        state["active"] = False


def install_proposal_kv_cache(head, *, enabled=False):
    """Install on one actual FlowmatchingActionHead; default stays disabled.

    Enabled calls still bypass cache unless called in outer no_grad/eval. The
    native @no_grad method, qwen_proj, ten Euler steps and RNG are unchanged.
    Unknown sources/processors or existing adapters fail before mutation.
    """
    state = head.__dict__.get(_STATE)
    if state is not None:
        _check_installed(head, state)
        if state["active"]: raise ValueError("Cannot configure an active proposal scope")
        state["enabled"] = bool(enabled)
        return proposal_kv_cache_status(head)
    if type(head) is not FlowmatchingActionHead or type(head.model) is not DiT:
        raise ValueError("Proposal K/V requires the actual registered GR00T/DiT classes")
    if "predict_action" in head.__dict__ or head.predict_action.__func__ is not FlowmatchingActionHead.predict_action:
        raise ValueError("Proposal method already has an adapter")
    if "forward" in head.model.__dict__ or head.num_inference_timesteps != 10:
        raise ValueError("Proposal K/V requires native DiT and original ten steps")
    sources = _attest_sources()
    targets = []
    for index, block in enumerate(head.model.transformer_blocks):
        attn = block.attn1
        if type(attn) is not Attention or type(attn.processor) is not AttnProcessor2_0:
            raise ValueError("Proposal K/V requires native Attention / AttnProcessor2_0")
        if "forward" in attn.__dict__ or "forward" in block.__dict__:
            raise ValueError("Attention/block already has an instance forward adapter")
        if block.cross_attention_dim is not None:
            for name in ("to_k", "to_v"):
                layer = getattr(attn, name)
                _native_linear(layer)
                targets.append((f"transformer_blocks.{index}.attn1.{name}", layer))
    if not targets or len({id(layer) for _, layer in targets}) != len(targets):
        raise ValueError("Expected independent native cross-attention K/V projections")
    state = {"original": head.predict_action, "targets": targets, "enabled": bool(enabled),
             "active": False, "last_scope": None, "provenance": {
                 "schema": "proposal_KV_native_Linear_instance_adapter_v1", "sources": sources,
                 "adapter_file": str(Path(__file__).resolve()), "adapter_sha256": _hash(Path(__file__).read_bytes()),
                 "cross_attention_layers": len(targets)//2, "default_enabled": False,
                 "processor_norm_cross_head_norm_masks_SDPA_Euler_rng": "native and unchanged",
                 "scope": "one original predict_action call; outer no_grad and all eval; inference_mode bypass",
                 "GPU_numerical_and_performance_acceptance": "pending; do not enable by default"}}
    @functools.wraps(FlowmatchingActionHead.predict_action)
    def wrapped(owner, *args, **kwargs):
        return _predict(state, owner, *args, **kwargs)
    state["function"] = wrapped
    head.predict_action = MethodType(wrapped, head)
    setattr(head, _STATE, state)
    return proposal_kv_cache_status(head)


def _check_installed(head, state):
    bound = head.__dict__.get("predict_action")
    if getattr(bound, "__self__", None) is not head or getattr(bound, "__func__", None) is not state["function"]:
        raise ValueError("Installed proposal adapter was replaced; refusing to overwrite it")


def proposal_kv_cache_status(head):
    state = head.__dict__.get(_STATE)
    if state is None: return {"installed": False, "enabled": False}
    _check_installed(head, state)
    return copy.deepcopy({"installed": True, "enabled": state["enabled"], "active": state["active"],
                          "provenance": state["provenance"], "last_scope": state["last_scope"]})


def uninstall_proposal_kv_cache(head):
    state = head.__dict__.get(_STATE)
    if state is None: return False
    _check_installed(head, state)
    if state["active"]: raise ValueError("Cannot remove an active proposal scope")
    delattr(head, "predict_action")
    delattr(head, _STATE)
    return True

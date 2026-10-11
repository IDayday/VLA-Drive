"""Instance-local Qwen3-VL vision split metadata adapter, Transformers 4.57.0.

Call install_qwen_vision_lengths(model.qwen_vl_interface.model.model.visual)
before workers/forwards start. Only audited forward source is transformed;
attention operators, precision, RNG, weights and buffers stay upstream-owned.
See qwen_execution.README.md for source hashes and the exact patch.
"""
from __future__ import annotations

import copy
import difflib
import functools
import hashlib
import inspect
from pathlib import Path
import textwrap
from types import MethodType

import transformers


_STATE = "_structured_world_qwen_lengths_v1"
_METADATA = "_structured_world_lengths_cpu"
_MODULE_HASHES = {
    "starVLA.model.modules.vlm.qwen3_vl.modeling_qwen3_vl":
        "0c8cb085e0da9f2a7ca1fe3c3d9921a0db9ef3d646980a682821644e5fea7661",
    "transformers.models.qwen3_vl.modeling_qwen3_vl":
        "dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da",
}
_FORWARD_HASHES = {
    "Qwen3VLVisionModel": "967c72a546faa603d7172adfa61d20ac6abe0f00ec03feb61d959dc723621aea",
    "Qwen3VLVisionAttention": "5a41ddeae90c047846ecdc78bd220c88a11e8bc9bed27a3f831fc6ae3d6e00c5",
}


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _lengths_cpu(cu_seqlens):
    # The same int32 subtraction and values as the original per-layer path.
    # One host transfer per vision forward, no cross-forward metadata cache.
    return tuple((cu_seqlens[1:] - cu_seqlens[:-1]).tolist())


def _replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Audited Qwen source patch anchor changed")
    return source.replace(before, after, 1)


def _patch_source(source, kind):
    if kind == "Qwen3VLVisionModel":
        source = _replace_once(
            source,
            "    hidden_states = self.patch_embed(hidden_states)\n",
            "    if '_structured_world_lengths_cpu' in kwargs:\n"
            "        raise ValueError('Vision split metadata is adapter-owned')\n"
            "    hidden_states = self.patch_embed(hidden_states)\n",
        )
        source = _replace_once(
            source,
            "    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)\n",
            "    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)\n"
            "    lengths_cpu = (_structured_world_make_lengths_cpu(cu_seqlens)\n"
            "                   if self.config._attn_implementation != 'flash_attention_2' else None)\n",
        )
        return _replace_once(
            source,
            "            position_embeddings=position_embeddings,\n",
            "            position_embeddings=position_embeddings,\n"
            "            _structured_world_lengths_cpu=lengths_cpu,\n",
        )
    if kind == "Qwen3VLVisionAttention":
        source = _replace_once(
            source,
            "    seq_length = hidden_states.shape[0]\n",
            "    lengths_cpu = kwargs.pop('_structured_world_lengths_cpu', None)\n"
            "    seq_length = hidden_states.shape[0]\n",
        )
        source = _replace_once(
            source,
            "        lengths = cu_seqlens[1:] - cu_seqlens[:-1]\n",
            "        if lengths_cpu is None:\n"
            "            lengths_cpu = _structured_world_make_lengths_cpu(cu_seqlens)\n",
        )
        return _replace_once(
            source,
            "torch.split(tensor, lengths.tolist(), dim=2)",
            "torch.split(tensor, lengths_cpu, dim=2)",
        )
    raise ValueError("Unsupported Qwen forward kind")


def _prepare_forward(owner, kind):
    cls = type(owner)
    if cls.__name__ != kind or cls.__module__ not in _MODULE_HASHES:
        raise ValueError("Unregistered Qwen class: " + cls.__module__ + "." + cls.__name__)
    original = owner.forward
    if "forward" in owner.__dict__ or getattr(original, "__func__", None) is not cls.forward:
        raise ValueError("Qwen forward already has an instance adapter")
    path = Path(inspect.getsourcefile(cls.forward)).resolve()
    file_hash = _sha256(path.read_bytes())
    if file_hash != _MODULE_HASHES[cls.__module__]:
        raise ValueError("Unregistered Qwen source hash: " + str(path))
    source = textwrap.dedent(inspect.getsource(cls.forward))
    source_hash = _sha256(source.encode())
    if source_hash != _FORWARD_HASHES[kind]:
        raise ValueError("Unregistered Qwen forward source hash: " + kind)
    patched = _patch_source(source, kind)
    namespace = dict(cls.forward.__globals__)
    namespace["_structured_world_make_lengths_cpu"] = _lengths_cpu
    # No original globals or class methods are mutated. Only pinned, local
    # source can reach this compile; no model/data text is executed.
    exec(compile(patched, str(Path(__file__).resolve()) + ":" + kind, "exec"), namespace)
    function = functools.update_wrapper(namespace["forward"], cls.forward)
    record = {
        "class": cls.__module__ + "." + kind,
        "source_file": str(path),
        "source_file_sha256": file_hash,
        "original_forward_sha256": source_hash,
        "patched_forward_sha256": _sha256(patched.encode()),
        "patch": "".join(difflib.unified_diff(
            source.splitlines(keepends=True), patched.splitlines(keepends=True),
            fromfile=kind + "/original", tofile=kind + "/instance_adapter",
        )),
    }
    return owner, original, function, record


def _check_installed(state):
    for owner, original, function, _ in state["bindings"]:
        bound = owner.__dict__.get("forward")
        if getattr(bound, "__self__", None) is not owner or getattr(bound, "__func__", None) is not function:
            raise ValueError("Installed Qwen adapter was replaced; refusing to overwrite it")


def install_qwen_vision_lengths(vision):
    """Install on one actual vision instance; return JSON-compatible provenance.

    Unknown versions/source and pre-existing instance forward adapters fail
    before mutation. Repeated installation on the same instance is idempotent.
    Flash Attention 2 retains its original path and does no metadata transfer.
    Install before forwards start, outside torch.compile/JIT/export.
    """
    state = vision.__dict__.get(_STATE)
    if state is not None:
        _check_installed(state)
        return copy.deepcopy(state["provenance"])
    if transformers.__version__ != "4.57.0":
        raise ValueError("Qwen split adapter requires audited Transformers 4.57.0")
    bindings = [_prepare_forward(vision, "Qwen3VLVisionModel")]
    if not len(vision.blocks):
        raise ValueError("Qwen vision has no attention blocks")
    for block in vision.blocks:
        if type(block).__name__ != "Qwen3VLVisionBlock" or type(block).__module__ not in _MODULE_HASHES:
            raise ValueError("Unregistered Qwen vision block")
        if "forward" in block.__dict__:
            raise ValueError("Qwen vision block already has an instance adapter")
        bindings.append(_prepare_forward(block.attn, "Qwen3VLVisionAttention"))
    if len({id(owner) for owner, _, _, _ in bindings}) != len(bindings):
        raise ValueError("Shared attention instances require a separate audited adapter")
    records = {record["class"]: record for _, _, _, record in bindings}
    provenance = {
        "schema": "qwen_vision_lengths_instance_adapter_v1",
        "transformers_version": transformers.__version__,
        "adapter_file": str(Path(__file__).resolve()),
        "adapter_sha256": _sha256(Path(__file__).read_bytes()),
        "vision_class": records[bindings[0][3]["class"]]["class"],
        "attention_layers": len(vision.blocks),
        "upstream": list(records.values()),
        "metadata": "one int32-derived tuple per non-FA2 vision forward; reused by every Q/K/V split",
        "shared_classes_and_site_packages_modified": False,
        "attention_backend_precision_rng_parameters_buffers": "unchanged",
        "GPU_equivalence_and_speed_validation": "pending; no numerical acceptance implied",
    }
    try:
        for owner, _, function, _ in bindings:
            owner.forward = MethodType(function, owner)
        setattr(vision, _STATE, {"bindings": bindings, "provenance": provenance})
    except BaseException:
        for owner, _, function, _ in bindings:
            if getattr(owner.__dict__.get("forward"), "__func__", None) is function:
                delattr(owner, "forward")
        raise
    return copy.deepcopy(provenance)


def uninstall_qwen_vision_lengths(vision):
    """Remove this instance's adapter, retaining weights/state/backend/modes."""
    state = vision.__dict__.get(_STATE)
    if state is None:
        return False
    _check_installed(state)
    for owner, _, _, _ in state["bindings"]:
        delattr(owner, "forward")
    delattr(vision, _STATE)
    return True

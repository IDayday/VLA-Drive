# Qwen vision lengths adapter provenance

This isolated engineering adapter changes only how the existing vision attention
split sizes reach Python. It is not a numerical acceptance receipt or a measured
speedup. Original strict CUDA comparison failures remain valid and unmodified.

The inspected runtime is `/tmp/structured-world-round1/envs/qwen/bin/python`,
an overlay on `/root/miniconda3/envs/ddp/bin/python`: Torch `2.5.1+cu124`,
Transformers `4.57.0`. A tiny CPU instance constructed through the actual
`starVLA.model.modules.vlm.QWen3.Qwen3VLForConditionalGeneration` import has
repository-owned model, vision and attention classes. Direct Transformers vision
instances resolve to site-packages. Installation inspects the actual instance,
accepts these two audited implementations, and rejects all other versions/hashes.

| Implementation file | SHA-256 |
| --- | --- |
| `starVLA/model/modules/vlm/qwen3_vl/modeling_qwen3_vl.py` | `0c8cb085e0da9f2a7ca1fe3c3d9921a0db9ef3d646980a682821644e5fea7661` |
| `/root/miniconda3/envs/ddp/lib/python3.10/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py` | `dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da` |

The two dedented forward implementations are identical across these files:

| Forward | SHA-256 of `textwrap.dedent(inspect.getsource(forward))` |
| --- | --- |
| `Qwen3VLVisionModel.forward` | `967c72a546faa603d7172adfa61d20ac6abe0f00ec03feb61d959dc723621aea` |
| `Qwen3VLVisionAttention.forward` | `5a41ddeae90c047846ecdc78bd220c88a11e8bc9bed27a3f831fc6ae3d6e00c5` |

Forward source is attributed to the Qwen Team and HuggingFace, Apache-2.0, as
recorded in each original file. The adapter compiles only these hash-verified
methods into a private globals dictionary and binds them on the supplied
instances. It writes no source file and changes no shared class or registry.
The return receipt includes source paths, original/patched hashes, adapter hash,
and complete unified diffs for both methods.

```python
from starVLA.model.modules.structured_world.qwen_execution import (
    install_qwen_vision_lengths, uninstall_qwen_vision_lengths,
)

receipt = install_qwen_vision_lengths(model.qwen_vl_interface.model.model.visual)
# Record receipt in the engineering execution provenance, before first forward.
# An optional rollback on the same instance:
# uninstall_qwen_vision_lengths(model.qwen_vl_interface.model.model.visual)
```

Exact patch: after the unchanged padded `cu_seqlens` construction, derive
`tuple((cu_seqlens[1:] - cu_seqlens[:-1]).tolist())` once per non-FA2 vision
forward. Pass that tuple through every vision block as private call metadata.
Attention removes the private keyword before calling the backend and replaces
only the three `lengths.tolist()` arguments to Q/K/V `torch.split` with that same
tuple. It retains the original subtraction dtype and values. A standalone
attention call derives its own tuple once. FA2 retains its original branch and
creates no tuple. No metadata is retained between forwards.

Patch embeddings, interpolation, rotary/mRoPE, attention calls and their order,
deepstack outputs, backend, autocast, mode, parameters, buffers and state-dict
inventory stay unchanged. Install before concurrent forwards begin, outside
compile/JIT/export. Existing instance forward wrappers and unknown sources fail
before mutation; lifecycle integration must install on reconstructed models,
not serialize entire adapted Python module objects.

CPU validation uses tiny actual repository and Transformers modules, 3/6-camera
and unequal/temporal grids, real SDPA/eager, BF16 CPU autocast, backward/AdamW/RNG,
both reentrant and non-reentrant activation checkpoint replay, a serialized
state-dict/optimizer/RNG restore with an identical next update, full
language/deepstack/mRoPE, instance isolation, idempotence and source drift
refusal. The FA2 test is a CPU backend spy validating
unchanged routing/kwargs; it does not validate a real Flash CUDA kernel.

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /tmp/structured-world-round1/envs/qwen/bin/python -B \
  tools/structured_world/test_qwen_execution.py
```

Local validation on 2026-10-11: **14 tests passed** (`unittest` reported 1.262s;
this is test execution time, not a model speed measurement). FP32/BF16 CPU
outputs, gradients, AdamW moments/updates and post-update RNG compare bitwise.
The three-layer fixture measures lengths `.tolist()` calls falling from nine to
one and checks that all nine Q/K/V splits receive the same tuple object.

The measured source paths above describe the local overlay. Target hosts must
retain the installation receipt from their own actual vision instance; the
installer verifies their source hashes again. No extra CPU grid payload is
required: the adapter derives lengths from the existing actual `cu_seqlens`.

GPU/full-size native ZeRO update, numerical continuation, exact recovery and
end-to-end speed profiles require separately authorized idle resources. CPU
results do not replace those gates. No active training code is changed here.

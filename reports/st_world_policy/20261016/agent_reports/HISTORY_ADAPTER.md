# History adapter implementation handoff

Status: **NEEDS_INDEPENDENT_REVIEW**. This report is implementation and CPU
evidence, not a real-S0 GPU qualification or formal training approval.

Source baseline: `a286aa17a9d5018d62070d476a6f2c8dc8f5d955`.
Branch: `agent/st-history-adapter-20261016`.
Owned implementation: `starVLA/model/modules/st_world/history_adapter.py`.
Owned tests: `tests/st_world/test_history_adapter.py`.
The original Qwen, DDPForesight, DiT/Flow loss, trainer and checkpoint files
were not modified. No GPU workload, training or navtest was launched.

## Interface and boundaries

```python
encoder = build_history_encoder(parent, initialization_seed=seed + 3700)
encoded = encoder.encode(parent, validated_model_inputs)
# encoded: W, action_queries; same downstream planner/Flow interfaces

native = build_history_encoder(parent, initialization_seed=seed + 3700,
                               mode="native_reference")
```

Construct the encoder before stripping the original auxiliary DINO readout:
its actual width and attention-head count define the new reader. It stores no
reference to the parent, so registering it does not duplicate Base parameters.
It uses the existing original frozen visual and the parent AMP boundary. It
rejects trainable visual parameters; visual evaluation mode and `no_grad`
include its final and DeepStack mergers. It does not force FP32 or BF16.

Budget mode preserves the original three-current-image processor path, then
inserts 96 ordinary memory positions immediately before original ego/W/action.
The placeholder is a verified ordinary tokenizer ID; the tokenizer vocabulary
and Base embedding rows are unchanged. History uses actual -1/-0.5-second
images, the same visual frontend, final merged tokens, true merged-grid cell
centres, camera, physical time and current-relative observed ego motion.
Two latent attention/FFN layers produce 16 queries per time/view. Valid past
groups can attend to all valid past groups. Missing groups have masked zero
memory; an all-missing sentinel prevents an undefined attention softmax.
Zero output remains connected to all history parameters for empty-mask ranks.

The inserted positions are not image placeholders. No fake image grid is
created. Real current image masks and all three current DeepStack streams
retain their correspondence. MRoPE and all driving slot offsets are rebuilt
using the real current grid and attention mask. All-missing memory preserves
old active token IDs and MRoPE exactly. Nonempty history intentionally changes
the model function; it is not claimed equivalent to the original S0.

Native-reference mode uses actual available history images plus the original
three current views through the real Qwen processor, grids, MRoPE and all
DeepStack streams. Missing images remain absent, with explicit missing text.
Observed time/camera/ego metadata are supplied. It is a layout reference,
not a numerically equivalent implementation of the budget reader.

Budget mode has two visual calls when any history exists (current and past),
one when all history is missing, and one language call. Native-reference mode
has one batched visual call and one language call. Both process additional
historical pixels; one shared visual module does not imply unchanged cost.

## Design evidence and new adaptations

The implementation follows independently reviewed evidence in
`/mnt/project/st-world-policy-20261016/audits/evidence_wave2/HISTORY_BUDGET_ADAPTER_EVIDENCE.md`.
The latent Q / media-plus-latent K/V and residual FFN structure is based on
OpenFlamingo commit `655f693fbfa04cd6e9a987d960654624d48917cf`,
`helpers.py::PerceiverAttention/PerceiverResampler`. The code is an independent
PyTorch adaptation; no OpenFlamingo weights or extra encoder are loaded.

Two layers, 96 slots, all-past cross-view fusion, physical time, xy, camera,
observed ego and explicit missing masks are project adaptations. They are not
capabilities attributed to the original helper. Ego translation uses a fixed
10-metre scale; yaw is sin/cos. Image xy and ego metadata do not claim 3D
correspondence or a depth-based warp.

## CPU evidence

Each command used one CPU thread for OpenMP and MKL, with these explicit assets:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ST_QWEN_PROCESSOR=/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct
export ST_HISTORY_AUDIT_ROOT=/mnt/project/st-world-policy-20261016/audits/history_data/smoke_train64_v1
export ST_CURRENT_DATA_ROOT=/mnt/project/ddp-foresight-artifacts/20260928/student_train_v1
/root/miniconda3/envs/ddp/bin/python -m pytest -vv tests/st_world/test_history_adapter.py
/root/miniconda3/envs/navsim/bin/python -m pytest -vv tests/st_world/test_history_adapter.py
```

| Environment | Versions | Result |
|---|---|---|
| Production `ddp` | Python 3.10.20, torch 2.5.1+cu124, Transformers 4.57.0 | 16 passed, 0 skipped, 9.19 s |
| CPU `navsim` | Python 3.9.25, torch 2.5.1+cu124, Transformers 4.57.6 | 16 passed, 0 skipped, 9.25 s |

Both installed and repository-vendored Qwen `get_rope_index` were executed.
Both actual Qwen TextModel implementations were instantiated at tiny random
dimensions for real causal-attention and backward tests. All-missing outputs
were compared with predeclared `rtol=1e-5, atol=1e-6` for those real language
implementations. The simpler fixture backbone uses `1e-6/1e-7`. These are not
pretrained S0 numerical tolerances or GPU kernel claims.

Tests cover partial/all-missing history, invalid frame/time/gradient inputs,
left padding, batch regrouping/tail batches, labels kept out of model inputs,
all history parameter gradients, zero graph on all-missing inputs, frozen
visual buffers, ordinary-slot and DeepStack separation, strict state restore,
wrong execution-contract rejection and deployment strip retaining history.

The first two scenes of the fixed 64-scene training manifest were additionally
read with verified current-record and history/image hashes. They have six real
past and three current images each. The actual processor gave `[1,36,64]` per
image, hence 576 merged tokens per image: 1728 real current image tokens plus
96 memory positions per scene in budget mode, versus 5184 real image tokens in
native mode. This checks actual pixels and layouts with an explicit fixture
visual output, not the pretrained visual numerical function. The fixture uses
two W and two action placeholders; its total sequence lengths are not real-S0
sequence-length measurements.

Using the actual Base config's visual/LLM width 2048 and original reader width
512 with 8 heads, the new budget reader has **9,211,392 trainable parameters**.
This is a CPU parameter count, not a measured GPU latency or quality result.

Complete logs and version/method hashes are under
`/mnt/project/st-world-policy-20261016/audits/history_adapter/`:
`pytest_ddp.log`, `pytest_navsim.log`, `CPU_IDENTITY_ddp.json`,
`CPU_IDENTITY_navsim.json`, and `capture_cpu_identity.py`.

## Checkpoint and integration handoff

`encoder.migration` records dimensions, slot semantics, seed and real runtime
method identities. The Lead should include this dictionary in the existing
wrapper/run migration identity. A persistent 32-byte execution-contract
buffer survives DeepSpeed's named-buffer FP32 reconstruction; strict loading
compares its exact values before accepting weights. Arbitrary extra-state was
not used because the actual ZeRO-to-FP32 utility drops it. The test includes
FP32 reconstruction of all state tensors, exact subsequent output equality and
rejection of a different initialization identity.

The Lead owns real pretrained S0 strict loading, GPU numeric tolerance,
DeepSpeed optimizer/resume, full loss gradients, peak memory and latency.
Those qualifications remain **NOT_RUN_BY_THIS_AGENT**. The native reference
may need a smaller GPU batch due to its actual longer image sequence. No
scientific effect or formal readiness is inferred from these CPU tests.

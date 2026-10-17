# ST-TEMPORAL-SCALE-001: one-shot shared-subset scale measurement

Source baseline: `7c4ef900d58128d3084e9ed27269880ad8ca5ff8`.
Implementation owner: `/root/st_temporal_scale_calibration`.
Status: implemented CPU reference; **independent review and real GPU measurement pending**.

The original plan §6.3 already requires one fixed training-batch calibration of
new future/delta weights from loss scales and shared gradients. It retains
current weight 1 and the existing raw feature-difference objective. This tool
fills that missing diagnostic; it does not replace the training framework.

Evidence actually consulted before implementation:

- Existing `SPARSE_TIME_DELTA_ADAPTATION` and `DINO_PCA_NONWHITEN` cards in
  `reports/st_world_policy/20261016/evidence/DESIGN_EVIDENCE_CARDS.json` bind
  DINO-Foresight `d98f765d2b743ec90fb38bc657a39c07c7ca4ab0`,
  `src/dino_f.py::{calculate_loss,pca_transform}`. They explicitly identify
  this project's sparse MSE and raw delta as adaptations, not upstream methods.
- [GradNorm, PMLR 80, full paper §3.1–3.2, equations 1–2 and Algorithm 1](https://proceedings.mlr.press/v80/chen18a/chen18a.pdf)
  measures weighted task gradients on a shared parameter subset, often the last
  shared layer. Its actual algorithm learns weights repeatedly using task
  training rates and renormalizes their sum. That dynamic algorithm is **not**
  implemented here. The relevant reusable point is measuring task-separated
  shared-layer gradients rather than treating scalar loss size as influence.
- [PyTorch v2.5.1 autograd source](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/autograd/__init__.py),
  `grad`, returns requested input gradients without accumulating `.grad`.
  Multiple objectives retain one forward graph; no optimizer is constructed.
  The installed version is already the campaign's pinned execution version.

New project hypothesis, declared before seeing any real calibration result:

```
w_current = 1
w_i = min(L_current / L_i, ||g_current||_2 / ||g_i||_2), i ∈ {future, delta}
```

Here `g_i` concatenates only the declared final shared language block and W
query parameters. All four unweighted losses (including original ego FM) and
norms/cosines for each group and their union are retained. The formula bounds
each new objective's scalar contribution and gradient norm by current on this
one measured batch. It does not bound the sum of gradients, account for future
optimization dynamics, establish an optimum, or promise planning improvement.
It is deliberately called `WEIGHT_PROPOSAL`, never a GradNorm reproduction.
No epsilon, post-hoc clipping, guessed coefficient or dev-score tuning occurs.
Missing valid future/delta supervision or zero auxiliary gradient in either
selected group makes calibration unavailable. A disconnected individual
parameter remains explicitly listed; valid pathways are not invented.

The reference may be the campaign's verified learned common S0 with the same
registered initialization of new ST/history/readout modules. It may instead
be a strictly reconstructed formal ST native endpoint with matching structure.
Changing that reference requires a new explicit calibration identity. Learned
SMOKE/PROFILE checkpoints are rejected. The first real4 proposal is an
engineering diagnostic, not an automatically adopted scientific coefficient.

Risks and controls:

- This is a shared **subset**, not the entire LLM gradient. Exact names, shapes,
  dtypes and initial bytes are reported; separate language/W norms expose
  domination by one group.
- One graph per microbatch shares dropout, FM noise/time and patch/horizon
  draws across tasks. Full-batch valid counts remain those of the original
  loss implementation. No repeat8 multiplier is added to feature losses.
- BF16 execution and BF16 gradient accumulation match the official model
  path; FP64 is used only for reporting dot products/norms. The diagnostic has
  no Adam/ZeRO communication/update and claims no distributed parity.
- Every parameter and buffer, including nonpersistent buffers, is hashed
  before/after; any mutation fails. Existing `.grad` is rejected, not cleared.
  Python/NumPy/Torch/CUDA/dedicated-generator states are restored even on error.
- The tiny tensor tests prove analytic reductions and isolation only. They do
  not establish real VLA gradient scale, resource feasibility or learned gain.
- No PCA refit, target normalization change, teacher execution or new tensor
  cache is performed. Existing teacher contract/data and pixel checks remain.

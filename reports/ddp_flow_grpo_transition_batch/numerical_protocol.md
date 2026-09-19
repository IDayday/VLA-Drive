# Fixed saved-transition layout qualification

This protocol does not relax historical atol=2e-6/rtol=2e-3, does not reclassify
serial/chunk1/chunk2 BF16 failures, and does not assert cross-layout exact resume.
The new native layout explicitly batches B,G,K saved transitions. Sampling remains
candidate_chunk=1 with ten dependent time steps. The complete objective, original
SFT replay, independent reference and all819,503,620 action-head trainables remain.

1. **Packing/objective math:** real F checkpoint values are first represented in
   BF16 and then exactly promoted to FP32 action parameters. Compare serial and
   batched full G16/K10 objectives on the same two predeclared official-reward
   chains (original behavior positions1,2). All359 mathematical gradients must
   satisfy the UNCHANGED original tolerance. Every raw transition statistic is
   reported, including failed near-zero elementwise log-density comparisons.
   Those raw diagnostic failures are not rewritten or called exact equivalence.
2. **Actual BF16 gradient target:** a separate FP32-leaf oracle preserves the
   original timestep sinusoid's explicit BF16 input quantization. This prevents
   head.float() from changing the forward function while testing accumulation.
   All velocity/mean/std/elementwise log-prob tensors must be BITWISE identical
   between this oracle and BF16 flat160. Every one of359 actual BF16 leaf gradients
   must be BITWISE equal to the oracle gradient rounded once to BF16. This is a
   representable-result test at zero tolerance, not a larger rtol inferred from
   failure magnitudes. Both fixed scenes must pass. No training forward uses the
   diagnostic hook or FP32-leaf oracle.
3. **Native updates:** actual target-world Accelerate/DeepSpeed ZeRO-2 must execute
   the unmodified joint RL+reference+original SFT loss, FP32 partition accumulation
   and communication, and both inner epochs. The existing full-epoch gate checks
   initial ratios, official nonzero group advantages, immutable reference/prefix,
   actual observed dtype inventory and full parameter coverage. Actual Adam
   moments, FP32 masters and BF16 forward weights are independently reconstructed
   using the existing CUDA Adam oracle and its unchanged bounds.
4. **Repeat/resume/export:** continuous and resumed inner-epoch boundaries must
   have exactly equal complete weights, optimizer, scheduler/controller, RNG,
   pending fixed behavior and stream states. Original infer.VLAAgent predictions
   must agree exactly with exported training weights under the same noise seed.
5. **Publication:** batch_profile.py verifies content and SHA of indexed real
   artifacts, kernel sources, checkpoint, scene identities, dtype observations,
   parameter names and target world. full_epoch.py retains all native pilot gates.
   The epoch publication is last; an incomplete batch proof cannot launch formal
   RL. Candidate chunks>=2, smaller K, changed G, new precision or topology do not
   inherit this qualification.

A mathematical interpretation of the evidence: serial BF16 leaves round many
separate contributions; the flat batch computes the summed matrix gradient before
its final BF16 cast. Thus matching the old serial leaf sum is not an independent
accuracy oracle. Native ZeRO FP32 partitions cannot undo rounding that already
occurred inside backward. The current successful BF16 oracle checks ALL tensors,
not just a cosine similarity or a final .float() conversion.

PyTorch explicitly documents that batched and sliced floating-point computation
need not be bitwise identical, including for FP32/FP64. This motivates reporting
both representable-gradient accuracy and cross-layout differences, not suppressing
failures: https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/numerical_accuracy.rst

Flow-GRPO's official examples batch candidates and use model-specific denoising
budgets (e.g. SD3 train10/eval40 and FLUX6/28). This change retains DDP's ten steps
and all sixteen candidates to avoid confounding computational improvements with
changed exploration/discretization: https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/config/grpo.py

Neither numerical qualification nor throughput proves a PDMS/EPDMS gain. The
registered full-navtest, five-seed, paired-to-own-SFT evaluation remains required.

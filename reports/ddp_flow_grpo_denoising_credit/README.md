# F-only denoising credit and runtime work — 2026-09-18

**Status: bounded research, not a production release.** Two fresh F-only runs of
64 actual optimizer updates are running. No new claim of stable PDMS/EPDMS
improvement is made before their full fixed development evaluation completes.
The historical BF16 candidate chunk1/2 failure remains FAIL. Chunk1 is used;
this work does not qualify chunk2. U is not trained in this investigation.

## Implemented change and basis

`starVLA/rl/flow_grpo/credit.py` applies `gamma**(K-1-k)` to the GRPO term of each
saved transition. Gamma1 reproduces the original objective/gradient exactly.
Gamma0.6 discounts early denoising transitions, using raw weights (last weight1,
ten-step mean0.2484883456), without rescaling the reference or action-SFT terms.
Both current-policy encoding and the complete independent SFT reference remain
unchanged. This changes the relative aggregate RL weighting too; the experiment
does not isolate only the direction of temporal credit assignment.

The rationale is diffusion-step credit assignment in
[DPPO](https://arxiv.org/abs/2409.00588), inspected at
[`cc7234a`](https://github.com/irom-princeton/dppo/blob/cc7234ad7ff39a8f32de3af903606723a16f0648/model/diffusion/diffusion_ppo.py).
The inspected SimWAM (`68b426c162827cb7701396895dbb3572d29f3420`) and ReCogDrive
(`6b8d8f5e01346c71094651c81dcaf66405dbc04e`) sources also use denoising gamma0.6.
These provide a hypothesis and implementation reference, not evidence that this
DDP task must improve. The broader primary-source review and exploration
experiments remain in the adjacent frozen-improvement and correlated-exploration
reports. This is not a reproduction of their entire algorithms.

The fixed four-scene, two-sampler saved-bank probe collected every predeclared
scene, including one constant-reward scene. In the three nonconstant Flow groups,
the last step contributes about49–55% of summed individual condition-gradient
norms. Mean-one discount normalization would amplify the aggregate condition
gradient about3.9–4.2 times. The raw discount gives norm ratios0.977–1.049 and
cosines0.953–0.988 relative to uniform weighting. Raw discount was chosen before
the longer runs; mean-one was not trained. These are condition-Jacobian
diagnostics, not a substitute for full-parameter optimizer evidence.

## Unchanged contract and registered experiment

- Original F frozen-visual step100000 checkpoint; SHA256
  `9a26685aa3838a2e1b89ab2d92997664afde4b259fed6683851f42781ad16eb4`.
- All672 source nonvisual trainable tensors,2,233,120,260 parameters. Qwen language,
  history/projector and action DiT train; F's own visual and original tied
  embedding/lm_head stay frozen. No auxiliary/video/depth tasks or LoRA.
- G16,K10, correlated Flow SDE rho0.8, eta0.1; unchanged input/normalization,
  official v2 single-scene reward, no candidate replacement/selection.
- Global scene batch16:8GPUs × accumulation2 × scene microbatch1. Inner epochs2;
  original behavior chain, old logprob and advantages stay fixed across both.
- LR1e-6, PPO clip0.02, reference KL0.01, action SFT0.1, training seed42.
- Each arm starts from original F-SFT, with identical token/replay/noise ordering.
  Uniform gamma1 versus raw gamma0.6 is the algorithm difference.
- Each arm:64 optimizer updates,512 fresh scene rollouts,8192 candidates and1024
  replay exposures. **512 is a scene budget, not batch size.** Save every16;
  evaluate32 and64. Step64 is primary; step32 is diagnostic. No score-based early
  selection, seed selection or Navtest use.
- Development evaluation:1696 scenes/16logs, stable token+seed42 initial noise,
  original single-candidate ten-step ODE. Genuine v1 PDMS and official v2 one-stage
  EPDMS are evaluated separately, with whole-log bootstrap. Development may have
  appeared in source SFT; it is not an SFT-unseen set.

The complete specification and placements are in `research64_manifest.json`.
Uniform trains on `training-vla-zt3` GPUs0–7; discount on `training-vla-zt2` GPUs0–7.
Evaluation uses `training-rl-zt4` GPUs0–3 and `training-rl-zt2` GPUs0–3 respectively.
These are distinct server names; rl-zt4 keeps GPUs6–7 free, rl-zt2 uses at most4.
Only verified pressure-script process groups were stopped. The unrelated local
ReCogDrive inference job and other users' CPU workloads were preserved.

## Actual validation and failures

| Check | Result / scope |
|---|---|
| Mathematical/implementation regression |240 passed in `final_source_tests.log`; does not itself certify production training |
| Full CPU regression after bounded-budget/CLI changes |251 passed,4 skipped, exit0 in `research_full_tests.log` |
| CLI/budget targeted regression |59 passed then15 passed, exit0; overlapping test sets |
| Actual async evaluation controller fixture |2 passed, exit0; control flow, not real-model evaluation |
| Native BF16/ZeRO2 pilot |2 actual updates,8GPUs, exit0; exact fixed-chain inner-epoch reuse |
| RL-only actual optimizer gradients |672/672 present, finite, nonzero FP32; frozen gradients empty; official reward11/16 nonconstant groups |
| RL-only ratios |Before update all1; after update range0.974828–1.020146 |
| Full reference / own visual / frozen tensors |Complete tensor hashes unchanged in native pilot, RL-only and resumed runs |
| Inner-boundary exact resume |Pilot update1 resumed to2;26 checkpoint files including optimizer/RNG/pending state compared with zero tolerance, PASS |
| Disable activation checkpointing |FAIL: actual ZeRO2 first update OOM;75.68GiB PyTorch allocation on79.21GiB device. Kept checkpointing enabled |
| Checkpointing on/off update comparison |NOT_RUN because off run failed before a complete update |
| Initial mistyped repeated CLI overrides |Failed attempt preserved; parser fixed with extend; new attempts use resolved configs |
| Historical candidate chunk1/2 BF16 equivalence |FAIL, unchanged |
| New64-update PDMS/EPDMS outcomes |RUNNING; not yet evidence of improvement |

Native execution digest:
`deb8ce31ca7d9e43b2538d1975f804c7338eaaec3a80792df493a40e2ee8b3b1`.
Training/controller code was published through commit
`6c4472c4b923f8fa0b6f4003ed7105b2b935092b` before the runs. Reports added later do not
change native source identity. Torch2.5.1+cu124, Transformers4.57.0,
Accelerate1.5.2, DeepSpeed0.16.9, flash-attn2.7.4.post1. No dependency upgrades.
The observed dtype inventory records BF16 stored weights/Qwen, inherited FP32
action computation, FP32 ZeRO2 accumulation/reduction/master weights/Adam states,
TF32off and deterministic flash attention. Declared profile fields are not used
as a replacement for observed dtype evidence.

Asset identity `04a0e93c1ea44902a4f73bf6973b236442ec5c8815ce6155c186a4d98254f420`;
processor identity `7e13bdb5ff27bd29bca3eaac59ee65ffbeb46364ae78dd436bb36db920d31d4d`.
The explicitly authorized completed full-asset receipt is reused without another
full corpus scan. Native processor, source, numerical and resume checks remain.
This receipt grants no model acceptance. The new research budget is semantically
bound to actual completed pilot/gradient/dtype artifacts and capped at64 updates.

## Runtime changes and limits

The CLI now retains repeated `--set` options. Native logs separate data fetching,
rollout, reward, reference, actor forward, backward/optimizer and post-update
probe, with per-rank spans and global maxima. These are host wall spans including
waiting and implicit CUDA synchronization, not kernel profiler measurements.
Gradient-dump observer time is nested inside backward and reported separately.
Checkpoint auditing/publication timing is separate; independent maxima should not
be added as though they were an exact critical-path trace.

`scripts/analysis/credit_research_run.py` runs training independently from complete
checkpoint export/evaluation on other GPUs. Evaluation never pauses training.
Checkpoint capture itself still requires a consistent optimizer boundary and is
not advertised as free/asynchronous. Save-every16 amortizes this cost. Full
gradient dumps are disabled in64-update runs. The pressure-released vla-zt3 host
has unrelated CPU contention, so its wall time is not an uncontended comparison
against vla-zt2. Local pilot timing also became contended by unrelated inference.

Additional runtime experiments are isolated in
`/mnt/project/DriveDreamer-Policy-perf`, branch
`fix/ddp-flow-grpo-inference-cache`. A scoped FP32 action-head snapshot saved only
about2% inference time and changed velocity/chain/logprob on all four fixed
scenes: FAIL, not deployed. CPU reward/reference overlap is being tested there;
neither experimental optimization modifies these running64-step arms.

## Real commands and output locations

Run from `/mnt/project/DriveDreamer-Policy-paired`, with the installed
`/root/miniconda3/envs/ddp/bin/python` and `PYTHONPATH=$PWD/navsim:$PWD`.
The controller is already running; a second instance is rejected by its lock.

```bash
# Registered research controller (fresh output locations required for a new run).
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python -m scripts.analysis.credit_research_run \
  --spec runs/denoising_credit/research64_manifest.json

# Read current train/evaluation state without loading models or scanning assets.
cat runs/denoising_credit/research64_control/progress.json

# Existing native CLI documents explicit train/resume/export/evaluation arguments.
/root/miniconda3/envs/ddp/bin/python -m starVLA.rl.flow_grpo.cli --help
```

Full checkpoints: `runs/denoising_credit/f_{uniform,discount}64/checkpoints`.
Per-scene trajectories and v2 scores:
`runs/denoising_credit/evaluation64/{uniform,discount}_{32,64}`.
After evaluation, genuine v1 rescoring and paired log-bootstrap reports are
generated at `runs/denoising_credit/research64_results/{v1_step32,v1_step64,
paired_step32,paired_step64}`. `performance_result.json` is written only after
this entire performance pipeline completes; a training completion alone is not
reported as a PDMS/EPDMS result. Check actual file completion, not these planned
directory names. No model weights are committed to git.

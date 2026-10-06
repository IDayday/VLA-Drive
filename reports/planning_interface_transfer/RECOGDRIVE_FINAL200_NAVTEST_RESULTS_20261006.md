# ReCogDrive Stage2 final200 Navtest (2026-10-06)

The complete canonical Navtest population scores **86.1176278112 PDMS points**:
12,146 scenes across 136 logs, no missing or failed scores. Exactly 946 scenes
have official `score == 0` (7.788572%). This is one training seed (0) and one
scene-bound inference seed (42), not a five-run estimate or a training-seed
stability result.

| Official NAVSIM v1 metric | Points, 0–100 |
| --- | ---: |
| PDMS | 86.117628 |
| NC | 95.920468 |
| DAC | 96.064548 |
| TTC | 88.333608 |
| EP | 85.013862 |
| Comfort | 100.000000 |
| DDC | 97.583567 |

The independent full unmodified official submission replay is **COMPLETE**.
Every scene and all seven official metric fields match exactly: maximum raw
error **0**, against the registered 1e-8 bound. All eight export completion
receipts and all 12,146 trajectory content hashes were separately verified.
Original-agent inference checks on eight scenes (one per rank) also have maximum
pose error **0**; this inference spot check is distinct from full CPU-score
parity. Aggregate completion evidence is in
`evidence/RECOGDRIVE_FINAL200_NAVTEST_COMPLETED_20261006.json`.
The unchanged low-level parallel scorer summary retains its pre-validation
`reference_parity_verified=false` flag; the separate completed official parity
certificate and export checks provide the final acceptance evidence.

The checkpoint is the preselected terminal `final.ckpt` after all 200 epochs,
not a checkpoint chosen using Navtest. Training completed 65,400 trainer steps,
65,372 actual Adam updates (28 AMP skipped steps), 16,729,600 scene exposures,
and 157.332601 formal-training GPU-hours. Earlier Stage1 extraction, smoke and
failed attempts remain separately recorded; formal training cost is not the
entire campaign cost. Full Navtest GPU export used 1.128847 GPU-hours on eight
A800 GPUs at training-vlawm-zt4; official CPU scoring and reference replay are
separate from that GPU allocation. Evaluation performed zero optimizer updates.

Inference uses the original ReCogDrive agent, current front RGB, navigation,
four allowed ego histories and current velocity/acceleration. Public frozen
Stage1 weights and computation are BF16; the full hidden interface and Stage2
FP32 master weights/computation are FP32, TF32 disabled. Original five-step
DDIM, one candidate and official clamp/denormalization remain unchanged. No
GT future, teacher, optimized label or metric cache enters the model input.
The newly registered scene-bound RNG is explicitly distinct from an unspecified
global sequential RNG. Original-agent/export-path inference parity is checked
on each of the eight rank partitions.

Scoring uses pinned unmodified NAVSIM v1.1, the complete official environment,
the official reference trajectory plus one selected ego prediction, and the
official scorer weights (EP 5, TTC 5, Comfort 2, DDC 0). The ReCogDrive GRPO
scorer's EP 10 setting is not used. The separate full official submission replay
compares every token and every official factor, with a maximum raw-error bound
of 1e-8. EPDMS is not reported here.

ReCogDrive training uses public driving Stage1 followed by independently
initialized official small Stage2, the official 200-epoch imitation recipe,
and the registered geometry-repair-v1 optimized labels. Its train population
is 83,636 scenes after excluding the existing development logs. This differs
from DDP's 101,592-scene population, three-view inputs, model initialization,
sampler and exposure; a raw difference from C/S or optimized-DDP scores is not
a matched architecture or optimized-label effect estimate. No method selection
or training change was made from this Navtest result.

Identities:

- Run: `final200_navtest_seed42_v1`.
- Training source: `8160b510fe2422587cd1409efb0e0a9cc0fabadd`.
- Evaluation source: `57130ca69b81b7652ec487f75a98ae518fa7f657`.
- Official ReCogDrive source: `6b8d8f5e01346c71094651c81dcaf66405dbc04e`.
- Checkpoint SHA256: `ba78c77ea6db3a9eb5b8ac7f3aca111a2239b8de904273d478141f9ca5f2e998`.
- Stage1 public source: `owl10/ReCogDrive-VLM-2B`, revision
  `16873acca08e3c04ab229b3d973f39aeba9db68d`.
- Stage1 weights SHA256: `79fb39297e322cd2d3dc68d4f23b86ff85806b336a4d5d0ed5db9b66e4034a3c`.
- Private artifact root:
  `/mnt/project/recogdrive-stage2-evaluation-artifacts/20261006/final200_navtest_seed42_v1`.

After the existing controller has exited, the same fixed registration supports
verified reuse/resume of its immutable trajectory bank:

```bash
cd /mnt/project/VLA-Drive-recogdrive-evaluation-source-57130ca
/root/miniconda3/envs/navsim/bin/python -m tools.recogdrive_stage2.run_evaluation \
  --plan /mnt/project/recogdrive-stage2-evaluation-artifacts/20261006/formal_registration_v1.json
```

Do not duplicate a live controller. Checkpoints, raw logs/images, scene-level
private data, trajectory banks and metric caches remain outside Git. The
separate started-run report preserves the initial PENDING state and failures.

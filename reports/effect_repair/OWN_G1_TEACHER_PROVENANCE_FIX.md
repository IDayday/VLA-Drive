# Own G1 training registration gate

Focused implementation requested after the independent Route A review. Source parent: `8a0ddcb6`; this change does not modify common training, collector, G0, Base weights, model forward or any running source snapshot. Independent author review is still required; this author does not self-approve the fix as an independent check.

`load_effect_teacher` gains optional `expected_registration_identity`. When supplied, it requires an actual checksummed complete TRAIN pair manifest from the G1 identity's `data_artifact`, an independently checksummed/self-identity-verified G0 registration with the expected identity, and a checksummed complete-population receipt. It verifies schema, train role, nondeployment-input status, requested scene count and matching Base/numeric/score-protocol fields. It also requires the loaded model mode to equal its recorded training arm. The verified dataset/registration/completion artifact pins are recorded in teacher provenance.

A1 and ScoreRefiner pass their actual repair manifest's registration identity. B1-effect carries the registration identity out of its already-verified repair-data parser and passes it to the same loader. Thus selecting a teacher from another G0 campaign, even with the same Base and architecture, fails before any guidance computation. The existing Base/mode/checkpoint/receipt/tensor-name/positive-step checks remain. The default optional-argument path remains usable for explicit legacy readers; formal A/B calls do not omit the new check.

The loader does not rerun simulation or rebuild the whole native bank. G1 already admitted its complete native-to-compact training bank; loading verifies its immutable metadata/artifact chain. `external_weights_loaded=false` is no longer the only source evidence on A/B paths. A forged administrator-rewritten entire history is outside this local immutable-artifact threat model.

CPU implementation tests:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/action_effect/test_effect_teacher_provenance.py
```

The targeted invocation including this file produced **19 passed, 8 deselected, 1 warning in4.23s**; the deselected tests were unrelated Flow tests included in the same command. Covered synthetic cases: correct same-registration load and provenance; legacy optional read; wrong-registration rejection with otherwise identical Base/mode/state; partial/DEV/wrong schema/deployment/numeric/Base/protocol mismatches; removed or changed pair/registration/completion files; partial/foreign completion; and A's actual `build_model` forwarding its repair registration to both teacher modes. Synthetic checkpoint fixtures are explicitly not evidence of real learning.

The existing Flow parser's focused join test is extended to assert its returned registration identity. It was executed separately; its actual result is recorded in the command output supplied with this commit. No whole matrix, model training, GPU or simulation was launched.

Next independent review should confirm all three call sites (A1, ScoreRefiner, B1-effect), negative role/source behavior and absence of changes to scientific loss or numerical execution.

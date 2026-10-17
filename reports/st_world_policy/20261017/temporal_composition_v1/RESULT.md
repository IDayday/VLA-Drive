# Narrow implementation result

- Original T2 provider restriction is retained; new explicit T3/T4 diagnostic
  composition allows only four updates, unchanged teacher width/future weight,
  original optimizer, precision, batch, seed, targets and 20k schedule.
- Existing HistoryEncoder, requests, temporal math, teacher computation and
  native training/resume are unchanged. Only the new composition validator,
  metadata preparation CLI and dataset attachment are wired.
- Author tests: **27 passed** in 3.49s; process wall 4.52s, sampled peak RSS
  636,604,416 B, child sampled rchar 139,457,474 B; parent post-wait rchar delta
  139,592,080 B includes the author's observation overhead. CPU8/9, no GPU,
  no real model/teacher/history record/image reads. `TEST.log` and
  `TEST_RESOURCES.json` retain the exact command. This is not independent review.
- Tests exercise real `teacher_provider.validate_profile_recipe` and
  `training_config.validate_config`, including both T3/T4, original T2 rejection,
  undeclared teacher/weight/optimizer/precision/step/batch changes, incomplete or
  wrong historical role/source/log/current identity, and the actual dataset hook.
  They do not replace pretrained-model integration or exact resume evidence.
- Metadata-only asset join: exact32 versus old history64 has **32 missing**.
  First assertion failure is retained in the summary; only the same small JSON
  metadata was reread to identify and record it. Matching causal metadata is
  `WAITING_INPUTS`; the future provider itself is not failing.
- `T3_RECIPE_PROPOSAL.json` and `T4_RECIPE_PROPOSAL.json` inherit the actual
  original exact32 config. They explicitly point at a **new pending** matched
  history namespace and lack the future newly generated provider/PCA pins.
  They are proposals, not runnable trainer configurations. The preparation CLI
  emits the validated final `training_config.json` after actual dependencies exist.
- No factory execution, GPU, SSH, native checkpoint read, full-index scan,
  ledger mutation, new training or dispatch occurred. Existing C jobs untouched.
  Formal/scientific T3/T4 status remains NOT_RUN. No independent PASS signed.

Known follow-up: independent review of this narrow diff; existing bounded causal
builder for exact32; regenerate control-source migration at final integrated
SHA; run the original provider/PCA validators once under bounded metadata
preparation; fresh training registration/resource lease with its own checkpoint
space; one admitted real four-update observation. No dependency on SP1 counter
lock repair, and no full-history build is required for this bounded integration.

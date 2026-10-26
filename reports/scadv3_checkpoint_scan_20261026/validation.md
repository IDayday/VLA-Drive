# Execution and integrity

An isolated evaluation branch was created from b2f57943c00c4f1e387695ed068fa45020277a22.
The live finite owner uses61ba85b0bee4302d62151e9d0daf3c8f7b4a83c3. Native
actor/evaluator worktrees remain unchanged and are bound in registration:
434cba8e(original/SCADV3),5a6bb6e(IL77-parent),22abe52(temporal/DINO).
Formal-v3 ee200bc supplies sealed full-population summaries. Later commits in
this branch add read-only reporting; they do not replace the running owner.

Executed:

```bash
python3 -m unittest tests.rl.test_finite_navtest_scan tests.rl.test_compare_rl_checkpoints -v
python3 -m py_compile scripts/analysis/qwen_sourceflow/finalize_navtest_scan.py scripts/analysis/qwen_sourceflow/compare_rl_checkpoints.py
python3 -m scripts.analysis.qwen_sourceflow.finalize_navtest_scan --help
```

Seven targeted CPU tests passed. These check finite scope/seed, distinct IL53
and IL77 parent wiring, actual temporal entry, refusal to launch training,
same-parent RL comparison, and rejection of mismatched source/parent or partial
populations. Five historical full summaries were verified against their seals.
Two actual12146-scene paired analyses completed, retaining scene seals,
strict source/protocol/parent/precision checks and log-cluster bootstrap.

The finite owner was actually launched once at2026-10-26 04:58:07 UTC:

```bash
/usr/bin/python3 -u scripts/analysis/qwen_sourceflow/finite_navtest_scan.py \
  --registration /mnt/project/qwen-sourceflow-native-20261018/scadv_and_latest_NAVTEST_20261026T044422Z/registration.json
```

Do not repeat that command while this owner exists. Actual jobs, process start
ticks, immutable source identities, native parity and completion receipts live
under the registration root. Reusing `owner.json` is deliberately rejected;
failures require preserved evidence and a separate explicit attempt.

Read-only completion/report command:

```bash
/usr/bin/python3 -u scripts/analysis/qwen_sourceflow/finalize_navtest_scan.py \
  --registration /mnt/project/qwen-sourceflow-native-20261018/scadv_and_latest_NAVTEST_20261026T044422Z/registration.json \
  --report-dir /mnt/project/VLA-Drive-navtest-scadv-latest-20261026/reports/scadv3_checkpoint_scan_20261026 \
  --wait
```

The report waits for all twelve complete populations, worker exits and lease
closure. It invokes existing strict paired analyses on CPU, launches no new
inference, and rejects incomplete results. Intermediate-versus-endpoint and
SCADV3-versus-CANONICAL comparisons are RL-to-RL comparisons, explicitly not
RL gains over IL. Additional analysis uses the exact IL77 parent control,
IL86 versus IL77, and fresh DINO14 versus fresh DINO5. It never mixes the
warm-start temporal diagnostic with the fresh run.

New helpers have no actor, sampling, decoding, reward or optimizer changes.
They are not evidence that any new checkpoint improves planning; that requires
the full evaluation results. Shared worker wall, job elapsed, model loading,
forward and scoring-wait costs are kept separate. Four reused export/parity
artifacts retain their original checkpoint/hash identities and historical costs.

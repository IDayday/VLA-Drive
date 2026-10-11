# C2 — prediction and official scoring retries

PASS on current source. `iqe/base_evaluation_cache.py` separates immutable predictions from individual scoring attempts. Successful score receipts bind prediction, observation/context/trajectory/checkpoint/protocol identity. Original images, observation JSON, target bytes and context are also checked. Completed evaluation receipts bind the scene list and result artifacts.

A timeout/worker/dependency failure remains `transient_scoring_failure`; bounded retries reuse the original trajectory without constructing the VLM. Every failed attempt retains its error and time. Budget exhaustion returns BLOCKED_RETRYABLE and CLI exit 2. `invalid_prediction` is a completed model-quality failure, never valid zero reward, never silently removed. Only the valid subset receives `PDMS_valid_subset_01`; full PDMS is unavailable when the requested population is incomplete. Qualify refuses incomplete labels and invalid predictions.

Legacy falsely COMPLETE artifacts are retained and imported into a new evaluation-schema identity. Valid labels are reused, failed labels retried. The report-before-receipt crash window is repaired without re-inference or rescoring.

Behavior tests in `test_closeout_qualification.py` inject one timeout among multiple scenes, verify exact backend call counts and prohibit model loading on retry; exhaust budgets; alter checkpoint/protocol/context/image/target bytes; inject nonfinite predictions; migrate legacy incomplete results; recover the completion-receipt window.

Actual Base evaluation used 32 fixed stage_val scenes from 32 logs: all 32 officially scored; zero transient failures and zero invalid predictions. The current source then reused and revalidated this result in 25.86s on CPU, without a new image forward or score invocation (`current_source_base_evaluation_reuse.log` in runtime). Faults are injected tests, not invented real worker failures. Original checkpoint/model/optimizer/RNG remain read-only.

A final negative regression covers direct qualify/freeze against a legacy receipt that omitted input bytes. `fixed_manifest` now revalidates observation/images, GT and metric context on every qualification/lock path. Old receipts remain immutable. Actual formal freeze was repeated with the current source and passed after this check.

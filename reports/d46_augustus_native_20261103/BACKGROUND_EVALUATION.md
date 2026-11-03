# Background evaluation and reporting

Two formal routes are maintained, with a32/16 GPU allocation. Scientific training remains frozen at78cea8f. CPU operations are frozen independently at247a9221975e989bcfbddec6f46c58eb512e1e9e; changing report code never changes the policy, Adam or scheduler.

The adopted native-monitor PID is321932 and evaluation-owner PID321933 (start ticks in the exact launch receipts). Both were verified live once after deployment. Do not launch duplicate owners. The original sole HTML watcher1859713 is preserved.

The native monitor adopts existing jobs and verifies native state. It does not launch another training task. A metadata problem is reported without stopping healthy GPU jobs; only an exact recorded child failure requests its own broken gang to save/exit.

Linux filesystem events on checkpoint/evaluation request/result directories drive the queue. Open training-journal writes do not trigger per-update checks, and self-published STATE/EVAL_STATE events are filtered to avoid a feedback loop. An idle service waits up to30minutes. Only a queued resource admission without active shards has a bounded five-minute retry. No manual training progress polling is needed.

Early complete NAVTEST requests are FULL native100/200 and LORA native200/400, followed by native epoch0.5/1/2/3/4/6 requests. The queue uses32 logical shards and only authorized spare physical GPUs, leaving the six training nodes dedicated. An exit0 partial shard cannot count as complete; only a verified pre-child75 resource refusal can be retried.

Official deployment stays B1, one trajectory, K10/shift5/seed2101. All12146 NAVTEST scenes,44 physical logs and invalid0 are required. Original summarization/pairing is reused. A canonical-sealed full result is recorded in EVAL_STATE before a one-shot notifier asks the existing report watcher to render. Notifications are idempotent per case/summary SHA; an incomplete/wrong-parent/qualified result is rejected. Report-notification errors cannot interrupt training or completed evaluation.

The HTML places current experiments first, refreshes every30minutes and immediately when a full evaluation completes. A new highest full score is highlighted with its checkpoint and parent/baseline. Public SOTA is not inferred from an internal score. The live path is `/mnt/project/qwen-world-performance-rl-20261103/report.html`.

State and logs: `/mnt/project/d46-priority-rl-20261103/operations/AUGUSTUS_CAMPAIGN_20261103/{STATE.json,EVAL_STATE.json,native_monitor.log,evaluation_daemon.log}`.

CPU checks: ten focused ownership/evaluation/event tests passed; three separate metadata notification tests passed in the report repository. Actual first32/16-GPU updates, own native resume, exact private RNG continuation and formal empty-Adam native0 are separate saved GPU/native evidence, not replaced by these CPU checks.

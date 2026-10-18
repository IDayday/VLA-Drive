# T0 actual capacity resource review

The exact four-update T0 capacity controller is admitted on Tsinghua-zt4 GPUs 0–3, CPU affinity 16–23. The recorded devices were empty, their native locks available, and approximately 903 GiB host RAM was available. The container CPU quota is unlimited and its reported memory limit exceeds host RAM; the controller's aggregate 192 GiB RSS stop remains required. CPU and shared-storage bandwidth are not exclusive.

Only one new 80 GiB inclusive output allowance is added. The remaining checkpoint reservation is the independently reconciled V3 receipt, not the stale 3.701 TB template value. All 768 GiB cache, 10 GiB metadata and 1 TiB floor reservations are retained. The same combined accounting covers the admitted SP3 continuation and potential future SP2 continuation already included in the fixed gross checkpoint allocation.

The inspected controller preserves the original monitor and ownership checks, four GPU locks, one-shot output creation, 1 TiB logical-read stop, 192 GiB aggregate RSS stop and bounded cleanup. Its maximum native run is 3,600 seconds, with the declared outer/cleanup margins. These are sampled stop limits rather than hard filesystem or kernel quotas. Quota availability was not independently exposed.

This is an independent resource review of the actual pinned controller and completed CPU inputs. It is not a new model review, throughput result, formal training approval, four-profile reservation or permission to retry. The independent correctness review is separate. Lead alone may run the resolved prepare/run commands. The controller must recheck idle devices and filesystem capacity before either action and during execution. If a guard fails, retain the failure evidence and do not weaken the guard.

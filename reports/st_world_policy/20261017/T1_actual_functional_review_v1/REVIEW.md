# Independent actual T1 native 2 → 4 functional review

**PASS_RECORDED_FUNCTIONAL_SCOPE**, with **FAIL_LIVE_DESCENDANT_SCOPE** retained independently. This review verifies the actual completed two-invocation receipts and their claims. It grants neither resource qualification nor formal training approval.

Reviewed Lead snapshot: `fa75ea66ad342f90a477499f7d19fbc45d7054ae`.
Actual trainer: `e85d11810bdf0240e7295d0994534f1df4ed21a4`.
Actual execution adapter: `59c256ec7c387d4b822a45787074db55386e2116`; its original file checksum `fe7b418dadb683c143f61d51508df32dae0c9d92f2726a4aade1950f9d112f93` was checked. This review did not substitute the later descendant-monitor fix for the actual adapter.

## Independently verified

- Both adapter invocations report COMPLETE, exit0 and two new updates. The underlying trainer result is correctly PAUSED / BUDGET_PAUSED at update2 and update4; it is not a completed20k scientific run. The new continuation differs only by stop_after2→4 and its explicit actual parent binding. Original max_updates20000, warmup1000, batch4/micro1/world2, data and optimization definitions remain identical.
- Actual step2 receipt `aae3d447…` and step4 receipt `c53c2199…` have valid metadata identities and agree with run identities, registrations, tags and client-state fields. Phase4 RESUME_RESOLUTION points to the actual step2 native checkpoint and its complete receipt. Its parent registration/config/run identity chain agrees throughout.
- Both rank0/rank1 LIVE_RESTORE receipts point to exactly that step2 folder/hash/receipt identity/client state. All five recorded comparison sections have no differences or nonfinite paths; every recorded live/native tensor hash matches. The unchanged existing observer compares actual live serialized model parameters/buffers, frozen fragments, full per-rank FP32 master/Adam state/options, engine/scheduler metadata and restored RNG before update3. Nonpersistent buffers are explicitly excluded from unavailable serialized-value comparison.
- Re-executed the unchanged original `epoch_batches(64,4,42,0)` on CPU. Its exact source bytes match the actual trainer commit. Applying the actual rank-stride rule reproduces every recorded rank's sample IDs for updates1–4. Epoch0 offsets1–4 and cumulative exposure4/8/12/16 agree. Thus the first resumed batches did not restart at update1 or skip ahead in the registered sample stream.
- The original learning-rate function reproduces 1e-8,2e-8,3e-8,4e-8 without shortening the20k timeline. Every update contains finite current and ego_fm losses, finite positive logged global gradient norm, positive sampled FP32-master change count and finite positive maximum sampled change. T1 has **two active loss terms on each of four updates**; this is not evidence that future/delta losses were run. Current valid-element count442368 and ego count4 agree with the recorded contract.
- Lead's derived update rows match the raw steps.jsonl exactly. Total is four updates / sixteen scene exposures, not a full64-scene epoch.

## Exact boundary of the evidence

This is a read-only review of actual run receipts, not a third execution or reloading the giant tensors. Full native payload integrity was checked by the actual trainer/observer at consumption; this review checks the receipt and recorded comparison chain without independently rereading those files now. It does not claim the checkpoint bytes can never change afterward.

Restore-boundary equality is supported. **Uninterrupted-versus-interrupted stepwise loss/gradient/parameter equivalence was NOT_RUN**, and there was no terminal step4 reload. No zero-update exit test was added. Normal FlashAttention execution was retained. A successful boundary check does not imply cross-kernel bitwise trajectories.

FP32-master changes are verified at the existing sampled indices, not asserted for every parameter. Finite recorded losses/global gradient norms do not constitute a new exhaustive gradient-tensor scan by this reviewer. Four timing points are insufficient for stable throughput; checkpoint times are separately reported by Lead.

The original session-only controller omitted torchrun ranks in new sessions. Its reported≈430MB peak is not training aggregate RSS and cannot support full-lifetime enforcement. **The independent resource FAIL remains unchanged.** This functional PASS cannot erase it or approve formal launch. Terminal inherited logical-read accounting and point observations do not reconstruct the missing true simultaneous peak.

No process signal, new training invocation, GPU call, SSH, image/teacher/model payload read or parameter modification occurred. Process-exit observations in Lead's summary remain its historical observations; this review did not perform new live PID ownership inspection.

## Executed review

```bash
taskset -c 10,11 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout 60s /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/T1_actual_functional_review_v1/audit_actual.py
```

Actual result: PASS,1.098s; peakRSS408195072 bytes; logical reads17104556 bytes; physical reads1363968 bytes. Twenty-four unique small artifacts total1245246 bytes. CPU10/11; CUDA hidden. The complete audit output, source pins, per-step sample comparisons and costs are retained in this directory. These remain within the assigned60s/256MiB/4GiB/16MiB scope.

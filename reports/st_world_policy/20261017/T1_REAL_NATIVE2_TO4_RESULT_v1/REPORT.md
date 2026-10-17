# Actual official T1 native 2 → 4 qualification

Two real A800 ranks completed updates 1–2, saved native model/master/Adam/RNG/cursor state, then a second process restored it and completed only updates 3–4. Both invocation receipts are COMPLETE. This used real current/history images and current FP32 DINO labels from the existing real64 assets; no full future tensor bank was created.

The configuration kept the original 20k/warmup1000 timeline; global batch4/microbatch1/world2 is a declared SMOKE override. The two active losses (current and ego_fm) and recorded global gradient norm are finite at each of the four updates; each update changed sampled FP32 master elements. Future/delta objectives were disabled in this T1 test. The next batches, including updates3/4, match the original fixed64 manifest and epoch_batches(seed42) exactly. Total exposure16, not a full64 epoch.

Both ranks compared actual native-restored BF16 model parameters, persistent buffers, FP32 master partitions/Adam, scheduler/engine state and restored RNG before update3. All recorded sections contain zero differences/nonfinite paths. Nonpersistent buffers are explicitly outside serialized-value comparison. This is a restore-boundary check, not a claim that uninterrupted and interrupted FlashAttention training were compared step for step.

| Update | Slowest rank update s | Data s (rank0) | Gradient norm |
|---:|---:|---:|---:|
| 1 | 3.237963 | 0.788242 | 18.875536 |
| 2 | 2.717749 | 0.698420 | 9.097044 |
| 3 | 3.456169 | 0.742728 | 4.161231 |
| 4 | 2.761168 | 0.735153 | 15.710287 |

Four timing points are insufficient for a stable throughput claim. Checkpoint saves took 54.056s, 56.710s. The two native outputs occupy 74989884642 bytes by stat, inside the separate120GiB allowance. Final inherited parent/child logical read charge is 299957970206 bytes, inside512GiB.

**Resource qualification remains failed:** independent actual inspection found the session-only monitor omitted torchrun ranks that started new sessions. The ~430MB reported peak is not aggregate training RSS, and live read/RSS enforcement for those ranks was incomplete. Historical results are preserved; no invented peak or retroactive PASS. Fresh point observations and per-process HWM are not a full-lifetime simultaneous peak. A narrow future process-scope fix is underway. All eight specifically recorded process PIDs have now exited; no signals were sent.

No third invocation, terminal reload, formal initialization, PDMS evaluation or navtest was started. The separate formal C jobs and existing metadata writer were not restarted or modified.

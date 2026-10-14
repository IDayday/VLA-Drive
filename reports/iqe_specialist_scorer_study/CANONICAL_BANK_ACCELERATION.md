# Fixed-bank execution acceleration

The six authorized experts have completed. This change only accelerates immutable
candidate reads and official-label preparation for the already selected pool.
It does not change models, targets, training budgets, source priority or score
contexts. The four Scorers still initialize independently from random weights.

The 16 temporary GPU workers completed all 3,114 canonical rich-endpoint batches,
99,486 records, in 717–811 seconds per worker. These are six own raw endpoint
columns on 16,581 D_S scenes; no network generated score labels. Coverage receipt:
`runtime/rich_precompute/COMPLETE.json`. Exact source duplicates are removed
before official scoring and cannot enlarge the 12-candidate budget.

Deployment precompute on permanent rl-zt2 GPUs0–3 completed the three selected
columns (49,743 candidate identities). The two selected raw endpoints reused
verified bytes of the identical checkpoint/model and canonical inference batch;
the road EMA endpoint was computed normally. Source records and their provenance
remain unchanged. The old D_S role alone was handed off after its Base labels were
complete and no scoring child was active. No optimizer was involved. The unique
completion controller remained alive. Receipts:
`runtime/deploy_precompute_handoff/{OLD_ROLE,HANDOFF,LAUNCH}.json`.

New rich-label preparation uses at most two concurrent original official jobs,
two primary CPU workers each. Existing record-locked helpers can assist. Before
submitting labels it follows the same fixed source ordering and trajectory hash
deduplication as the original builder. It binds complete scene records, candidate
keys, Base/pool/protocol hashes and original job names. The main builder verifies
the sealed column receipt and all score joins. A failed, expired or corrupted
preparation is rejected; it does not create a new experiment or silently switch
to different predictions. All candidate bytes are checked using up to eight
ordered read threads. Complete rich columns no longer instantiate an expert or
read frozen W/A features again.

After the user stopped MOPD, eight CPU helpers were deployed on released vla-zt,
PIDs2706766–2706773, with a six-hour limit. Each handles one immutable metric record
at a time; aggregate expected peak is24 scoring CPU processes including existing
helpers and primary workers. Launch:
`runtime/local_score_helpers_20261014T1615Z/LAUNCH.json`.
Initial real progress was60 records per helper, then about660 per helper, with
zero failures. Counts include strictly validated receipt reuse.

Both temporary GPU hosts subsequently acquired genuine Flow jobs. Live capacity
checks rejected another GPU launch there, without signalling those jobs. All16
finished rich GPU workers exited naturally; the eight lease CPU helpers retain
the original23:07:18 UTC expiry. Long Scorers remain on permanent rl-zt2.

Validation: focused canonical-byte reuse, duplicate/capacity and corruption
tests:8 passed in0.58s. An initial test collection typo and relative-import error
were corrected before dispatch; neither launched scientific work. Full regression
before this narrow execution change:502 passed,3 skipped,53.43s. The extra matched
factorial analysis test passed1/1; its bootstrap analysis reads only completed
5000-step validation labels and does not choose or replace any endpoint.

Inspect running receipts before using the following continuation command; the
RUN.lock rejects concurrent preparers. It preserves all completed labels:

```bash
cd /mnt/project/DriveVLA-M0-iqe-specialist-scorer-study
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
 /root/miniconda3/envs/ddp/bin/python -m iqe.specialist_study.rich_labels \
 --config /mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1/resolved_config.json \
 --parallel-columns 2 --score-workers 2 --hours 6
```

# Full-topology runtime guard: independent narrow review

**PASS only for runtime guard commit `210be55caa9b3395434504b5e310f7274d09c35a`.**
The pending controller, actual resource admission and real measurement are not
approved by this report.

Nine CPU-only checks passed: the five provided small-source cases were
independently executed, with four additional checks for the exact difference,
actual original trainer resume rejection, original timing windows, and
unchanged downstream safeguards. No model, Torch, checkpoint, database or
real payload was loaded. Only test/report files were added in this worktree.

The new branch accepts exactly PROFILE, world 8, microbatch 4, global batch
32, the full 101,592-scene population, 240 updates, original 20,000-update
schedule with 1,000 warmup, and timing mode. This means 7,680 scene exposures,
960 per rank, accumulation 1. The existing TimingPlan supplies steady steps
21–120 and 141–240, after 20 warmup steps in each window. Existing global-2
diagnostics remain available under their prior contract.

The consumption receipt's no-resume text is backed by the unchanged actual
trainer guard: timing/trace plus `--resume` raises before beginning its
measurement windows. That guard was executed as an AST-extracted statement,
without model construction. A config containing continuation is also rejected
by the new gate. This does not test real distributed execution.

Full-population order/identity validation and the original mathematical recipe
checks have identical ASTs to `a8b0036f…`. The original training, timing,
fixed-batch teacher and indexed-provider source bytes are unchanged. RAM-only
teacher behavior and zero new teacher-tensor disk declarations are inherited
source guarantees, not measurements from this review. PROFILE weights remain
ineligible for formal initialization under the unchanged approval guard.

Recorded test cost: 1.211 seconds, 30,494,720-byte peak parent RSS,
2,906,520 parent logical-read bytes including startup and git stdout, CPUs
2/3. Git's internal reads are not separately measured. See `TEST.log`,
`TEST_RESULT.json` and exact binding in `REVIEW.json`.

```bash
taskset -c 2,3 env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  /root/miniconda3/envs/ddp/bin/python -B \
  reports/st_world_policy/20261017/full_topology_runtime_guard_correctness_v1/run_review.py
```

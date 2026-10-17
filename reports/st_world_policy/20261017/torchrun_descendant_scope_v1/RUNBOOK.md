The following is the actual CPU verification command (already run; do not
automatically rerun it to manufacture additional evidence):

```bash
cd /mnt/project/VLA-Drive-st-torchrun-process-scope-20261017
taskset -c 10,11 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 120s /root/miniconda3/envs/ddp/bin/python -m pytest -q reports/st_world_policy/20261017/torchrun_descendant_scope_v1/test_owned_descendants.py reports/st_world_policy/20261017/official_T1_native4_execution_v1/test_run_once.py
```

Future reuse is a narrow change before Popen in a separately reviewed launcher:

```python
ownership = load_module('owned_descendants', reviewed_ownership_helper_path)
s = ownership.ancestry_supervisor(original_pidfd_subreaper_helper)
# Only then enter the existing lifetime_scope and make the one task Popen.
owner = s.OwnedSession(actual_new_popen)
# Discover before each live accounting sample. Fail on unknown_adopted.
# Existing live_owned_usage consumes all owner.members across sessions.
# finish always receives the original absolute cleanup deadline.
```

The patched T1 CLI still exposes `--admission` and `--admission-sha256`, but its
old phase2/phase4 namespace and invocation budget are already consumed. **Do not
run it against the old admissions.** It now rejects admissions without the exact
`ownership_helper` pin and independent review bindings. There is intentionally
no generated execution admission, new run namespace, new checkpoint reservation
or automatic training command in this implementation-only deliverable.

For an independently authorized future bounded job, bind the actual helper
SHA256 and future adapter SHA256 from `SOURCE_AUDIT.json`, re-review only the
ownership/resource change, and retain the original scientific/configuration and
absolute lease boundaries. A new invocation must use its own existing job
registration and reviewed resources. This fix does not itself authorize another
T1 qualification or consume additional training updates.

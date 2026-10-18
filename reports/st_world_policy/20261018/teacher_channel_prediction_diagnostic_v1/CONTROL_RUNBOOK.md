# Fixed diagnostic execution handoff

This is a control-only addition to diagnostic71011454. The diagnostic, BINDING, original model loader, target reader, requests and channel metric remain unchanged. No GPU, SSH, checkpoint payload, model or teacher data was accessed while implementing this wrapper. The proposal is deliberately invalid for launch until Lead records fresh actual host, CPU and GPU identity, independent reviewer and time window.

`control.py` directly reuses the original `run_build_once.py::monitor`, SHA4b2f0fccfd4bf3d655198f61179c88f6b2a51840731465c720b8addb6169b1a2. It does not create a new monitor. Its parent-only metadata module is pinned to aaa36f5383451eadcd92c160c6f3ffb8704110ab44eb12ee433b3207995e71ba; the child starts afresh with original fdf25f99 runtime and only calls fixed `diagnose.py --execute` via runpy. Native stdout goes to the existing bounded stderr pipe; stdout contains exactly one original accounting envelope. Before/final/delta `/proc/self/io` and original post-wait accounting remain intact.

Bounds are work1800s plus existing cleanup40s, two caller-admitted CPUs, sampled aggregate RSS96GiB and sampled cumulative rchar320GiB. These are monitored stop thresholds, not cgroup hard limits. The320GiB planning allowance includes repeated strict checkpoint inventories, master reconstruction, parent construction and metadata. Memory-mapped page accesses are not completely represented by rchar; no physical I/O claim is made. Native logs retain the original1MiB per-stream in-memory cap; stored control reports preserve byte counts, hashes and4KiB tails so log content cannot exhaust the1MiB control allocation.

Fresh admission requires the actual A80080GiB UUID, at least68GiB currently free device memory and96GiB MemAvailable. The device threshold is conservative, not a measured diagnostic peak. Sharing with unrelated jobs is allowed only when this capacity check passes. Existing global one-loader and per-device flock locks remain held. Only the original monitor-owned new session/verified descendants may be terminated; unrelated jobs are not owned. No checkpoint allocation or live ledger write is made.

The diagnostic alone claims its new15MiB root. The wrapper creates only the independent sibling1MiB controls root; the child root is never pre-created. Both are stat-censused, including partial files and directory allocation. Existing output/control paths reject repeated invocation. Child output failures retain a separate bounded failure receipt. No automatic retries or resume are provided for this read-only one-shot diagnostic.

Exact help and eventual Lead command:

```bash
D=/mnt/project/VLA-Drive-st-teacher-channel-prediction-diagnostic-20261018/reports/st_world_policy/20261018/teacher_channel_prediction_diagnostic_v1
/root/miniconda3/envs/ddp/bin/python -B "$D/control.py" --help
# ACTUAL_ADMISSION and ACTUAL_ADMISSION_SHA must identify the newly reviewed,
# immutable filled proposal; this repository supplies no APPROVED admission.
bash "$D/launch_once.sh" --admission "$ACTUAL_ADMISSION" --admission-sha256 "$ACTUAL_ADMISSION_SHA"
```

The shell's GNUtimeout covers Python/bootstrap with1800s and at most40s kill grace. The original finite absolute alarm also starts before the first pinned source read. Admission requires at least1840s remaining from launch. All literal source hashes are in CONTROL_PINS.json and the proposal; no runtime path or arbitrary executable is caller-selectable.

Focused validation used CPU2,3 and no GPU:8/8 PASS,0.1900s wall,29,712KiB maximum RSS,4,629,121B parent plus reaped-child rchar. Two fixtures executed real tiny child processes through the original4b2f monitor, testing multiple native prints, SystemExit(0), SystemExit(3), sole envelope, exact terminal accounting and ownedremaining0. Other fixtures cover stale/wrong admission, actual helper pins, fresh UUID/free/RAM gates, separate disk caps, bounded logs, failed-output retention and one-shot refusal. This is implementation evidence, not independent approval or an actual diagnostic result. The initial attempt to collect `/usr/bin/time` statistics exited127 because that executable is absent; the recorded successful run used Python resource/proc counters and did not repeat any real diagnostic.

Reproduce the focused tests (fixtures only):

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 timeout --signal=TERM --kill-after=5s 120s taskset -c 2,3 \
 /root/miniconda3/envs/ddp/bin/python -B "$D/test_control.py"
```

No computation, PCA comparison, training-loss interpretation or model initialization is changed by this controller.

# Probe control-file pin boundary

Lead change on 2026-10-17, following independent Correctness's real JSON
replacement counterexample against source `278f55c7`.

`spatial_probe.pinned` previously hashed a file then reopened it with
`read_text`. A replacement between those operations could supply a different
protocol or label-manifest reference than the approved bytes. It now reads one
bounded byte sequence, checks the open file and path versions, and hashes and
parses those same bytes. The one-MiB limit applies only to protocol, fixed
population, label-manifest and training-registration JSON. The actual largest
registered control is the 128,955-byte label manifest; native checkpoints and W
tensors do not pass through this function.

No probe input, loss, labels, training recipe, native loader or active runtime
was modified. This is not a claim that real W extraction or probe fitting ran.

Lead test: `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/root/miniconda3/envs/ddp/bin/python -m pytest -q
tests/st_world/test_spatial_probe.py`: 14 passed, one existing pynvml warning,
8.53 seconds. Independent replacement/failure tests still require review on
this fixed source. No GPU was used.

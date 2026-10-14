# Preserve training state while avoiding unowned GPU load

On 2026-10-14 around07:52 UTC, tsinghua-zt6 acquired about19.3GiB per GPU
and90–100% utilization unaccounted for by visible container processes. The
two visible extension trainers used only about1.28GiB each. No matching pressure
script was visible. No unknown owner was signalled or classified as idle.

Only this campaign's waiting controller, two native training processes and
unstarted bank waiters were handled. The trainers received their existing
SIGTERM optimizer-boundary request after the old controller exited. Both saved
step1465 and exited. Their complete receipts verify:

| Cell | Source checkpoint SHA256 | EMA updates | Sampler consumed |
|---|---|---:|---:|
| EXT_interaction | c1c047f6c9573b43d4756cefe615d3d74fc42b840e79ea66a30cea168a12dd71 |1465|187520|
| EXT_safe_progress |84171299d2346f158b5c492c10dfdc95a19843097a8ab1e6aafce10594cb5e6c|1465|187520|

Each includes210 Adam parameter states with FP32 moments and step1465, raw
and EMA parameters, scheduler, both CPU/CUDA RNG and consumed sampler state.
The original global batch128, microbatch16, source hash, data/target hashes and
fixed5000-update learning-rate timeline are unchanged. Both environments use
the previously verified identical Python3.10.20/PyTorch2.5.1+cu124 installation.
The source checkpoint bytes were not rewritten.

Destination: tsinghua-zt5 GPU5/PID22204 and GPU6/PID22328, resumed by the
sole controller PID22108 with `--gpus 5 6 1 4`. The first new step is1466.
All128 next sample IDs match reconstruction from each source checkpoint;
the shared resolver reports EXACT_OPTIMIZER_BOUNDARY. The observed new losses
and gradients are finite. No random reinitialization, scheduler reset or new
expert run was introduced.

Measured pure optimizer-update windows, excluding validation, logging and
checkpoint pauses:

| Cell | Source final100 mean/p50/p95(s) | Destination after5 warmup mean/p50/p95(s) | Destination n |
|---|---|---|---:|
|interaction|1.3092 /1.3339 /1.4782|0.4266 /0.4255 /0.4362|126|
|safe_progress|1.3072 /1.3226 /1.4800|0.4399 /0.4385 /0.4467|116|

This is an observed same-recipe resource migration, not a model-performance
claim. The parameter/data/RNG continuation is audited separately from wall time.
Validation, candidate-bank construction and four-Scorer results remain pending.

Future launches now inspect physical memory, utilization and visible compute
owners. Unexplained occupancy greater than1GiB, utilization above60%, or less
than8GiB free causes a bounded WAITING_CAPACITY state, without process signals.
The bank roles are versioned onto zt5 GPUs2(D_S),7(dev),1(stage_val),4(cal);
they have not started model work until the final pool is locked. The old route
is preserved under `runtime/bank_routing_versions/`. New routing SHA256:
`569aa371abb7dbe99512d0dc962e3a4cd576be7b6473c4f5edb4a66525fce0a2`.

All20,325 spatial contexts are now COMPLETE under the same original capture
hash and six-shard identity. Neither Base, G2, MOPD nor their caches/jobs were
changed. This campaign has no remaining GPU process on zt6.

Evidence: campaign root `runtime/zt6_unowned_load_migration/` contains source
process identities, verified complete boundaries, versioned route changes,
destination launch/capacity records and `RESUMED_VERIFICATION.json`.
Source commit for the launch is `a28148c3ced22790236d17109ba4ac7a3822e488`.

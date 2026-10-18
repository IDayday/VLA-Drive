# CPU evidence reproduction only

This directory does not expose a formal trainer or simulator command. It cannot
start a GPU experiment, alter a checkpoint or approve a formal run.

The actual reference used `/root/miniconda3/envs/ddp/bin/python` with an isolated
`/tmp/st-dfql-reference-deps-20261018` import prefix. Nothing was installed into
that conda environment. Three CPU packages were installed with `--no-deps`:
`jax==0.4.35`, `jaxlib==0.4.35`, `ml_dtypes==0.5.0`; existing NumPy/SciPy/
opt_einsum were reused. No Flax, simulator, robot suite, network weights or GPU
plugin was installed. See SOURCE_MANIFEST.json for actual bytes and versions.

Budget for a reference re-run: one CPU affinity, 120 seconds, no visible GPU,
2 GiB process RAM budget, 512 MiB isolated dependency storage, 1 MiB output. The source is a 12 KiB pinned
file; the checked paper is below 4 MiB. This is not a whole-repository scan.

From the isolated worktree, reproduce with a NEW output path:

```bash
cd /mnt/project/VLA-Drive-st-consequence-proxy-evidence-20261018
PYTHONPATH=/tmp/st-dfql-reference-deps-20261018 \
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
timeout --signal=TERM --kill-after=5 120 \
/root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261018/consequence_proxy_evidence_v1/reference.py \
  --official-source /mnt/project/st-world-policy-20261016/audits/evidence/public_sources/dfql/agents/dfql.py \
  --output /tmp/dfql_cpu_reference_recheck_20261018.json
```

The program checks the official file SHA before extracting the unmodified
`actor_loss` AST. An altered file fails before computation. An existing output
fails instead of overwriting old evidence. Outputs are synthetic values only.

Tests cover four exact-source/JAX versus PyTorch combinations (mean/min Q,
normalization off/on), paper shifted-target gradient identity, source reduction
scale, full BPTT versus analytic recurrence/finite difference, a shared-parameter
counterexample, the actual navigation raw-to-physical decoder derivative, and
constant-score non-gradient behavior. `REFERENCE_EXECUTED` is not independent
correctness approval, navigation integration or scientific success.

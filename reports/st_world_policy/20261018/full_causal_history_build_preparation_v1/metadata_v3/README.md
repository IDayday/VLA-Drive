# Exact metadata package v3 preparation

This is an unissued proposal, not an admission or live policy migration.
The implementation base is `676e3e88d48d9444ab2cb928242d196381631b26`.
Only `metadata_storage.py` changes: the explicit 8→10 GiB version chain and its
existing additive migration condition. Builder, history reader, model, PCA,
ledger locking, checkpoint accounting and per-run limits are unchanged.

The v3 package retains the original v1 hash in `previous_package_sha256` and
binds the entire immediately preceding v2 package in
`previous_v2_package_sha256`. Reconstructing either parent must match its hash.
`original_package(v3)` still returns exactly the same v1 package used by frozen-W
claims. Skipping v2, unknown versions, changed baseline, changed campaign cap,
removed reservations and changed claims fail. No generic compatibility map is
introduced.

`PACKAGE_ONLY_POLICY_PROPOSAL.json` changes only the metadata envelope of the
pinned `c77dca47…` policy. It does not add train/dev claims, change the live ledger
or assert current writer quiescence. The eventual full-role registration must use
this new immutable runtime, and retain the already reviewed SP0/SP1 common-source
bridge before loading the storage guards. Running the unbridged native CLI against
those mixed-source claims must continue to fail; this patch does not bypass that
independent source check.

The retained conservative budget is 7,904 MiB, including the failed diagnostic's
entire 16 MiB. Holding another 16 MiB for a possible corrected diagnostic plus
2,048 MiB train and 128 MiB dev (both inclusive of their 1 MiB controls) gives
10,096/10,240 MiB, leaving 144 MiB. The corrected diagnostic hold is not a claim
that it has started. No unused staging or failed allocation is released.

The full-history source and command evidence remains commit `09b4ea91`,
`PREPARATION_PLAN.json` SHA `cf2ca70906592baba626ed03282466919dd6c58277467ccbd9b5745054dfac3a`.
Independent storage evidence is commit `33c040e8`, REVIEW SHA
`8c89dc59cebd22436f3ab69eaeec598ae7b73d2d7dea058d36f54141ec049568`.
Failed-diagnostic accounting is commit `5247ffee`, REVIEW SHA
`bdd81fa7baa1df92ebf8c1098b7aad22e3efe5acfb416470099e3f11d6343ba7`.

## Executed local checks

```sh
taskset -c 10,11 timeout 120s env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  python -m pytest -q -p no:cacheprovider \
  tests/st_world/test_metadata_envelope_v3.py \
  tests/st_world/test_metadata_envelope_v2.py
```

40 tests passed. The measured equivalent in-process pytest invocation took
1.572 s, peak RSS 53,228 KiB, `/proc/self/io` final `rchar` 16,026,094 bytes,
including interpreter/import reads. Tests call the real migration on temporary
ledgers, exercise all four interrupted publication windows, preserve frozen-W
and claims, reject malformed compatibility chains, and retain 4 GiB per-run
disk/RSS and 14,400 s limits. A prior measurement command failed before tests
because `/usr/bin/time` is absent; the stdlib measurement replaced it.
No real image, log, tensor, checkpoint or live ledger was opened.

## Existing native migration API

```sh
python -m tools.st_world_policy.metadata_storage migrate --help
python -m tools.st_world_policy.metadata_storage run --help
```

After independent review, the Lead supplies the existing migration arguments:
`--old-policy`, `--old-policy-sha256`, `--new-policy`, `--new-policy-sha256`,
`--expected-old-ledger-sha256`, `--quiescence-receipt`, `--quiescence-sha256`,
`--pilot-inventory`, `--pilot-inventory-sha256`, `--checkpoint-receipt`,
`--checkpoint-sha256`. Empty pilot inventory is supported: no old-cache scan or
allowed-root extension is needed. These arguments must be passed through the
source-bridged full-role control binding; this source patch alone is not a launch
command. A fresh CP receipt must cover the full role's work plus 40 s cleanup.
The old 4→8 transition and its tests remain supported unchanged.

# Metadata CLI contract pin: atomic replacement fix

Independent review of 9c7743f5 demonstrated a race between hashing the contract
receipt and reopening it for JSON parsing. An atomic replacement with another
valid teacher contract could publish a request index under the old approved SHA.
Original failing evidence remains at the private audit path
`audits/correctness/metadata_index_review_v1/CONTRACT_PIN_COUNTEREXAMPLE.log`.

`index_teacher_metadata.run` now reads at most 1 MiB + 1 from one binary open,
checks before/after file version and exact length, hashes those bytes and parses
those same bytes. It never opens the receipt again for the contract. Size and
schema guards remain; no teacher/cache precision or population semantics changed.

New synthetic negative test swaps the pathname atomically after the bounded read
but before verification. It rejects the changed file and writes no COMPLETE.
Other tests cover a single read, wrong SHA, oversized and empty receipts.

CPU only, affinity 0,1, OMP/MKL/OpenBLAS threads 2, CUDA visibility empty:

```
/root/miniconda3/envs/ddp/bin/python -m pytest \
  tests/st_world/test_metadata_contract_pin.py \
  tests/st_world/test_teacher_metadata_index.py -q
```

29 passed in 4.00 s; full log in `evidence/metadata_contract_pin_fix_v1/`.
No real metadata scan, teacher load, GPU or tensor build. The concurrently drafted
future-ledger implementation is excluded from this fix commit and its unchanged
legacy branch was present during local tests; independent review will rerun the
pinned commit. No formal training permission is implied.

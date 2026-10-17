# Indexed PCA scope and connection-budget boundary fix

Independent review of `5e0e2ec3` found two failing counterexamples despite the
original 28 indexed-path tests passing. A re-signed legacy PCA scope could bind
an indexed provider without the new fitted-cache index. Also, registration
opened 272 MiB of declared index caches before rejecting a 256 MiB combined
budget. This was a declaration/connection admission failure, not measured RSS.

`provider_pca.py` now rejects indexed provider identities or descriptors in the
legacy scope before reading fitted legacy metadata or creating a preview.
The indexed path checks the fitted-index plus provider connection budgets
before opening a new connection, both during registration and when validating
a scope with an already owned provider. The check remains after construction
as well. Original basis, source identity, tensor hashes and old valid eager
paths are unchanged. No running runtime or real artifact was modified.

Lead regression command, with no visible GPU, two CPU threads/affinity and a
180-second outer limit:

```bash
python -m pytest -q --tb=short \
  tests/st_world/test_indexed_provider_factory.py \
  tests/st_world/test_provider_training.py \
  tests/st_world/test_fixed_provider_training.py
```

Actual result: **61 passed, one existing pynvml warning, 21.70 seconds**.
Independent counterexample revalidation is pending. This document does not
grant real indexed-training or formal population admission. The original
independent failure evidence is retained separately.

# Independent fixed export host extension review

PASS for the exact source hash recorded in REVIEW.json. This review was performed by `/root/spatial20k_evidence`, which did not implement the adapter.

The original pinned v5 exporter permits three fixed hosts; current live policy and campaign authorization independently already permit Tsinghua zt4. The adapter adds only that fourth host to `ROUTE_HOSTS_V3` and executes the original AST. Reversing that one assignment yields an AST identical to the pinned original, including all original main code, scientific inputs, registration/native checkpoint checks, actual environment/GPU checks, slots/locks, budgets, and both execution-window checks. `__file__` binds the actual wrapper into the launch receipt.

Six source-only tests passed (0.043 seconds): exact one-value delta, changed-source rejection, current zt4 authorization with original rejection, absent authorization/ban/unknown-host rejection, original time rejection, actual wrapper identity. No heavy imports, model/payload reads or launches occurred.

Lead may use this exact adapter for fresh fixed C1/C4 routes after the unchanged live checks succeed. No assertion is made that target GPU/memory/disk remains available at launch time. No old route or evidence was changed.

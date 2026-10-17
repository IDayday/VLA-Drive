# FIFO control-file bounded-open fix

Parent implementation: `ba09bf88d677e62b9503e1391ca47e9d02f5a8c7`. Independent counterexample and original FAIL evidence: `12467ce7f2332c5cb72160e1fa0a6db255196b6b`, `reports/st_world_policy/20261017/correctness/w_storage_bridge_review_v1/TEST_FAIL.log`. That historical evidence is preserved unchanged.

The only production edit is `Reads.raw`: add `os.O_NONBLOCK` to the existing `os.O_RDONLY | os.O_NOFOLLOW` open. A FIFO with no writer now opens without waiting and is rejected by the unchanged descriptor regular/private-file check. Existing file-size, version, SHA, source, budget and import checks are unchanged. No other helper or runtime implementation changed.

`fifo_review/test_independent.py` is byte-for-byte the independent test from the above commit (SHA256 `aba9c9235de67196a46f4a79c370f528d4594016f356cdd5289dda3d70832695`). Its placement makes its original relative module path resolve to this revised implementation; the test body is not modified.

Current verification: **45 passed in 6.06 s**: original44 owner tests plus the independent FIFO test, CPUs0–1, 2GiB per-process address-space ceiling and120s outer timeout. `FIFO_FIX_TESTS.log` contains the complete result; `FIFO_FIX_RESOURCES.json` records6.533s wrapper wall and68,907,008B child maximum RSS (not an aggregate tree measurement). No GPU, model/image/native tensor, actual ledger, policy or frozen compute runtime was accessed or changed.

New `storage_bridge.py` SHA256: `a6aa3a2ba5430ef588cd9811a4da373d64d05c17f042cade40324c810909089f`. Any future compatibility admission must pin this revised helper; the old EVIDENCE/FINAL_TESTS documents remain historical rather than being overwritten. Independent re-review is still required before actual extraction.

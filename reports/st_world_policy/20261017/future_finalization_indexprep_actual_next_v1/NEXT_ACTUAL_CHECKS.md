# Finalized TRAIN metadata: minimum next actual checks

This is a bounded receipt/source review, not a launch approval. The earlier result remains **17 complete PASS plus one reviewer order-tail assertion incomplete**. No tests were rerun, no Torch imported, no source payload/request body/ledger/model checkpoint opened, and no job dispatched.

The actual finalizer receipt and independent terminal agree: **101592 scenes,1176 logs,COMPLETE**, returncode0, no monitor failure, cleanup0 and complete read accounting. `RECORDED_BINDINGS.json` records exact actual pins. The original failed build remains bound by its unchanged `FAILED_NO_AUTO_RETRY` terminal SHA; finalization success does not relabel that build.

The actual original PLAN is285070 bytes and explicitly has `selected_rows=null`, because full population derives from the original source index/selected manifest. This explains the earlier review-tail assertion error without making that assertion pass. The PLAN's hashes are:

- Ordered scene tokens: `9cd301e72bf2a9e46f5f9199dc0322c9fa7db92071e3aea4bed1558c77b89c3a`.
- Entire original selection: `bcbbb5b45ec32dc0c9f767fb5b76250059f2358b61e5c04c7fb9007826715b2a`.
- Finalized request semantic identity: `0c8132c3b3ac97aa8adf88736a46594427f845a03712e34f2e53ea91ce5be57b`.
- Finalized request file SHA: `5f43583a4a0f35a2fc9374d3b6c59442332b8f00721f63bd1ef1fdc9f5bb71ef`.

## Can the existing index runner consume it?

**Yes, using the already implemented direct-request route and reviewed immutable runtime.** `index_teacher_metadata.run` accepts `--requests`, independently of a teacher cache `COMPLETE`. It hashes the request dependency and `build_index` streams its complete semantic identity, inserts each row with original ordinal `pos`, checks each row, preserves order, validates raw-log provenance coverage, and publishes its own immutable index receipt. No fake old `BUILD_COMPLETE`, no teacher payload extraction, and no weakening is needed. The new preparation helper supplies the finalized request path and identity; its exact bytes at integrated commit3dd7eb16 match the reviewed source99ff3af1.

The remaining minimum actual sequence is:

1. Run the reviewed helper's `--check-only` in an explicitly bounded CPU process that includes validator/Torch import allowance. Require both roles `ACTUAL_COMPLETE_BOUND`: TRAIN from the new `FINALIZATION_COMPLETE.json` and separate actual finalization terminal; DEV from unchanged original attempt2 receipts. Preserve original failure pins. The strict shared validator must really execute; do not replace it with this receipt summary.
2. Prepare new index controls using fresh policy/ledger, current CP/free-space obligations and exact TRAIN runtime source pins. Retain7701MiB total planned metadata including both pending indexes once and existing control addenda. Only Lead and the independent resource reviewer may authorize mutation/build. This report neither reads nor approves the live ledger.
3. Build and verify each registered index. TRAIN must report **exactly101592 scenes** and DEV1696; `max-scenes` is only an upper bound and must not be mistaken for an equality check. TRAIN receipt `request_identity` and `dependencies.requests.sha256` must equal the actual finalizer values above; its source-log ledger must match finalized provenance. No opened image/tensor files and no teacher/model creation.
4. Resolve the full-population order limitation using the existing `teacher_provider._validate_full_population` path, after indexes exist, in a separately bounded metadata-only consumer invocation. Supply the new future completion, new future index, original current index, and pinned original `SELECTED_SCENES.json`; use the compatible immutable provider implementation. This routine already streams all selected/current/future rows together, compares `(token,log)`, timestamp and current frame, checks no trailing/missing rows, checks101592 scenes/1176 logs, recomputes ordered-token and full-selection hashes, and validates original source/index/partition identities. Record those actual checks rather than adding a duplicate generic validator or claiming count alone proves order.
5. Record each actual controller terminal, cleanup, cost and resulting index/consumer identities. A partial/failed build never supplies readiness. Earlier synthetic harness errors remain preserved; successful actual full-order consumer evidence is a separate later result.

This sequence needs no student model checkpoint. `source_student_identity` here identifies the immutable data source, not current optimizer progress. The CP commitment receipt is a storage-reservation constraint; it does not require inspecting weight tensors or requalifying the model. Teacher feature contract remains pinned, but model training/teacher extraction are not authorized by metadata completion.

A PROFILE descriptor/runtime still needs the compatible provider new-receipt branch. An older frozen PROFILE runtime cannot be silently patched. Loading the bounded metadata consumer is distinct from creating a teacher executor or calling training; the above does not authorize GPU work or formal training.

## Cost and evidence boundary

This follow-up read only the19,383-byte completion/terminal pair, the285,070-byte original PLAN and small source code; the final evidence capture reread304,453 bytes. Including both receipt reads, two PLAN reads and small source excerpts, reads remain below4MiB. The capture itself used CPU6/7, stdlib only, with costs in `RECORDED_BINDINGS.json`. No actual fullpopulation request/selection/index scan was performed here, so **FULL_ORDER_CONSUMER_PENDING** remains explicit.

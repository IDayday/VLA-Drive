# CPU fit terminal resource audit

Original 300-update CPU fit completed with its frozen W identity preserved. The pinned supervisor and fit receipts agree. Wall 10.826 s, sampled RSS 793,411,584 bytes, charged logical reads 312,542,045 bytes, cleanup 1 verified / 0 live. Terminal fit output is 12,454,544 bytes; controls are 8,504 bytes. No GPU, native model loading or torch.load calls were reported.

The review read only bounded JSON and file metadata. It did not read W tensors, checkpoints or per-scene metric payloads separately, and did not repeat fitting. Physical reads (1,350,967,296 bytes) remain a separate counter. Historical extraction failure is preserved; this does not imply a scientific or generalization PASS.

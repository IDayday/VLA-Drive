# S0 contract

Framework reuse only. No learned DiT/S0 driving weights are loaded.

The original Qwen3-VL2B input pipeline is retained; the flow-matching head is replaced by a singleton Query Decoder. The new Query S0 requires its own GT IL base training before incremental expert cloning. Original auxiliary supervision remains during base training and is excluded from deployment.

Source and all actual dimensions, normalization constants, file hashes, metric and input identities are in S0_CONTRACT.json. There is no output-equivalence claim between the DiT head and the newly initialized Query head.

Framework source: `1493deda247107efe076b9294863a6753391298a`. VLM hidden width: 2048. Scene memory:16x256. Raw trajectory:8x4 normalized XY/sin/cos; physical:8x3 rear-axle relative; dt=0.5s.

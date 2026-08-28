# All-log Candidate Scoring and Target Construction

- Eligible logs/scenes: 1,192 / 45,378
- Complete logs: 1,191/1,192
- Candidate rows: 726,032/726,048 (99.998%)
- Metric rows: 726,032/726,048 (99.998%)
- Official offline scoring success: 100.000%
- v3 target scenes: 45,377/45,378 (99.998%)
- Sampled target schema/leakage audit: 512/512
- Audited failure examples: [{'log_name': '2021.06.09.19.40.26_veh-12_01241_01510', 'scene_token': 'f47b259046405a8d', 'error': 'GEOSException: TopologyException: side location conflict at 664469.44046492805 3997510.1089300122. This can occur if the input geometry is invalid.'}]
- Gate target-v3: PASS

The model-target NPZ files contain no official aggregate score, official factor,
candidate family or candidate index semantics beyond the deterministic tensor row
index. Official scores remain in physically separate offline-evaluation Parquet
files. Dynamic targets are candidate-conditioned relabeling of the shared logged
future under the non-reactive assumption.

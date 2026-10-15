# Process-label feasibility

Read-only audit of the actually locked NAVSIM v1.1 source. No new simulator calls, no process-label model, and no unfreezing of VLA.

Official simulated ego grid: {'dt': 0.1, 'states': 41, 'start': 0.0, 'end': 4.0}; Query output grid: {'dt': 0.5, 'points': 8, 'start': 0.5}.

- at_fault_collision: _collision_time_idcs: earliest attributed collision; inf if none; not all event times
- offroad: _ego_areas[..., NON_DRIVABLE_AREA]: per-simulated-time area test; global DAC uses any(time)
- TTC: _ttc_time_idcs: earliest dangerous current ego simulation time, using projected offsets 0,.3,.6,.9 s
- simulated_ego: _states and _ego_coords on simulator 0.1s grid
- global_scores: _multi_metrics and _weighted_metrics; global scores do not identify a complete event process

earliest event is not per-time hazard; do not assign dense simulated outcomes to nearest 8-point action token

Dense off-road flags are directly available. Full collision/TTC event intervals would require explicit instrumentation of the existing checks; earliest indices alone cannot supply all per-time labels.
The metric reference and future traffic are supervision-only. A future representation-adaptation experiment must keep the fixed-bank diagnosis separate from regeneration and evaluation of changed candidates.
Updating shared E changes D_base(E(x)) even with a frozen Base head. It cannot promise unchanged old candidates and an unchanged single shared forward at the same time.

Implementation hashes: {'scorer': '02ab075017d4edb1c8bba6476ef1674b84d43dbca0f599df87cfe052b2dd5211', 'simulator': 'ae07b533ada10bc37e973e5685a46a7cafc51de8d922ed02add558452ff04807', 'official': '07c3fdf737388f4ee219bba4de5187a2140d8e6411151d686302666be5f063c8', 'cache_processor': '164c6be26ce4b1286d1260886f2124afddc5aae8476692fb43b0120b78ff4895', 'current_annotations': '28f0ee850c35157ab5a9600a3aa68f0f5b0a848b278e7b322b10a000916723f9'}

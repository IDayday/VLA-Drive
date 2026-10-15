"""Read-only process-supervision feasibility audit of the locked simulator source."""
from pathlib import Path
from iqe.contracts import require
from iqe.io import read_json,atomic_json,atomic_bytes,file_hash,digest
from .binding import verify_binding


def run(config):
    binding=verify_binding(config);source=config.source()
    protocol=read_json(source.root/'SCORE_PROTOCOL.json')
    root=Path(source.navsim_root)
    files={name:root/path for name,path in dict(
        scorer='navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py',
        simulator='navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py',
        official='navsim/evaluate/pdm_score.py',
        cache_processor='navsim/planning/metric_caching/metric_cache_processor.py',
        current_annotations='navsim/planning/scenario_builder/navsim_scenario_utils.py').items()}
    text=files['scorer'].read_text()
    expected=['_collision_time_idcs','_ttc_time_idcs','_ego_areas','_ego_coords','_states','NON_DRIVABLE_AREA']
    require(all(v in text for v in expected),'locked simulator process fields changed')
    sampling=protocol['config']['proposal_sampling']
    audit=dict(status='SOURCE_AUDIT_COMPLETE',binding_hash=binding['hash'],score_protocol=protocol['protocol_hash'],
        implementation_hashes={k:file_hash(p) for k,p in files.items()},
        simulator_times=dict(dt=sampling['interval_length'],states=sampling['num_poses']+1,start=0.,end=sampling['num_poses']*sampling['interval_length']),
        action_times=dict(dt=binding['trajectory']['dt'],points=binding['trajectory']['horizon'],start=binding['trajectory']['dt']),
        fields=dict(at_fault_collision='_collision_time_idcs: earliest attributed collision; inf if none; not all event times',
                    offroad='_ego_areas[..., NON_DRIVABLE_AREA]: per-simulated-time area test; global DAC uses any(time)',
                    TTC='_ttc_time_idcs: earliest dangerous current ego simulation time, using projected offsets 0,.3,.6,.9 s',
                    simulated_ego='_states and _ego_coords on simulator 0.1s grid',
                    global_scores='_multi_metrics and _weighted_metrics; global scores do not identify a complete event process'),
        timestamp_warning='earliest event is not per-time hazard; do not assign dense simulated outcomes to nearest 8-point action token',
        input_label_separation=True,process_fields_as_deployment_inputs=False,
        new_feedback_exports=0,new_process_models=0,export_budget_limit_if_needed=512,
        prior_metric_t0_caveat='one-frame objects can be repeated into observation[0] by cache interpolation; require original metadata timestamp==t0',
        joint_training_not_executed=True)
    atomic_json(config.root/'PROCESS_LABEL_FEASIBILITY.json',audit,immutable=True)
    lines=['# Process-label feasibility','',
        'Read-only audit of the actually locked NAVSIM v1.1 source. No new simulator calls, no process-label model, and no unfreezing of VLA.',
        '',f"Official simulated ego grid: {audit['simulator_times']}; Query output grid: {audit['action_times']}.",
        '',*['- '+k+': '+v for k,v in audit['fields'].items()],
        '',audit['timestamp_warning'],
        '', 'Dense off-road flags are directly available. Full collision/TTC event intervals would require explicit instrumentation of the existing checks; earliest indices alone cannot supply all per-time labels.',
        'The metric reference and future traffic are supervision-only. A future representation-adaptation experiment must keep the fixed-bank diagnosis separate from regeneration and evaluation of changed candidates.',
        'Updating shared E changes D_base(E(x)) even with a frozen Base head. It cannot promise unchanged old candidates and an unchanged single shared forward at the same time.',
        '', 'Implementation hashes: '+str(audit['implementation_hashes'])]
    atomic_bytes(config.root/'PROCESS_LABEL_FEASIBILITY.md',('\n'.join(lines)+'\n').encode(),immutable=True)
    return dict(status='COMPLETE',report=str(config.root/'PROCESS_LABEL_FEASIBILITY.md'),new_scoring_calls=0)

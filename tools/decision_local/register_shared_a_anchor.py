"""Register only the completed, exposure-matched A-only S0 endpoint.

The historical anchor supplies immutable dataset descriptors, never weights.
Model-runtime SHA, wrapper SHA, fitted teacher and normalization are explicit.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','tag','registration','population-contract','historical-anchor','output'):
        p.add_argument('--'+k,required=True)
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('_a_anchor_helpers',Path(__file__).with_name('common.py'))
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    h.check_host()
    if Path(a.output).exists():raise FileExistsError('Anchor registrations are immutable')
    contract=h.read(a.population_contract);registered=h.read(a.registration);old=h.read(a.historical_anchor)
    audit=h.read(contract['source_audit']);proof=h.read(contract['recording_frame_audit'])
    if (contract['schema']!='a_only_s0_training_population_v1' or contract['role']!='A_train' or
        contract['source_audit_identity']!=audit['identity'] or not proof['population_ready'] or
        proof['source_audit_identity']!=audit['identity']):raise ValueError('Unverified A-only source population')
    legacy=Path(old['source_root'])
    if h.source_sha(legacy)!='1493deda247107efe076b9294863a6753391298a':raise ValueError('Wrong S model runtime')
    sys.path.insert(0,str(legacy))
    from tools.foresight.checkpoints import checkpoint_identity
    training,checkpoint=checkpoint_identity(a.run,a.tag)
    complete=h.read(Path(a.run)/'checkpoints'/a.tag/'COMPLETE.json')
    status=h.read(Path(a.run)/'status.json')
    config=training['config'];provenance=config['natural_error_provenance']
    if (status['status']!='COMPLETE' or training['scope']!='formal' or
        training['source_sha']!=registered['training_source_sha'] or
        training['registration_sha256']!=h.digest(a.registration) or
        complete['completed']!=registered['updates'] or complete['exposure']!=registered['expected_exposure'] or
        checkpoint['completed']!=training['updates'] or training['scene_count']!=registered['scene_count'] or
        training['selected_index_hash']!=contract['index_identity'] or
        provenance['population_contract_sha256']!=h.digest(a.population_contract)):
        raise ValueError('Shared base has not completed the registered common exposure')
    if (config['foresight']['planner_condition_mode']!='action_only' or
        config['foresight']['future_action_condition']!='none' or
        config['foresight']['future_target_type']!='dino_sequence'):
        raise ValueError('This registration covers original S0 only')
    init=h.read(Path(a.run)/'initialization.json')
    if init['driving_weights_loaded']:raise ValueError('A-only base inherited a driving checkpoint')
    if training['data']['population_contract']!=contract:raise ValueError('Actual trained population differs')
    if h.digest(contract['action_codec'])!=registered['action_codec_sha256']:raise ValueError('Changed A-only codec')
    codec=h.read(contract['action_codec'])
    if codec['identity']!=contract['action_codec_identity']:raise ValueError('Wrong action codec identity')
    indices={}
    for role in ('A_train','A_calibration','B_predictor','B_refiner'):
        path=Path(contract['source_audit']).parent/(role+'.json')
        rows=h.read(path)
        if any(audit['roles'][r['log'].rsplit('_',2)[0]]!=role for r in rows):raise ValueError('Cross-role recording')
        indices[role]={'path':str(path),'sha256':h.digest(path),'scenes':len(rows),'index_identity':h.identity(rows)}
    for row in old['data'].values():
        if h.read(Path(row['current_root'])/'identity.json')!=row['identity']:raise ValueError('Current inputs changed')
    meters=list((Path(registered['campaign_root'])/'runs').glob('*/status.json'))
    own=[h.read(x) for x in meters if h.read(x).get('run_id_parent')==registered['run_id']]
    declaration=dict(schema='natural_error_shared_A_anchor_v1',registered_anchor='one_shared_A_only_S0_terminal_not_reselected',
        training_role='A_train',source_root=str(legacy),runtime_source_sha=h.source_sha(legacy),
        training_source_sha=training['source_sha'],run=a.run,tag=a.tag,checkpoint=checkpoint,
        training_identity=training['identity'],data=old['data'],role_indices=indices,
        source_audit_identity=audit['identity'],source_audit=contract['source_audit'],
        population_contract=a.population_contract,population_contract_sha256=h.digest(a.population_contract),
        recording_separation_proof_sha256=h.digest(contract['recording_frame_audit']),
        action_codec=contract['action_codec'],action_codec_identity=codec['identity'],
        action_codec_sha256=h.digest(contract['action_codec']),
        teacher_identity=training['interaction']['teacher_identity'],fitted_teacher_source=contract['teacher_training_identity'],
        original_base_source_sha=old['training_source_sha'],original_base_weights_loaded=False,
        base_training_GPU_hours=sum(x.get('gpu_hours',0.) for x in own),base_training_meters=[str(x) for x in meters if h.read(x).get('run_id_parent')==registered['run_id']],
        shared_base_training_seed=config['seed'],training_exposure=complete['exposure'],
        precision='FP32 native optimizer master reconstruction, strict=True, eval, TF32 off',
        proposal_seeds_train=[42,43],primary_sampling_seed=42,registration_source_sha=h.source_sha(ROOT))
    declaration['identity']=h.identity(declaration);h.atomic_json(a.output,declaration)
    print(declaration['identity'])


if __name__=='__main__':main()

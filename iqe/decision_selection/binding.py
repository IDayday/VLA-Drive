from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from iqe.contracts import require
from iqe.io import atomic_json, read_json, digest, file_hash, lock
from iqe.specialist_study.bank import pool
from .data import ROLES, canonical_bank


def r00_prediction_source(config, role):
    root = Path(config.reference_run)
    choice = read_json(root/'MODEL_CHOICE.json')
    row = next(r for r in choice['cells'] if r['cell'] == 'R00')
    require(file_hash(row['checkpoint']) == row['checkpoint_hash'], 'R00 checkpoint changed')
    if role == 'stage_val':
        receipt = root/'scorers/R00'/f"validation_{row['step']:06d}.json"
        path = root/'scorers/R00'/f"predictions_stage_val_{row['step']:06d}.npz"
        r = read_json(receipt)
        require(r['scorer_state_hash'] == row['state_hash'], 'R00 selected validation identity')
        checksum = r['prediction_hash']
    else:
        receipt = root/'evaluation/R00'/role/'COMPLETE.json'; r = read_json(receipt)
        path = receipt.parent/'predictions.npz'; checksum = r['checksum']
        require(r['identity']['checkpoint_hash'] == row['checkpoint_hash'], 'R00 prediction checkpoint mismatch')
        require(r['identity']['role'] == role and r['identity']['selection_hash'] == file_hash(root/'MODEL_CHOICE.json'), 'R00 prediction role/choice')
    require(file_hash(path) == checksum, 'R00 prediction artifact corrupted')
    return dict(path=str(path), checksum=checksum, receipt=str(receipt), receipt_hash=file_hash(receipt),
                checkpoint=row['checkpoint'], checkpoint_hash=row['checkpoint_hash'], endpoint=row['step'])


def bind(config):
    source = config.source()
    with lock(config.root/'BIND.lock'):
        path = config.root/'SELECTION_STUDY_BINDING.json'
        if path.exists():
            r = read_json(path)
            require(r['config_hash'] == config.identity and r['hash'] == digest({k:v for k,v in r.items() if k != 'hash'}), 'binding/config conflict')
            for p,h in r['artifacts'].items(): require(file_hash(p) == h, 'bound artifact changed: '+p)
            return r
        base = read_json(source.root/'BASE_BINDING.json'); p = pool(source)
        require(base['checkpoint_hash'] == config.expected_base_hash and p['hash'] == config.expected_pool_hash, 'wrong Base70k/fixed pool')
        require(base['optimizer_step'] == 70000 and len(p['experts']) == 3, 'fixed Query70k and three existing experts')
        require(file_hash(base['checkpoint']) == base['checkpoint_hash'], 'Base weights corrupted')
        require(file_hash(base['checkpoint_receipt']) == base['checkpoint_receipt_hash'], 'Base receipt changed')
        artifacts = {}
        for f in ('BASE_BINDING.json','OFFLINE_BASE_CONTRACT.json','CAMPAIGN_BINDING.json','SCORE_PROTOCOL.json',
                  'SERVING_CANDIDATE_POOL.json','DATA_PARTITION.json','scenes.json','BASE_PLANNER.pt','BASE_PLANNER.json','resolved_config.json'):
            artifacts[str(source.root/f)] = file_hash(source.root/f)
        experts = []
        for e in p['experts']:
            require(file_hash(e['checkpoint']) == e['checksum'], 'expert checkpoint corrupted')
            artifacts[e['checkpoint']] = e['checksum']
            experts.append({k:e[k] for k in ('expert_id','cell','checkpoint','checksum','model_hash','endpoint','step','private_reader')})
        banks = {}; reference = {}
        for role in ROLES:
            meta, _, _ = canonical_bank(config, role, verify_records=True)
            folder = source.root/'banks/deploy'/role
            for f in ('BANK.json','arrays.npz'): artifacts[str(folder/f)] = file_hash(folder/f)
            banks[role] = {k:meta[k] for k in ('hash','array_hash','requested_scenes','scene_ids','source_logs','expert_ids','canonical_order','score_protocol_hash')}
            reference[role] = r00_prediction_source(config, role)
            artifacts[reference[role]['path']] = reference[role]['checksum']
            artifacts[reference[role]['receipt']] = reference[role]['receipt_hash']
        geom = Path(config.geometric_run)
        final = geom/'FINAL_RESULTS.json'
        if not final.exists(): final = geom/'evaluation/FINAL_RESULTS.json'
        geometric = dict(status='NOT_VERIFIED', root=str(geom))
        # Existing result only: never start an old finalizer from this study.
        if final.exists():
            geometric.update(status='EXISTING_ADDITIONAL_RESULT', report=str(final), checksum=file_hash(final))
            artifacts[str(final)] = file_hash(final)
        contract = read_json(source.root/'OFFLINE_BASE_CONTRACT.json')
        repo = Path(__file__).parents[2]
        r = dict(schema='fixed_K4_decision_selection_v1', config_hash=config.identity, config=asdict(config),
                 created_utc=datetime.now(timezone.utc).isoformat(), audit_commit='1375ad2f67ed841f44bc7e6ec42bcc0f6d301bf0',
                 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                 Base=base, experts=experts, pool_hash=p['hash'], expert_ids=['expert_0']+sorted(e['expert_id'] for e in experts),
                 trajectory=contract['trajectory'], runtime=base['inference_runtime'], artifacts=artifacts,
                 banks=banks, R00=reference, geometric=geometric,
                 new_experts=0, generator_updates=0, new_navtest=False, ACTIVE_updated=False,
                 models=['M_scene','M_set'], output_semantics='decision_logits/probabilities; not predicted PDMS',
                 data_source='official candidate rewards; no network teacher',
                 planned_budget=dict(diagnostic=2*3000,refit_max=2*5000,overfit_max=2*300,current_state_probe_max=2000),
                 calibration=dict(units='decision_logit_difference',grid=[0.,.25,.5,1.,2.],new_NC_budget=0,new_DAC_budget=0,
                                  fallback='always_Base_is_not_success'),
                 scientific_rule='report all fixed outcomes; no dev-driven refit or checkpoint reselection')
        r['hash'] = digest(r); atomic_json(path,r,immutable=True)
        return r


def verify_binding(config):
    r = read_json(config.root/'SELECTION_STUDY_BINDING.json')
    require(r['config_hash'] == config.identity and r['hash'] == digest({k:v for k,v in r.items() if k!='hash'}), 'study binding changed')
    # These small authoritative identities transitively bind the immutable banks.
    for f in ('BASE_BINDING.json','SERVING_CANDIDATE_POOL.json','DATA_PARTITION.json','OFFLINE_BASE_CONTRACT.json','SCORE_PROTOCOL.json'):
        p = str(config.source().root/f); require(file_hash(p) == r['artifacts'][p], 'source identity changed')
    return r

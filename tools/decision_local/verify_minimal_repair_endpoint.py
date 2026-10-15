"""Check the versioned B-fold common endpoint before its locked final Navtest.

This does not choose a model by score, launch training, or relax the previous
full-population decision_local endpoint contract. It covers this distinct study.
"""
import argparse,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import read,digest,atomic_json,source_sha,check_host


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method-lock',type=Path,required=True)
    p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();check_host();lock=read(a.method_lock);registry=read(a.registry)
    if lock['schema']!='minimal_repair_final_lock_v1' or lock['main_endpoint_epochs']!=24 or lock['proposal_sampling_seed']!=42:
        raise ValueError('Wrong versioned common endpoint')
    if registry['population']!='navtest' or registry['method_lock_sha256']!=digest(a.method_lock):
        raise ValueError('Wrong final registry or changed lock')
    locked={x['id']:x for x in lock['camera_only_runs']}
    if set(locked)!={x['id'] for x in registry['jobs']}:raise ValueError('Final jobs differ from locked models')
    missing=[];checkpoints={};training=[];development=[]
    for path in lock['training_registries']:
        for job in read(path)['jobs']:
            root=Path(job['output']);complete=root/'COMPLETE.json';ckpt=root/'epoch_24.pt'
            if not complete.exists() or not ckpt.exists():missing.append(str(complete));continue
            info=read(root/'identity.json');done=read(complete);fold=info['repair_fold'];n=fold['scenes']
            if (info['protocol']!='minimal_repair_v1' or info['frozen_base_optimizer_updates']!=0 or
                info['source_sha']!=job['training_source_sha'] or info['anchor_identity']!=lock['anchor_identity'] or
                done['epoch']!=24 or done['completed']!=math.ceil(n/128)*24 or done['exposure']!=n*24 or
                info['batch']!=128 or info['microbatch']!=16 or info['compute_precision']!='FP32' or info['TF32']):
                raise ValueError('Incomplete or foreign registered training endpoint')
            checkpoints[str(ckpt)]=digest(ckpt);training.append({'id':job['id'],'source_sha':info['source_sha'],'fold':fold})
    for path in lock['development_registries']:
        for job in read(path)['jobs']:
            complete=Path(job['scores'])/'COMPLETE.json'
            if not complete.exists():missing.append(str(complete));continue
            done=read(complete);meta=read(Path(job['scores'])/'identity.json');pred=read(Path(job['predictions'])/'identity.json')
            if meta['population']!='dev' or done['identity']!=meta['identity'] or meta['prediction_identity']!=pred['identity']:
                raise ValueError('Foreign development evaluation protocol')
            for variant in ('q0','q_final'):
                if not done[variant]['valid'] or done[variant]['failed'] or done[variant]['scenes']!=1696:
                    raise ValueError('Incomplete development population')
            if job['epoch']==24:
                if job['checkpoint'] not in checkpoints:missing.append(job['checkpoint']);continue
                if pred['checkpoint_sha256']!=checkpoints[job['checkpoint']]:raise ValueError('Different evaluated checkpoint')
                development.append(job['scores'])
    verification=[]
    for job in registry['jobs']:
        entry=locked[job['id']];checkpoint=entry['checkpoint'];proof_path=Path(entry['online_acceptance'])
        if checkpoint not in checkpoints or not proof_path.exists():missing.append(str(proof_path));continue
        if job['checkpoint']!=checkpoint or checkpoints[checkpoint]!=entry['checkpoint_sha256']:
            raise ValueError('Locked final checkpoint changed')
        if entry['kind'] not in ('LOG_NONE','MIN_NONE','R_MIN_CURRENT','R_MIN_ABS','R_MIN_DIFF'):
            raise ValueError('Oracle cannot enter camera-only final table')
        proof=read(proof_path);online=proof['online_receipt']
        if proof['checkpoint_sha256']!=checkpoints[checkpoint] or online['scenes']<4 or not online['python_training_file_access_guard'] or online['opened_GT_relation_or_feature_cache']:
            raise ValueError('Unverified current-image deployment')
        if len(proof['comparisons'])<4 or any(not v['exact'] for row in proof['comparisons'].values() for v in row.values()):
            raise ValueError('Current-image and cache outputs differ')
        if any(any(x[k]!=1 for k in ('proposal','qwen_language','qwen_visual')) for x in online['actual_module_forward_checks']):
            raise ValueError('Deployment did not use exactly one frozen S0 forward')
        if digest(Path(job['predictions']).parent/'FINAL_METHOD_LOCK.json')!=digest(a.method_lock):
            raise ValueError('Export does not see the same immutable method lock')
        verification.append({'path':str(proof_path),'sha256':digest(proof_path)})
    feature_path=Path(lock['navtest_features']);features=read(feature_path/'identity.json');feature_done=read(feature_path/'COMPLETE.json')
    if (features['identity']!=lock['navtest_feature_identity'] or features['anchor_identity']!=lock['anchor_identity'] or
        features['split']!='navtest' or len(features['index'])!=12146 or feature_done['scenes']!=12146):
        raise ValueError('Incomplete or foreign unchanged Navtest feature population')
    decision=read(lock['stage_b_decision'])
    if decision['status'] not in ('NOT_RUN','COMPLETE') or not decision['evidence']:
        raise ValueError('Conditional stage must have an explicit evidence-based disposition')
    result={'accepted':not missing,'pending':sorted(set(missing)),'method_lock_sha256':digest(a.method_lock),
        'training_runs':training,'common_development_endpoints':development,'online_verification':verification,
        'checkpoint_sha256':checkpoints,'source_sha':source_sha(ROOT),'tool_sha256':digest(__file__),
        'no_score_threshold_or_checkpoint_selection':True,'scope':'Frozen S0, new heads only; complete registered A/B-fold population per head, not full-VLA OOF'}
    atomic_json(a.output if not missing else a.output.with_name('READINESS_PENDING.json'),result)
    print('accepted' if not missing else 'pending '+str(len(set(missing))))


if __name__=='__main__':main()

"""Freeze the user-requested fastest single main run, preserving the old matrix plan."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.campaign import charged_gpu_hours
from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256


def checked_extra_host(candidate, world, host, inventory):
    """Only the explicitly authorized C4 extension can add rl-zt2 to this run."""
    if candidate != 'C4':
        if host or inventory:
            raise ValueError('Additional-host registration is only for the requested C4 run')
        return None
    if world != 8 or host != 'training-rl-zt2' or inventory is None:
        raise ValueError('C4 requires the explicitly authorized standalone rl-zt2 inventory')
    if (inventory['host_alias'] != host or inventory['gpu_count'] != 8 or
            not inventory['all_gpus_free'] or
            {g['index'] for g in inventory['gpus']} != set(range(8)) or
            len({g['uuid'] for g in inventory['gpus']}) != 8 or
            any(g['memory_total_mib'] < 81920 or g['memory_used_mib'] >= 512
                for g in inventory['gpus'])):
        raise ValueError('Missing/occupied/mismatched authorized eight-A80080GB allocation')
    return {'host_alias': host, 'hostname': inventory['hostname'],
            'gpu_uuids': [g['uuid'] for g in inventory['gpus']],
            'authorization': inventory['authorization']}


def main():
    p = argparse.ArgumentParser(__doc__)
    for key in ('previous-registration', 'training-worktree', 'decision', 'output', 'campaign-root', 'run-id'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--candidate',choices=('C0','C1','C4'),default='C0',
                   help='C0/C1/C4 share144-query student shape; C4 requires explicit extra-host inventory')
    p.add_argument('--additional-host',choices=('training-rl-zt2',))
    p.add_argument('--resource-inventory')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():raise FileExistsError('Immutable registration already exists')
    work = Path(a.training_worktree)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=work).strip():raise ValueError('Training source must be clean')
    old = json.loads(Path(a.previous_registration).read_text())
    if identity_hash({k:v for k,v in old.items() if k != 'identity'}) != old['identity']:raise ValueError('Old registration hash changed')
    decision = json.loads(Path(a.decision).read_text())
    if decision['objective'] != 'fastest_single_complete_C0' or not decision['measurement_complete']:
        raise ValueError('Actual full-model speed evidence required')
    world = decision['selected_world_size']
    if world not in (8,16):raise ValueError('Use only the two authorized eight-GPU servers')
    if a.candidate!='C0' and world!=8:raise ValueError('Companion only uses the spare standalone host')
    extra_host = checked_extra_host(a.candidate,world,a.additional_host,
        json.loads(Path(a.resource_inventory).read_text()) if a.resource_inventory else None)
    if file_sha256(decision['selected_profile_path']) != decision['selected_profile_sha256']:
        raise ValueError('Measured profile changed')
    profile = json.loads(Path(decision['selected_profile_path']).read_text())
    if not profile['measurement_complete'] or profile['world_size'] != world or profile['candidate'] != 'C0' or not profile['all_four_losses']:
        raise ValueError('Wrong measured main experiment')
    reg = copy.deepcopy(old);reg.pop('identity')
    reg.update(schema='ddp_full_priority_registration_v2', created_unix=time.time(),
        parent_registration_sha256=file_sha256(a.previous_registration),
        training_source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=work,text=True).strip(),
        nodes=world//8, priority='Complete C0 first; C1 companion can use the second standalone host',
        runs={a.run_id:dict(candidate=a.candidate,seed=42,updates=old['updates'],schedule_updates=old['updates'],
                           global_batch=32,gpus=world,micro_batch=32//world)},
        topology_decision=decision, topology_decision_sha256=file_sha256(a.decision),
        planned_optimizer_updates=old['updates'],
        estimated_campaign_gpu_hours=charged_gpu_hours(Path(a.campaign_root)) + profile['optimizer_gpu_hours_per_1000_updates']*old['updates']/1000*1.35+64,
        old_matrix_status='PAUSED_USER_PRIORITY; old run weights and ledgers preserved, not reused for initialization',
        screen_behavior='25000 remains a common observation point; priority C0 continues to100000 without waiting for matrix ranking',
        secondary_budget='Remaining C1-C5, second seeds and ablations deferred until main result; no extra resource authorization')
    if reg['estimated_campaign_gpu_hours'] > reg['gpu_hours_cap']:raise ValueError('Measured main plan exceeds existing cap')
    if extra_host:
        reg.update(priority='User-authorized C4 on rl-zt2; existing C0/C1 continue unchanged',
                   additional_host=extra_host,
                   resource_inventory_sha256=file_sha256(a.resource_inventory),
                   authorized_hosts=sorted(set(old['authorized_hosts']) | {extra_host['host_alias']}),
                   old_matrix_status='C0/C1 remain active; C4 starts independently from generic/random initialization',
                   screen_behavior='Same100000 schedule and fixed development milestones as C0/C1',
                   measured_cost_scope='C0/C1/C4 have identical144W/8x6target student shapes; inherited8GPU cost estimate, actual rl-zt2 throughput observed in formal run')
    reg['identity'] = identity_hash(reg);atomic_json(out,reg)
    print(json.dumps({'registration':str(out),'identity':reg['identity'],'world_size':world,'estimated_gpu_hours':reg['estimated_campaign_gpu_hours']}))


if __name__ == '__main__':main()

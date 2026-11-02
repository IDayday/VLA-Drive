"""Distinct-scene exploration diagnostics; no oracle is reported as deployment."""
import torch
import torch.distributed as dist

STRICT=('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort')


def strict(record):
    return all(float(record['metrics'][k])==1. for k in STRICT) and float(record['metrics']['score'])>=.9


def window_counts(window):
    names=('valid_scenes','sampled_G16_strict_good','sampled_main_group_strict_good',
        'nominal_B1_scored','nominal_B1_strict_good','sampled_good_nominal_bad',
        'rescue_enabled','extra_G32_scenes','extra_G32_strict_good','nominal_zero','main_group_all_zero')
    counts=dict.fromkeys(names,0)
    for s in window['samples']:
        if not s['valid_scene']:continue
        good=any(strict(r) for r in s['records']);prefix=any(strict(r) for r in s['records'][:16])
        counts['valid_scenes']+=1;counts['sampled_G16_strict_good']+=int(prefix)
        counts['sampled_main_group_strict_good']+=int(good)
        counts['main_group_all_zero']+=int(all(float(r['metrics']['score'])==0. for r in s['records']))
        p=s.get('nominal_improvement',{});nominal=p.get('nominal')
        if nominal:
            b1good=strict(nominal['feedback']);counts['nominal_B1_scored']+=1
            counts['nominal_B1_strict_good']+=int(b1good);counts['sampled_good_nominal_bad']+=int(good and not b1good)
            counts['nominal_zero']+=int(float(nominal['feedback']['metrics']['score'])==0.)
        counts['rescue_enabled']+=int(p.get('enabled',False))
        if p.get('extra'):
            counts['extra_G32_scenes']+=1
            counts['extra_G32_strict_good']+=int(good or any(strict(r) for r in p['extra']['records']))
    values=torch.tensor(list(counts.values()),device=window['rewards'].device,dtype=torch.int64)
    if dist.is_initialized():dist.all_reduce(values)
    return dict(zip(names,values.cpu().tolist()))

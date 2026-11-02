"""Positive verified targets are supervised off-policy, never put in PG ratio."""
import torch
import torch.nn.functional as F


def verified_index(records, nominal, *, improvement=.02):
    eligible=[]
    for i,r in enumerate(records):
        m=r['metrics']
        if (all(m[k]==1 for k in ('no_at_fault_collisions','drivable_area_compliance',
                'time_to_collision_within_bound','comfort')) and m['score']>=.90
                and m['score']-nominal>=improvement):eligible.append(i)
    return max(eligible,key=lambda i:(records[i]['metrics']['score'],-i)) if eligible else None


def positive_loss(prediction, teacher):
    return F.smooth_l1_loss(prediction.float(),teacher.detach().float(),beta=.1)

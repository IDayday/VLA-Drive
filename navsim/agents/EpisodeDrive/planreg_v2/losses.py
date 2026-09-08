import torch
import torch.distributed as dist
from .normalizers import wrap_angle
from .predictor import state_norm


def global_valid_mean(values,valid,total_count=None):
    """Global value AND correct gradient under DDP's average reduction, including empty ranks."""
    numerator = torch.where(valid,values,0.).sum()
    denominator = valid.sum().to(values.dtype) if total_count is None else torch.as_tensor(total_count,device=values.device,dtype=values.dtype)
    world = 1
    global_numerator = numerator.detach().clone()
    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size()
        if total_count is None: dist.all_reduce(denominator)
        dist.all_reduce(global_numerator)
    den = denominator.clamp_min(1)
    return global_numerator/den + (numerator-numerator.detach())*world/den


def trajectory_stage_loss(proposals,target,valid,std,total_count=None):
    """Physical WTA followed by sum-coordinate/time-mean scaled regression."""
    target=torch.where(valid[...,None],target,0.)
    error = torch.where(valid[:,None,:,None],proposals,0.)-target[:,None]
    error = torch.cat((error[...,:2],wrap_angle(error[...,2:3])),-1)
    error = torch.where(valid[:,None,:,None],error,0.)
    distance = error.abs().sum(-1).sum(-1)/valid.sum(-1).clamp_min(1)[:,None]
    selected = distance.argmin(-1)
    chosen = error[torch.arange(len(error),device=error.device),selected]
    value = (chosen.abs()/std).sum(-1).sum(-1)/valid.sum(-1).clamp_min(1)
    return global_valid_mean(value,valid.any(-1),total_count)


def trajectory_loss(stages,targets,std,weights=(.25,.25,.25,.25),long_weight=1.,valid_counts=None):
    losses = []
    for proposals in stages:
        original = trajectory_stage_loss(proposals,targets['trajectory'],targets['trajectory_valid'],std,
                                        None if valid_counts is None else valid_counts['trajectory'])
        long_valid = targets['trajectory_long_valid'][:,None].expand(-1,8)
        extended = trajectory_stage_loss(proposals,targets['trajectory_long'],long_valid,std,
                                        None if valid_counts is None else valid_counts['long'])
        losses.append(original+long_weight*extended)
    return sum(w*l for w,l in zip(weights,losses)),losses


def world_model_loss(tf,ro,targets,token_valid,future_valid,action_valid,valid_counts=None):
    target_mask=future_valid[:,:,None,None]&token_valid[:,None,:,None]
    target = state_norm(torch.where(target_mask,targets,0.)).detach()
    prefix_actions = action_valid.long().cumprod(-1).bool()
    tf_valid = future_valid & prefix_actions
    tf_valid[:,1:] &= future_valid[:,:-1].long().cumprod(-1).bool()
    ro_valid = future_valid & prefix_actions
    def branch(pred,valid,label):
        token_error = (pred-target).abs().mean(-1)
        # A sample with many tiles does not receive more weight.
        error = (token_error*token_valid[:,None]).sum(-1)/token_valid.sum(-1).clamp_min(1)[:,None]
        valid = valid & token_valid.any(-1)[:,None]
        total=None if valid_counts is None else valid_counts[label].sum()
        return global_valid_mean(error,valid,total),[global_valid_mean(error[:,i],valid[:,i],
            None if valid_counts is None else valid_counts[label][i]) for i in range(3)]
    tf_loss,tf_each = branch(tf,tf_valid,'tf')
    ro_loss,ro_each = branch(ro,ro_valid,'ro')
    return dict(wm_tf_loss=tf_loss,wm_ro_loss=ro_loss,wm_loss=tf_loss+ro_loss,
                **{'wm_tf_'+label:value for label,value in zip(('0p5','1p5','4p0'),tf_each)},
                **{'wm_ro_'+label:value for label,value in zip(('0p5','1p5','4p0'),ro_each)})


def wm_weight(step,total):
    return .01+.09*min(1.,max(0.,step/max(1,total*.10)))

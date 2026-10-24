"""Same FP32 AdamW arithmetic, locally sharded states and explicit parent map."""
import copy
import torch
from torch.distributed.optim import ZeroRedundancyOptimizer
from .il_optimization import parameter_groups


def build_optimizer(model,cfg):
    groups=parameter_groups(model,cfg)
    head=list(model.current_dino_head.parameters());ids={id(p) for p in head}
    for group in groups:group['params']=[p for p in group['params'] if id(p) not in ids]
    groups.append(dict(params=head,lr=cfg['learning_rates']['action_interfaces'],name='current_dino'))
    return ZeroRedundancyOptimizer(groups,optimizer_class=torch.optim.AdamW,
        betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'],
        foreach=False,fused=False,overlap_with_ddp=False,parameters_as_bucket_view=False)


def local_parameter_names(model,optimizer):
    names={id(p):n for n,p in model.named_parameters()}
    return [[names[id(p)] for p in g['params']] for g in optimizer.optim.param_groups]


def inherit_parent_adam(model,optimizer,payload):
    old=payload['optimizer'];groups=optimizer.param_groups
    if len(old['param_groups'])!=3 or len(groups)!=4:raise ValueError('Explicit original3 plus new head group required')
    state=copy.copy(old);state['state']=dict(old['state']);state['param_groups']=copy.deepcopy(old['param_groups'])
    offset=0
    for old_group,new_group in zip(old['param_groups'],groups[:3]):
        if old_group['name']!=new_group['name'] or len(old_group['params'])!=len(new_group['params']):
            raise ValueError('Parent parameter group/order changed')
        if old_group['params']!=list(range(offset,offset+len(new_group['params']))):
            raise ValueError('Parent global Adam index contract changed')
        offset+=len(new_group['params'])
    head={k:v for k,v in groups[3].items() if k!='params'}
    head['params']=list(range(offset,offset+len(groups[3]['params'])));head['initial_lr']=groups[3]['lr']
    state['param_groups'].append(head)
    # Installed torch2.5 partitions this full state into the deterministic local
    # optimizer; no moments are fabricated for the two new projection tensors.
    optimizer.load_state_dict(state)
    # torch2.5 load_state_dict also populates the exposed Optimizer.state with
    # duplicate local moments. step() uses optimizer.optim exclusively. Our
    # checkpoints store that exact local state, so release only the duplicates.
    optimizer.state.clear()
    del state


def validate_local_adam(model,optimizer,updates,parent_steps):
    names={id(p):n for n,p in model.named_parameters()}
    for group in optimizer.optim.param_groups:
        for p in group['params']:
            n=names[id(p)];expected=parent_steps.get(n,0)+updates;s=optimizer.optim.state.get(p)
            if expected and (s is None or float(s['step'])!=expected):raise ValueError('Adam count mismatch: '+n)
            if s:
                if s['exp_avg'].dtype!=torch.float32 or s['exp_avg_sq'].dtype!=torch.float32:
                    raise ValueError('Adam moments must remain FP32')

"""Optional light head1-3 supervision; the fixed scorer loss is untouched."""
import torch


def intermediate_trajectory_loss(proposal_list, targets):
    if len(proposal_list) != 5:
        raise ValueError('Light deep supervision requires exactly four refinement stages')
    losses = []
    for proposals in proposal_list[1:4]:
        loss = torch.linalg.vector_norm(proposals - targets['trajectory'][:,None], ord=1, dim=-1).mean(-1).amin(1).mean()
        if 'trajectory_long' in targets:
            loss += torch.linalg.vector_norm(proposals - targets['trajectory_long'][:,None], ord=1, dim=-1).mean(-1).amin(1).mean()
        losses.append(loss)
    return .2 * torch.stack(losses).mean()

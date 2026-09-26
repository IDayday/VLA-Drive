import torch


def token_positions(input_ids, token_ids):
    ids = torch.as_tensor(token_ids,device=input_ids.device,dtype=input_ids.dtype)
    if ids.ndim != 1 or len(ids) != len(torch.unique(ids)) or (ids < 0).any():
        raise ValueError('Token IDs must be nonnegative and unique')
    matches = input_ids[...,None] == ids[None,None]
    if not (matches.sum(1) == 1).all():
        raise ValueError('Required token missing or repeated')
    return matches.long().argmax(1)


def insert_world_tokens(ids, embeddings, mask, positions, action_positions, world, placeholder_id):
    """Insert continuous text-type queries before actions; preserve native visual grids."""
    b,l,h = embeddings.shape
    n = world.shape[1]
    new_ids=[];new_embeddings=[];new_masks=[];new_positions=[];world_positions=[]
    for i in range(b):
        at = int(action_positions[i,0])
        if at <= 0 or at >= l or not mask[i,at]:
            raise ValueError('Invalid action insertion position')
        if not (positions[:,i,at] == positions[0,i,at]).all():
            raise ValueError('Action insertion must be at a text position')
        new_ids.append(torch.cat([ids[i,:at],ids.new_full((n,),placeholder_id),ids[i,at:]]))
        new_embeddings.append(torch.cat([embeddings[i,:at],world[i].to(embeddings.dtype),embeddings[i,at:]]))
        new_masks.append(torch.cat([mask[i,:at],mask.new_ones(n),mask[i,at:]]))
        pos = positions[:,i,at,None] + torch.arange(n,device=ids.device)[None]
        suffix = positions[:,i,at:] + n
        new_positions.append(torch.cat([positions[:,i,:at],pos,suffix],-1))
        world_positions.append(torch.arange(at,at+n,device=ids.device))
    return torch.stack(new_ids),torch.stack(new_embeddings),torch.stack(new_masks),torch.stack(new_positions,1),torch.stack(world_positions),action_positions+n

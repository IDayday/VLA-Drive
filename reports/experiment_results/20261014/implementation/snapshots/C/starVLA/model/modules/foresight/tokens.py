"""Continuous text-query replacement with explicit batch shape and unique slots."""
import torch


def replace_query_embeddings(embeddings,positions,values):
    if embeddings.ndim!=3 or positions.ndim!=2 or positions.dtype!=torch.long:
        raise ValueError('Embedding/query position contract')
    batch,length,hidden=embeddings.shape
    if positions.shape[0]!=batch or values.shape[-2:]!=(positions.shape[1],hidden):
        raise ValueError('Query value/position shape mismatch')
    if ((positions<0)|(positions>=length)).any() or (positions[:,1:]<=positions[:,:-1]).any():
        raise ValueError('Query positions must be ordered, unique and in bounds')
    if values.ndim==2:values=values[None].expand(batch,-1,-1)
    if values.shape!=(batch,positions.shape[1],hidden):raise ValueError('Query batch mismatch')
    # PyTorch2.5.1 deterministic CUDA index_put cannot reliably broadcast a
    # [1,Q,H] RHS over multiple batch indices. Explicit scatter has no implicit
    # value broadcast and no duplicate writes, including in its backward.
    return embeddings.scatter(1,positions[...,None].expand(-1,-1,hidden),values.to(embeddings.dtype))

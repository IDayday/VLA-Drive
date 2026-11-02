"""Explicit D adaptation of CFM-energy ratio; NOT a trajectory likelihood."""
import torch


def velocity_energy(state, prediction, target, sigma):
    if (sigma<=0).any():raise ValueError('Positive sigma required')
    velocity=(state.float()-prediction.float())/sigma.float()[:,None,None]
    return (velocity-target.detach().float()).square().mean((-2,-1))


def awm_loss(energy, advantage, reference_prediction, prediction, reference_coefficient=1.):
    # Conservative positive-only adaptation, NOT the full signed AWM theorem.
    weight=advantage.detach().float().clamp(0,5)
    positive=(weight*energy).mean()
    keep=(prediction.float()-reference_prediction.detach().float()).square().mean()
    return positive+reference_coefficient*keep

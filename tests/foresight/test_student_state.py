import torch
from tools.foresight.student_state import optimizer_batch_counts,learning_rate


def test_loss_denominators_use_entire_optimizer_batch_and_valid_views():
    valid=torch.zeros(3,3,3,dtype=torch.bool)
    valid[0,0,:2]=True;valid[1,2,0]=True  # third scene stays in ego despite no auxiliary label
    targets={'ego':torch.zeros(3,8,4),'future_latent':torch.zeros(3,3,3,2,4,6),'future_valid':valid,
             'interaction_latent':torch.zeros(3,8,512),'interaction_valid':torch.tensor([True,False,True])}
    counts=optimizer_batch_counts(targets,torch.Generator().manual_seed(42),'cpu')
    assert counts=={'ego_scenes':3.,'visual':3*2*4*6.,'interaction':2*8*512.}
    assert targets['visual_horizon'][:2].tolist()==[0,2]
    # Explicit task RNG calls cannot change the main Torch noise RNG.
    before=torch.get_rng_state().clone()
    optimizer_batch_counts(targets,torch.Generator().manual_seed(93),'cpu')
    assert torch.equal(before,torch.get_rng_state())


def test_schedule_continues_across_pause_and_respects_declared_horizon():
    uninterrupted=[learning_rate(i,1e-5,5e-7,2,8) for i in range(8)]
    resumed=[learning_rate(i,1e-5,5e-7,2,8) for i in list(range(4))+list(range(4,8))]
    assert uninterrupted==resumed and uninterrupted[0]==5e-6
    assert uninterrupted[4]>uninterrupted[7]>5e-7

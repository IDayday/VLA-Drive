"""Crash window tests execute the real trainer; no synthetic claims about S0."""
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from iqe.io import atomic_json, atomic_torch, file_hash, read_json
from iqe.training.trainer import train
from iqe.training.checkpoint import capture_rng, restore_rng
from iqe.data.sampler import ConsumedSampler
from test_training_resume_ddp import cfg, plan, network, fetch, loss_fn, denominator


def fail_result(monkeypatch):
    import iqe.training.trainer as trainer
    original = trainer.atomic_json
    def write(path, *args, **kw):
        if Path(path).name == 'result.json':
            raise OSError('FAULT_AFTER_TERMINAL_CHECKPOINT')
        return original(path, *args, **kw)
    monkeypatch.setattr(trainer, 'atomic_json', write)


@pytest.mark.parametrize('early', [True, False])
@pytest.mark.parametrize('legacy', [True, False])
def test_terminal_crash_does_not_consume_or_update(tmp_path, monkeypatch, early, legacy):
    config = cfg(4) | {'validation_every_steps': 1, 'early_stop_patience': 1}
    validation = (lambda m, s: {'selection_key': [-s]}) if early else None
    module = network(.3)
    with monkeypatch.context() as patch:
        fail_result(patch)
        with pytest.raises(OSError, match='FAULT_AFTER'):
            train(module, loss_fn, fetch, ConsumedSampler(plan()), config, {'data':'unit'}, tmp_path,
                  validate=validation, denominator_function=denominator)
    step = 2 if early else 4
    path = tmp_path/f'step_{step:06d}.pt'
    saved = torch.load(path, weights_only=False)
    expected = 'EARLY_STOPPED' if early else 'MAX_STEPS'
    assert saved['trainer_state']['termination']['termination_reason'] == expected
    if legacy:
        del saved['trainer_state']['termination']
        atomic_torch(path, saved)
        receipt = read_json(str(path)+'.COMPLETE.json'); receipt['checksum']=file_hash(path)
        atomic_json(str(path)+'.COMPLETE.json', receipt)
    checksum = file_hash(path)
    sampler = ConsumedSampler(plan())
    restored = network(.3)
    def forbidden(*a): raise AssertionError('terminal resume consumed a batch')
    history = train(restored, loss_fn, forbidden, sampler, config, {'data':'unit'}, tmp_path, resume='auto',
                    validate=validation, denominator_function=denominator)
    assert history == [] and file_hash(path) == checksum
    assert sampler.state_dict() == saved['sampler']
    assert torch.equal(capture_rng()['cpu'], saved['rng_by_rank'][0]['cpu'])
    assert capture_rng()['python'] == saved['rng_by_rank'][0]['python']
    assert all(torch.equal(restored.state_dict()[k], v) for k,v in saved['module'].items())
    result=read_json(tmp_path/'result.json')
    assert result['new_optimizer_steps'] == 0 and result['termination']['termination_reason'] == expected
    if early:
        assert result['best_checkpoint_hash'] == saved['trainer_state']['best_snapshot']['checksum']


def test_operator_pause_is_resumable(tmp_path):
    config=cfg(4)
    train(network(),loss_fn,fetch,ConsumedSampler(plan()),config,{'data':'unit'},tmp_path,stop_after=2)
    saved=torch.load(tmp_path/'step_000002.pt',weights_only=False)
    assert saved['trainer_state']['termination']['termination_reason']=='OPERATOR_PAUSED'
    history=train(network(),loss_fn,fetch,ConsumedSampler(plan()),config,{'data':'unit'},tmp_path,resume='auto')
    assert [r['optimizer_step'] for r in history]==[3,4]


def terminal_worker(rank, rendezvous, folder):
    import iqe.training.trainer as trainer
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    torch.manual_seed(9)
    config=cfg(2)|{'validation_every_steps':1,'early_stop_patience':1}
    original=trainer.atomic_json
    def failure(path,*a,**kw):
        if Path(path).name=='result.json':raise OSError('FAULT_AFTER_TERMINAL_CHECKPOINT')
        return original(path,*a,**kw)
    trainer.atomic_json=failure
    try:
        train(network(.2),loss_fn,fetch,ConsumedSampler(plan()),config,{'data':'unit'},folder,
              validate=lambda m,s:{'selection_key':[-s]})
    except (OSError,ValueError):pass
    else:raise AssertionError('fault not delivered')
    trainer.atomic_json=original
    def forbidden(*a):raise AssertionError('terminal DDP fetched')
    history=train(network(.2),loss_fn,forbidden,ConsumedSampler(plan()),config,{'data':'unit'},folder,resume='auto')
    assert history==[]
    dist.destroy_process_group()


def test_gloo_terminal_resume_all_ranks(tmp_path):
    mp.spawn(terminal_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path/'run')),nprocs=2,join=True)

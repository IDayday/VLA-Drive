import os
from pathlib import Path
import subprocess
import sys
import torch
import pytest


def test_global_gap_stream_objective_two_process_gloo(tmp_path):
    output=tmp_path/'result.pt'
    worker=Path(__file__).with_name('internal_ddp_worker.py')
    result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',str(worker),str(output)],
                          env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONPATH=str(Path(__file__).resolve().parents[2])),capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    assert torch.load(output,weights_only=False)['status']=='PASS'


@pytest.mark.skipif(os.environ.get('IQE_RUN_GPU_TESTS')!='1',reason='requires two authorized idle GPUs')
def test_global_gap_stream_objective_two_process_nccl(tmp_path):
    from iqe.resources import qualify
    qualify('cuda')
    output=tmp_path/'nccl_result.pt'
    worker=Path(__file__).with_name('internal_ddp_worker.py')
    result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',str(worker),str(output)],
                          env=dict(os.environ,IQE_INTERNAL_DDP_BACKEND='nccl',OMP_NUM_THREADS='1',
                                   PYTHONPATH=str(Path(__file__).resolve().parents[2])),capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    state=torch.load(output,weights_only=False)
    assert state['status']=='PASS' and state['backend']=='nccl'

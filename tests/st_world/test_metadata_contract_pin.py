"""Narrow contract receipt atomic-replacement/size regression; synthetic only."""
from copy import deepcopy
from pathlib import Path
import pytest

from tools.st_world_policy import index_teacher_metadata as cli
from starVLA.dataloader.st_world_targets import atomic_json
from starVLA.dataloader.st_world_history import file_sha256
from tests.st_world.test_ondemand_teacher import fixture


def inputs(tmp_path):
    c,m=fixture(tmp_path)
    request=tmp_path/'requests.json'; atomic_json(request,m)
    receipt=tmp_path/'contract.json'
    approved={'schema':'st_fixed_teacher_qualification_plan_v1','contract':c.to_dict()}
    atomic_json(receipt,approved)
    args=cli.parser().parse_args(['build','--requests',str(request),
        '--expected-request-identity',m['identity'],'--contract-receipt',str(receipt),
        '--contract-receipt-sha256',file_sha256(receipt),'--output',str(tmp_path/'index'),
        '--max-scenes','1'])
    return args,approved


def test_contract_replacement_during_read_rejected(tmp_path,monkeypatch):
    args,approved=inputs(tmp_path)
    original=Path.open
    replacement=deepcopy(approved); replacement['contract']['teacher_identity']='9'*64
    class SwappingRead:
        def __init__(self,stream): self.stream=stream
        def __enter__(self): return self
        def __exit__(self,*unused): self.stream.close()
        def read(self,n):
            data=self.stream.read(n)
            atomic_json(args.contract_receipt,replacement)
            return data
    def intercept(self,*a,**kw):
        f=original(self,*a,**kw)
        return SwappingRead(f) if self==args.contract_receipt and a==('rb',) else f
    monkeypatch.setattr(Path,'open',intercept)
    with pytest.raises(ValueError,match='changed'): cli.run(args)
    assert not (args.output/'COMPLETE.json').exists()


def test_contract_parses_pinned_bytes_without_reopening(tmp_path,monkeypatch):
    args,approved=inputs(tmp_path)
    actual=Path.open; reads=[]
    def counted(self,*a,**kw):
        if self==args.contract_receipt: reads.append(a)
        return actual(self,*a,**kw)
    monkeypatch.setattr(Path,'open',counted)
    assert cli.run(args)['scene_count']==1
    assert reads==[('rb',)]


@pytest.mark.parametrize('kind',['changed_sha','oversize','empty'])
def test_invalid_contract_pin_never_publishes(tmp_path,kind):
    args,_=inputs(tmp_path)
    if kind=='changed_sha': args.contract_receipt_sha256='a'*64
    elif kind=='oversize': args.contract_receipt.write_bytes(b'x'*(cli.MiB+1))
    else: args.contract_receipt.write_bytes(b'')
    with pytest.raises(ValueError,match='checksum|small pinned'): cli.run(args)
    assert not (args.output/'COMPLETE.json').exists()

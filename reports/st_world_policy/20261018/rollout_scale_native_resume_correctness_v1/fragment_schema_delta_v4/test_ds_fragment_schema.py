"""Real installed dataclass; only tiny trusted fixture pickle roundtrips."""
import copy
import importlib.util
import json
from pathlib import Path
import pickle

import pytest
from deepspeed.utils.tensor_fragment import fragment_address

SOURCE=Path('/mnt/project/VLA-Drive-st-rollout-native-resume-smoke-v4-20261018/resources/st_rollout_resume_smoke_v1.py')
spec=importlib.util.spec_from_file_location('independent_real_fragment_v4',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


@pytest.mark.parametrize('numel,start',[(393216,0),(0,0)])
def test_real_dataclass_stable_order_tiny_roundtrip_and_both_fields_matter(numel,start):
    # 393216/0 is the integer-opcode witness from the explicitly approved native
    # data.pkl. No tensor of this size is allocated; this is two integer fields.
    obj=fragment_address(start=start,numel=numel)
    before=dict(vars(obj));encoded=m.logical_state(obj)
    assert encoded=={'kind':'deepspeed_fragment_address_v1','type':'deepspeed.utils.tensor_fragment.fragment_address','numel':numel,'start':start}
    assert m.logical_state(copy.deepcopy(obj))==encoded
    # Only our own tiny class fixture is unpickled, never the native checkpoint.
    assert m.logical_state(pickle.loads(pickle.dumps(obj)))==encoded
    assert json.loads(json.dumps(encoded,sort_keys=True))==encoded and vars(obj)==before
    obj.start=start+1;assert m.logical_state(obj)!=encoded
    obj.start=start;obj.numel=numel+1;assert m.logical_state(obj)!=encoded


def test_fake_same_qualified_name_is_rejected_before_inspecting_unknown_fields():
    def trap(self):raise AssertionError('Unknown object attributes must never be inspected')
    Fake=type('fragment_address',(),{'__module__':'deepspeed.utils.tensor_fragment','__dict__':property(trap),
        'start':property(trap),'numel':property(trap)})
    with pytest.raises(ValueError,match='Unknown'):m.logical_state(Fake())

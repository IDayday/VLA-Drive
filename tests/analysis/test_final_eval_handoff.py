import json
import signal
from pathlib import Path
import pytest
from scripts.analysis.final_eval_handoff import training_finished, retire_controller


def test_no_signal_before_native_success_and_complete_checkpoint(tmp_path):
    result = tmp_path / 'result.json'; checkpoint = tmp_path / 'checkpoint';checkpoint.mkdir()
    assert training_finished(result, checkpoint, 12912) is False
    result.write_text(json.dumps({'status':'FAIL','exit_codes':[1]}))
    with pytest.raises(RuntimeError): training_finished(result, checkpoint, 12912)
    result.write_text(json.dumps({'status':'PASS','exit_codes':[0]}))
    with pytest.raises(ValueError, match='complete'): training_finished(result, checkpoint, 12912)
    (checkpoint/'COMPLETE').write_text('complete')
    (checkpoint/'trainer_state.json').write_text(json.dumps({'update':12911,'world_size':8}))
    with pytest.raises(ValueError, match='budget'): training_finished(result, checkpoint, 12912)
    (checkpoint/'trainer_state.json').write_text(json.dumps({'update':12912,'world_size':8}))
    assert training_finished(result, checkpoint, 12912)


def test_signal_only_exact_owned_controller(tmp_path):
    process=tmp_path/'123';process.mkdir();calls=[]
    command=b'python\0-m\0scripts.analysis.accelerated_epoch_run\0--spec\0/owned.json\0'
    (process/'cmdline').write_bytes(command)
    retire_controller(123,Path('/owned.json'),proc_root=tmp_path,send=lambda *a:calls.append(a))
    assert calls==[(123,signal.SIGINT)]
    with pytest.raises(ValueError,match='unrecognized'):
        retire_controller(123,'/other.json',proc_root=tmp_path,send=lambda *a:calls.append(a))
    assert len(calls)==1
    assert retire_controller(456,'/owned.json',proc_root=tmp_path,send=lambda *a:calls.append(a))=='already_exited'

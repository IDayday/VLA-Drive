"""The deployment command must never inherit qualification/RL weights."""
import pytest
from tools.d46_augustus_ops.qualify_aligned import guard_new_root,formal_command


def test_nonempty_formal_root_is_never_reinitialized(tmp_path):
    out=tmp_path/'formal';out.mkdir();(out/'identity.json').write_text('already used')
    with pytest.raises(ValueError,match='Nonempty'):guard_new_root(out,[])
    guard_new_root(out,[{'durable':'existing launch'}])


@pytest.mark.parametrize('exp,hosts',[('AUG3_FULL_ALIGNED',['a','b','c','d']),('AUG4_LORA_ALIGNED',['e','f'])])
def test_formal_is_same_config_fresh_new_root_without_qualification_resume(exp,hosts):
    r=dict(experiment=exp,hosts=hosts,master_addr='private',config='cfg',formal_output='new_formal')
    command=formal_command(r,0)
    assert '--qualification'not in command and '--resume'not in command
    assert command[command.index('--output')+1]=='new_formal'
    assert '--nnodes='+str(len(hosts))in command

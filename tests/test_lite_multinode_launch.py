"""Resource changes do not silently shrink the declared DDP world."""
from pathlib import Path
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def resolve(nodes, peers=None):
    env = {k:v for k,v in os.environ.items() if k not in ('PLANREG_PEER_HOSTS','PLANREG_PEER_HOST')}
    if peers is not None:
        env['PLANREG_PEER_HOSTS'] = peers
    return subprocess.run(['bash','-c',
        'source local_planreg_wm_v1/formal_runtime.sh; '
        f'planreg_formal_resolve_peers {nodes} || exit $?; '
        'printf "%s\\n" "${PLANREG_RUNTIME_PEERS[@]}"'],
        cwd=ROOT, env=env, capture_output=True, text=True)


def test_two_node_legacy_peer_default():
    result=resolve(2)
    assert result.returncode==0 and result.stdout.strip()=='training-vla-zt2'


def test_four_nodes_require_all_explicit_peers():
    assert resolve(4).returncode==2
    assert resolve(4,'a,b').returncode==2
    result=resolve(4,'a,b,c')
    assert result.returncode==0 and result.stdout.splitlines()==['a','b','c']


def test_duplicate_and_shell_peer_names_rejected():
    assert resolve(4,'a,b,a').returncode==2
    assert resolve(2,'a;false').returncode==2


def test_launchers_have_no_hardcoded_two_node_torchrun():
    for name in ('benchmark_formal_common.sh','formal_launch_common.sh'):
        path=ROOT/'local_planreg_wm_v1'/name
        subprocess.run(['bash','-n',str(path)],check=True)
        text=path.read_text()
        assert '--nnodes=2' not in text
        assert '--nnodes="${num_nodes}"' in text

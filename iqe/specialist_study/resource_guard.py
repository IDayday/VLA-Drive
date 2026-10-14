"""Respect the shared workspace's live host policy before dispatching work."""
import os,socket
from pathlib import Path
from iqe.io import read_json,file_hash,atomic_json
from iqe.contracts import require


def check(config,command):
    path=Path('/mnt/project/server_dispatch_policy.json')
    if not path.exists():return {'policy':'not_present_on_portable_host','hostname':socket.gethostname()}
    policy=read_json(path);host=socket.gethostname()
    allowed={name for name in policy['allowed_hosts']}
    for item in policy['allowed_hosts'].values():allowed.update(item.get('aliases',[]))
    require(host in allowed,'host forbidden by current server dispatch policy: '+host)
    if command in ('train-expert-cell','train-scorer-cell','cache-score-context','run-specialist-scorer-study'):
        require(not host.startswith('training-vla-zt2-') and host!='training-vla-zt2','Base training host is reserved')
    receipt={'hostname':host,'policy_hash':file_hash(path),'command':command,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
             'no_other_process_signals':True}
    atomic_json(config.root/'runtime_resource_checks'/(command+'_'+host+'.json'),receipt)
    return receipt

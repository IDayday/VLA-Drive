"""Execute the registered TRAIN monitor with an immutable real action policy."""
import argparse
import copy
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from starVLA.rl.constrained_flow_noise.contracts import require_dispatch, file_sha
from starVLA.rl.constrained_flow_noise.condition_cache import ConditionStore
from starVLA.rl.constrained_flow_noise.loading import make_noise_policy
from starVLA.rl.constrained_flow_noise.monitor import run_monitor
from starVLA.rl.constrained_flow_noise.rollout_graph import configure
from starVLA.rl.constrained_flow_noise.safety_costs import ConstrainedRewardService
from starVLA.rl.flow_grpo import distributed as control
from starVLA.rl.flow_grpo.reproducibility import configure_numerics


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('variant','population','condition-cache','calibration','registration','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--metric-cache',action='append',required=True);p.add_argument('--checkpoint')
    a=p.parse_args();require_dispatch();configure_numerics()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('nccl');control.initialize_control_group(1800)
    rank,world=dist.get_rank(),dist.get_world_size()
    if world!=8:raise ValueError('Monitor requires registered independent eight-rank execution')
    pop,cal,reg=[json.loads(Path(path).read_text()) for path in (a.population,a.calibration,a.registration)]
    store=ConditionStore(a.condition_cache,json.loads((Path(a.condition_cache)/'identity.json').read_text()),readonly=False)
    actor,_,_=make_noise_policy(a.variant,cal['noise_std_bounds'])
    if actor.noise_head is not None:
        if file_sha(cal['noise_initial_path'])!=cal['noise_initial_sha256']:raise ValueError('Initial noise identity changed')
        actor.noise_head.load_state_dict(torch.load(cal['noise_initial_path'],weights_only=True,map_location='cpu'),strict=True)
    reference=copy.deepcopy(actor).eval().requires_grad_(False)
    multipliers=torch.tensor([1.,1.,1.,.1,.1]);update=raw=0
    if a.checkpoint:
        root=Path(a.checkpoint)
        if not (root/'COMPLETE').exists():raise ValueError('Incomplete checkpoint')
        identity=json.loads((root/'identity.json').read_text())
        if identity['variant']!=a.variant or identity['population_identity']!=pop['identity_sha256']:raise ValueError('Checkpoint population/variant changed')
        weight=next((root/'pytorch_model').glob('*model_states.pt'))
        actor.load_state_dict(torch.load(weight,map_location='cpu',weights_only=False,mmap=True)['module'],strict=True)
        state=torch.load(root/f'rank{rank}.pt',map_location='cpu',weights_only=False)['state']
        from starVLA.rl.constrained_flow_noise.lagrange import CostMultipliers
        dual=CostMultipliers(cal['budgets']);dual.load_state_dict(state['dual']);multipliers=dual.snapshot()
        update,raw=state['updates'],state['raw_cursor']
    actor.eval().requires_grad_(False);configure(actor,True)
    version=dict(variant=a.variant,checkpoint=str(a.checkpoint),optimizer_update=update,raw_exposure=raw,
                 condition=store.identity_sha,calibration_sha256=file_sha(a.calibration),kernel=vars(actor.spec),mode='eval',collection='monitor')
    service=ConstrainedRewardService(dict(repo=str(Path.cwd()),maps_root='/mnt/navsim/maps',cache_roots=a.metric_cache),workers=2)
    try:run_monitor(actor,reference,store,service,pop,reg,cal,multipliers,version,a.output,rank,world)
    finally:service.close();dist.destroy_process_group()


if __name__=='__main__':main()

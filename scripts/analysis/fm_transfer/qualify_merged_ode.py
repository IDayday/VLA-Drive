"""Merged LoRA must preserve both full ten-step single-candidate ODE readouts."""
import argparse
import json
from pathlib import Path
import torch
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.fm_transfer.rl_adapter import make_transfer_actor
from starVLA.rl.fm_transfer.model import TransferPolicy,make_policy
from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters,merged_state_dict
from starVLA.rl.simwam_recipe.contracts import RAW_CONDITION,POPULATION,file_sha,seed_for
from starVLA.rl.constrained_flow_noise.condition_cache import ConditionStore
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();host=require_dispatch();configure_numerics();torch.cuda.set_device(0)
    checkpoint=Path(args.checkpoint);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    seal=json.loads((checkpoint/'COMPLETE').read_text());files=json.loads((checkpoint/'files.json').read_text())
    if (file_sha(checkpoint/'files.json')!=seal['files_sha256'] or
            file_sha(checkpoint/'optimizer.pt')!=files['optimizer.pt'] or
            file_sha(checkpoint/'identity.json')!=files['identity.json']):raise ValueError('Changed native RL checkpoint')
    identity=json.loads((checkpoint/'identity.json').read_text())
    if identity['purpose']!='qualification_not_formal_initialization':raise ValueError('Qualification lineage required')
    sft=identity['recipe']['supervised_transfer']['SFT_identity']
    actor,loaded=make_transfer_actor(sft['checkpoint'],identity['recipe'],qualification=True)
    if sft!=loaded:raise ValueError('Changed supervised reference')
    state=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=False)
    restore_adapters(actor,state['adapters'])
    architecture=sft['config']['architecture'];wrapped=TransferPolicy(actor.action_expert,architecture).eval()
    plain=make_policy(architecture,device='cuda').eval().requires_grad_(False)
    plain.head.load_state_dict(merged_state_dict(actor.action_expert),strict=True)
    cache=ConditionStore(RAW_CONDITION);tokens=json.loads(Path(POPULATION).read_text())['train_tokens'][:2];checks=[]
    for token in tokens:
        raw=cache.get(token).cuda().float()
        root=torch.randn((1,8,4),device='cuda',dtype=torch.bfloat16,
            generator=torch.Generator(device='cuda').manual_seed(seed_for('original_ode_initial',2101,token)))
        for grid in ('native','shift5'):
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                before=wrapped.sample_ode(raw,root,grid=grid);after=plain.sample_ode(raw,root,grid=grid)
            torch.testing.assert_close(before,after,rtol=1e-4,atol=1e-4)
            checks.append(dict(grid=grid,max_abs=float((before-after).abs().max())))
    result=dict(status='PASS',host=host,architecture=architecture,checkpoint=str(checkpoint),
        updates=seal['updates'],checks=checks,scope='actual nonzero real-reward LoRA, complete ODE before/after merge',
        uses_training_only_conditions=True,qualification_only=True,driving_effect_claim=False)
    atomic_json(out,result);print(json.dumps(result))


if __name__=='__main__':main()

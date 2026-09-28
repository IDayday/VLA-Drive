"""Load the real public Qwen once per arm and compare every shared driving tensor."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import torch
from omegaconf import OmegaConf
from starVLA.model.framework.DDPForesight import DDPForesight
from starVLA.model.modules.vehicle_joint.initialization import module_manifest,tensor_hash,identity_hash
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config-root',default='configs/foresight')
    p.add_argument('--output',required=True);p.add_argument('--campaign-root',required=True);p.add_argument('--run-id',required=True)
    a=p.parse_args();torch.set_num_threads(4);output=Path(a.output)
    if output.exists():raise FileExistsError('Evidence is immutable')
    with metered_run(a.campaign_root,a.run_id,0,{'kind':'real_generic_initialization_check','real_optimizer_updates':0}):
        arms={};shared=None;query=None
        for arm in ('R','A','B','C','D'):
            config=OmegaConf.load(Path(a.config_root)/(arm+'.yaml'));torch.manual_seed(7832)
            before=torch.get_rng_state().clone();model=DDPForesight(config)
            if not torch.equal(before,torch.get_rng_state()):raise AssertionError('Model construction consumed caller main RNG')
            inventory={'action':module_manifest(model.action_model),'state':module_manifest(model.action_input_model),
                       'driving_tokens':model.qwen_vl_interface.driving_token_initialization}
            # Full common Qwen blocks are pinned public tensors; also compare their
            # loaded values. Exclude embedding rows reserved for the added W tokens.
            language=model.qwen_vl_interface.model.model.language_model
            common={n:tensor_hash(t) for n,t in language.named_parameters() if 'embed_tokens' not in n}
            signature=identity_hash({'driving':inventory,'language':common})
            if shared is None:shared=signature
            elif signature!=shared:raise AssertionError('Shared initialization differs for '+arm)
            W=tensor_hash(model.foresight_queries) if hasattr(model,'foresight_queries') else None
            if W is not None:
                if query is None:query=W
                elif query!=W:raise AssertionError('W initialization differs')
            arms[arm]={'shared_parameter_hash':signature,'W_hash':W,
                'full_training_parameters':sum(t.numel() for t in model.parameters()),
                'trainable_parameters':sum(t.numel() for t in model.parameters() if t.requires_grad),
                'main_rng_preserved':True,'driving_weights_loaded':False,
                'auxiliary_heads':[key for key in ('future_head','interaction_head') if hasattr(model,key)]}
            del model;gc.collect()
        atomic_json(output,{'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                           'real_generic_models':True,'arms':arms,'passed':True})


if __name__=='__main__':main()
